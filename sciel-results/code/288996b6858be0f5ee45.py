import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
A_REF = np.sqrt(3) / 4.0

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_x = None
    best_score = -np.inf
    
    def get_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Dirichlet distribution for barycentric coords
        # Shape (N, 3)
        bary = rng.dirichlet(np.ones(N), size=N).T
        return bary.flatten()

    def cartesian_from_barycentric(x):
        u = x[0::3]
        v = x[1::3]
        w = x[2::3]
        px = v + 0.5 * w
        py = (np.sqrt(3)/2) * w
        return np.column_stack((px, py))

    def compute_areas_and_grads(points):
        """
        Returns array of areas (size M) and list of gradients for each area.
        Gradients are (3N, ) vectors.
        """
        areas = []
        grads = []
        triples = list(combinations(range(N), 3))
        
        for i, j, k in triples:
            p1, p2, p3 = points[i], points[j], points[k]
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            
            # Determinant D
            D = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
            A = 0.5 * abs(D)
            sign = 1.0 if D >= 0 else -1.0
            
            areas.append(A)
            
            grad = np.zeros(3*N)
            
            # dD/dx1 = -(y3-y1), dD/dy1 = (x3-x1)
            # dD/dx2 = (y3-y1), dD/dy2 = -(x3-x1)
            # dD/dx3 = -(y2-y1), dD/dy3 = (x2-x1)
            
            # Note: indices in x are 3*i, 3*i+1, 3*i+2
            # But x is barycentric!
            # We need gradient w.r.t barycentric coords u,v,w.
            # P = (v + 0.5w, (sqrt(3)/2)w)
            # dP/du = (0,0)
            # dP/dv = (1,0)
            # dP/dw = (0.5, sqrt(3)/2)
            
            # So grad_bary = J^T grad_cart
            # J = [[0, 1, 0.5], [0, 0, sqrt(3)/2]] for each point
            
            # Let's compute grad_cart first
            gc1 = np.array([-sign * 0.5 * (y3-y1), sign * 0.5 * (x3-x1)])
            gc2 = np.array([sign * 0.5 * (y3-y1), -sign * 0.5 * (x3-x1)])
            gc3 = np.array([-sign * 0.5 * (y2-y1), sign * 0.5 * (x2-x1)])
            
            # Map to barycentric
            # du: 0
            # dv: gc_x
            # dw: 0.5*gc_x + sqrt(3)/2*gc_y
            
            # For point 1
            grad[3*i+1] = gc1[0]
            grad[3*i+2] = 0.5*gc1[0] + (np.sqrt(3)/2)*gc1[1]
            
            # For point 2
            grad[3*j+1] = gc2[0]
            grad[3*j+2] = 0.5*gc2[0] + (np.sqrt(3)/2)*gc2[1]
            
            # For point 3
            grad[3*k+1] = gc3[0]
            grad[3*k+2] = 0.5*gc3[0] + (np.sqrt(3)/2)*gc3[1]
            
            grads.append(grad)
            
        return np.array(areas), grads

    def smooth_min_and_grad(x, tau):
        points = cartesian_from_barycentric(x)
        areas, grads = compute_areas_and_grads(points)
        
        # Smooth min: -tau log sum exp(-A/tau)
        # Shift for stability
        max_A = np.max(areas)
        exp_terms = np.exp(-(areas - max_A) / tau)
        sum_exp = np.sum(exp_terms)
        val = max_A - tau * np.log(sum_exp)
        
        # Grad: - (1/sum_exp) * sum (exp_term * (1/tau) * grad_A) * tau ?
        # f = -tau log(S)
        # df/dx = -tau * (1/S) * dS/dx
        # dS/dx = sum (d exp(-A_i/tau)/dx) = sum (-1/tau exp(-A_i/tau) grad_A_i)
        # df/dx = -tau * (1/S) * sum (-1/tau exp_i grad_i) = (1/S) sum (exp_i grad_i)
        
        weights = exp_terms / sum_exp
        grad = np.sum([w * g for w, g in zip(weights, grads)], axis=0)
        
        return -val, -grad # We want to maximize min area, so minimize -val

    def objective(x):
        # Annealing tau? 
        # For simplicity in this step, use a fixed small tau
        # But SLSQP might struggle if tau is too small and landscape is rough
        val, grad = smooth_min_and_grad(x, 1e-3)
        return val, grad

    A_eq = np.zeros((N, 3*N))
    b_eq = np.ones(N)
    for i in range(N):
        A_eq[i, 3*i] = 1.0
        A_eq[i, 3*i+1] = 1.0
        A_eq[i, 3*i+2] = 1.0
    
    cons = ({'type': 'eq', 'fun': lambda x: A_eq.dot(x) - b_eq, 'jac': lambda x: A_eq})
    bounds = [(0, 1) for _ in range(3*N)]

    n_starts = 30
    for s in range(n_starts):
        if time.time() - start_time > time_limit:
            break
            
        x0 = get_initial_guess(s)
        
        try:
            res = minimize(objective, x0, method='SLSQP', jac=True, bounds=bounds, constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                x_opt = res.x
                # Evaluate exact min area
                points = cartesian_from_barycentric(x_opt)
                min_a = np.inf
                for i in range(N):
                    for j in range(i+1, N):
                        for k in range(j+1, N):
                            p1, p2, p3 = points[i], points[j], points[k]
                            area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                            if area < min_a:
                                min_a = area
                if min_a > best_score:
                    best_score = min_a
                    best_x = x_opt
        except Exception:
            pass

    if best_x is None:
        x0 = get_initial_guess(0)
        points = cartesian_from_barycentric(x0)
    else:
        points = cartesian_from_barycentric(best_x)
        
    return points
