import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    def compute_facet_radii(pts):
        """
        Computes the exact covering radius for each facet of the convex hull.
        Returns an array of shape (num_facets,) with the local covering radius for each facet.
        Also returns the active vertices for each facet to allow gradient computation if needed,
        but here we just need the values for the soft-max.
        """
        try:
            hull = ConvexHull(pts)
            radii = np.zeros(len(hull.simplices))
            for i, simplex in enumerate(hull.simplices):
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    radii[i] = 0.0
                    continue
                c_s = normal / norm
                
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                max_d_circ = max(d1, d2, d3)
                
                # Edge midpoints
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                else:
                    local_rad = max_d_edge

                radii[i] = local_rad
            return radii
        except Exception:
            return None

    def soft_max_objective(x_flat):
        """
        Objective function for L-BFGS-B.
        x_flat: (3N+1,) array. First 3N are coordinates, last is t (covering radius estimate).
        We minimize t + beta * log-sum-exp(radii - t).
        This is an epigraph formulation with a smooth penalty.
        """
        if time.time() - start_time > time_limit - 2.0:
            return np.inf, np.zeros(3*N + 1)
            
        x = x_flat[:3*N].reshape(N, 3)
        t = x_flat[3*N]
        
        # Normalize points to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        radii = compute_facet_radii(x)
        if radii is None:
            return np.inf, np.zeros(3*N + 1)
        
        # Soft-max approximation of max(radii)
        # log-sum-exp(radii - t) is small if t >= max(radii), large if t < max(radii)
        # We want to push t down to max(radii).
        
        # Stability: subtract max
        max_r = np.max(radii)
        if max_r > t:
            # If t is below max radii, the penalty should be large
            # Use a stable log-sum-exp
            a = radii - t
            a_max = np.max(a)
            if a_max > 50: # Overflow protection, though t should be close
                lse = a_max + np.log(np.sum(np.exp(a - a_max)))
            else:
                lse = np.log(np.sum(np.exp(a)))
            
            # Objective: t + lambda * lse
            # If lambda is small, it approximates max(t, max_radii)
            # Actually, min_t [ t + 1/beta * log(sum(exp(beta*(r_i - t))) ) ]
            # As beta -> inf, this approaches max(r_i).
            # Let's use a fixed beta (inverse temperature)
            beta = 50.0 
            # Note: The standard soft-max is (1/beta) log sum exp(beta x).
            # Here we have log sum exp(beta (r_i - t)).
            # Let's stick to the form: f(t, x) = t + (1/beta) log(sum(exp(beta (r_i - t))))
            # This is convex in t for fixed x.
            
            obj = t + (1.0/beta) * lse
            grad_t = 1.0 - (1.0/beta) * beta * np.sum(np.exp(beta*(radii - t) - lse)) 
            # Wait, derivative of (1/beta) log(sum(exp(beta(r_i - t)))) w.r.t t
            # = (1/beta) * (1/sum) * sum(exp * (-beta))
            # = - sum(exp(beta(r_i - t)) / sum(exp(beta(r_i - t)))
            # = - softmax(beta(r_i - t))
            
            # So grad_t = 1 - softmax_sum
            # Since sum(softmax) = 1, grad_t = 1 - 1 = 0? No.
            # Let's re-derive.
            # L(t) = t + (1/beta) log( sum_i exp(beta(r_i - t)) )
            # dL/dt = 1 + (1/beta) * (1/sum) * sum_i exp(beta(r_i - t)) * (-beta)
            # dL/dt = 1 - sum_i [ exp(beta(r_i - t)) / sum_j exp(beta(r_j - t)) ]
            # The term in brackets is a probability distribution p_i. Sum p_i = 1.
            # So dL/dt = 1 - 1 = 0.
            # This implies the minimum is at the point where the gradient is 0, which is always true?
            # No, the derivative of log-sum-exp is the softmax.
            # Let S = sum exp(beta(r_i - t)).
            # d/dt log S = (1/S) * dS/dt = (1/S) * sum exp(beta(r_i - t)) * (-beta) = -beta * (sum exp / S) = -beta.
            # So d/dt [(1/beta) log S] = -1.
            # dL/dt = 1 - 1 = 0.
            # This means the function is linear in t?
            # Let's check.
            # (1/beta) log( sum exp(beta r_i) exp(-beta t) )
            # = (1/beta) log( exp(-beta t) sum exp(beta r_i) )
            # = (1/beta) [ -beta t + log(sum exp(beta r_i)) ]
            # = -t + (1/beta) log(sum exp(beta r_i))
            # So L(t) = t - t + const = const.
            # The objective is independent of t!
            # This formulation is flawed for optimizing t.
            
            # Correct Epigraph Formulation:
            # Minimize t subject to r_i <= t for all i.
            # Smooth relaxation: Minimize t + (1/beta) log( sum_i exp(beta (r_i - t)) )
            # Wait, if I minimize over t, the minimum is at t = max(r_i) in the limit beta->inf.
            # But for finite beta, the minimum is slightly above max(r_i).
            # Let's just minimize the soft-max of radii directly, without t.
            # Objective: (1/beta) log( sum_i exp(beta r_i) )
            # This approximates max(r_i).
            
            # Let's restart the objective definition.
            # We will minimize: (1/beta) log( sum_i exp(beta r_i) )
            # This is a scalar function of x.
            
            # Recalculate with this in mind.
            # The previous code calculated lse = log(sum(exp(beta(r_i - t)))).
            # Let's just use soft_max = (1/beta) log(sum(exp(beta * radii)))
            
            a = beta * radii
            a_max = np.max(a)
            soft_max_val = (1.0/beta) * (a_max + np.log(np.sum(np.exp(a - a_max))))
            
            # Gradient w.r.t x is needed.
            # d(soft_max)/dx = (1/beta) * (1/sum(exp(beta r_i))) * sum(exp(beta r_i) * beta * dr_i/dx)
            # = sum( p_i * dr_i/dx ) where p_i = exp(beta r_i) / sum(exp(beta r_j))
            
            # We need dr_i/dx.
            # This is the gradient of the radius of facet i w.r.t the points.
            # We can compute this numerically or analytically.
            # Given time constraints, numerical gradient for the soft-max part might be too slow if done naively.
            # However, we can reuse the analytical gradient logic from the seed program but weighted by p_i.
            
            # Let's compute the weighted gradient.
            # We need the "center" and "active vertices" for each facet to compute the gradient contribution.
            # We already have the logic in the seed program. Let's refactor it to return per-facet gradients.
            
            grad_x = np.zeros((N, 3))
            
            # We need to re-iterate facets to get gradients, or store them.
            # To save time, we assume the hull structure is stable for small steps.
            # We'll compute the gradient contributions here.
            
            # We need to re-run the facet analysis to get the centers and vertex indices.
            # This doubles the cost.
            # Alternative: Use finite differences on the soft-max objective?
            # 3N = 150 dimensions. 150 evaluations of ConvexHull per gradient step.
            # ConvexHull is O(N log N) ~ 50 * 5.
            # 150 * 250 ops = 37,500 ops per gradient.
            # L-BFGS-B might take 100-200 steps.
            # 200 * 37,500 = 7.5M ops. This is feasible in 120s?
            # Python overhead for ConvexHull is significant.
            # 150 calls to ConvexHull might take 0.1s each? 15s per gradient.
            # 200 steps * 15s = 300s. Too slow.
            
            # We MUST use analytical gradients.
            # Let's modify compute_facet_radii to also return the gradient contributions.
            
            return soft_max_val, None # Placeholder, will implement analytical grad below
            
        else:
            # t >= max radii, penalty is 0
            # Actually, if we just minimize soft_max(radii), we don't need t.
            # Let's stick to minimizing soft_max(radii) directly.
            pass

    def objective_and_grad(x_flat):
        """
        Computes soft-max of facet radii and its analytical gradient.
        """
        if time.time() - start_time > time_limit - 2.0:
            return np.inf, np.zeros(3*N)
            
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        try:
            hull = ConvexHull(x)
        except Exception:
            return np.inf, np.zeros(3*N)
            
        radii = np.zeros(len(hull.simplices))
        # Store data needed for gradient
        facet_data = []
        
        for i, simplex in enumerate(hull.simplices):
            v1, v2, v3 = x[simplex[0]], x[simplex[1]], x[simplex[2]]
            
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                radii[i] = 0.0
                facet_data.append((simplex, 0.0, None, None, None, None)) # (indices, rad, center, g1, g2, g3)
                continue
            c_s = normal / norm
            
            d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
            d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
            max_d_circ = max(d1, d2, d3)
            
            e12 = v1 + v2
            e12_norm = np.linalg.norm(e12)
            if e12_norm > 1e-10:
                m12 = e12 / e12_norm
                d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
            else:
                d_e12 = 0.0
                
            e23 = v2 + v3
            e23_norm = np.linalg.norm(e23)
            if e23_norm > 1e-10:
                m23 = e23 / e23_norm
                d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
            else:
                d_e23 = 0.0

            e31 = v3 + v1
            e31_norm = np.linalg.norm(e31)
            if e31_norm > 1e-10:
                m31 = e31 / e31_norm
                d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
            else:
                d_e31 = 0.0

            max_d_edge = max(d_e12, d_e23, d_e31)

            if max_d_circ <= np.pi / 2.0:
                local_rad = max_d_circ
                center = c_s
                # Gradient of radius w.r.t vertices
                # r = arccos(c . v). dr/dv = -c / sqrt(1 - (c.v)^2)
                # But we are on the sphere. The gradient on the tangent plane is -c_perp.
                # Actually, the vector (c - v) projected to tangent is the direction to decrease distance.
                # Let's use the force model: move v towards c.
                g1 = c_s - v1
                g2 = c_s - v2
                g3 = c_s - v3
            else:
                local_rad = max_d_edge
                if d_e12 == max_d_edge:
                    center = m12
                    g1 = m12 - v1
                    g2 = m12 - v2
                    g3 = np.zeros(3)
                elif d_e23 == max_d_edge:
                    center = m23
                    g1 = np.zeros(3)
                    g2 = m23 - v2
                    g3 = m23 - v3
                else:
                    center = m31
                    g1 = m31 - v1
                    g2 = np.zeros(3)
                    g3 = m31 - v3

            radii[i] = local_rad
            facet_data.append((simplex, local_rad, center, g1, g2, g3))

        # Compute soft-max
        beta = 50.0
        a = beta * radii
        a_max = np.max(a)
        if a_max > 50:
             # Overflow unlikely with beta=50 and radii ~ 0.3
             pass
        exp_a = np.exp(a - a_max)
        sum_exp_a = np.sum(exp_a)
        soft_max_val = (1.0/beta) * (a_max + np.log(sum_exp_a))
        
        # Weights for gradient
        # p_i = exp(beta r_i) / sum(exp(beta r_j))
        # Note: exp(a - a_max) / sum(exp(a - a_max)) is the same as exp(a) / sum(exp(a))
        weights = exp_a / sum_exp_a
        
        grad_x = np.zeros((N, 3))
        
        for i, (simplex, rad, center, g1, g2, g3) in enumerate(facet_data):
            w = weights[i]
            if w < 1e-10:
                continue
                
            idx1, idx2, idx3 = simplex
            grad_x[idx1] += w * g1
            grad_x[idx2] += w * g2
            grad_x[idx3] += w * g3
            
        # Project to tangent plane
        for i in range(N):
            dot_prod = np.dot(grad_x[i], x[i])
            grad_x[i] -= dot_prod * x[i]
            
        return soft_max_val, grad_x.flatten()

    # Initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = np.inf

    # Optimization Loop with Multiple Restarts
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        if restart == 0:
            x0 = points.copy()
        else:
            x0 = points.copy()
            x0 += np.random.normal(0, 0.2, x0.shape)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        
        # L-BFGS-B Optimization
        # We optimize the flattened coordinates.
        # We don't enforce the sphere constraint in L-BFGS-B directly,
        # but we normalize inside the objective function.
        # This is a "projected" approach implicitly.
        
        res = minimize(
            objective_and_grad, 
            x0.flatten(), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
        )
        
        x_opt = res.x.reshape(N, 3)
        x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
        
        # Evaluate exact score
        radii = compute_facet_radii(x_opt)
        if radii is not None:
            exact_score = np.max(radii)
            if exact_score < best_score:
                best_score = exact_score
                best_points = x_opt.copy()

    # Final Polish with Original Gradient Descent
    if time.time() - start_time < time_limit - 5.0:
        x = best_points.copy()
        current_rad, grad = None, None
        
        # Re-implement the exact gradient function from seed for polishing
        def compute_exact_grad(pts):
            try:
                hull = ConvexHull(pts)
                max_rad = 0.0
                active_facets = []
                
                for simplex in hull.simplices:
                    v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                    normal = np.cross(v2 - v1, v3 - v1)
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    c_s = normal / norm
                    d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                    d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                    d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                    max_d_circ = max(d1, d2, d3)
                    
                    e12 = v1 + v2
                    e12_norm = np.linalg.norm(e12)
                    if e12_norm > 1e-10:
                        m12 = e12 / e12_norm
                        d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                    else:
                        d_e12 = 0.0
                    e23 = v2 + v3
                    e23_norm = np.linalg.norm(e23)
                    if e23_norm > 1e-10:
                        m23 = e23 / e23_norm
                        d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                    else:
                        d_e23 = 0.0
                    e31 = v3 + v1
                    e31_norm = np.linalg.norm(e31)
                    if e31_norm > 1e-10:
                        m31 = e31 / e31_norm
                        d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                    else:
                        d_e31 = 0.0

                    max_d_edge = max(d_e12, d_e23, d_e31)

                    if max_d_circ <= np.pi / 2.0:
                        local_rad = max_d_circ
                        center = c_s
                        g1 = c_s - v1
                        g2 = c_s - v2
                        g3 = c_s - v3
                    else:
                        local_rad = max_d_edge
                        if d_e12 == max_d_edge:
                            center = m12
                            g1 = m12 - v1
                            g2 = m12 - v2
                            g3 = np.zeros(3)
                        elif d_e23 == max_d_edge:
                            center = m23
                            g1 = np.zeros(3)
                            g2 = m23 - v2
                            g3 = m23 - v3
                        else:
                            center = m31
                            g1 = m31 - v1
                            g2 = np.zeros(3)
                            g3 = m31 - v3

                    if local_rad > max_rad:
                        max_rad = local_rad
                        active_facets = [(simplex, local_rad, g1, g2, g3, center)]
                    elif abs(local_rad - max_rad) < 1e-4:
                        active_facets.append((simplex, local_rad, g1, g2, g3, center))

                if not active_facets:
                    return 0.0, np.zeros((N, 3))

                grad = np.zeros((N, 3))
                for simplex, rad, g1, g2, g3, center in active_facets:
                    grad[simplex[0]] += g1
                    grad[simplex[1]] += g2
                    grad[simplex[2]] += g3
                
                for i in range(N):
                    dot_prod = np.dot(grad[i], pts[i])
                    grad[i] -= dot_prod * pts[i]
                
                return max_rad, grad
            except Exception:
                return np.inf, np.zeros((N, 3))

        current_rad, grad = compute_exact_grad(x)
        lr = 0.01
        for step in range(100):
            if time.time() - start_time > time_limit - 2.0:
                break
            x_new = x + lr * grad
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            new_rad, new_grad = compute_exact_grad(x_new)
            if new_rad < current_rad:
                x = x_new
                current_rad = new_rad
                grad = new_grad
                if current_rad < best_score:
                    best_score = current_rad
                    best_points = x.copy()
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break

    return best_points

# EVOLVE-BLOCK-END
