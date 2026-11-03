import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def parametric_to_step(params):
    """
    Convert parametric representation to 600-step function.
    params: [height, width, center]
    height: amplitude of the Gaussian
    width: sigma of the Gaussian
    center: mean of the Gaussian in [-0.25, 0.25]
    """
    h, w, c = params
    if h < 0: h = 0
    if w <= 1e-6: w = 1e-6
    
    # Define the grid for the step function
    # The intervals are equal on [-1/4, 1/4]. 
    # The step function value is constant on each interval.
    # We evaluate the Gaussian at the center of each interval.
    x_edges = np.linspace(-0.25, 0.25, N + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    
    f = h * np.exp(-0.5 * ((x_centers - c) / w) ** 2)
    return f

def objective(params):
    """
    Objective function for optimization.
    We minimize the score.
    """
    f = parametric_to_step(params)
    return compute_score(f)

def local_refinement(params, time_limit):
    """
    Use Nelder-Mead to refine the parameters.
    """
    start_time = time.time()
    
    # Initial simplex
    # Perturb the starting point
    h, w, c = params
    simplex = [
        [h, w, c],
        [h * 1.1, w * 1.1, c],
        [h * 1.1, w * 0.9, c],
        [h, w, c + 0.01]
    ]
    
    # Convert to numpy array
    x0 = np.array(simplex[0])
    
    # Use minimize with Nelder-Mead
    # We wrap the objective to ensure it returns a float
    def obj_wrapper(p):
        val = objective(p)
        if np.isinf(val) or np.isnan(val):
            return 1e10
        return val
        
    try:
        res = minimize(obj_wrapper, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
        
        if time.time() - start_time > time_limit:
            return params, objective(params)
            
        best_params = res.x
        best_score = objective(best_params)
        
        if best_score < objective(params):
            return best_params, best_score
        else:
            return params, objective(params)
            
    except Exception:
        return params, objective(params)

def construct():
    start_time = time.time()
    
    # Initial best score is the constant function score
    # f = 1, conv = 600 * triangular, max = 600, sum = 600. Score = 600 / 360000 = 1/600 ~ 0.00166
    # Wait, let's check the constant function score properly.
    f_const = np.ones(N)
    best_score = compute_score(f_const)
    best_params = [1.0, 0.1, 0.0] # Just a placeholder, we'll update
    
    # Simulated Annealing
    # Parameters: [height, width, center]
    # Ranges:
    # height: [0.1, 10] (scale doesn't matter for score, but helps SA)
    # width: [0.01, 0.25]
    # center: [-0.25, 0.25]
    
    # Start with a reasonable guess: a Gaussian centered at 0, width covering half the domain
    current_params = [1.0, 0.15, 0.0]
    current_score = objective(current_params)
    
    if current_score < best_score:
        best_score = current_score
        best_params = current_params.copy()
    
    # SA Parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter_sa = 5000
    
    current_params = np.array(current_params)
    
    for i in range(max_iter_sa):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
            
        # Generate neighbor
        neighbor = current_params.copy()
        # Random perturbation
        neighbor[0] += random.gauss(0, 0.1) # height
        neighbor[1] += random.gauss(0, 0.01) # width
        neighbor[2] += random.gauss(0, 0.01) # center
        
        # Clip to valid ranges
        neighbor[0] = np.clip(neighbor[0], 0.01, 10.0)
        neighbor[1] = np.clip(neighbor[1], 0.01, 0.25)
        neighbor[2] = np.clip(neighbor[2], -0.25, 0.25)
        
        neighbor_score = objective(neighbor)
        
        # Acceptance probability
        if neighbor_score < current_score:
            current_params = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = neighbor_score - current_score
            if random.random() < np.exp(-delta / T):
                current_params = neighbor
                current_score = neighbor_score
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Try multiple restarts if time permits
    # The Gaussian family might be too restrictive. 
    # Let's also try a "Plateau" family or a "Two-bump" family?
    # For now, stick to Gaussian but restart with different initial widths/centers.
    
    restarts = 5
    for r in range(restarts):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Random restart
        h = random.uniform(0.5, 2.0)
        w = random.uniform(0.05, 0.20)
        c = random.uniform(-0.1, 0.1)
        init_params = np.array([h, w, c])
        
        init_score = objective(init_params)
        
        # Run a shorter SA or just local refinement
        # Let's run a short SA to escape local minima, then NM
        T = 0.005
        for i in range(500):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            neighbor = init_params.copy()
            neighbor[0] += random.gauss(0, 0.05)
            neighbor[1] += random.gauss(0, 0.005)
            neighbor[2] += random.gauss(0, 0.005)
            neighbor[0] = np.clip(neighbor[0], 0.01, 10.0)
            neighbor[1] = np.clip(neighbor[1], 0.01, 0.25)
            neighbor[2] = np.clip(neighbor[2], -0.25, 0.25)
            
            n_score = objective(neighbor)
            if n_score < init_score:
                init_params = neighbor
                init_score = n_score
            else:
                if random.random() < np.exp(-(n_score - init_score) / T):
                    init_params = neighbor
                    init_score = n_score
            T *= 0.99
            
        # Local refinement
        refined_params, refined_score = local_refinement(init_params, time_limit=5)
        
        if refined_score < best_score:
            best_score = refined_score
            best_params = refined_params
    
    # Final local refinement on the best found
    if time.time() - start_time < TIME_LIMIT - 5:
        final_params, final_score = local_refinement(best_params, time_limit=TIME_LIMIT - 5)
        if final_score < best_score:
            best_score = final_score
            best_params = final_params
            
    # Convert best params to step function
    f_final = parametric_to_step(best_params)
    
    # Ensure non-negative
    f_final = np.maximum(f_final, 0.0)
    
    # One last check: does a simple "box" or "triangle" beat this?
    # Triangle function (conv of two boxes)
    # f = |1 - 4x| for |x|<0.25? No, that's a triangle on [-0.25, 0.25] with height 1 at 0.
    # Let's test a few standard shapes just in case the Gaussian is suboptimal.
    
    # 1. Constant
    f_const = np.ones(N)
    s_const = compute_score(f_const)
    
    # 2. Triangle (linear decay from center)
    x_centers = (np.linspace(-0.25, 0.25, N+1)[:-1] + np.linspace(-0.25, 0.25, N+1)[1:]) / 2.0
    f_tri = np.maximum(0, 1 - 4*np.abs(x_centers)) # Height 1, base 0.5
    s_tri = compute_score(f_tri)
    
    # 3. Box (middle half)
    f_box = np.zeros(N)
    f_box[N//4 : 3*N//4] = 1.0
    s_box = compute_score(f_box)
    
    # 4. Cosine bump
    f_cos = np.zeros(N)
    mask = np.abs(x_centers) < 0.25
    f_cos[mask] = np.cos(np.pi * x_centers[mask] / 0.5) ** 2 # Smooth bump
    
    candidates = [
        (f_final, best_score),
        (f_const, s_const),
        (f_tri, s_tri),
        (f_box, s_box),
        (f_cos, compute_score(f_cos))
    ]
    
    best_f = min(candidates, key=lambda x: x[1])[0]
    
    # Ensure the returned function is valid
    best_f = np.maximum(best_f, 0.0)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
