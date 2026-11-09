"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Objective: Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Equivalent to minimizing -log( ||h||_2^2 / (||h||_1 * ||h*h||_inf) )
    # To avoid log(0) and numerical issues, we optimize the raw ratio 
    # but handle scaling carefully.
    # Note: The function is scale-invariant. 
    # f(c*h) = (c^2 ||h||_2^2) / (c ||h||_1 * c ||h*h||_inf) = f(h)
    # So we can fix ||h||_1 = 1 to simplify the search space, 
    # or just let the optimizer handle it. 
    # Fixing ||h||_1=1 reduces the effective dimensionality and helps convergence.
    # However, standard minimizers work on unconstrained or box-constrained variables.
    # We can parameterize h_i = exp(x_i) / sum(exp(x_j)) to enforce non-negativity and sum=1.
    # But log-sum-exp can be unstable. 
    # Alternative: Optimize h >= 0 directly with bounds, and normalize inside the objective.
    
    def compute_ratio(h):
        h = np.asarray(h)
        if np.any(h < 0):
            return 0.0
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution h*h
        # For discrete sequences, convolution is np.convolve(h, h, mode='full')
        # Supremum norm is the max value of the convolution
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        if sup_norm == 0:
            return 0.0
        return (l2_sq / (l1 * sup_norm))
    
    def neg_ratio(h):
        # Maximize ratio => Minimize negative ratio
        # Add small penalty if any h is negative (though bounds prevent this)
        return -compute_ratio(h)

    # Initial guess: Uniform
    h0 = np.ones(N)
    
    # Bounds: [0, 10] (Scale doesn't matter, but large values can cause overflow in conv)
    # Since it's scale invariant, we can restrict to [0, 1] to keep numbers small.
    # If the optimal is a spike, it will be at the boundary 1.
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Options for L-BFGS-B
    options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'disp': False
    }
    
    best_h = h0.copy()
    best_score = compute_ratio(best_h)
    
    try:
        # Try L-BFGS-B
        res = minimize(
            neg_ratio, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options=options
        )
        
        if time.time() < deadline:
            candidate_h = res.x
            candidate_score = compute_ratio(candidate_h)
            if candidate_score > best_score:
                best_h = candidate_h
                best_score = candidate_score
                
        # If time remains, try a few random restarts to escape local minima
        # The landscape might have local minima.
        # A "spike" is a strong candidate.
        # Let's test specific structural guesses if optimization is slow or stuck.
        
        # 1. Delta function (single peak)
        h_delta = np.zeros(N)
        h_delta[N//2] = 1.0
        score_delta = compute_ratio(h_delta)
        if score_delta > best_score:
            best_h = h_delta
            best_score = score_delta
            
        # 2. Cosine bell
        x = np.linspace(-1, 1, N)
        h_cos = (np.cos(np.pi * x / 2) + 1e-9)
        h_cos /= np.sum(h_cos) # Normalize to sum 1, though ratio is invariant
        score_cos = compute_ratio(h_cos)
        if score_cos > best_score:
            best_h = h_cos
            best_score = score_cos
            
        # 3. Gaussian
        x = np.linspace(-3, 3, N)
        h_gauss = np.exp(-x**2)
        h_gauss /= np.sum(h_gauss)
        score_gauss = compute_ratio(h_gauss)
        if score_gauss > best_score:
            best_h = h_gauss
            best_score = score_gauss

        # 4. Random restarts if time permits
        np.random.seed(42)
        restarts = 0
        while time.time() < deadline - 5.0 and restarts < 50:
            # Random mixture of delta and uniform or random vector
            if restarts % 2 == 0:
                # Random sparse
                h_rand = np.zeros(N)
                idx = np.random.choice(N, size=np.random.randint(1, 5), replace=False)
                h_rand[idx] = 1.0
            else:
                # Random smooth
                h_rand = np.random.rand(N)
                # Smooth it
                kernel = np.array([0.2, 0.6, 0.2])
                h_rand = np.convolve(h_rand, kernel, mode='same')
            
            # Local optimization from this start
            res_r = minimize(
                neg_ratio, 
                h_rand, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            score_r = compute_ratio(res_r.x)
            if score_r > best_score:
                best_h = res_r.x
                best_score = score_r
            restarts += 1

    except Exception as e:
        # Fallback to delta if anything goes wrong
        best_h = np.zeros(N)
        best_h[N//2] = 1.0
        best_score = compute_ratio(best_h)

    # Ensure return type is list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
