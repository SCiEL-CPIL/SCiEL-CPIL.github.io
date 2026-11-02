import numpy as np
from scipy.optimize import differential_evolution
import time
import warnings

N = 50
warnings.filterwarnings('ignore')

def _compute_score(h):
    """
    Compute the ratio: (L2^2) / (L1 * Sup_AutoConv)
    Using the discrete interpretation which matches the observed scores (1.0 for uniform, 1.1944 for [2/3, 1/3]).
    
    L2^2 = sum(h_i^2)
    L1 = sum(h_i)
    Sup_AutoConv = max_k (sum_i h_i h_{i+k})
    
    Note: If the evaluator uses continuous integration, the dw factors cancel out in the ratio:
    (sum h^2 dw) / (sum h dw * max_conv_dw) 
    where max_conv_dw is roughly max_disc_conv * dw.
    So the ratio is invariant to the choice of dw if consistent.
    """
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    h = np.maximum(h, 0.0) # Ensure non-negative
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
        
    l2_sq = np.sum(h**2)
    if l2_sq < 1e-15:
        return 0.0

    # Discrete autoconvolution (full mode)
    # conv[k] = sum_{i} h[i] * h[k-i] (with appropriate bounds)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    
    if sup_conv < 1e-15:
        return 0.0

    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (limit is 150s)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Initialize with known good candidates
    candidates = []
    
    # Uniform (Seed)
    h_uniform = np.ones(N)
    candidates.append(h_uniform)
    
    # [2/3, 1/3, 0...] (Known best from prompt)
    h_known = np.zeros(N)
    h_known[0] = 2.0/3.0
    h_known[1] = 1.0/3.0
    candidates.append(h_known)
    
    # Delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    candidates.append(h_delta)
    
    # Random sparse
    for _ in range(5):
        h_rand = np.zeros(N)
        idx = np.random.choice(N, size=3, replace=False)
        h_rand[idx] = 1.0/3.0
        candidates.append(h_rand)

    # Evaluate initial candidates
    for h in candidates:
        score = _compute_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
            
    # 2. Global Optimization using Differential Evolution
    # We want to maximize score. DE minimizes, so we minimize -score.
    # Constraints: h_i >= 0. We can use bounds [0, 1] and normalize inside?
    # No, the score is scale-invariant?
    # Let's check:
    # R(c*h) = (c^2 sum h^2) / (c sum h * c max conv) = (c^2 A) / (c^2 B) = A/B.
    # Yes, the score is homogeneous of degree 0. Scale doesn't matter.
    # So we can constrain sum(h) = 1 to reduce dimensionality, or just use bounds [0, 1] and let DE find the shape.
    # Using bounds [0, 1] is safer for DE.
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    def objective(x):
        # x is the height vector
        # To avoid DE getting stuck in zero regions, we can add a tiny penalty or just rely on the score being 0 if sum is 0.
        # But if x is all zeros, score is 0.
        # If x is very small, score is stable.
        # We just return -score.
        if time.time() > deadline:
            return 1e6 # Penalty for timeout
        h = np.asarray(x)
        s = _compute_score(h)
        return -s

    # Strategy: Run DE for a while, then refine.
    # DE can be slow for N=50. Let's limit population size and generations.
    # popsize=10, maxiter=500 might be too slow.
    # Let's try popsize=5, maxiter=200.
    # Or use a smaller subset of variables? No, we need the full shape.
    
    # Alternative: Use the "Spectral Mask" idea from inspiration 1?
    # The prompt suggests combining:
    # Current: [2/3, 1/3]
    # Insp 1: DE on sparse frequency mask / Multiplicative updates.
    # Insp 2: Continuous integration model (which turned out to be the same ratio).
    
    # Let's stick to DE on the heights, but use a good initial population.
    # DE requires an initial population. We can provide `x0`? No, DE generates its own.
    # We can use `strategy='best1bin'` or similar.
    
    # To ensure we don't waste time, we'll run DE in chunks or with a tight budget.
    # N=50 is high for DE. Let's try a hybrid:
    # 1. DE on a reduced set of "active" indices?
    # 2. Or just DE on all 50 with low popsize.
    
    # Let's try DE with popsize=4 (minimum is usually 4*dim? No, popsize is multiplier).
    # scipy differential_evolution popsize is a multiplier of dimension.
    # So popsize=4 means 200 individuals. That's a lot.
    # Let's use popsize=2 -> 100 individuals.
    # maxiter=100.
    # Tolerance 1e-5.
    
    # Actually, let's look at the "Spectral Mask" idea more closely.
    # "Model the height vector as the squared magnitude of the inverse DFT of a binary mask..."
    # This is a specific parametrization: h = |IDFT(M)|^2.
    # This guarantees h >= 0.
    # The variables are the binary mask M (or continuous weights).
    # This reduces the search space significantly and enforces non-negativity.
    # However, not all h can be represented this way (only those with non-negative autocorrelation? No, |IDFT|^2 always has non-negative autocorrelation? No, autocorrelation of h is IDFT of |H|^2. If h = |IDFT(M)|^2, then H = DFT(h) = DFT(|IDFT(M)|^2). This is complicated.)
    
    # Simpler structural change from Inspiration 1:
    # "Iterative Multiplicative Updates (Sinkhorn-style)"
    # This is a local search method.
    # Let's combine DE (global) with Multiplicative Updates (local refinement).
    
    # Phase 1: Global Search (DE)
    # Since N=50 is large, DE might be slow.
    # Let's try a coarser search: assume h is non-zero only on a small subset of indices?
    # The known best [2/3, 1/3] has support size 2.
    # Maybe the optimal support is small.
    # Let's search over support size K and the values on that support.
    
    # Strategy:
    # 1. Try support sizes K = 1, 2, 3, 4, 5.
    # 2. For each K, choose K indices and K values.
    # 3. This is still a huge combinatorial space.
    
    # Let's stick to the full vector but use a faster optimizer.
    # `scipy.optimize.differential_evolution` is robust but slow.
    # `scipy.optimize.minimize` with `method='Nelder-Mead'` or `Powell` might be faster for local refinement.
    
    # Plan:
    # 1. Generate a diverse set of initial guesses (random sparse, random smooth, etc.)
    # 2. Run a quick local optimization (Nelder-Mead) on each.
    # 3. Keep the best.
    # 4. If time remains, run DE on the best few.
    
    initial_guesses = []
    
    # Known best
    initial_guesses.append(h_known)
    
    # Random sparse (support size 2 to 5)
    for _ in range(20):
        h = np.zeros(N)
        k = np.random.randint(2, 6)
        idx = np.random.choice(N, size=k, replace=False)
        # Random weights
        w = np.random.rand(k)
        w /= np.sum(w)
        h[idx] = w
        initial_guesses.append(h)
        
    # Random smooth (low frequency)
    for _ in range(10):
        h = np.random.rand(N)
        # Smooth it
        h = np.convolve(h, np.ones(5)/5, mode='same')
        h = np.maximum(h, 0)
        h /= np.sum(h)
        initial_guesses.append(h)
        
    # Run local optimization on each guess
    # Nelder-Mead is good for non-smooth? No, it's derivative-free.
    # Our function is non-smooth due to max(conv).
    # Nelder-Mead might struggle.
    # Let's use the Multiplicative Update from Inspiration 1 as the local refiner.
    
    def refine_multiplicative(h0, max_iters=100, eta=0.1):
        """
        Local refinement using multiplicative updates to reduce peak autoconvolution.
        """
        h = h0.copy()
        h = np.maximum(h, 1e-10) # Avoid zeros
        h /= np.sum(h)
        
        best_local = h.copy()
        best_local_score = _compute_score(h)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            conv = np.convolve(h, h, mode='full')
            sup_conv = np.max(conv)
            if sup_conv < 1e-15:
                break
            
            k_star = np.argmax(conv)
            
            # Gradient of peak conv w.r.t h
            grad_h = np.zeros(N)
            for i in range(N):
                idx1 = k_star - i
                idx2 = i - k_star
                val = 0.0
                if 0 <= idx1 < N:
                    val += h[idx1]
                if 0 <= idx2 < N:
                    val += h[idx2]
                grad_h[i] = val
                
            # Multiplicative update
            # We want to reduce h_i where grad_h_i is large (contributes to peak).
            # h_i <- h_i * exp(-eta * grad_h_i)
            update = np.exp(-eta * grad_h)
            h_new = h * update
            s = np.sum(h_new)
            if s < 1e-15:
                break
            h_new /= s
            
            score_new = _compute_score(h_new)
            if score_new > best_local_score:
                best_local_score = score_new
                best_local = h_new.copy()
                h = h_new
            else:
                # Reduce eta if stuck? Or break?
                # Let's just keep h and try again with smaller eta?
                # For simplicity, break if no improvement in a few steps?
                # Or just continue with same h?
                # If we don't update h, we loop.
                # Let's decay eta.
                eta *= 0.9
                if eta < 1e-4:
                    break
    
        return best_local, best_local_score

    # Apply refinement to initial guesses
    for h0 in initial_guesses:
        if time.time() > deadline - 5.0:
            break
        h_ref, score_ref = refine_multiplicative(h0)
        if score_ref > best_score:
            best_score = score_ref
            best_h = h_ref

    # If time remains, try DE on the best few
    if time.time() < deadline - 20.0:
        # Use DE to explore the neighborhood of the best solution
        # We can't easily constrain DE to a neighborhood, but we can use it on the full space
        # with a small population.
        
        # Let's try DE with popsize=2 (100 individuals) and maxiter=50.
        # This is still expensive.
        # Maybe just do a random perturbation search (Simulated Annealing-like)?
        
        # Simulated Annealing on the height vector
        current_h = best_h.copy()
        current_score = best_score
        T = 0.1
        T_min = 1e-4
        
        for _ in range(5000):
            if time.time() > deadline:
                break
                
            # Perturb
            h_new = current_h.copy()
            # Add noise
            noise = np.random.randn(N) * T
            h_new += noise
            h_new = np.maximum(h_new, 0.0)
            s = np.sum(h_new)
            if s < 1e-15:
                continue
            h_new /= s
            
            new_score = _compute_score(h_new)
            
            # Accept if better or with probability exp((new - old)/T)
            if new_score > current_score:
                current_h = h_new
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_h = h_new.copy()
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_h = h_new
                        current_score = new_score
            
            T *= 0.999 # Cool down
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_score(h))
