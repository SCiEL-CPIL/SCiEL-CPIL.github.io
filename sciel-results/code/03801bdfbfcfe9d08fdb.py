import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0 # 120s limit, 5s margin

def compute_ratio(h):
    h = np.array(h)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def smooth_max_conv(h, temperature=1.0):
    """
    Compute a smooth approximation of the max autoconvolution using log-sum-exp.
    This helps with gradient-based optimization by smoothing the kink at the maximum.
    """
    h_conv = np.convolve(h, h, mode='full')
    # Scale to prevent overflow/underflow in log-sum-exp
    scaled = h_conv / temperature
    # Use log-sum-exp trick
    max_val = np.max(scaled)
    if not np.isfinite(max_val):
        return 0.0
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return temperature * log_sum_exp

def compute_smooth_ratio(h, temperature=0.01):
    """
    Compute a smooth surrogate of the ratio for gradient-based optimization.
    Uses log-sum-exp to smooth the max in the autoconvolution.
    """
    h = np.array(h)
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Smooth max of autoconvolution
    max_conv_smooth = smooth_max_conv(h, temperature)
    
    if max_conv_smooth < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv_smooth))

def is_sidon(arr):
    """Check if a set of integers is a Sidon set (B2 set)."""
    arr = sorted(set(arr))
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums[s] = (i, j)
    return True

def find_sidon_set_greedy(n, max_val=1000):
    """Find a Sidon set of size n using a greedy algorithm."""
    sidon = []
    for i in range(max_val):
        candidate = sidon + [i]
        if is_sidon(candidate):
            sidon.append(i)
            if len(sidon) == n:
                break
    return sidon

def generate_near_sidon_support(k, n=N):
    """
    Generate a support set of size k that minimizes pairwise sum collisions.
    We greedily pick indices that introduce the fewest new collisions.
    """
    support = []
    # Precompute collision counts? Or just check incrementally.
    # For small k (<=15), incremental check is fast.
    for _ in range(k):
        best_idx = -1
        min_collisions = float('inf')
        
        for i in range(n):
            if i in support:
                continue
            
            # Count collisions if we add i
            # Collisions occur if i + s = s1 + s2 for existing s, s1, s2 in support
            # Or more simply, check if the new set is "less Sidon-like"
            # Let's just count the number of duplicate sums in the new set.
            
            new_support = support + [i]
            sums = {}
            collisions = 0
            for a in range(len(new_support)):
                for b in range(a, len(new_support)):
                    s = new_support[a] + new_support[b]
                    if s in sums:
                        collisions += 1
                    else:
                        sums[s] = 1
                        
            if collisions < min_collisions:
                min_collisions = collisions
                best_idx = i
                
        if best_idx != -1:
            support.append(best_idx)
            
    return sorted(support)

def local_search_swap(h, max_iters=1000, time_limit=10.0):
    """
    Perform local search by swapping values at two indices.
    This helps escape local minima in the continuous optimization by exploring discrete permutations of support weights.
    """
    h = h.copy()
    l1 = np.sum(h)
    if l1 < 1e-12:
        return h
    
    best_val = compute_ratio(h)
    start_time = time.time()
    no_improve = 0
    
    for _ in range(max_iters):
        if time.time() > time_limit:
            break
        if time.time() > DEADLINE - 5:
            break
            
        # Pick two random indices to swap
        i, j = np.random.choice(N, 2, replace=False)
        
        if i == j:
            continue
            
        # Try swapping
        h_new = h.copy()
        h_new[i], h_new[j] = h[j], h[i]
        
        val_new = compute_ratio(h_new)
        
        if val_new > best_val + 1e-12:
            h = h_new
            best_val = val_new
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 100:
                # Restart with random perturbation? Or just break?
                # Let's try a small random perturbation to escape
                h += np.random.randn(N) * 1e-3
                h = np.clip(h, 0, None)
                if np.sum(h) < 1e-12:
                    break
                best_val = compute_ratio(h)
                no_improve = 0
                
    return h

def optimize_weights_on_support(support, time_limit=5.0):
    """
    Optimize the weights of a fixed support set to maximize the ratio.
    We use log-space optimization to ensure positivity.
    """
    k = len(support)
    if k == 0:
        return None, -1.0
        
    def objective(log_weights):
        w = np.exp(log_weights)
        h = np.zeros(N)
        h[support] = w
        # Normalize to L1=1
        s = np.sum(h)
        if s < 1e-12:
            return 1000.0
        h /= s
        # Use smooth surrogate for gradient computation
        r = compute_smooth_ratio(h, temperature=0.01)
        if r <= 0:
            return 1000.0
        return -np.log(r)
        
    # Initial guess: uniform weights
    x0 = np.zeros(k)
    
    start_time = time.time()
    best_val = -1.0
    best_h = None
    
    # Try a few random initializations in log-space
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > time_limit:
            break
        if time.time() > DEADLINE - 2:
            break
            
        if i == 0:
            x_start = np.zeros(k)
        else:
            x_start = np.random.randn(k) * 0.5
            
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12})
            h_opt = np.zeros(N)
            w_opt = np.exp(res.x)
            h_opt[support] = w_opt
            s = np.sum(h_opt)
            if s > 1e-12:
                h_opt /= s
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass
            
    return best_h, best_val

def high_precision_polish(h, time_limit=10.0):
    """
    Perform a high-precision polish pass on the given heights.
    Uses coordinate descent with tight tolerance to refine the solution.
    """
    h = np.array(h, dtype=float)
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return h
    
    h = h / l1  # Normalize to L1=1
    best_val = compute_ratio(h)
    start_time = time.time()
    
    # Coordinate descent: optimize each coordinate while keeping others fixed
    for coord_idx in range(1000):
        if time.time() > time_limit:
            break
        if time.time() > DEADLINE - 3:
            break
            
        # Cycle through coordinates
        i = coord_idx % N
        
        # Try different values for h[i]
        # Use a grid search around the current value
        current_val = h[i]
        
        # Define a range to search
        # Scale relative to the mean
        mean_val = np.mean(h[h > 1e-8]) if np.any(h > 1e-8) else 0.01
        search_range = max(0.001, mean_val * 0.5)
        
        # Try values from 0 to current_val + search_range
        n_grid = 50
        candidates = np.linspace(0, current_val + search_range, n_grid)
        
        best_candidate = current_val
        best_candidate_val = best_val
        
        for c in candidates:
            h_test = h.copy()
            h_test[i] = c
            
            # Renormalize to L1=1
            s = np.sum(h_test)
            if s < 1e-12:
                continue
            h_test = h_test / s
            
            val = compute_ratio(h_test)
            if val > best_candidate_val:
                best_candidate_val = val
                best_candidate = c
        
        # Update h[i]
        h[i] = best_candidate
        h = h / np.sum(h)  # Renormalize
        
        # Update best
        val_now = compute_ratio(h)
        if val_now > best_val:
            best_val = val_now
    
    return h

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # 2. Spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    candidates.append(h_spike)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-5 * x**2)
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # 4. Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= np.sum(h_cos)
    candidates.append(h_cos)
    
    # 5. Near-Sidon supports of various sizes
    # We will optimize weights for these specific supports.
    # This is the core of the new strategy.
    
    best_h = candidates[0]
    best_val = compute_ratio(best_h)
    
    for h in candidates[1:]:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h
            
    # Strategy: Iterate over different support sizes and near-Sidon sets
    # For each, optimize the weights.
    
    # Time budget management
    time_spent = time.time() - start_time
    time_remaining = DEADLINE - 5 - time_spent
    
    # Try support sizes from 4 to 20
    # Smaller supports are more likely to be Sidon-like and have higher potential ratio.
    # Larger supports might allow for more complex structures.
    
    # We will prioritize smaller supports first as they are faster to optimize and often better.
    support_sizes = list(range(4, 21))
    
    # Shuffle to add diversity? Or keep structured?
    # Let's keep structured but randomize the specific set generation slightly.
    
    for k in support_sizes:
        if time.time() > DEADLINE - 5:
            break
            
        # Generate a few different near-Sidon sets for this size
        n_sets = 3 if k < 10 else 1
        
        for s_idx in range(n_sets):
            if time.time() > DEADLINE - 5:
                break
                
            # Generate near-Sidon support
            # Add some randomness to the greedy selection to find different basins
            if s_idx == 0:
                support = generate_near_sidon_support(k)
            else:
                # Perturb the greedy selection
                # Randomly swap a few elements or start with a random seed
                # For simplicity, let's just use a different random permutation of indices for the greedy check
                # But generate_near_sidon_support is deterministic.
                # Let's modify it to accept a random seed or just rely on the weight optimization to find the best weights.
                # Actually, different supports are crucial.
                # Let's try a simple random support for the other iterations.
                support = sorted(np.random.choice(N, k, replace=False))
                
            # Optimize weights on this support
            # Time limit per optimization: proportional to remaining time
            # Rough estimate: 50 total time. 17 sizes * 3 sets = 51 optimizations.
            # ~1s per optimization? Might be too slow.
            # Let's limit to 0.5s per optimization.
            
            t_limit = min(1.0, (DEADLINE - 5 - time.time()) / 5.0) # Divide remaining by 5 to be safe
            
            h_opt, val_opt = optimize_weights_on_support(support, time_limit=t_limit)
            
            if h_opt is not None and val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
                
    # Final polish on the best found so far
    if time.time() < DEADLINE - 5:
        # Use the support of the best_h (indices with non-zero values)
        support_best = np.where(best_h > 1e-6)[0]
        if len(support_best) > 0:
            # Re-optimize on this support with more time
            t_limit = min(5.0, DEADLINE - 5 - time.time())
            h_opt, val_opt = optimize_weights_on_support(support_best, time_limit=t_limit)
            if h_opt is not None and val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
                
        # Apply local search swap
        if time.time() < DEADLINE - 2:
            t_limit = min(5.0, DEADLINE - 2 - time.time())
            h_swapped = local_search_swap(best_h, max_iters=500, time_limit=t_limit)
            val_swapped = compute_ratio(h_swapped)
            if val_swapped > best_val:
                best_val = val_swapped
                best_h = h_swapped
        
        # High-precision polish pass
        if time.time() < DEADLINE - 3:
            t_limit = min(15.0, DEADLINE - 3 - time.time())
            h_polished = high_precision_polish(best_h, time_limit=t_limit)
            val_polished = compute_ratio(h_polished)
            if val_polished > best_val:
                best_val = val_polished
                best_h = h_polished

    # Ensure non-negative
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
         best_h = np.ones(N) / N
         
    return list(best_h)
