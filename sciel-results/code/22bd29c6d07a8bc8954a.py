import time
import numpy as np
from scipy.optimize import minimize_scalar

# Global deadline to ensure we finish well before the 120s/150s limit
_GLOBAL_DEADLINE = time.time() + 110 

def _compute_sup_norm_refined(coeffs, num_samples=100000, refine=True):
    """
    Compute the sup norm of the polynomial on the unit circle with high precision.
    1. Dense sampling to find candidate peaks.
    2. Golden-section refinement (via scipy) around the top peaks.
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
        
    # 1. Dense sampling
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    # Evaluate polynomial P(z) = sum c_k z^k
    # We use Horner's method in reverse for stability and speed
    # coeffs are c_0, ..., c_{n-1}
    # P(z) = c_0 + c_1 z + ... + c_{n-1} z^{n-1}
    # Reversed Horner: start with 0, for c in reversed(coeffs): val = val*z + c
    # Wait, standard Horner is: c_{n-1} + z(c_{n-2} + z(...))
    # If coeffs = [c_0, ..., c_{n-1}], reversed is [c_{n-1}, ..., c_0]
    # Loop: v=0; v = v*z + c_{n-1}; v = v*z + c_{n-2}; ...; v = v*z + c_0
    # This computes c_0 + c_1 z + ... + c_{n-1} z^{n-1}. Correct.
    
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
        
    mag = np.abs(vals)
    
    if not refine:
        return float(np.max(mag))
        
    # 2. Refine top peaks
    # Find the indices of the largest magnitudes
    top_k = 20
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = float(np.max(mag))
    window = 2 * np.pi / num_samples
    
    # Helper for fast eval at a single point
    def eval_poly(z):
        res = 0.0 + 0.0j
        for c in reversed(coeffs):
            res = res * z + c
        return res

    for idx in peak_indices:
        center = angles[idx]
        
        # Objective for minimization (we want to maximize magnitude, so minimize -mag)
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(eval_poly(z))
            
        try:
            # Bounded minimization
            res = minimize_scalar(obj, bounds=(center - window, center + window),
                                  method='bounded', options={'xatol': 1e-12})
            if res.success:
                peak_mag = -res.fun
                if peak_mag > best_val:
                    best_val = peak_mag
        except Exception:
            pass
            
    return best_val

def _local_search_fast(coeffs, time_limit=10.0):
    """
    Local search with vectorized single flips and sampled pair flips.
    Returns the best sequence found and its fast sup norm.
    """
    start_time = time.time()
    current = np.array(coeffs, dtype=np.int8)
    n = len(current)
    
    # Pre-compute angles for fast evaluation
    num_samples = 2048
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    
    def fast_sup_norm(c_arr):
        """Vectorized sup norm calculation for a single coefficient array."""
        vals = np.zeros(num_samples, dtype=complex)
        for c in reversed(c_arr):
            vals = vals * zs + c
        return float(np.max(np.abs(vals)))
        
    current_score = fast_sup_norm(current)
    
    # Pre-compute the base value for single flips?
    # P(z) = c_i z^i + P_rest(z)
    # Flipping c_i changes P by -2 c_i z^i.
    # This is hard to vectorize efficiently for all i at once without significant memory.
    # Instead, we loop through single flips but evaluate them quickly.
    
    improved = True
    while improved:
        improved = False
        if time.time() > start_time + time_limit or time.time() > _GLOBAL_DEADLINE:
            break
        
        # Single flips
        for i in range(n):
            if time.time() > _GLOBAL_DEADLINE:
                return current.tolist(), current_score
            
            old_val = current[i]
            current[i] = -old_val
            new_score = fast_sup_norm(current)
            if new_score < current_score - 1e-9:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
                
        # Pair flips (sampled to save time)
        # Only try pair flips if we have time budget left
        if time.time() < start_time + time_limit * 0.5:
            for i in range(n):
                if time.time() > _GLOBAL_DEADLINE:
                    return current.tolist(), current_score
                for j in range(i + 1, n):
                    if time.time() > _GLOBAL_DEADLINE:
                        return current.tolist(), current_score
                    
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = fast_sup_norm(current)
                    if new_score < current_score - 1e-9:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj
                    if time.time() > start_time + time_limit:
                        break
                if time.time() > start_time + time_limit:
                    break

    return current.tolist(), current_score

def _grs_sequence(length=24):
    """
    Generate Rudin-Shapiro sequence of given length.
    """
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def construct():
    # 1. Generate diverse initial candidates
    grs = _grs_sequence(24)
    
    candidates = []
    # Rudin-Shapiro and its symmetries
    candidates.append(grs)
    candidates.append([-c for c in grs])
    candidates.append(list(reversed(grs)))
    candidates.append([-c for c in reversed(grs)])
    
    # Add a few random walks to explore other basins
    np.random.seed(42)
    for _ in range(5):
        random_seq = np.random.choice([1, -1], size=24).tolist()
        candidates.append(random_seq)

    best_seq = None
    best_score = float('inf')
    
    # 2. Run local search on each candidate
    # We allocate time dynamically based on remaining time
    num_candidates = len(candidates)
    time_per_candidate = 8.0  # Seconds
    
    for i, cand in enumerate(candidates):
        if time.time() > _GLOBAL_DEADLINE - 20: # Stop if we only have 20s left for final refinement
            break
            
        # Run local search
        searched_seq, local_score = _local_search_fast(cand, time_limit=time_per_candidate)
        
        if local_score < best_score:
            best_score = local_score
            best_seq = searched_seq
            
        # If we found a very good score early, we can stop searching other candidates
        # The theoretical minimum is ~1.258. If we get below 1.26, it's likely optimal.
        if best_score < 1.26:
            break

    if best_seq is None:
        # Fallback to the first candidate if no search completed
        best_seq = candidates[0]
        best_score = _compute_sup_norm_refined(best_seq, refine=False)

    # 3. Final high-precision refinement and verification
    # We use the best sequence found.
    # If we have time, we might try to refine the top 2 candidates further, 
    # but local search usually converges quickly.
    
    # Calculate the final high-precision score for the best sequence
    final_score = _compute_sup_norm_refined(best_seq)
    
    # Optional: If time permits, try to improve the best sequence with a very tight local search pass
    if time.time() < _GLOBAL_DEADLINE - 10:
        # Try one more round of single flips with high precision metric?
        # No, high precision is too slow. 
        # Instead, ensure the sequence is "locally optimal" with respect to the fast metric.
        # The local search above already did this.
        pass

    return best_seq

# Test block (optional, usually removed in final submission but good for sanity)
if __name__ == "__main__":
    start = time.time()
    coeffs = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Coeffs: {coeffs}")
    score = _compute_sup_norm_refined(coeffs)
    print(f"Score: {score}")
    print(f"Target: 1.2583758")
    print(f"Ratio: {score / 1.2583758}")
