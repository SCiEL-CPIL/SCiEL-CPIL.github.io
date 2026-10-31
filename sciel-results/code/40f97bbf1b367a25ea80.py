"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
def _score(x):
    """Compute the score: max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    ac = np.convolve(x, x, mode='full')
    return np.max(np.abs(ac)) / (s * s)

def _flip_effect(x, i):
    """Compute the change in objective if we flip sign of x[i].
    We need max|ac| / s^2. Flipping x[i] changes:
      s -> s - 2*x[i]
      ac changes: ac[k] -= 2*x[i]*x[k-i] for all k where valid.
    We approximate by computing the new max|ac| efficiently.
    """
    xi = x[i]
    s_new = np.sum(x) - 2 * xi
    if s_new == 0:
        return float('inf')
    # The autoconvolution of x is ac = x * x (convolution).
    # Flipping x[i] means x_new = x - 2*xi*e_i.
    # ac_new = (x - 2xi e_i) * (x - 2xi e_i) = ac - 2xi*(e_i*x) - 2xi*(x*e_i) + 4xi^2*(e_i*e_i)
    # e_i * x is x shifted by i. x * e_i is x shifted by -i (reversed).
    # For real sequences, e_i * x = shift(x, i) and x * e_i = shift(x, -i) in convolution index.
    # Actually, convolution: (a*b)[k] = sum_j a[j]b[k-j].
    # (e_i * x)[k] = x[k-i]. (x * e_i)[k] = x[k-i]. So both are shift(x, i).
    # Wait: (e_i * x)[k] = sum_j e_i[j] x[k-j] = x[k-i].
    # (x * e_i)[k] = sum_j x[j] e_i[k-j] = x[k-i].
    # So ac_new = ac - 2xi * shift(x,i) - 2xi * shift(x,i) + 4xi^2 * e_i*e_i
    # e_i * e_i is a delta at position 2i (in full convolution of length 2N-1).
    # Actually e_i has 1 at index i. e_i * e_i has 1 at index 2i.
    # So ac_new = ac - 4xi * shift(x, i) + 4xi^2 * delta_{2i}
    
    # We need max |ac_new|. This is expensive to compute fully each time.
    # Instead, we'll just compute the full score for candidate flips in a batched way,
    # but that's O(N^2) per flip. Too slow.
    
    # Alternative: use the fact that we only need the maximum. 
    # For now, use a simpler heuristic: try flipping the element that contributes most to the peak.
    # This is a placeholder; the actual search below uses a different approach.
    return None  # Not used in the main loop

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s out of 150s budget
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)
    
    # Initialize with several diverse starting points
    rng = np.random.default_rng(42)
    
    initializations = []
    # All ones
    initializations.append(np.ones(N, dtype=np.float64))
    # Random signs
    initializations.append(np.sign(rng.standard_normal(N)) * 1.0)
    # Block pattern: alternating blocks of +1 and -1
    for block_size in [4, 8, 16, 32]:
        x = np.ones(N, dtype=np.float64)
        for b in range(0, N, block_size * 2):
            x[b + block_size : b + 2 * block_size] = -1.0
        initializations.append(x)
    # Chirp-like: sign based on quadratic phase
    t = np.arange(N)
    initializations.append(np.sign(np.cos(2 * np.pi * 0.01 * t * t)))
    initializations.append(np.sign(np.cos(2 * np.pi * 0.02 * t * t)))
    # Sparse: mostly +1, few -1
    x_sparse = np.ones(N, dtype=np.float64)
    x_sparse[rng.choice(N, size=N//10, replace=False)] = -1.0
    initializations.append(x_sparse)
    
    # Evaluate all initializations and pick the best
    for x0 in initializations:
        s = _score(x0)
        if s < best_score:
            best_score = s
            best_x = x0.copy()
    
    # Local search: greedy sign flips
    # To make this efficient, we'll use a coordinate descent approach.
    # At each step, find the flip that most reduces the score.
    # Computing the exact effect of a flip is O(N) if we precompute the autoconvolution.
    # ac = np.convolve(x, x, 'full')  # O(N^2)
    # Flipping x[i]: ac_new = ac - 4*x[i]*shift(x, i) + 4*x[i]^2*delta_{2i}
    # We can compute this in O(N) per flip.
    # But trying all N flips each iteration is O(N^2) per iteration.
    # With N=400, that's 160,000 operations per iteration. Should be fast.
    
    x = best_x.copy()
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        ac = np.convolve(x, x, mode='full')
        s = np.sum(x)
        if s == 0:
            break
        s2 = s * s
        
        best_improvement = 0
        best_flip_idx = -1
        
        for i in range(N):
            xi = x[i]
            if xi == 0:
                continue
            s_new = s - 2 * xi
            if s_new == 0:
                continue
            s2_new = s_new * s_new
            
            # ac_new = ac - 4*xi*shift(x, i) + 4*xi^2*delta_{2i}
            # shift(x, i)[k] = x[k - i] for valid k, 0 otherwise
            # In the full convolution of length 2N-1, index k ranges from 0 to 2N-2.
            # x[k-i] is valid when 0 <= k-i < N, i.e., i <= k < i+N.
            
            # Compute the change: delta_ac[k] = -4*xi*x[k-i] for i <= k < i+N
            # and +4*xi^2 at k=2i.
            
            # We need max |ac_new[k]| over all k.
            # ac_new[k] = ac[k] - 4*xi*x[k-i] for i <= k < i+N (and k != 2i, where we add 4xi^2)
            # For k outside [i, i+N), ac_new[k] = ac[k] (except k=2i if 2i is outside [i,i+N), but 2i is always in [i, i+N) for i>=0).
            
            # This is still O(N) per flip to compute the max.
            # Let's just compute it directly but efficiently.
            
            # Create a copy of ac and apply the change
            # This is O(N) memory and O(N) time per flip, so O(N^2) total per iteration.
            
            # Optimization: we can compute the max without creating a full copy.
            # max |ac_new| = max(
            #   max over k not in [i, i+N) of |ac[k]|,   (these are unchanged, except possibly k=2i)
            #   max over k in [i, i+N) of |ac[k] - 4*xi*x[k-i] + 4*xi^2*delta(k,2i)|
            # )
            
            # For the unchanged part, we can precompute the max of |ac| excluding a range.
            # But that's complex. Let's just do the O(N) computation per flip.
            
            # Actually, let's just compute the new ac for the affected range and take the max.
            
            # The affected indices are [i, i+N-1] (since x[k-i] is valid for k in [i, i+N-1]).
            # Wait, x has indices 0..N-1. x[k-i] is valid for 0 <= k-i <= N-1, so i <= k <= i+N-1.
            # So the range is [i, i+N-1].
            
            lo = i
            hi = min(i + N - 1, 2 * N - 2)
            
            # For k in [lo, hi]: ac_new[k] = ac[k] - 4*xi*x[k-i]
            # Plus at k=2i: ac_new[2i] += 4*xi^2
            # But 2i is in [i, i+N-1] only if i <= 2i <= i+N-1, i.e., i <= N-1, which is true.
            
            # Compute the max in the affected range
            affected = ac[lo:hi+1] - 4 * xi * x[0:N-(hi-lo)]  # x[k-i] for k in [lo,hi]
            # Wait, x[k-i] for k in [i, i+N-1] is x[0] to x[N-1]. So it's just x.
            affected = ac[lo:hi+1] - 4 * xi * x
            
            # Add the delta at position 2i
            delta_pos = 2 * i - lo  # position within the affected slice
            if 0 <= delta_pos < len(affected):
                affected[delta_pos] += 4 * xi * xi
            
            max_affected = np.max(np.abs(affected))
            
            # For the unaffected part
            if lo > 0:
                max_unaffected_left = np.max(np.abs(ac[:lo]))
            else:
                max_unaffected_left = 0
            if hi < 2 * N - 2:
                max_unaffected_right = np.max(np.abs(ac[hi+1:]))
            else:
                max_unaffected_right = 0
            max_unaffected = max(max_unaffected_left, max_unaffected_right)
            
            new_max_ac = max(max_affected, max_unaffected)
            new_score = new_max_ac / s2_new
            
            improvement = current_score - new_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            x[best_flip_idx] *= -1
            current_score = _score(x)
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Occasionally try a random restart from the best found so far
        if no_improve_count >= 20 and time.time() < deadline:
            x = best_x.copy()
            # Randomly flip a few signs
            n_flips = rng.integers(1, 10)
            flip_indices = rng.choice(N, size=n_flips, replace=False)
            x[flip_indices] *= -1
            current_score = _score(x)
            no_improve_count = 0
    
    return best_x
# EVOLVE-BLOCK-END
