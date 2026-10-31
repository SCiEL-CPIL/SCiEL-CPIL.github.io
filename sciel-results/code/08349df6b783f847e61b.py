import time
import math
import numpy as np
import random

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150s kill

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    u: list of nonnegative integers
    max_val: max(u)
    Returns: (size_sum, size_diff)
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    M = max_val
    # N needs to be at least 2*M + 1 to avoid circular convolution aliasing
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    F = np.fft.fft(f)
    
    # Sum set: Convolution f * f
    conv_sum = np.fft.ifft(F * F).real
    # Valid range for sums is [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set: Autocorrelation
    # r[k] = sum_x f[x] f[x+k]
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # r[k] for k in [1, M] corresponds to positive differences
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0
    
    size_sum, size_diff = _compute_sets_fft(u, m)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _is_golomb(u):
    """Check if u is a Golomb ruler (all pairwise differences distinct)."""
    diffs = set()
    n = len(u)
    for i in range(n):
        for j in range(i+1, n):
            d = u[j] - u[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _gen_golomb(n, max_val):
    """
    Generate a random Golomb ruler of size n with elements < max_val.
    Uses a greedy randomized approach with backtracking if necessary.
    """
    if n <= 1:
        return [0]
    
    # Start with a random permutation of a subset of [0, max_val]
    # This is slow for large n, so we use a constructive heuristic.
    # Simple heuristic: start with 0, then add random points, checking conflicts.
    
    u = [0]
    used_diffs = set()
    
    # To make it faster, we can try to construct it in a compact range.
    # A known lower bound for max val of Golomb ruler of size n is ~ n^2.
    # We restrict max_val to be reasonable.
    
    # Randomized greedy insertion
    # We want to keep values small to minimize M.
    # Let's try to fill [0, max_val] sparsely.
    
    # A better approach: start with a known good Golomb ruler or random restarts.
    # Given time constraints, we'll do a simple random walk.
    
    for _ in range(n - 1):
        # Try to find a spot for the next element
        # To keep M small, we should try small values first, but randomize to escape traps.
        # Actually, for Golomb rulers, larger gaps are often needed.
        # Let's pick a random value in [1, max_val] and check.
        # If max_val is too small, it will fail.
        
        success = False
        # Try a few random candidates
        for _ in range(100):
            candidate = random.randint(1, max_val)
            # Check if adding candidate maintains Golomb property
            # New diffs: candidate - u[i] for all i
            # Also need to check if any new diff conflicts with existing diffs or other new diffs
            # Since we add one at a time, we just check against existing u.
            
            valid = True
            new_diffs = []
            for x in u:
                d = candidate - x
                if d < 0:
                    d = -d # Golomb property is symmetric for differences magnitude? 
                           # Actually U-U contains both d and -d. 
                           # The set of absolute differences must be unique for a Golomb ruler.
                           # Wait, standard Golomb ruler definition: all pairwise differences are distinct.
                           # Since u is sorted, u[j]-u[i] > 0. 
                           # We just need to ensure no duplicate positive differences.
                
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Check if new_diffs have duplicates among themselves?
                # No, because u is a set, and we are adding one element.
                # The new differences are candidate - u[i].
                # Are they distinct? 
                # candidate - u[i] = candidate - u[j] implies u[i] = u[j], which is false.
                # So new_diffs are distinct.
                # We just need to check against used_diffs.
                
                u.append(candidate)
                u.sort()
                for d in new_diffs:
                    used_diffs.add(d)
                success = True
                break
        
        if not success:
            # Failed to add. Return None or restart?
            # For robustness, let's return None and let the caller retry.
            return None
            
    return u

def _local_search(u, max_val, time_limit=5.0):
    """
    Improve a Golomb ruler by moving elements to minimize M and potentially |U+U|.
    Since Golomb property is rigid, we can only move if it remains a Golomb ruler.
    Actually, moving an element might break Golomb property.
    So we try to find a "better" Golomb ruler via random restarts or small perturbations.
    
    Strategy:
    1. Start with a valid Golomb ruler.
    2. Try to reduce M by shifting the largest element down if possible while maintaining Golomb property.
    3. Or, perform a random walk in the space of Golomb rulers.
    """
    best_u = list(u)
    best_s = _score(best_u)
    best_m = max(best_u)
    
    start_time = time.time()
    
    # Try to reduce max value by checking if the largest element can be moved down
    # or if we can find a different Golomb ruler with smaller max.
    
    # Since finding a Golomb ruler with minimal max (optimal Golomb ruler) is hard,
    # we will rely on random restarts of _gen_golomb with varying max_val constraints.
    
    # However, _local_search is called with a specific u. 
    # Let's try to "compress" u.
    
    # A simple compression: try to map u to a smaller range by checking if 
    # there's a bijection to a smaller set of integers that preserves Golomb property.
    # This is equivalent to finding an optimal Golomb ruler.
    
    # Given time, we'll just do random restarts of the generator with tighter bounds
    # and keep the best one found.
    
    # Actually, the caller should handle restarts. 
    # Here, we can try small perturbations: 
    # Pick an element (not 0), try moving it to a nearby empty spot, check Golomb.
    
    n = len(u)
    if n <= 1:
        return best_u, best_s
    
    used_diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            used_diffs.add(u[j] - u[i])
            
    # To allow moving, we need to recompute diffs.
    # This is O(n^2) per move. For n~10, it's fast.
    
    for _ in range(10000): # Max iterations
        if time.time() > start_time + time_limit:
            break
            
        # Pick a random index to move
        idx = random.randint(1, n-1) # Don't move 0
        old_val = u[idx]
        
        # Try to move it to a random nearby value
        # Range: [1, max_val]
        # To reduce M, bias towards smaller values if idx is the max index
        if idx == n-1:
            # Try to reduce max
            new_val = random.randint(1, max(1, old_val - 10))
        else:
            new_val = random.randint(1, max_val)
            
        if new_val == old_val:
            continue
            
        # Check if new_val is already in u
        if new_val in u:
            continue
            
        # Tentatively move
        u[idx] = new_val
        u.sort()
        
        # Check Golomb property
        valid = True
        new_diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                d = u[j] - u[i]
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                break
        
        if valid:
            s = _score(u)
            if s > best_s:
                best_s = s
                best_u = list(u)
                best_m = max(u)
                # Update used_diffs for next iteration? 
                # We recompute in the loop, so no need to maintain global state.
        else:
            # Revert
            u[idx] = old_val
            u.sort()
            
    return best_u, best_s

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 115.0)
    
    random.seed(42)
    
    best_u = [0]
    best_s = 1.0
    
    # We will try a range of sizes n.
    # For each n, we will try to generate Golomb rulers with various max_val bounds.
    # Smaller max_val is better for the score denominator.
    
    # Based on previous insights, n=5 to n=15 seems promising.
    sizes_to_try = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 10:
            break
            
        # Try different max_val limits for generation.
        # Optimal Golomb ruler of size n has max val ~ n^2.
        # We can try limits from n^2 to 2*n^2.
        
        base_max = n * n
        max_vals_to_try = [base_max, int(base_max * 1.2), int(base_max * 1.5), int(base_max * 2)]
        max_vals_to_try = sorted(set([m for m in max_vals_to_try if m >= n]))
        
        for M_limit in max_vals_to_try:
            if time.time() > _DEADLINE - 5:
                break
            
            # Try to generate a Golomb ruler
            # We might need multiple attempts if random generation fails
            # or to find a good one.
            
            found_u = None
            for attempt in range(50):
                if time.time() > _DEADLINE - 5:
                    break
                u_cand = _gen_golomb(n, M_limit)
                if u_cand:
                    found_u = u_cand
                    break
            
            if not found_u:
                continue
                
            # Improve via local search
            u_improved, s_improved = _local_search(found_u, M_limit, time_limit=2.0)
            
            if s_improved > best_s:
                best_s = s_improved
                best_u = u_improved
                
            # If we found a very good one, we might break early?
            # No, let's keep trying other M_limits.

    # Final polish: ensure format
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u

# EVOLVE-BLOCK-END
