import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses NumPy for vectorized computation of sum and difference sets.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Vectorized sumset and diffset calculation
    # This creates arrays of size n*n, which is fast for n < 200
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    # Count unique elements
    # For small n, sorting and counting unique is fast
    # np.unique is efficient for 1D arrays
    num_sums = len(np.unique(sums))
    num_diffs = len(np.unique(diffs))
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / float(num_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _generate_geometric(n, base, jitter=0.0):
    """Generate a geometric-like set."""
    if n < 2:
        return [0, 1]
    vals = [0]
    for i in range(1, n):
        val = int(round(base ** i))
        if jitter > 0:
            val += random.randint(-jitter, jitter)
        if val <= vals[-1]:
            val = vals[-1] + 1
        vals.append(val)
    return sorted(set(vals))

def _generate_random_sorted(n, max_val):
    """Generate a random sorted set of size n within [0, max_val]."""
    if n < 2:
        return [0, 1]
    # Ensure 0 is included
    other_vals = random.sample(range(1, max_val + 1), n - 1)
    return sorted([0] + other_vals)

def _local_search(u, max_iters=2000, time_limit=10.0):
    """
    Perform local search with simulated annealing logic.
    Moves: 
    1. Replace an element with a nearby value.
    2. Add a new element.
    3. Remove an element.
    """
    current_u = sorted(list(u))
    current_s = _score_from_set(current_u)
    best_u = current_u[:]
    best_s = current_s
    
    m = max(current_u) if current_u else 1
    # Temperature for SA
    T = 0.01
    T_min = 0.001
    cooling = 0.99
    
    start_time = time.time()
    iter_count = 0
    
    while iter_count < max_iters and time.time() - start_time < time_limit and time.time() < _DEADLINE - 5.0:
        iter_count += 1
        new_u = current_u[:]
        move_type = random.choice(['replace', 'add', 'remove'])
        
        if move_type == 'replace' and len(new_u) > 1:
            idx = random.randint(1, len(new_u) - 1) # Don't touch 0 usually, or maybe allow it? 
            # Actually, 0 must be in the set. Let's keep 0 fixed.
            old_val = new_u[idx]
            # Search range: near old_val or near boundaries
            search_range = max(10, int(0.1 * m))
            delta = random.randint(-search_range, search_range)
            new_val = old_val + delta
            if new_val < 0:
                new_val = 0
            # Check if new_val is already in set (excluding the one being replaced)
            if new_val in new_u and new_val != old_val:
                continue
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            if len(new_u) != len(current_u):
                continue
            if 0 not in new_u:
                # If 0 was removed, add it back? Or reject move? 
                # Contract says U must contain 0.
                # If we replace 0, we must ensure 0 is still there.
                # Since we don't touch index 0 (assuming sorted, 0 is at 0), this is safe.
                # But if old_val was 0? We skipped index 0.
                pass
        
        elif move_type == 'add' and len(new_u) < 100:
            # Add a random value
            max_val = max(new_u)
            new_val = random.randint(1, max_val * 2 + 10)
            if new_val not in new_u:
                new_u.append(new_val)
                new_u = sorted(new_u)
        
        elif move_type == 'remove' and len(new_u) > 2:
            # Remove a random non-zero element
            idx = random.randint(1, len(new_u) - 1)
            new_u.pop(idx)
        
        else:
            continue
            
        new_s = _score_from_set(new_u)
        
        # Simulated Annealing acceptance
        if new_s >= current_s:
            current_u = new_u
            current_s = new_s
            if new_s > best_s:
                best_s = new_s
                best_u = new_u[:]
        else:
            # Accept with probability exp((new_s - current_s) / T)
            prob = math.exp((new_s - current_s) / T)
            if random.random() < prob:
                current_u = new_u
                current_s = new_s
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_u, best_s

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_s = _score_from_set(best_u)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Geometric bases
    for base in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for n in [10, 15, 20, 30]:
            u = _generate_geometric(n, base, jitter=1)
            candidates.append(u)
            
    # Random sets
    for _ in range(20):
        n = random.randint(5, 50)
        max_val = random.randint(100, 5000)
        u = _generate_random_sorted(n, max_val)
        candidates.append(u)
        
    # Small exhaustive-like (brute force small sizes to find good local structures)
    # We won't do full exhaustive, but random small sets
    for _ in range(20):
        n = random.randint(3, 8)
        max_val = random.randint(10, 50)
        u = _generate_random_sorted(n, max_val)
        candidates.append(u)
        
    # Evaluate all candidates
    for u in candidates:
        s = _score_from_set(u)
        if s > best_s:
            best_s = s
            best_u = u[:]
            
    # Phase 2: Local Search from top candidates
    # Sort candidates by score
    scored_candidates = []
    for u in candidates:
        s = _score_from_set(u)
        scored_candidates.append((s, u))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Run local search on the top 5 candidates
    top_k = 5
    for i in range(top_k):
        if time.time() > _DEADLINE - 10.0:
            break
            
        init_u = scored_candidates[i][1]
        # Allocate time based on remaining budget
        time_left = _DEADLINE - time.time() - 5.0
        # Give more time to better starting points? Or equal time?
        # Let's give each run a chunk of time
        run_time = min(15.0, max(2.0, time_left / (top_k - i)))
        
        improved_u, improved_s = _local_search(init_u, max_iters=5000, time_limit=run_time)
        
        if improved_s > best_s:
            best_s = improved_s
            best_u = improved_u[:]
            
    # Phase 3: Final Polish
    # Run local search on the best found so far with more iterations
    if time.time() < _DEADLINE - 5.0:
        final_u, final_s = _local_search(best_u, max_iters=10000, time_limit=_DEADLINE - time.time() - 2.0)
        if final_s > best_s:
            best_s = final_s
            best_u = final_u[:]
            
    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        best_u.sort()
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
