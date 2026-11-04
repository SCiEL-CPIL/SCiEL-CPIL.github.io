import numpy as np
import time

N = 50
dx = 0.5 / N

def _compute_ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return -np.inf
    
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    
    l2_sq = np.sum(h * h)
    if l2_sq <= 0:
        return 0.0
    
    # Discrete convolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    
    return l2_sq / (dx * l1 * max_conv)

def _find_sidon_set(size, max_iter=500, n_attempts=100):
    """
    Find a set of 'size' indices in [0, N-1] such that pairwise sums are distinct.
    Uses a greedy randomized search with multiple restarts.
    """
    best_set = None
    best_score = -np.inf
    
    for _ in range(max_iter):
        current_set = []
        used_sums = set()
        
        # Try to build the set
        for _ in range(size):
            found = False
            for _ in range(n_attempts):
                p = np.random.randint(0, N)
                if p in current_set:
                    continue
                
                # Check if adding p causes sum collision
                new_sums = set()
                valid = True
                for q in current_set:
                    s = p + q
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.add(s)
                
                if not valid:
                    continue
                
                # Self-sum p+p
                s_self = 2 * p
                if s_self in used_sums:
                    valid = False
                    continue
                new_sums.add(s_self)
                
                if valid:
                    current_set.append(p)
                    used_sums.update(new_sums)
                    found = True
                    break
            
            if not found:
                break
        
        if len(current_set) == size:
            # Evaluate this Sidon set
            h = np.zeros(N)
            for p in current_set:
                h[p] = 1.0
            val = _compute_ratio(h)
            if val > best_score:
                best_score = val
                best_set = current_set[:]
                
            if time.time() > _deadline - 5:
                break
    
    return best_set

def _refine_positions(indices, max_iter=200):
    """
    Try to improve a set of spike positions by local search.
    For each spike, try moving it to a nearby position if it doesn't break the Sidon property.
    """
    current = list(indices)
    current_val = _compute_ratio(np.zeros(N) + np.array([1.0 if i in current else 0.0 for i in range(N)]))
    
    for _ in range(max_iter):
        if time.time() > _deadline - 3:
            break
            
        improved = False
        for idx in range(len(current)):
            old_pos = current[idx]
            # Try moving to nearby positions
            for new_pos in [old_pos - 1, old_pos + 1, old_pos - 2, old_pos + 2]:
                if new_pos < 0 or new_pos >= N:
                    continue
                if new_pos in current:
                    continue
                
                # Check Sidon property
                test_set = current[:idx] + [new_pos] + current[idx+1:]
                used_sums = set()
                valid = True
                for i in range(len(test_set)):
                    for j in range(i, len(test_set)):
                        s = test_set[i] + test_set[j]
                        if s in used_sums:
                            valid = False
                            break
                        used_sums.add(s)
                    if not valid:
                        break
                
                if valid:
                    test_h = np.zeros(N)
                    for p in test_set:
                        test_h[p] = 1.0
                    test_val = _compute_ratio(test_h)
                    if test_val > current_val:
                        current[idx] = new_pos
                        current_val = test_val
                        improved = True
                        break
        
        if not improved:
            break
    
    return current

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 145.0
    
    best_h = None
    best_val = -np.inf
    
    # Try different sizes, starting with larger ones
    for size in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > _deadline - 15:
            break
            
        sidon_set = _find_sidon_set(size, max_iter=300, n_attempts=150)
        
        if sidon_set is None:
            continue
        
        # Refine the positions
        refined = _refine_positions(sidon_set, max_iter=100)
        
        h = np.zeros(N)
        for p in refined:
            h[p] = 1.0
            
        val = _compute_ratio(h)
        
        if val > best_val:
            best_val = val
            best_h = h.copy()
            
        if size >= 9:
            break

    if best_h is None:
        best_h = np.zeros(N)
        best_h[25] = 1.0
        best_val = _compute_ratio(best_h)

    return [float(v) for v in best_h]

if __name__ == "__main__":
    res = construct()
    h = np.array(res)
    val = _compute_ratio(h)
    print(f"Final score: {val}")
