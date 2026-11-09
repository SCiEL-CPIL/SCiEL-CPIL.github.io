"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _exists_sidon_subset(s, k):
    """Check if there exists a Sidon subset of size k in s."""
    n = len(s)
    if k == 1:
        return True
    if k > n:
        return False
    
    # Sort to ensure consistent ordering for backtracking
    s = sorted(s)
    used_sums = set()
    
    def backtrack(index, current_subset):
        if len(current_subset) == k:
            return True
        
        # Pruning: not enough elements left to reach size k
        if len(current_subset) + (n - index) < k:
            return False
            
        for i in range(index, n):
            val = s[i]
            # Check if adding val creates a collision
            # New sums: val + x for x in current_subset, and 2*val
            new_sums = []
            valid = True
            for x in current_subset:
                sm = val + x
                if sm in used_sums:
                    valid = False
                    break
                new_sums.append(sm)
            diag = 2 * val
            if diag in used_sums:
                valid = False
            else:
                new_sums.append(diag)
                
            if valid:
                # Add new sums to used_sums
                for sm in new_sums:
                    used_sums.add(sm)
                
                # Recurse
                if backtrack(i + 1, current_subset + [val]):
                    return True
                
                # Remove new sums
                for sm in new_sums:
                    used_sums.remove(sm)
                    
        return False
        
    return backtrack(0, [])

def _max_sidon_subset_size(s):
    """Find the largest subset of s with all pair sums (incl. diagonals) unique."""
    n = len(s)
    # Iterate from largest possible size down to 1
    for k in range(n, 0, -1):
        if _exists_sidon_subset(s, k):
            return k
    return 1

def _evaluate_set(s):
    """Return the largest Sidon subset size for set s (lower is better)."""
    return _max_sidon_subset_size(s)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe
    
    random.seed(42)
    
    # Initial candidates: Dense sets (APs) are likely to have small max Sidon subsets
    best_set = list(range(12))
    best_score = _evaluate_set(best_set)
    
    # Local search function
    def local_search(start_set, max_iters=200):
        current = list(start_set)
        current_score = _evaluate_set(current)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            new_set = list(current)
            idx = random.randint(0, 11)
            old_val = new_set[idx]
            new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in new_set:
                continue
                
            new_set[idx] = new_val
            
            if any(x < 0 or x > 128 for x in new_set):
                continue
                
            score = _evaluate_set(new_set)
            
            if score < current_score:
                current = new_set
                current_score = score
            elif score == current_score and random.random() < 0.3:
                # Plateau move to escape local minima
                current = new_set
                
        return current, current_score

    # Run local search from several starting points
    start_points = []
    
    # 1. Simple AP
    start_points.append(list(range(12)))
    
    # 2. Clustered APs
    for base in range(0, 100, 10):
        s = [base + i for i in range(12)]
        if max(s) <= 128:
            start_points.append(s)
            
    # 3. Two clusters
    for b1 in range(0, 50, 10):
        for b2 in range(b1 + 20, 120, 10):
            c1 = [b1 + i for i in range(6)]
            c2 = [b2 + i for i in range(6)]
            if max(c2) <= 128:
                start_points.append(c1 + c2)

    # Evaluate and search
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        
        # Quick check if this start point is already better
        sp_score = _evaluate_set(sp)
        if sp_score < best_score:
            best_score = sp_score
            best_set = list(sp)
        
        # Run local search from this point
        if time.time() > deadline - 5:
            continue
            
        found_set, found_score = local_search(sp)
        
        if found_score < best_score:
            best_score = found_score
            best_set = list(found_set)
            
        if best_score <= 5:
            # Very good result, likely near optimal
            break

    return best_set
# EVOLVE-BLOCK-END
