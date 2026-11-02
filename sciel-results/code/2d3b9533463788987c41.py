"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    n = len(U)
    sum_set = set()
    diff_set = set()
    
    # Optimize loops for speed
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sum_set.add(ui + U[j])
        for j in range(n):
            diff_set.add(ui - U[j])
    
    # Note: The problem definition says U-U is the difference set.
    # Usually |U-U| counts all a-b for a,b in U.
    # My loop above adds ui - U[j] for all i,j. This is correct.
    # sum_set adds ui + U[j] for i<=j. Since addition is commutative, 
    # this covers all unique sums.
    
    if len(sum_set) == 0:
        return 0.0
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences distinct)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _greedy_golomb(n):
    """
    Generate a Golomb ruler of order n using greedy construction.
    U = {0, ...}
    """
    if n <= 1:
        return [0]
    
    # Known optimal Golomb rulers for small n
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }
    
    if n in known:
        return known[n]
    
    ruler = [0]
    used_diffs = set()
    
    for _ in range(1, n):
        pos = 1
        while True:
            # Check if adding pos to ruler keeps it a Golomb ruler
            valid = True
            for i in range(len(ruler)):
                d = pos - ruler[i]
                if d in used_diffs:
                    valid = False
                    break
                used_diffs.add(d)
            if valid:
                ruler.append(pos)
                break
            else:
                # Rollback diffs added in this attempt
                for i in range(len(ruler)):
                    d = pos - ruler[i]
                    used_diffs.discard(d)
                pos += 1
        
    return ruler

def _local_search(U, deadline, max_iter=20000):
    """
    Local search to improve the score of a set U.
    Tries swapping elements, adding, removing, and perturbing.
    """
    U = sorted(set(U))
    if len(U) < 2:
        return U
    
    best_U = U[:]
    best_score = _score(best_U)
    rng = random.Random(42)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        improved = False
        n = len(best_U)
        
        # Strategy 1: Perturb a random element (not 0)
        if n > 1:
            idx = rng.randint(1, n - 1)
            old_val = best_U[idx]
            
            # Try a range of new values
            # Range based on local density
            search_range = max(10, int(0.1 * best_U[-1]))
            
            # Sample candidates
            candidates = set()
            for _ in range(50):
                delta = rng.randint(-search_range, search_range)
                new_val = old_val + delta
                if new_val >= 0:
                    candidates.add(new_val)
            
            # Also try values near neighbors
            if idx > 0:
                candidates.add(best_U[idx-1] + 1)
            if idx < n - 1:
                candidates.add(best_U[idx+1] - 1)
                
            for new_val in candidates:
                if new_val == old_val:
                    continue
                new_U = best_U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    continue
                if len(new_U) != len(best_U):
                    # Size changed, might be valid but usually we want to keep size
                    # unless we are specifically doing add/remove
                    pass
                
                s = _score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # Strategy 2: Add an element
        if not improved and n < 30:
            # Try adding a value
            max_u = best_U[-1]
            candidates = set()
            for _ in range(50):
                v = rng.randint(1, max_u + 50)
                candidates.add(v)
            # Try filling gaps
            for i in range(n - 1):
                gap = best_U[i+1] - best_U[i]
                if gap > 1:
                    candidates.add(best_U[i] + rng.randint(1, gap - 1))
            
            for v in candidates:
                if v in best_U:
                    continue
                new_U = sorted(best_U + [v])
                s = _score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # Strategy 3: Remove an element
        if not improved and n > 2:
            idx = rng.randint(1, n - 1) # Don't remove 0
            new_U = best_U[:idx] + best_U[idx+1:]
            s = _score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
        
        # If no improvement in a while, try a larger random restart/perturbation
        if not improved and it % 100 == 0:
             # Try a completely random set of same size?
             # Or just a larger perturbation
             if n > 1:
                 idx = rng.randint(1, n - 1)
                 old_val = best_U[idx]
                 new_val = rng.randint(0, best_U[-1] + 100)
                 new_U = best_U[:]
                 new_U[idx] = new_val
                 new_U = sorted(set(new_U))
                 if 0 in new_U and len(new_U) == n:
                     s = _score(new_U)
                     if s > best_score + 1e-12:
                         best_score = s
                         best_U = new_U
                         improved = True

    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Iterate over different sizes of Golomb rulers
    # Small sizes often have better scores due to the log(max) penalty
    for n in range(3, 20):
        if time.time() > deadline - 20:
            break
            
        # Generate Golomb ruler
        gr = _greedy_golomb(n)
        
        # Evaluate
        s = _score(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
            
        # Refine with local search if we have time
        if time.time() < deadline - 15:
            refined = _local_search(gr, deadline, max_iter=5000)
            s_ref = _score(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_U = refined
        
    # Try a few random sets as well, maybe they beat Golomb?
    # Unlikely, but cheap to try
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            n = random.randint(3, 15)
            max_val = random.randint(10, 100)
            U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    # Final polish
    if time.time() < deadline - 5:
        final = _local_search(best_U, deadline, max_iter=10000)
        s_final = _score(final)
        if s_final > best_score:
            best_U = final

    # Ensure 0 is in the set and all are nonnegative
    best_U = sorted(set([x for x in best_U if x >= 0]))
    if 0 not in best_U:
        best_U = [0] + best_U
        
    return best_U

# EVOLVE-BLOCK-END
