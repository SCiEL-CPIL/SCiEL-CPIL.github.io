import time
import random
import math
import numpy as np

def _score(A):
    """Compute the objective value: max(|A+A|, |A*A|)"""
    A_arr = np.array(A)
    sums = np.add.outer(A_arr, A_arr)
    prods = np.multiply.outer(A_arr, A_arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _generate_candidates(n=20, max_val=10000):
    """
    Generate initial candidates based on the structural idea of
    union of geometric progressions (powers of 2 and powers of 3)
    to maximize multiplicative collisions.
    """
    candidates = []
    
    # 1. Pure Geometric Progressions (Powers of 2)
    # 2^19 = 524288 > 10000, so we can only have up to 2^13 = 8192 (14 elements: 1..13)
    # Actually 2^0 to 2^13 is 14 elements.
    gp2 = [2 ** i for i in range(14)]
    # Need 6 more. Add powers of 3?
    # Let's try a mix.
    
    # Strategy: Union of two GP clusters.
    # Cluster 1: Powers of 2. 2^0 ... 2^k
    # Cluster 2: Powers of 3. 3^0 ... 3^m
    # We need 20 distinct numbers <= 10000.
    
    # Candidate 1: Maximize small numbers for sumset, but use GP for product set.
    # The "best" sets for product set are GPs. The "best" for sumset are Aritmetic Progressions.
    # We want to balance.
    
    # Let's generate specific structural candidates.
    
    # 1. [1, 2, 4, ..., 8192] (14 elements) + [3, 9, 27, 81, 243, 729] (6 elements)
    # Check collisions: 3 is not a power of 2. 9 is not. etc.
    # All distinct? Yes.
    # Max val 8192 <= 10000.
    cand1 = [2 ** i for i in range(14)] + [3 ** i for i in range(1, 7)]
    if len(set(cand1)) == n and max(cand1) <= max_val:
        candidates.append(cand1)
        
    # 2. More balanced: Powers of 2 up to 2^10 (11 elements) + Powers of 3 up to 3^8 (9 elements)
    # 2^10 = 1024. 3^8 = 6561.
    # 11 + 9 = 20.
    cand2 = [2 ** i for i in range(11)] + [3 ** i for i in range(1, 10)]
    # Check distinctness: 3^i is odd, 2^i is even (except 1). 1 is in both?
    # 2^0 = 1. 3^0 = 1.
    # So remove 1 from one.
    cand2 = [2 ** i for i in range(1, 11)] + [3 ** i for i in range(0, 10)]
    # 2^1=2 ... 2^10=1024 (10 elements)
    # 3^0=1 ... 3^9=19683 > 10000.
    # 3^8 = 6561.
    # So 3^0 to 3^8 is 9 elements.
    # Total 19. Need 1 more.
    # Add 5? Or 2^11 = 2048?
    # Let's just use a generic generator for "Mixed GP"
    
    def _make_mixed_gp(count_2, count_3, max_val=10000):
        set_2 = set(2 ** i for i in range(count_2) if 2 ** i <= max_val)
        set_3 = set(3 ** i for i in range(count_3) if 3 ** i <= max_val)
        # Ensure 1 is unique
        if 1 in set_2 and 1 in set_3:
             # Remove 1 from set_3, add next power
             set_3.discard(1)
             # Find smallest 3^k not in set_2
             k = 1
             while True:
                 val = 3 ** k
                 if val <= max_val and val not in set_2:
                     set_3.add(val)
                     break
                 k += 1
        combined = list(set_2 | set_3)
        if len(combined) >= n:
             # Pick n smallest to keep values low (good for sumset)
             combined.sort()
             return combined[:n]
        return None

    # Try various splits
    for c2 in range(10, 15):
        for c3 in range(5, 10):
            if c2 + c3 >= n:
                cand = _make_mixed_gp(c2, c3)
                if cand:
                    candidates.append(cand)

    # 3. Pure powers of 2 (as much as possible) + random small numbers
    # 2^0..2^13 (14 nums). Need 6 more.
    # Add 3, 5, 7, 11, 13, 17 (primes) to minimize product collisions with 2s?
    # No, primes * 2^k are distinct.
    # Actually, adding small numbers helps sumset.
    base_gp = [2 ** i for i in range(14)]
    extras = [3, 5, 7, 11, 13, 17]
    cand3 = base_gp + extras
    if len(set(cand3)) == n and max(cand3) <= max_val:
        candidates.append(cand3)

    # 4. Arithmetic Progression (good for sumset, bad for product set)
    # A = {1, 2, ..., 20}
    candidates.append(list(range(1, n + 1)))
    
    # 5. Random smooth numbers
    def _generate_smooth_numbers(limit=10000, primes=[2, 3, 5, 7, 11]):
        smooths = set([1])
        current = [1]
        for p in primes:
            new_smooths = set()
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    new_smooths.add(val)
            smooths.update(new_smooths)
            current = list(smooths)
        return sorted([x for x in smooths if x <= limit])

    smooths = _generate_smooth_numbers(max_val)
    if len(smooths) >= n:
        # Log-uniform sampling
        log_smooths = np.log(smooths)
        indices = np.linspace(0, len(log_smooths)-1, n).astype(int)
        cand = [smooths[i] for i in indices]
        if len(set(cand)) == n:
            candidates.append(cand)
            
        # Random sampling
        for _ in range(5):
            idx = random.sample(range(len(smooths)), n)
            cand = [smooths[i] for i in idx]
            if len(set(cand)) == n:
                candidates.append(cand)

    # Validate and deduplicate
    valid_candidates = []
    seen = set()
    for c in candidates:
        if len(c) == n and len(set(c)) == n and all(1 <= x <= max_val for x in c):
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key)
                valid_candidates.append(c)
                
    if not valid_candidates:
        valid_candidates = [list(range(1, n + 1))]
        
    return valid_candidates

def _local_search(A, max_iter=5000, deadline=None, seed=None):
    """
    Local search with Replace and Swap moves.
    Refinement machinery: 
    - Uses a pool of smooth numbers for replacement to maintain structure.
    - Tries local perturbations.
    - Includes swap moves to rearrange elements (though set is unordered, 
      swapping in the list representation doesn't change the set, so this is 
      only useful if we treat A as an ordered list for some reason, or if we 
      are doing something like coordinate descent on a sorted list. 
      Since A+A and A*A are set operations, order doesn't matter. 
      So "Swap" moves are redundant for the objective function unless we are 
      optimizing a different metric. We will focus on Replace moves which 
      change the set content.
    """
    if seed is not None:
        random.seed(seed)
        
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    n = len(current_A)
    max_val = 10000
    
    # Pre-compute a pool of "good" values to try for replacement
    # Use smooth numbers and random values
    def _generate_smooth_numbers(limit=10000, primes=[2, 3, 5, 7, 11]):
        smooths = set([1])
        current = [1]
        for p in primes:
            new_smooths = set()
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    new_smooths.add(val)
            smooths.update(new_smooths)
            current = list(smooths)
        return sorted([x for x in smooths if x <= limit])

    smooths = _generate_smooth_numbers(max_val)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        
        # 1. Try Replace moves
        for k in range(n):
            old_val = current_A[k]
            # Generate a few candidate replacements
            candidates = []
            # Local perturbations
            for delta in [-5, -2, -1, 1, 2, 5, -10, 10, -50, 50]:
                nv = old_val + delta
                if 1 <= nv <= max_val and nv not in current_A:
                    candidates.append(nv)
            # Random smooth number
            if smooths:
                # Pick a smooth number close to old_val or random
                if random.random() < 0.5:
                    # Find closest smooth
                    idx = np.searchsorted(smooths, old_val)
                    if idx < len(smooths):
                        candidates.append(smooths[idx])
                    if idx > 0:
                        candidates.append(smooths[idx-1])
                else:
                    candidates.append(random.choice(smooths))
            # Random number
            candidates.append(random.randint(1, max_val))
            
            for nv in candidates:
                if nv in current_A and nv != old_val:
                    continue
                if nv < 1 or nv > max_val:
                    continue
                
                current_A[k] = nv
                new_score = _score(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_A[k] = old_val
            
            if improved:
                if current_score < best_score:
                    best_score = current_score
                    best_A = current_A[:]
                break # Restart loop after an improvement
        
        if not improved:
            # Local minimum reached
            # Try a random kick to escape?
            # For now, just break and let the outer loop try another start
            break
            
    return best_A, best_score

def _sa_search(A, start_temp=10.0, end_temp=0.1, steps=50000, seed=None, deadline=None):
    """Simulated Annealing search with deadline guard."""
    if seed is not None:
        random.seed(seed)
    
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    
    n = len(current_A)
    
    def _generate_smooth_numbers(limit=10000, primes=[2, 3, 5, 7, 11]):
        smooths = set([1])
        current = [1]
        for p in primes:
            new_smooths = set()
            for x in current:
                val = x
                while val * p <= limit:
                    val *= p
                    new_smooths.add(val)
            smooths.update(new_smooths)
            current = list(smooths)
        return sorted([x for x in smooths if x <= limit])
        
    smooths = _generate_smooth_numbers(10000)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        # Cooling schedule
        T = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Generate neighbor
        k = random.randint(0, n - 1)
        old_val = current_A[k]
        
        # Choose new value - mix of local and global moves
        if random.random() < 0.7:
            # Local perturbation - smaller steps
            delta = random.randint(-20, 20)
            new_val = old_val + delta
        elif random.random() < 0.5:
            # Medium jump
            new_val = random.randint(1, max(1, old_val // 2)) if old_val > 10 else random.randint(1, 100)
        else:
            # Global jump
            new_val = random.randint(1, 10000)
            
        # Ensure bounds and distinctness
        if new_val < 1 or new_val > 10000:
            continue
            
        # Check distinctness efficiently
        if new_val in current_A and new_val != old_val:
            continue
            
        # Apply move
        current_A[k] = new_val
        new_score = _score(current_A)
        
        # Acceptance probability
        delta_score = new_score - current_score
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A[:]
        else:
            # Reject move
            current_A[k] = old_val
            
    return best_A, best_score

def construct():
    n = 20
    max_val = 10000
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_A = None
    best_score = float('inf')
    
    candidates = _generate_candidates(n, max_val)
    
    # Evaluate initial candidates
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_A = c[:]
            
    # Run local search on the best candidates
    # Sort candidates by score to prioritize good starts
    scored_candidates = []
    for c in candidates:
        s = _score(c)
        scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on top 5 distinct candidates
    seen_structures = set()
    search_count = 0
    for score, c in scored_candidates:
        if time.time() > deadline - 10: # Leave 10s for SA and final polish
            break
            
        # Check if we've already searched a very similar structure
        key = tuple(sorted(c))
        if key in seen_structures:
            continue
        seen_structures.add(key)
        
        A_res, score_res = _local_search(c, max_iter=2000, deadline=deadline, seed=search_count)
        if score_res < best_score:
            best_score = score_res
            best_A = A_res[:]
            
        search_count += 1
        if search_count >= 5:
            break
            
    # Run SA from best solution found so far
    if best_A is not None and time.time() < deadline - 5:
        A_sa, score_sa = _sa_search(best_A, start_temp=5.0, end_temp=0.1, steps=20000, seed=123, deadline=deadline)
        if score_sa < best_score:
            best_score = score_sa
            best_A = A_sa[:]
            
    # Final polish on the best solution found
    if best_A is not None and time.time() < deadline - 2:
        A_polish, score_polish = _local_search(best_A, max_iter=5000, deadline=deadline, seed=999)
        if score_polish < best_score:
            best_score = score_polish
            best_A = A_polish
            
    if best_A is None:
        best_A = list(range(1, n + 1))
        
    return best_A
