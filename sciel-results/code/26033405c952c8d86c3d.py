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

def _generate_smooth_numbers(limit=10000, primes=[2, 3, 5, 7, 11]):
    """Generate all numbers <= limit with prime factors in 'primes'."""
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

def _generate_candidates(n=20, max_val=10000):
    candidates = []
    
    # 1. Geometric Progressions
    for base in [2, 3, 4, 5]:
        gp = [base ** i for i in range(n)]
        if gp[-1] <= max_val:
            candidates.append(gp)
            
    # 2. Two Geometric Progressions
    # e.g., small powers of 2 and large powers of 3
    base1, base2 = 2, 3
    n1, n2 = n // 2, n - n // 2
    gp1 = [base1 ** i for i in range(n1)]
    # Scale gp2 to be larger than gp1
    start_val = max(gp1) * 2
    gp2 = []
    val = start_val
    # Find a base for the second part
    # Just use random smooth numbers for the second part to ensure distinctness and range
    smooths = _generate_smooth_numbers(max_val)
    # Pick n2 smooths that are > max(gp1)
    large_smooths = [x for x in smooths if x > max(gp1)]
    if len(large_smooths) >= n2:
        # Pick evenly spaced from the log scale of large_smooths
        indices = np.linspace(0, len(large_smooths)-1, n2).astype(int)
        gp2 = [large_smooths[i] for i in indices]
        candidates.append(gp1 + gp2)
        
    # 3. Smooth Numbers (Random subsets)
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

    # 4. Arithmetic Progressions
    for d in [2, 3, 5, 10, 20]:
        for a in [1, 2, 3]:
            ap = [a + i * d for i in range(n)]
            if ap[-1] <= max_val:
                candidates.append(ap)
                
    # 5. Mixed: Small consecutive + Large smooth
    small = list(range(1, 11))
    large_smooths = [x for x in smooths if x > 100]
    if len(large_smooths) >= 10:
        idx = np.linspace(0, len(large_smooths)-1, 10).astype(int)
        large = [large_smooths[i] for i in idx]
        cand = small + large
        if len(set(cand)) == n:
            candidates.append(cand)

    # 6. Random
    for _ in range(5):
        r = random.sample(range(1, max_val + 1), n)
        candidates.append(r)

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

def _local_search(A, max_iter=2000, deadline=None, seed=None):
    """
    Local search with Replace and Swap moves.
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
            for delta in [-5, -2, -1, 1, 2, 5]:
                nv = old_val + delta
                if 1 <= nv <= max_val and nv not in current_A:
                    candidates.append(nv)
            # Random smooth number
            if smooths:
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
        
        # 2. If no replace improvement, try Swap moves
        if not improved:
            for i in range(n):
                for j in range(i + 1, n):
                    # Swap A[i] and A[j]
                    current_A[i], current_A[j] = current_A[j], current_A[i]
                    new_score = _score(current_A)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_A = current_A[:]
                        break
                    else:
                        current_A[i], current_A[j] = current_A[j], current_A[i]
                    
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Local minimum reached
            # Try a random kick to escape?
            # For now, just break and let the outer loop try another start
            break
            
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
        if time.time() > deadline - 5: # Leave 5s for final polish
            break
            
        # Check if we've already searched a very similar structure
        # Simple check: if the sorted tuple is the same, skip
        key = tuple(sorted(c))
        if key in seen_structures:
            continue
        seen_structures.add(key)
        
        A_res, score_res = _local_search(c, max_iter=1000, deadline=deadline, seed=search_count)
        if score_res < best_score:
            best_score = score_res
            best_A = A_res[:]
            
        search_count += 1
        if search_count >= 5:
            break
            
    # Final polish on the best solution found
    if best_A is not None and time.time() < deadline - 2:
        A_polish, score_polish = _local_search(best_A, max_iter=5000, deadline=deadline, seed=999)
        if score_polish < best_score:
            best_score = score_polish
            best_A = A_polish
            
    if best_A is None:
        best_A = list(range(1, n + 1))
        
    return best_A
