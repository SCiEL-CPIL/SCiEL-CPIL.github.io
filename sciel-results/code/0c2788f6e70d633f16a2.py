import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True

def _get_hamming_numbers(n):
    """Generate first n Hamming numbers (5-smooth)."""
    H = [1] * n
    i2 = i3 = i5 = 0
    for i in range(1, n):
        H[i] = min(H[i2]*2, H[i3]*3, H[i5]*5)
        if H[i] == H[i2]*2: i2 += 1
        if H[i] == H[i3]*3: i3 += 1
        if H[i] == H[i5]*5: i5 += 1
    return H

def _generate_candidates(deadline):
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Rational Geometric Progressions
    # Search p/q for small p, q
    # To fit 20 terms in 10000: a * (p/q)^19 <= 10000 => a <= 10000 * (q/p)^19
    # If p/q > 1, a must be small. If p/q < 1, we can reverse the set.
    # We will generate sets and sort them, so order doesn't matter.
    # We want distinct integers.
    
    for p in range(1, 60):
        for q in range(1, 60):
            if p == q: continue
            r = p / q
            if r == 1: continue
            
            # Determine max starting value 'a'
            # If r > 1: a * r^19 <= 10000 => a <= 10000 / r^19
            # If r < 1: a * r^0 = a <= 10000, and a * r^19 >= 1 => a >= 1 / r^19
            # Since we can pick any 'a', let's try to maximize the spread or minimize collisions.
            # Actually, for GPs, the structure is fixed by r. 
            # The set is {a, ar, ar^2, ..., ar^19}.
            # We need integers. So we pick 'a' such that all terms are integers.
            # This implies a must be divisible by q^19 if p/q is irreducible? 
            # No, we just need the final values to be integers.
            # Let's just try integer 'a' values.
            
            # To ensure distinct integers, we can iterate 'a' from 1 to some limit.
            # Limit for 'a' if r>1: 10000 / r^19
            # Limit for 'a' if r<1: 10000 (but min term a*r^19 >= 1 => a >= r^-19)
            
            max_a = 10000
            min_a = 1
            
            if r > 1:
                max_a = int(10000 / (r ** 19)) + 1
                if max_a < 1: continue
            else:
                min_a = int(1 / (r ** 19)) + 1
                if min_a > 10000: continue
            
            # Try a few 'a' values: smallest, largest, and some in between
            # Smallest 'a' minimizes the product set magnitude but might increase sum set if terms are close?
            # Actually, smaller terms => smaller sums. So smallest 'a' is usually good for sumset.
            # Largest 'a' => larger terms => larger sums.
            # However, the *number* of distinct sums/prods depends on the ratio.
            # For a fixed ratio, the structure of collisions is the same.
            # So we just need one valid 'a'. Smallest 'a' is safest for bounds.
            
            a_start = min_a
            A = []
            val = float(a_start)
            valid = True
            for i in range(20):
                iv = int(val + 0.5)
                if iv < 1 or iv > 10000:
                    valid = False
                    break
                if iv in A:
                    valid = False
                    break
                A.append(iv)
                val *= r
            
            if valid and len(A) == 20:
                A.sort()
                if _valid(A):
                    candidates.append(A)
                    
            # Also try a larger 'a' if possible, to see if it helps (unlikely for sumset, but maybe for prods?)
            # Prods scale with a^2, but distinctness is invariant to scaling if integers hold.
            # So smallest 'a' is generally best for keeping numbers small, but distinctness is key.
            
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    # 2. Hamming Number Subsets
    # Hamming numbers have very few product collisions.
    # We want to pick 20 from the first ~100 to minimize sumset.
    # Smaller Hamming numbers => smaller sums.
    # So the first 20 Hamming numbers are a strong candidate.
    H_pool = _get_hamming_numbers(100)
    H_pool = [x for x in H_pool if x <= 10000]
    
    # Candidate 1: First 20
    if len(H_pool) >= 20:
        candidates.append(H_pool[:20])
        
    # Candidate 2: Greedy selection from first 50
    # Start with smallest, add next that minimizes score increase
    if len(H_pool) >= 20:
        A_greedy = [H_pool[0]]
        for _ in range(19):
            best_h = None
            best_s = float('inf')
            for h in H_pool:
                if h in A_greedy: continue
                if h > 10000: break
                new_A = A_greedy + [h]
                # Optimization: only check if h is not too large
                s = _score(new_A)
                if s < best_s:
                    best_s = s
                    best_h = h
            if best_h is not None:
                A_greedy.append(best_h)
            else:
                break
        if len(A_greedy) == 20 and _valid(A_greedy):
            candidates.append(A_greedy)
            
    # 3. Random Hamming Subsets
    rng = random.Random(42)
    for _ in range(20):
        subset = rng.sample(H_pool, 20)
        if _valid(subset):
            candidates.append(subset)

    # 4. Arithmetic Progressions (for comparison)
    for d in range(1, 100):
        A = list(range(1, 20*d + 1, d))
        if _valid(A):
            candidates.append(A)
            
    return candidates

def _local_search(A, max_iter=50000, deadline=None):
    """Aggressive local search."""
    current_A = list(A)
    current_score = _score(current_A)
    best_A = list(current_A)
    best_score = current_score
    rng = random.Random(42)
    n = len(current_A)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        move_type = rng.choice(['single', 'swap', 'kick'])
        
        if move_type == 'single':
            k = rng.randrange(n)
            old_val = current_A[k]
            # Try to replace with a value that is "close" to existing values to maintain structure?
            # Or random.
            if rng.random() < 0.7:
                # Local perturbation
                delta = rng.randint(-20, 20)
                if delta == 0: delta = 1
                new_val = old_val + delta
            else:
                new_val = rng.randint(1, 10000)
                
            if 1 <= new_val <= 10000 and new_val not in current_A:
                current_A[k] = new_val
                new_score = _score(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_A = list(current_A)
                else:
                    current_A[k] = old_val
                    
        elif move_type == 'swap':
            k1, k2 = rng.sample(range(n), 2)
            v1, v2 = current_A[k1], current_A[k2]
            current_A[k1], current_A[k2] = v2, v1
            new_score = _score(current_A)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(current_A)
            else:
                current_A[k1], current_A[k2] = v1, v2
                
        else: # kick
            # Change 2-3 elements
            num_changes = rng.randint(2, 4)
            indices = rng.sample(range(n), num_changes)
            old_vals = [current_A[i] for i in indices]
            for i in indices:
                current_A[i] = rng.randint(1, 10000)
            if len(set(current_A)) == 20 and all(1 <= x <= 10000 for x in current_A):
                new_score = _score(current_A)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_A = list(current_A)
                else:
                    for i, v in zip(indices, old_vals):
                        current_A[i] = v
            else:
                for i, v in zip(indices, old_vals):
                    current_A[i] = v

    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    candidates = _generate_candidates(deadline)
    
    if not candidates:
        candidates = [list(range(1, 21))]

    # Score and sort candidates
    scored_candidates = []
    for A in candidates:
        if _valid(A):
            s = _score(A)
            scored_candidates.append((s, A))
            
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [A for _, A in scored_candidates[:15]]
    
    if not top_candidates:
        return list(range(1, 21))

    best_A = top_candidates[0]
    best_score = scored_candidates[0][0]
    
    # Local Search on top candidates
    # Allocate time to the best few candidates
    time_left = deadline - time.time()
    if time_left > 10:
        # Run LS on top 5
        ls_deadline = deadline - 2.0
        time_per_ls = (ls_deadline - time.time()) / 5.0
        
        for i in range(min(5, len(top_candidates))):
            sub_deadline = time.time() + time_per_ls
            if sub_deadline > ls_deadline:
                sub_deadline = ls_deadline
                
            A_ls, s_ls = _local_search(top_candidates[i], max_iter=100000, deadline=sub_deadline)
            if s_ls < best_score:
                best_score = s_ls
                best_A = A_ls
                
            if time.time() > deadline - 1.0:
                break
                
    # Final Polish
    if time.time() < deadline - 0.5:
        final_deadline = deadline - 0.5
        A_polish, s_polish = _local_search(best_A, max_iter=100000, deadline=final_deadline)
        if s_polish < best_score:
            best_score = s_polish
            best_A = A_polish

    return best_A
