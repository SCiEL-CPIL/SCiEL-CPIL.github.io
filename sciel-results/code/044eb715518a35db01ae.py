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
    
    # 1. Hamming Numbers based
    # The first 20
    H = _get_hamming_numbers(20)
    if _valid(H):
        candidates.append(H)
        
    # Try scaling Hamming numbers? 
    # Scaling doesn't change |A+A| or |A*A|. 
    # But maybe picking a shifted set of Hamming numbers helps avoid small integer collisions?
    # No, collisions are structural.
    
    # Try random subsets of a larger pool of Hamming numbers
    H_pool = _get_hamming_numbers(100)
    # Filter to <= 10000
    H_pool = [x for x in H_pool if x <= 10000]
    
    rng = random.Random(123)
    for _ in range(10):
        # Pick 20 random distinct Hamming numbers
        # Ensure they are distinct
        subset = rng.sample(H_pool, 20)
        if _valid(subset):
            candidates.append(subset)
            
    # 2. Geometric Progressions (rational ratios)
    for p in range(2, 50):
        for q in range(1, p):
            r = p / q
            max_a = int(10000 / (r ** 19))
            if max_a < 1:
                continue
            # Try largest 'a'
            a = max_a
            A = []
            val = float(a)
            valid = True
            for i in range(20):
                iv = int(val + 0.5)
                if iv > 10000 or iv < 1:
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
            
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    # 3. Arithmetic Progressions
    for d in range(1, 500):
        A = list(range(1, 20*d + 1, d))
        if _valid(A):
            candidates.append(A)
            
    return candidates

def _local_search(A, max_iter=50000, deadline=None):
    """Aggressive local search with re-heating."""
    current_A = list(A)
    current_score = _score(current_A)
    best_A = list(current_A)
    best_score = current_score
    rng = random.Random(42)
    n = len(current_A)
    no_improve_count = 0
    max_no_improve = 1000
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        move_type = rng.choice(['single', 'swap', 'kick'])
        
        if move_type == 'single':
            k = rng.randrange(n)
            old_val = current_A[k]
            # Generate candidate replacements
            # Mix of local and global
            if rng.random() < 0.5:
                # Local perturbation
                delta = rng.randint(-10, 10)
                if delta == 0: delta = 1
                new_val = old_val + delta
            else:
                # Global random
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
                    no_improve_count = 0
                else:
                    current_A[k] = old_val
                    no_improve_count += 1
                    
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
                no_improve_count = 0
            else:
                current_A[k1], current_A[k2] = v1, v2
                no_improve_count += 1
                
        else: # kick
            # Randomly change 1-3 elements to escape local minima
            num_changes = rng.randint(1, 3)
            indices = rng.sample(range(n), num_changes)
            old_vals = [current_A[i] for i in indices]
            for i in indices:
                current_A[i] = rng.randint(1, 10000)
            # Check validity (distinctness)
            if len(set(current_A)) == 20 and all(1 <= x <= 10000 for x in current_A):
                new_score = _score(current_A)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_A = list(current_A)
                    no_improve_count = 0
                else:
                    # Revert
                    for i, v in zip(indices, old_vals):
                        current_A[i] = v
                    no_improve_count += 1
            else:
                # Revert
                for i, v in zip(indices, old_vals):
                    current_A[i] = v
                no_improve_count += 1

        if not improved:
            # If stuck for too long, do a bigger kick?
            pass

    return best_A, best_score

def construct():
    start_time = time.time()
    # Use 85s deadline to be safe
    deadline = start_time + 85.0

    # 1. Generate Candidates
    candidates = _generate_candidates(deadline)
    
    if not candidates:
        # Fallback
        candidates = [list(range(1, 21))]

    # Score candidates and keep top 10
    scored_candidates = []
    for A in candidates:
        s = _score(A)
        scored_candidates.append((s, A))
        
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [A for _, A in scored_candidates[:10]]
    
    # 2. Local Search
    best_A = None
    best_score = float('inf')
    
    # Allocate remaining time evenly among top candidates?
    # Or just run until deadline?
    # Let's run LS on each top candidate with a sub-deadline.
    
    time_left = deadline - time.time()
    if time_left < 5:
        # Not enough time for LS, return best candidate
        if top_candidates:
            return top_candidates[0]
        else:
            return list(range(1, 21))
            
    # Run LS on top 5 candidates
    ls_deadline = deadline - 2.0 # Reserve 2s for final checks
    time_per_ls = (ls_deadline - time.time()) / len(top_candidates[:5])
    
    for i, A in enumerate(top_candidates[:5]):
        sub_deadline = time.time() + time_per_ls
        # Adjust sub_deadline to not exceed global deadline
        if sub_deadline > ls_deadline:
            sub_deadline = ls_deadline
            
        A_ls, s_ls = _local_search(A, max_iter=50000, deadline=sub_deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_A = A_ls
            
        # Check if we have time for more
        if time.time() > deadline - 1.0:
            break
            
    # 3. Final Polish on the best result
    if time.time() < deadline - 0.5:
        final_deadline = deadline - 0.5
        A_polish, s_polish = _local_search(best_A, max_iter=50000, deadline=final_deadline)
        if s_polish < best_score:
            best_score = s_polish
            best_A = A_polish

    return best_A
