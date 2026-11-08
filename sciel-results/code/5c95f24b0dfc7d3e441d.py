import time
import math
import random
import numpy as np

def compute_score(U):
    """Compute the score for set U."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    n = len(U)
    
    if n <= 50:
        # Use Python sets for small sets (faster overhead)
        sum_set = set()
        for i in range(n):
            for j in range(i, n):
                sum_set.add(U[i] + U[j])
        
        diff_set = set()
        for i in range(n):
            for j in range(n):
                diff_set.add(U[i] - U[j])
    else:
        # Use numpy for larger sets
        U_arr = np.array(U, dtype=np.int64)
        # Sum set: use broadcasting for pairs i<=j
        sums = U_arr[:, None] + U_arr[None, :]
        # Only take upper triangle (i<=j) to avoid duplicates, but we need unique values
        # Actually, we need all pairwise sums including i=j, so just take all and unique
        sum_set = set(np.unique(sums).tolist())
        
        # Difference set: all pairwise differences
        diffs = U_arr[:, None] - U_arr[None, :]
        diff_set = set(np.unique(diffs).tolist())
    
    num_sum = len(sum_set)
    num_diff = len(diff_set)
    
    if num_sum == 0:
        return 0.0
    
    ratio = num_diff / num_sum
    if ratio <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_base = math.log(2 * maxU + 1)
    
    if log_base <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_base

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Initial set
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Try various initial constructions
    initial_candidates = []
    
    # Geometric progression
    for base in [2, 3, 4, 5, 6, 7, 8, 10]:
        for length in range(3, 18):
            s = [0]
            val = 1
            for _ in range(length):
                s.append(val)
                val *= base
            initial_candidates.append(s)
    
    # Arithmetic progression
    for n in range(3, 25):
        initial_candidates.append(list(range(n)))
    
    # Mixed: small AP + sparse large elements
    for small_n in [2, 3, 4, 5, 6]:
        for large_count in [2, 3, 4, 5, 6]:
            for gap in [5, 10, 20, 50, 100, 200]:
                s = list(range(small_n))
                for i in range(1, large_count + 1):
                    s.append(i * gap)
                initial_candidates.append(s)
    
    # Power-of-2 based constructions
    for k in range(3, 12):
        s = [0]
        val = 1
        for i in range(k):
            s.append(val)
            val *= 2
        initial_candidates.append(s)
    
    # Fibonacci-like
    for k in range(3, 15):
        s = [0, 1]
        for i in range(k - 2):
            s.append(s[-1] + s[-2])
        initial_candidates.append(s)
    
    # Binary representation patterns
    for n in range(3, 12):
        for mask in range(1, min(1 << n, 100)):
            s = [0]
            for i in range(n):
                if mask & (1 << i):
                    s.append(1 << i)
            if len(s) >= 2:
                initial_candidates.append(s)
    
    # Evaluate all initial candidates
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        score = compute_score(cand)
        if score > best_score:
            best_score = score
            best_set = cand[:]
    
    # Local search
    def try_modify(U, modification):
        """Try a modification and return (new_set, new_score) or None."""
        new_U = U[:]
        
        if modification[0] == 'add':
            elem = modification[1]
            if elem in new_U:
                return None
            new_U.append(elem)
        elif modification[0] == 'remove':
            elem = modification[1]
            if elem not in new_U:
                return None
            new_U.remove(elem)
            if len(new_U) < 2:
                return None
        elif modification[0] == 'replace':
            old, new = modification[1], modification[2]
            if old not in new_U or new in new_U:
                return None
            idx = new_U.index(old)
            new_U[idx] = new
        
        new_U = sorted(set(new_U))
        if 0 not in new_U:
            return None
        score = compute_score(new_U)
        return (new_U, score)
    
    # Local search iterations
    max_iters = 8000
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        U = best_set[:]
        maxU = max(U)
        n = len(U)
        
        # Generate candidate modifications
        candidates = []
        
        # Add elements
        # Try elements near existing ones
        for elem in U:
            for delta in [-1, 1, -2, 2, -3, 3, -5, 5, -10, 10]:
                new_elem = elem + delta
                if new_elem >= 0 and new_elem not in U:
                    candidates.append(('add', new_elem))
        
        # Try some random elements in various ranges
        for _ in range(8):
            new_elem = random.randint(1, max(1, maxU * 2))
            if new_elem not in U:
                candidates.append(('add', new_elem))
        
        # Try larger elements
        for _ in range(5):
            new_elem = random.randint(maxU + 1, maxU * 4 + 20)
            if new_elem not in U:
                candidates.append(('add', new_elem))
        
        # Try small elements
        for _ in range(3):
            new_elem = random.randint(1, max(1, min(maxU, 100)))
            if new_elem not in U:
                candidates.append(('add', new_elem))
        
        # Remove elements (except 0)
        for elem in U:
            if elem != 0:
                candidates.append(('remove', elem))
        
        # Replace elements
        for i, elem in enumerate(U):
            if elem == 0:
                continue
            for delta in [-1, 1, -2, 2, -3, 3, -5, 5]:
                new_elem = elem + delta
                if new_elem >= 0 and new_elem not in U:
                    candidates.append(('replace', elem, new_elem))
            # Random replacement
            for _ in range(3):
                new_elem = random.randint(1, max(1, maxU * 2))
                if new_elem not in U:
                    candidates.append(('replace', elem, new_elem))
        
        # Evaluate candidates and pick the best improvement
        best_cand = None
        best_cand_score = best_score
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            result = try_modify(U, cand)
            if result is not None:
                new_U, new_score = result
                if new_score > best_cand_score:
                    best_cand_score = new_score
                    best_cand = new_U
        
        if best_cand is not None:
            best_set = best_cand
            best_score = best_cand_score
        else:
            # No improvement, try random perturbations
            improved = False
            for _ in range(15):
                if time.time() > deadline:
                    break
                # Random perturbation
                new_U = best_set[:]
                if len(new_U) > 2:
                    # Randomly modify 1-3 elements
                    for _ in range(random.randint(1, 3)):
                        idx = random.randint(1, len(new_U) - 1)  # Don't modify 0
                        new_elem = random.randint(1, max(new_U) * 3)
                        if new_elem not in new_U:
                            new_U[idx] = new_elem
                new_U = sorted(set(new_U))
                if 0 in new_U and len(new_U) >= 2:
                    score = compute_score(new_U)
                    if score > best_score:
                        best_set = new_U
                        best_score = score
                        improved = True
                        break
            
            if not improved:
                # Accept some non-improving moves to escape local optima
                if random.random() < 0.15:
                    new_U = best_set[:]
                    if len(new_U) > 2:
                        idx = random.randint(1, len(new_U) - 1)
                        new_elem = random.randint(1, max(new_U) * 4)
                        new_U[idx] = new_elem
                    new_U = sorted(set(new_U))
                    if 0 in new_U and len(new_U) >= 2:
                        score = compute_score(new_U)
                        if score > best_score * 0.97:  # Accept slightly worse
                            best_set = new_U
                            best_score = score
    
    # Final polish: fine-grained local search
    polish_iters = 2000
    for _ in range(polish_iters):
        if time.time() > deadline:
            break
        
        U = best_set[:]
        maxU = max(U)
        n = len(U)
        
        candidates = []
        
        # Fine-grained additions
        for elem in U:
            for delta in [-1, 1, -2, 2, -3, 3]:
                new_elem = elem + delta
                if new_elem >= 0 and new_elem not in U:
                    candidates.append(('add', new_elem))
        
        # Fine-grained replacements
        for elem in U:
            if elem == 0:
                continue
            for delta in [-1, 1, -2, 2, -3, 3]:
                new_elem = elem + delta
                if new_elem >= 0 and new_elem not in U:
                    candidates.append(('replace', elem, new_elem))
        
        # Small random additions
        for _ in range(5):
            new_elem = random.randint(1, max(1, maxU + 10))
            if new_elem not in U:
                candidates.append(('add', new_elem))
        
        best_cand = None
        best_cand_score = best_score
        
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            result = try_modify(U, cand)
            if result is not None:
                new_U, new_score = result
                if new_score > best_cand_score:
                    best_cand_score = new_score
                    best_cand = new_U
        
        if best_cand is not None:
            best_set = best_cand
            best_score = best_cand_score
        else:
            break  # No improvement in polish, stop
    
    return sorted(best_set)
