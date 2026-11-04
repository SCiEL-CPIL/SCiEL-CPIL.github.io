import time
from math import log, floor, ceil
import itertools
import numpy as np
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster scoring using set operations for small sets, numpy for larger."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    if n <= 25:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums_arr = U_arr[:, None] + U_arr[None, :]
        sums = set(np.unique(sums_arr).tolist())
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        diffs = set(np.unique(diffs_arr).tolist())
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def _score_set_fast_numpy(U):
    """Numpy-only scoring path for repeated calls in local search."""
    U = np.asarray(U, dtype=np.int64)
    n = len(U)
    if n < 2:
        return -1.0
    U_sorted = np.sort(U)
    if U_sorted[0] != 0:
        return -1.0
    sums_arr = U_sorted[:, None] + U_sorted[None, :]
    diffs_arr = U_sorted[:, None] - U_sorted[None, :]
    n_sums = len(np.unique(sums_arr))
    n_diffs = len(np.unique(diffs_arr))
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1.0
    D = 2 * int(U_sorted[-1]) + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set_fast(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        
        if not improved:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        
        if current_score > best_local_score:
            best_local_score = current_score
            best_local_set = list(U)
            
    return best_local_set, best_local_score

def construct():
    # Deadline: evaluator kills at 150s, use 140s safety margin
    deadline = time.time() + 140
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Exhaustive search for small sets (n up to 5, values up to 40)
    max_val = 40
    for n in range(3, 6):
        if time.time() > deadline - 30:
            break
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline - 30:
                break
            U = (0,) + combo
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Structured constructions
    if time.time() < deadline - 15:
        # Power of 2
        for m in range(3, 12):
            U = [0] + [2**k for k in range(1, m + 1)]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Mixed: powers of 2 with offsets
        for m in range(3, 10):
            for offset in range(0, 5):
                U = [0] + [(2**k + offset) for k in range(1, m + 1)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
        
        # Fibonacci-like
        for m in range(3, 15):
            fib = [0, 1]
            for _ in range(m - 1):
                fib.append(fib[-1] + fib[-2])
            U = fib[:m + 1]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Squares
        for m in range(3, 12):
            U = [k*k for k in range(m)]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Triangular numbers
        for m in range(3, 12):
            U = [k*(k+1)//2 for k in range(m)]
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Binary basis vectors (powers of 2) with perturbations
        for m in range(3, 10):
            base = [0] + [2**k for k in range(1, m)]
            for perturb in range(0, 4):
                U = [x + (perturb if i > 0 and i % 2 == 0 else 0) for i, x in enumerate(base)]
                U = sorted(set(U))
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
        
        # Sparse geometric: b^k with small b
        for b in [3, 4, 5, 7, 11]:
            for m in range(3, 10):
                U = [0] + [b**k for k in range(1, m)]
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Phase 3: Geometric progression search with ternary refinement
    if time.time() < deadline - 10:
        for m in range(2, 12):
            if time.time() > deadline - 10:
                break
            
            b_low = 1.01
            b_high = 25.0
            
            # Coarse grid search
            best_b = None
            best_s_coarse = -1
            for i in range(80):
                if time.time() > deadline - 10:
                    break
                b = b_low + (b_high - b_low) * i / 79.0
                U = geometric_set(b, m)
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_s_coarse:
                    best_s_coarse = s
                    best_b = b
            
            if best_b is not None and time.time() < deadline - 10:
                # Ternary search around best_b
                lo = max(b_low, best_b * 0.7)
                hi = min(b_high, best_b * 1.3)
                for _ in range(40):
                    if time.time() > deadline - 10:
                        break
                    m1 = lo + (hi - lo) / 3
                    m2 = hi - (hi - lo) / 3
                    
                    U1 = geometric_set(m1, m)
                    s1 = score_set_fast(U1) if len(U1) >= 2 else -1
                    
                    U2 = geometric_set(m2, m)
                    s2 = score_set_fast(U2) if len(U2) >= 2 else -1
                    
                    if s1 < s2:
                        lo = m1
                    else:
                        hi = m2
                
                b_opt = (lo + hi) / 2
                U = geometric_set(b_opt, m)
                if len(U) >= 2:
                    s = score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # Phase 4: Local perturbation search around best set
    if time.time() < deadline - 3:
        current_set = set(best_set)
        
        # Try adding elements
        for new_elem in range(1, 400):
            if time.time() > deadline - 3:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set_fast(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
        
        # Try removing elements
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline - 3:
                    break
                if elem == 0:
                    continue
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set_fast(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
        
        # Try shifting elements
        for _ in range(8):
            if time.time() > deadline - 3:
                break
            improved = False
            for i, elem in enumerate(list(best_set)):
                if time.time() > deadline - 3:
                    break
                if elem == 0:
                    continue
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_elem = elem + delta
                    if new_elem < 0 or new_elem in best_set:
                        continue
                    new_set = list(best_set)
                    new_set[i] = new_elem
                    new_set = sorted(set(new_set))
                    if 0 not in new_set or len(new_set) < 2:
                        continue
                    s = score_set_fast(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        improved = True
                        break
            if not improved:
                break
        
        # Scale the set
        for scale in [2, 3, 5]:
            if time.time() > deadline - 3:
                break
            scaled = [0] + [x * scale for x in best_set if x > 0]
            s = score_set_fast(scaled)
            if s > best_score:
                best_score = s
                best_set = scaled
    
    # Phase 5: Dynamic local search with candidate generation (from inspiration 2)
    if time.time() < deadline - 5:
        best_set_ls, best_score_ls = local_search_dynamic(
            best_set, 
            deadline - 5,
            max_val_limit=3000,
            max_n_limit=30
        )
        if best_score_ls > best_score:
            best_score = best_score_ls
            best_set = best_set_ls
    
    # Phase 6: Simulated annealing for final polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        T = 0.5
        T_min = 0.01
        
        for step in range(500):
            if time.time() > deadline - 2:
                break
            if T < T_min:
                break
            T *= 0.97
            
            # Random move: add, remove, or shift an element
            move_type = np.random.choice(3)
            
            if move_type == 0 and len(current) < 15:
                # Add a random element
                new_elem = np.random.randint(1, 500)
                if new_elem not in current:
                    new_set = sorted(set(current) | {new_elem})
                    s = score_set_fast(new_set)
                    if s > current_score or np.random.random() < np.exp((s - current_score) / T):
                        current = new_set
                        current_score = s
            elif move_type == 1 and len(current) > 2:
                # Remove a random element
                idx = np.random.randint(1, len(current))
                new_set = sorted(set(current) - {current[idx]})
                if len(new_set) >= 2:
                    s = score_set_fast(new_set)
                    if s > current_score or np.random.random() < np.exp((s - current_score) / T):
                        current = new_set
                        current_score = s
            else:
                # Shift a random element
                idx = np.random.randint(1, len(current))
                delta = np.random.randint(-20, 21)
                new_elem = current[idx] + delta
                if new_elem >= 0 and new_elem not in current:
                    new_set = list(current)
                    new_set[idx] = new_elem
                    new_set = sorted(set(new_set))
                    if 0 in new_set and len(new_set) >= 2:
                        s = score_set_fast(new_set)
                        if s > current_score or np.random.random() < np.exp((s - current_score) / T):
                            current = new_set
                            current_score = s
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
    
    # Phase 7: High-precision local polish — exhaustive single-element shifts with wider range
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i, elem in enumerate(list(best_set)):
                if time.time() > deadline - 1:
                    break
                if elem == 0:
                    continue
                for delta in range(-50, 51):
                    if delta == 0:
                        continue
                    new_elem = elem + delta
                    if new_elem < 0 or new_elem in best_set:
                        continue
                    new_set = list(best_set)
                    new_set[i] = new_elem
                    new_set = sorted(set(new_set))
                    if 0 not in new_set or len(new_set) < 2:
                        continue
                    s = score_set_fast(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
        
        # Try adding one element at a time with wider range
        if time.time() < deadline - 1 and len(best_set) < 20:
            current_set = set(best_set)
            for new_elem in range(1, 1000):
                if time.time() > deadline - 1:
                    break
                if new_elem in current_set:
                    continue
                new_set = current_set | {new_elem}
                s = score_set_fast(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
                    current_set = new_set
    
    # Phase 8: Final geometric refinement around best set's implicit base
    if time.time() < deadline - 1:
        # Estimate effective base from best_set
        if len(best_set) >= 3:
            ratios = []
            for i in range(1, len(best_set)):
                if best_set[i] > best_set[i-1] > 0:
                    r = best_set[i] / best_set[i-1]
                    if r > 1.01 and r < 20:
                        ratios.append(r)
            if ratios:
                median_ratio = float(np.median(ratios))
                for m in range(max(2, len(best_set)-2), len(best_set)+3):
                    if time.time() > deadline - 1:
                        break
                    for frac in range(-20, 21):
                        b = median_ratio * (1.0 + frac * 0.005)
                        if b <= 1.01:
                            continue
                        U = geometric_set(b, m)
                        if len(U) < 2:
                            continue
                        s = score_set_fast(U)
                        if s > best_score:
                            best_score = s
                            best_set = list(U)
    
    return best_set
