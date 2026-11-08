import time
import random
import math
import numpy as np

def _score_set(U):
    """Compute the score for a given set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set using NumPy for speed
    arr = np.array(sorted(U_set), dtype=np.int64)
    n = len(arr)
    # Sumset: all a+b for a,b in U
    sums = np.add.outer(arr, arr).ravel()
    # Diffset: all a-b for a,b in U
    diffs = np.subtract.outer(arr, arr).ravel()
    
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _score_set_fast(U):
    """Faster scoring for small sets using set operations."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _generate_ap_candidate(start, diff, length, max_val=200):
    """Generate an arithmetic progression candidate."""
    if start < 0 or diff <= 0 or length < 1:
        return None
    U = set()
    for i in range(length):
        val = start + i * diff
        if val > max_val:
            break
        U.add(val)
    if 0 not in U:
        return None
    return sorted(U)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_score = _score_set([0, 1, 3])
    best_set = [0, 1, 3]
    
    # Generate diverse initial candidates
    initial_candidates = []
    
    # 1. Small exhaustive search
    from itertools import combinations
    for size in range(3, 7):
        for combo in combinations(range(1, 12), size - 1):
            U = [0] + list(combo)
            s = _score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            initial_candidates.append((s, U[:]))
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # 2. Arithmetic progression based candidates
    for start in range(0, 5):
        for diff in range(1, 15):
            for length in range(2, 8):
                U = _generate_ap_candidate(start, diff, length, max_val=100)
                if U and len(U) >= 2:
                    s = _score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
                    initial_candidates.append((s, U[:]))
    
    # 3. Union of two APs with growing differences
    for d1 in range(1, 10):
        for l1 in range(2, 6):
            for d2 in range(d1 * 2, d1 * 8):
                for l2 in range(2, 5):
                    U = set()
                    for i in range(l1):
                        U.add(i * d1)
                    base2 = l1 * d1
                    for i in range(l2):
                        U.add(base2 + i * d2)
                    if 0 in U and len(U) >= 3:
                        U_list = sorted(U)
                        s = _score_set_fast(U_list)
                        if s > best_score:
                            best_score = s
                            best_set = U_list[:]
                        initial_candidates.append((s, U_list[:]))
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Sort and keep top candidates
    initial_candidates.sort(key=lambda x: -x[0])
    top_candidates = [c[1] for c in initial_candidates[:15]]
    
    # Simulated annealing with better moves
    random.seed(42)
    
    for init_set in top_candidates[:6]:
        if time.time() > deadline:
            break
        
        current_set = init_set[:]
        current_score = _score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
        
        temperature = 0.05
        no_improve = 0
        
        for step in range(8000):
            if time.time() > deadline:
                break
            
            # Generate a move
            move_type = random.random()
            new_set = current_set[:]
            
            if move_type < 0.25:
                # Add an element - try positions that might help
                max_elem = max(current_set) if current_set else 0
                # Try adding near gaps or at structured positions
                candidates = []
                for d in range(1, 10):
                    candidates.append(max_elem + d)
                    candidates.append(max_elem - d)
                # Also try random
                candidates.append(random.randint(1, max_elem + 20))
                # Filter valid
                valid = [x for x in candidates if x > 0 and x not in new_set]
                if valid:
                    new_set.append(random.choice(valid))
            elif move_type < 0.4:
                # Remove an element
                if len(new_set) > 2:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        new_set.remove(random.choice(non_zero))
            elif move_type < 0.6:
                # Replace an element with nearby value
                if len(new_set) > 2:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        idx = random.choice(non_zero)
                        new_val = idx + random.choice([-2, -1, 1, 2, 3])
                        if new_val > 0 and new_val not in new_set:
                            new_set.remove(idx)
                            new_set.append(new_val)
            elif move_type < 0.8:
                # Shift an element
                if len(new_set) > 2:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        idx = random.choice(non_zero)
                        new_val = idx + random.choice([-1, 1])
                        if new_val > 0 and new_val not in new_set:
                            new_set.remove(idx)
                            new_set.append(new_val)
            elif move_type < 0.9:
                # Try to create AP structure: pick two elements and insert midpoint or extend
                if len(new_set) >= 2:
                    non_zero = [x for x in new_set if x != 0]
                    if len(non_zero) >= 2:
                        a, b = random.sample(non_zero, 2)
                        if b > a:
                            # Try to fill in AP
                            if (b - a) % 2 == 0:
                                mid = (a + b) // 2
                                if mid > 0 and mid not in new_set:
                                    new_set.append(mid)
                            # Or extend
                            diff = b - a
                            new_val = b + diff
                            if new_val not in new_set and new_val < max(new_set) + 50:
                                new_set.append(new_val)
            else:
                # Random perturbation of multiple elements
                if len(new_set) > 3:
                    non_zero = [x for x in new_set if x != 0]
                    if len(non_zero) >= 2:
                        for _ in range(2):
                            idx = random.choice(non_zero)
                            new_val = idx + random.randint(-3, 3)
                            if new_val > 0 and new_val not in new_set:
                                new_set.remove(idx)
                                new_set.append(new_val)
            
            if not new_set or 0 not in new_set:
                continue
            
            new_score = _score_set(new_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_set = new_set
                current_score = new_score
                no_improve = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
                    no_improve = 0
            else:
                no_improve += 1
            
            # Cool down
            temperature *= 0.9995
            
            # If stuck, restart from best
            if no_improve > 500:
                if best_score > current_score:
                    current_set = best_set[:]
                    current_score = best_score
                no_improve = 0
                temperature = 0.05
    
    # Final polish: try local improvements on best set
    if time.time() < deadline - 10:
        current_set = best_set[:]
        current_score = best_score
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Try single element changes
            improved = False
            for i, elem in enumerate(current_set):
                if elem == 0:
                    continue
                for delta_val in range(-5, 6):
                    if delta_val == 0:
                        continue
                    new_elem = elem + delta_val
                    if new_elem <= 0 or new_elem in current_set:
                        continue
                    new_set = current_set[:]
                    new_set[i] = new_elem
                    new_score = _score_set(new_set)
                    if new_score > current_score + 1e-10:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try adding or removing
                if len(current_set) > 2 and random.random() < 0.5:
                    non_zero = [x for x in current_set if x != 0]
                    if non_zero:
                        idx = random.choice(non_zero)
                        new_set = [x for x in current_set if x != idx]
                        new_score = _score_set(new_set)
                        if new_score > current_score + 1e-10:
                            current_set = new_set
                            current_score = new_score
                else:
                    max_elem = max(current_set) if current_set else 0
                    for d in range(1, 10):
                        new_elem = max_elem + d
                        if new_elem not in current_set:
                            new_set = current_set + [new_elem]
                            new_score = _score_set(new_set)
                            if new_score > current_score + 1e-10:
                                current_set = new_set
                                current_score = new_score
                                break
                    if time.time() > deadline:
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
    
    return sorted(best_set)
