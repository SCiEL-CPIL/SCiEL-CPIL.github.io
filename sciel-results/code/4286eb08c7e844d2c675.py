import time
from math import log, floor
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    # Compute all sums and differences efficiently
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = int(U_arr.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    val = 1.0
    for k in range(1, m + 1):
        val = val * b
        iv = int(round(val))
        if iv <= 0:
            iv = 1
        if iv not in seen:
            U.append(iv)
            seen.add(iv)
    return sorted(U)

def local_search(U, deadline, max_val_limit=2000, max_n_limit=50):
    """
    Local search with element perturbation, addition, and removal.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 40
    temperature = 1.0
    cooling = 0.995
    
    while time.time() < deadline - 3:
        if no_improvement_count > max_no_improvement:
            temperature *= cooling
            if temperature < 0.01:
                break
            no_improvement_count = 0
            max_no_improvement = int(40 / temperature)
            max_no_improvement = min(max_no_improvement, 200)
        
        improved = False
        n = len(U)
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.6:
            # Perturb an element
            if n > 2:
                idx = random.randint(1, n - 1)
                old_val = U[idx]
                # Try several nearby values
                deltas = sorted(random.sample(range(-20, 21), min(15, 41)))
                found = False
                for delta in deltas:
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    new_score = score_set(new_U)
                    if new_score > current_score or (new_score > current_score - 0.001 and random.random() < temperature):
                        U = new_U
                        current_score = new_score
                        improved = True
                        found = True
                        break
                if not found:
                    no_improvement_count += 1
        elif move_type < 0.8:
            # Add an element
            if n < max_n_limit:
                max_val = max(U)
                # Generate candidates near existing structure
                candidates = set()
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val * 2 and s not in U:
                            candidates.add(s)
                        if 0 < d <= max_val * 2 and d not in U:
                            candidates.add(d)
                # Also try some random values
                for _ in range(10):
                    rv = random.randint(1, max_val * 2)
                    if rv not in U:
                        candidates.add(rv)
                
                if candidates:
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for c in cand_list[:15]:
                        new_U = sorted(U + [c])
                        new_score = score_set(new_U)
                        if new_score > current_score or (new_score > current_score - 0.002 and random.random() < temperature):
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                if not improved:
                    no_improvement_count += 1
        else:
            # Remove an element
            if n > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score or (new_score > current_score - 0.001 and random.random() < temperature * 0.5):
                        U = new_U
                        current_score = new_score
                        improved = True
                    else:
                        no_improvement_count += 1
        
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def fine_polish(U, deadline):
    """Fine-grained local search with small perturbations."""
    current = sorted(set(U))
    current_score = score_set(current)
    best = list(current)
    best_score = current_score
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 1 and iteration < 200:
        improved = False
        iteration += 1
        n = len(current)
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() > deadline - 1:
                break
            old_val = current[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if new_val in current:
                    continue
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element near the max
        if not improved and n < 30 and time.time() < deadline - 1:
            max_val = max(current)
            for new_val in range(max(1, max_val - 5), max_val + 6):
                if new_val <= 0:
                    continue
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

        # Try removing an element
        if not improved and n > 3 and time.time() < deadline - 1:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

        if current_score > best_score:
            best_score = current_score
            best = list(current)

    return best_score, best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    best_score = score_set(best_set)

    # 1. Systematic geometric search with fine-grained base
    b_candidates = [b_int / 10.0 for b_int in range(11, 40)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 60:
            break
        for m in range(2, 30):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 2. Fine-tune the best base
    if best_b is not None and time.time() < deadline - 30:
        b_fine = [best_b + (i - 10) * 0.005 for i in range(21)]
        b_fine = [b for b in b_fine if b > 1.0]
        for b in b_fine:
            if time.time() > deadline - 30:
                break
            for m in range(max(2, best_m - 3), best_m + 5):
                U = geometric_set(b, m)
                if len(U) < 2:
                    continue
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                    best_b = b
                    best_m = m

    # 3. Structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        [0, 1, 3, 7, 15, 31, 63, 127, 255],
        [0, 1, 3, 9, 27, 81, 243, 729],
        [0, 1, 2, 5, 12, 29, 70, 169],
        [0, 1, 4, 10, 20, 35, 56, 84, 120],
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45],
        [0, 2, 5, 10, 18, 30, 48, 73],
    ]
    
    for cand in structured_candidates:
        if time.time() > deadline - 40:
            break
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 4. Local search from multiple seeds
    seeds = [best_set]
    for cand in structured_candidates[:4]:
        if cand != best_set:
            seeds.append(cand)
    # Add geometric variants
    if best_b is not None:
        for dm in [-2, -1, 1, 2]:
            m = best_m + dm
            if m >= 2:
                seeds.append(geometric_set(best_b, m))
    
    max_val_limit = 2000
    max_n_limit = 40
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds
            if len(seeds) < 8:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Final polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        polished_score, polished_set = fine_polish(current, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_set

    # Ensure valid output
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
