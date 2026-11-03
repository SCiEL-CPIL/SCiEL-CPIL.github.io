import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    U = tuple(sorted(set(U)))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def fast_score(U):
    """Fast scoring using numpy for vectorized set size computation."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if 0 not in U or len(U) < 2:
        return -1.0
    max_u = int(U.max())
    if max_u == 0:
        return -1.0
    
    n = len(U)
    
    # For small n, NumPy broadcasting is very fast
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    # Count unique elements
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def incremental_score_update(U, sums_set, diffs_set, add_val=None, remove_val=None):
    """
    Update score incrementally when adding/removing an element.
    Returns (new_score, new_sums_set, new_diffs_set)
    """
    U = list(U)
    sums_set = set(sums_set)
    diffs_set = set(diffs_set)
    
    if add_val is not None and add_val not in U:
        U.append(add_val)
        for a in U[:-1]:  # don't double-count with self
            new_sum = a + add_val
            new_diff = a - add_val
            new_diff_rev = add_val - a
            if new_sum not in sums_set:
                sums_set.add(new_sum)
            if new_diff not in diffs_set:
                diffs_set.add(new_diff)
            if new_diff_rev not in diffs_set:
                diffs_set.add(new_diff_rev)
        # Also add add_val + add_val and add_val - add_val
        if add_val + add_val not in sums_set:
            sums_set.add(add_val + add_val)
        # add_val - add_val = 0, already there since 0 in U
    
    if remove_val is not None and remove_val in U:
        # Remove contributions involving remove_val
        U_copy = [x for x in U if x != remove_val]
        # We need to recompute sums/diffs without remove_val
        # For efficiency, we'll just recompute if removal is requested
        # (removal is less common than addition in our search)
        pass
    
    if not U:
        return -1.0, sums_set, diffs_set
    
    U_sorted = sorted(U)
    if 0 not in U_sorted:
        return -1.0, sums_set, diffs_set
    
    max_u = max(U_sorted)
    if max_u == 0:
        return -1.0, sums_set, diffs_set
    
    # Recompute sums and diffs from scratch for correctness when removing
    if remove_val is not None:
        sums_set = set()
        diffs_set = set()
        for a in U_sorted:
            for b in U_sorted:
                sums_set.add(a + b)
                diffs_set.add(a - b)
    
    n_sums = len(sums_set)
    n_diffs = len(diffs_set)
    
    if n_sums == 0:
        return -1.0, sums_set, diffs_set
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return 1.0, sums_set, diffs_set
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1), sums_set, diffs_set

def generate_smart_candidates(U, sums_set, diffs_set, max_u, num_candidates=50):
    """
    Generate candidate values that are likely to improve the score.
    Focus on values that create new differences without creating many new sums.
    """
    candidates = set()
    
    # Strategy 1: Values that are far from existing elements (maximize new diffs)
    # Try values beyond current max
    for offset in range(1, 20):
        candidates.add(max_u + offset)
    
    # Strategy 2: Values in gaps between existing elements
    U_sorted = sorted(U)
    if len(U_sorted) >= 2:
        for i in range(len(U_sorted) - 1):
            gap_start = U_sorted[i]
            gap_end = U_sorted[i + 1]
            gap_size = gap_end - gap_start
            if gap_size > 2:
                # Try middle of gap
                mid = (gap_start + gap_end) // 2
                if mid not in U:
                    candidates.add(mid)
                # Try quarter points
                q1 = gap_start + gap_size // 4
                q3 = gap_start + 3 * gap_size // 4
                if q1 not in U:
                    candidates.add(q1)
                if q3 not in U:
                    candidates.add(q3)
    
    # Strategy 3: Values that are differences of existing elements but not in U
    # These might create interesting structure
    for a in U_sorted:
        for b in U_sorted:
            d = abs(a - b)
            if d > 0 and d not in U and d <= max_u + 10:
                candidates.add(d)
    
    # Strategy 4: Random sampling
    for _ in range(20):
        candidates.add(random.randint(1, max_u + 30))
    
    # Filter: only keep non-negative, not already in U
    candidates = {c for c in candidates if c >= 0 and c not in U}
    
    # Score candidates by potential improvement
    scored_candidates = []
    for c in list(candidates)[:num_candidates * 2]:
        # Quick estimate: how many new diffs vs new sums
        new_diffs = 0
        new_sums = 0
        for a in U:
            d1 = a - c
            d2 = c - a
            s = a + c
            if d1 not in diffs_set:
                new_diffs += 1
            if d2 not in diffs_set:
                new_diffs += 1
            if s not in sums_set:
                new_sums += 1
        # Net gain in ratio: we want high new_diffs, low new_sums
        # Approximate score improvement
        n_sums = len(sums_set)
        n_diffs = len(diffs_set)
        new_ratio = (n_diffs + new_diffs) / (n_sums + new_sums) if (n_sums + new_sums) > 0 else 1
        old_ratio = n_diffs / n_sums if n_sums > 0 else 1
        gain = new_ratio - old_ratio
        scored_candidates.append((gain, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored_candidates[:num_candidates]]

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin
    
    def time_left():
        return time.time() < deadline
    
    best_set = [0, 1, 3]
    best_score = fast_score(best_set)
    
    random.seed(42)

    # Phase 1: Generate diverse seeds
    seeds = []
    
    # Geometric progressions
    for m in range(2, 12):
        for b in [1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            U = [0]
            val = 1
            for k in range(1, m+1):
                val = int(round(b**k))
                if val in U:
                    continue
                U.append(val)
            if len(U) >= 2:
                seeds.append(U)
                
    # Powers of 2
    for m in range(2, 15):
        U = [1 << i for i in range(m)]
        seeds.append(U)
        
    # Super-increasing sequences with random gaps
    for _ in range(30):
        U = [0]
        current = 1
        for _ in range(random.randint(3, 12)):
            current += random.randint(1, 8)
            U.append(current)
        seeds.append(U)

    # Evaluate and sort seeds
    scored_seeds = []
    for U in seeds:
        U = sorted(set(U))
        if 0 not in U:
            U.insert(0, 0)
        s = fast_score(U)
        scored_seeds.append((s, U))
    
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top 5 seeds
    top_seeds = [U for s, U in scored_seeds[:5]]
    
    # Update best
    if top_seeds:
        s, U = scored_seeds[0]
        if s > best_score:
            best_score = s
            best_set = U

    # Phase 2: Beam Search with Incremental Scoring
    beam = [(best_score, best_set)]
    for s, U in scored_seeds[1:5]:
        beam.append((s, U))
    
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:10]
    
    max_size = 15  # Slightly larger to allow more exploration
    
    for step in range(25):
        if not time_left():
            break
            
        new_beam = []
        for score, U in beam:
            if len(U) >= max_size:
                new_beam.append((score, U))
                continue
                
            # Compute current sums and diffs
            U_arr = np.array(U, dtype=np.int64)
            sums_np = U_arr[:, None] + U_arr[None, :]
            diffs_np = U_arr[:, None] - U_arr[None, :]
            sums_set = set(np.unique(sums_np).tolist())
            diffs_set = set(np.unique(diffs_np).tolist())
            
            max_u = max(U)
            
            # Generate smart candidates
            candidates = generate_smart_candidates(U, sums_set, diffs_set, max_u, num_candidates=40)
            
            # Also add some basic candidates
            for i in range(1, min(10, max_u + 1)):
                candidates.append(i)
            for i in range(max(1, max_u - 3), max_u + 15):
                candidates.append(i)
            
            # Deduplicate
            candidates = list(set(candidates))
            
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_U.sort()
                s = fast_score(new_U)
                if s > -1:
                    new_beam.append((s, new_U))
        
        if not new_beam:
            break
            
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:10]
        
        if beam[0][0] > best_score:
            best_score = beam[0][0]
            best_set = beam[0][1]

    # Phase 3: Local Search with Dynamic Candidate Generation
    U = list(best_set)
    current_score = best_score
    
    improved = True
    while improved and time_left():
        improved = False
        n = len(U)
        max_u = max(U)
        
        # Compute current sums and diffs for candidate generation
        U_arr = np.array(U, dtype=np.int64)
        sums_np = U_arr[:, None] + U_arr[None, :]
        diffs_np = U_arr[:, None] - U_arr[None, :]
        sums_set = set(np.unique(sums_np).tolist())
        diffs_set = set(np.unique(diffs_np).tolist())
        
        # 1. Try modifying existing elements
        for i in range(1, n):
            old_val = U[i]
            # Try nearby values with wider range
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0:
                    continue
                if new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > current_score:
                    current_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        
        # 2. Try adding a new element using smart candidates
        if not improved and len(U) < max_size:
            candidates = generate_smart_candidates(U, sums_set, diffs_set, max_u, num_candidates=60)
            
            best_add_score = -1
            best_add_val = -1
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_U.sort()
                s = fast_score(new_U)
                if s > best_add_score:
                    best_add_score = s
                    best_add_val = c
            
            if best_add_score > current_score:
                current_score = best_add_score
                U.append(best_add_val)
                U.sort()
                improved = True
                
        # 3. Try removing an element
        if not improved and len(U) > 3:
            best_rem_score = -1
            best_rem_idx = -1
            for i in range(1, n):
                new_U = U[:i] + U[i+1:]
                s = fast_score(new_U)
                if s > best_rem_score:
                    best_rem_score = s
                    best_rem_idx = i
            
            if best_rem_score > current_score:
                current_score = best_rem_score
                U = U[:best_rem_idx] + U[best_rem_idx+1:]
                improved = True

        if current_score > best_score:
            best_score = current_score
            best_set = U[:]

    # Phase 4: Simulated Annealing with Better Moves
    U = np.array(best_set, dtype=np.int64)
    current_score = best_score
    
    T = 0.02
    T_min = 0.001
    cooling = 0.999
    
    for iteration in range(60000):
        if not time_left():
            break
            
        if T < T_min:
            T = T_min
            
        n = len(U)
        move_type = random.random()
        
        if move_type < 0.5 and n > 2:
            # Modify with larger perturbation
            idx = random.randint(1, n-1)
            old_val = U[idx]
            # Larger Gaussian noise for more exploration
            sigma = max(2, old_val * 0.15)
            new_val = max(0, int(old_val + random.gauss(0, sigma)))
            if new_val == old_val:
                continue
            new_U = U.copy()
            new_U[idx] = new_val
            if len(set(new_U)) != len(new_U):
                continue
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        elif move_type < 0.8 and n < max_size:
            # Add using smart candidate generation
            U_list = U.tolist()
            U_arr = U
            sums_np = U_arr[:, None] + U_arr[None, :]
            diffs_np = U_arr[:, None] - U_arr[None, :]
            sums_set = set(np.unique(sums_np).tolist())
            diffs_set = set(np.unique(diffs_np).tolist())
            max_u = int(U.max())
            
            candidates = generate_smart_candidates(U_list, sums_set, diffs_set, max_u, num_candidates=30)
            if not candidates:
                candidates = [random.randint(1, max_u + 30)]
            
            new_val = random.choice(candidates)
            if new_val in U_list:
                new_val = random.randint(1, max_u + 30)
                if new_val in U_list:
                    continue
            new_U = np.append(U, new_val)
            new_U = np.sort(new_U)
            s = fast_score(new_U)
            delta = s - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                U = new_U
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = sorted(U.tolist())
                    
        else:
            # Remove
            if n > 3:
                idx = random.randint(1, n-1)
                new_U = np.delete(U, idx)
                s = fast_score(new_U)
                delta = s - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    U = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = sorted(U.tolist())
                        
        T *= cooling

    # Final Polish with wider search
    U = list(best_set)
    improved = True
    polish_count = 0
    while improved and time_left() and polish_count < 50:
        improved = False
        polish_count += 1
        n = len(U)
        for i in range(1, n):
            old_val = U[i]
            for delta in range(-3, 4):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                s = fast_score(new_U)
                if s > best_score:
                    best_score = s
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        if improved:
            best_set = U[:]

    return best_set

if __name__ == "__main__":
    res = construct()
    print(compute_score(res))
