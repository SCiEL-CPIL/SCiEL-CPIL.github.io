import time
import random
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_iter=500):
    """
    Local search with move/swap/add/remove operations.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        move_type = random.choice(['move', 'swap', 'add', 'remove', 'move', 'swap'])
        
        new_set = current[:]
        
        if move_type == 'move' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Move within a window proportional to the gap structure
            window = max(3, int(max(current) * 0.05))
            new_val = old_val + random.randint(-window, window)
            if new_val <= 0:
                new_val = old_val + random.randint(1, window)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
            
        elif move_type == 'swap' and n > 1:
            i = random.randint(1, n-1)
            max_val = max(current) + 10
            new_val = random.randint(1, max_val)
            if new_val in current:
                continue
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
        elif move_type == 'add' and n < 25:
            # Try adding near existing elements or in gaps
            max_val = max(current) + 20
            if random.random() < 0.5:
                # Add near a random existing element
                base = random.choice(current[1:])
                new_val = base + random.randint(1, 5)
            else:
                new_val = random.randint(1, max_val)
            if new_val in current or new_val <= 0:
                continue
            new_set.append(new_val)
            new_set = sorted(new_set)
            
        elif move_type == 'remove' and n > 4:
            i = random.randint(1, n-1)
            new_set.pop(i)
            
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            no_improve += 1
            # Accept worse moves with simulated annealing style
            if random.random() < max(0.05, 0.3 * (1 - no_improve / 100.0)):
                current = new_set
                current_score = new_score
        
        if no_improve > 200:
            break
    
    return best, best_score

def greedy_build(deadline, max_elems=20):
    """
    Greedily build a set by adding the element that maximizes score improvement.
    """
    current = [0]
    current_score = 1.0  # Degenerate case
    
    for step in range(max_elems):
        if time.time() > deadline:
            break
        
        n = len(current)
        max_val = max(current) * 2 + 50 if n > 1 else 100
        if max_val < 50:
            max_val = 50
        
        best_candidate = None
        best_candidate_score = current_score
        
        # Generate diverse candidates
        candidates = set()
        
        # Random candidates
        for _ in range(80):
            candidates.add(random.randint(1, max_val))
        
        # Near existing elements
        for elem in current:
            for d in range(1, 6):
                candidates.add(elem + d)
        
        # In gaps
        for i in range(1, n):
            gap = current[i] - current[i-1]
            if gap > 3:
                candidates.add((current[i] + current[i-1]) // 2)
        
        # Beyond max
        for d in range(1, 15):
            candidates.add(max(current) + d)
        
        current_set = set(current)
        candidates = [c for c in candidates if c not in current_set and c >= 1]
        
        if not candidates:
            break
        
        # Evaluate top candidates
        if len(candidates) > 60:
            candidates = random.sample(candidates, 60)
        
        for c in candidates:
            new_U = sorted(current_set | {c})
            s = fast_score(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = c
        
        if best_candidate is not None and best_candidate_score > current_score:
            current = sorted(current_set | {best_candidate})
            current_score = best_candidate_score
        else:
            # Try a few more random restarts of the candidate generation
            if step > 5:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    random.seed(12345)
    
    best_set = [0, 1, 3]
    best_score = fast_score(best_set)
    
    # Phase 1: Try good initial candidates
    # Small sets with structure
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 9, 16],
        [0, 1, 3, 6, 10],
        [0, 1, 4, 8, 12],
        [0, 1, 3, 5, 8],
        [0, 1, 4, 6, 9],
        [0, 1, 2, 5, 8],
        [0, 1, 3, 6, 9, 12],
    ]
    
    # Add some random small sets
    for _ in range(20):
        n_elem = random.randint(3, 8)
        max_val = random.randint(15, 60)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        initial_candidates.append(U)
    
    scored_candidates = []
    for U in initial_candidates:
        s = fast_score(U)
        if s > 0:
            scored_candidates.append((s, U))
    
    scored_candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Phase 2: Local search from top candidates
    for score, U in scored_candidates[:8]:
        if time.time() > deadline - 30:
            break
        polished, polished_score = local_search(U, deadline, max_iter=200)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished
    
    # Phase 3: Greedy build from scratch, then local search
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        g_set, g_score = greedy_build(deadline - 10)
        if g_score > best_score:
            best_score = g_score
            best_set = g_set
        
        # Polish the greedy result
        if time.time() < deadline - 15:
            polished, polished_score = local_search(g_set, deadline, max_iter=150)
            if polished_score > best_score:
                best_score = polished_score
                best_set = polished
    
    # Phase 4: Final polish of the best set
    if time.time() < deadline - 5:
        final_polished, final_score = local_search(best_set, deadline, max_iter=300)
        if final_score > best_score:
            best_score = final_score
            best_set = final_polished
    
    return best_set
