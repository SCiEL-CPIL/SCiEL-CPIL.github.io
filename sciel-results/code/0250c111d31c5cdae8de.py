import itertools
import numpy as np
import time
import random
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    block_to_tsubs = [[] for _ in range(n_k)]
    tsub_to_blocks = [[] for _ in range(n_t)]

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            block_to_tsubs[j].append(i)
            tsub_to_blocks[i].append(j)
            
    # Precompute sets for fast subset checks
    block_to_tsubs_set = [set(lst) for lst in block_to_tsubs]

    # 2. Initial Greedy Solution
    # Build sparse matrix for fast gain calculation
    rows = []
    cols = []
    for j in range(n_k):
        for i in block_to_tsubs[j]:
            rows.append(i)
            cols.append(j)
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_t, n_k))

    covered = np.zeros(n_t, dtype=bool)
    selected = []
    selected_set = set()
    
    # Greedy
    while not np.all(covered):
        if time.time() > deadline - 20:
            break
        
        not_covered = ~covered
        gains = M.T.dot(not_covered)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        
        best_j = int(np.argmax(gains))
        if best_j not in selected_set:
            selected.append(best_j)
            selected_set.add(best_j)
            covered |= M[:, best_j].toarray().astype(bool)
            
    # Repair if greedy timed out
    if not np.all(covered):
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            not_covered = ~covered
            gains = M.T.dot(not_covered)
            if np.max(gains) == 0: break
            best_j = int(np.argmax(gains))
            if best_j not in selected_set:
                selected.append(best_j)
                selected_set.add(best_j)
                covered |= M[:, best_j].toarray().astype(bool)

    if not np.all(covered):
        # Fallback
        return [list(k_subsets[i]) for i in range(200)]

    best_solution = selected
    best_size = len(best_solution)
    
    # 3. Local Search
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in best_solution:
        for s in block_to_tsubs[b]:
            cover_count[s] += 1
            
    # Ensure validity
    while not np.all(cover_count >= 1):
        if time.time() > deadline - 5: break
        uncovered = np.where(cover_count == 0)[0]
        if len(uncovered) == 0: break
        s_target = uncovered[0]
        candidates = tsub_to_blocks[s_target]
        if not candidates: break
        b_new = random.choice(candidates)
        if b_new not in selected_set:
            best_solution.append(b_new)
            selected_set.add(b_new)
            best_size += 1
            for s in block_to_tsubs[b_new]:
                cover_count[s] += 1

    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 2:
        if no_improve > max_no_improve:
            no_improve = 0
            max_no_improve += 100
            
        if not best_solution:
            break
            
        b_out = random.choice(best_solution)
        
        # Check redundancy
        is_redundant = True
        critical_subs = []
        for s in block_to_tsubs[b_out]:
            if cover_count[s] == 1:
                is_redundant = False
                critical_subs.append(s)
            # If cover_count[s] > 1, it's safe to remove for this s
            
        if is_redundant:
            for s in block_to_tsubs[b_out]:
                cover_count[s] -= 1
            best_solution.remove(b_out)
            selected_set.remove(b_out)
            best_size -= 1
            no_improve = 0
            continue
        
        # Try Swap
        if critical_subs:
            c_set = set(critical_subs)
            # Pick a random critical sub to find candidates
            c_sub = random.choice(critical_subs)
            candidates = tsub_to_blocks[c_sub]
            random.shuffle(candidates)
            
            found = False
            for b_in in candidates:
                if b_in in selected_set:
                    continue
                if c_set.issubset(block_to_tsubs_set[b_in]):
                    # Execute swap
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    best_solution.remove(b_out)
                    selected_set.remove(b_out)
                    best_solution.append(b_in)
                    selected_set.add(b_in)
                    
                    found = True
                    no_improve = 0
                    break
            
            if not found:
                no_improve += 1

    # 4. Final Deletion Pass
    for _ in range(5):
        if time.time() > deadline - 1:
            break
        changed = True
        while changed:
            changed = False
            indices = list(range(len(best_solution)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline - 1:
                    break
                b = best_solution[idx]
                is_redundant = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                    best_solution.pop(idx)
                    selected_set.remove(b)
                    best_size -= 1
                    changed = True
                    break

    return [list(k_subsets[idx]) for idx in best_solution]
