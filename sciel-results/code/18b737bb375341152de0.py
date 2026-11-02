import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Precompute neighbors (ball contents)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        covered = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            orig_val = w_list[i]
            for val in range(q):
                if val != orig_val:
                    w_list[i] = val
                    covered.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig_val
        neighbors.append(covered)
    
    # Precompute neighbors as numpy arrays for vectorized operations
    neighbors_np = np.array(neighbors, dtype=np.int32)  # shape (N, 21)
    ball_size = neighbors_np.shape[1]  # 1 + q*n = 1 + 21 = 22? No: 1 + n*(q-1) = 1+14=15
    
    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx in range(N):
        for pt in neighbors[cw_idx]:
            coverers[pt].append(cw_idx)
    
    # Convert coverers to numpy arrays for faster intersection
    coverers_np = []
    for i in range(N):
        coverers_np.append(np.array(coverers[i], dtype=np.int32))
    
    # 1. Initial Construction: Greedy Set Cover using Bitsets
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
        
    all_mask = (1 << N) - 1
    current_cover_mask = 0
    codewords = []
    is_chosen = np.zeros(N, dtype=bool)
    
    # Greedy Set Cover - optimized with vectorized bit_count
    while current_cover_mask != all_mask:
        if time.time() > deadline - 60:
            break
            
        best_i = -1
        best_count = 0
        
        # Vectorized scan: compute new coverage for all candidates at once
        # But Python ints for bitsets... let's keep the loop but optimize
        for i in range(N):
            if is_chosen[i]:
                continue
            new_bits = ball_bitsets[i] & ~current_cover_mask
            if new_bits:
                count = new_bits.bit_count()
                if count > best_count:
                    best_count = count
                    best_i = i
                    
        if best_i == -1:
            break
            
        codewords.append(best_i)
        is_chosen[best_i] = True
        current_cover_mask |= ball_bitsets[best_i]
        
    # Ensure validity
    if current_cover_mask != all_mask:
        while current_cover_mask != all_mask:
            i = random.randint(0, N-1)
            if not is_chosen[i]:
                codewords.append(i)
                is_chosen[i] = True
            current_cover_mask |= ball_bitsets[i]
            if time.time() > deadline - 10:
                break

    # 2. Local Search Setup
    cover_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        cover_count[neighbors[cw]] += 1
        
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Precompute: for each codeword, which points are "critical" (count==1)
    # We'll recompute this as needed since it changes
    
    # Helper: Try to remove redundant codewords - vectorized
    def try_remove_redundant(codewords, cover_count):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            # Check if all points in ball(cw) have count > 1
            b = neighbors_np[cw]
            if cover_count[b].min() > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Precompute ball_sum: for each potential new codeword, sum of cover_count in its ball
    # This is expensive to maintain, so we compute on the fly but cache
    
    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 30  # Lower threshold for more perturbations
    
    # Precompute: for each point, the set of codewords that cover it (as a set for fast intersection)
    coverers_set = [set(c) for c in coverers]
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue
        
        # 2. Try Swaps
        if codewords:
            # Pick a random codeword to try swapping
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find critical points: points in ball(old_cw) with count == 1
            b_old = neighbors_np[old_cw]
            critical_mask = (cover_count[b_old] == 1)
            critical_points = b_old[critical_mask].tolist()
            
            if not critical_points:
                # Can remove
                cover_count[b_old] -= 1
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
            
            # Find candidates that cover ALL critical points
            # Use set intersection
            candidate_set = set(coverers_set[critical_points[0]])
            for p in critical_points[1:]:
                candidate_set &= coverers_set[p]
            
            # Remove old_cw and existing codewords
            candidate_set.discard(old_cw)
            candidate_set -= set(codewords)
                
            if candidate_set:
                # Pick the best candidate by minimum cost
                # Cost = sum of cover_count in its ball (lower = less redundancy)
                best_new_cw = -1
                min_cost = float('inf')
                
                candidates_list = list(candidate_set)
                if len(candidates_list) > 30:
                    random.shuffle(candidates_list)
                    candidates_list = candidates_list[:30]
                
                for new_cw in candidates_list:
                    b_new = neighbors_np[new_cw]
                    cost = cover_count[b_new].sum()
                    if cost < min_cost:
                        min_cost = cost
                        best_new_cw = new_cw
                        
                if best_new_cw != -1:
                    # Perform swap
                    cover_count[b_old] -= 1
                    b_new = neighbors_np[best_new_cw]
                    cover_count[b_new] += 1
                    codewords[idx] = best_new_cw
                    improved = True
                    no_improvement_count = 0
                    # Try to remove redundancies after swap
                    try_remove_redundant(codewords, cover_count)
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    continue
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                # Perturbation: Add a random codeword
                rand_cw = random.randint(0, N - 1)
                if rand_cw not in codewords:
                    codewords.append(rand_cw)
                    cover_count[neighbors_np[rand_cw]] += 1
                    no_improvement_count = 0
                else:
                    no_improvement_count = 0

    # Final Polish
    final_codewords = list(best_codewords)
    final_cover_count = np.zeros(N, dtype=np.int32)
    for cw in final_codewords:
        final_cover_count[neighbors_np[cw]] += 1
        
    # Run redundancy removal on the best solution found
    if final_cover_count.min() > 0:
        try_remove_redundant(final_codewords, final_cover_count)
        
    # Verify final validity
    if final_cover_count.min() > 0:
        return [list(all_words[cw]) for cw in final_codewords]
    else:
        return [list(all_words[cw]) for cw in codewords]
