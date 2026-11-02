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
    
    # Precompute neighbors (ball contents) as numpy arrays for vectorized ops
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
    
    # Convert to numpy arrays
    neighbors_np = [np.array(b, dtype=np.int32) for b in neighbors]
    neighbors_set = [set(b) for b in neighbors]
    
    # Precompute coverers: coverers[point] = list of codewords that cover point
    coverers = [[] for _ in range(N)]
    for cw_idx, ball in enumerate(neighbors):
        for pt in ball:
            coverers[pt].append(cw_idx)
    
    # Convert coverers to numpy arrays
    coverers_np = [np.array(c, dtype=np.int32) for c in coverers]
    
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
    
    # Greedy Set Cover
    while current_cover_mask != all_mask:
        if time.time() > deadline - 60:
            break
            
        best_i = -1
        best_count = 0
        
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
        cover_count[neighbors_np[cw]] += 1
        
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Helper: Try to remove redundant codewords (vectorized)
    def try_remove_redundant(codewords, cover_count, deadline):
        removed = 0
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 30:
                break
            cw = codewords[i]
            b = neighbors_np[cw]
            if np.min(cover_count[b]) > 1:
                cover_count[b] -= 1
                codewords.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    # Initial redundancy removal
    try_remove_redundant(codewords, cover_count, deadline)
    if len(codewords) < best_size:
        best_size = len(codewords)
        best_codewords = list(codewords)
        
    # Local Search Loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try Removals
        removed = try_remove_redundant(codewords, cover_count, deadline)
        if removed > 0:
            improved = True
            no_improvement_count = 0
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue
        
        # 2. Try Swaps with improved candidate selection
        if codewords:
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            b_old = neighbors_np[old_cw]
            critical_points = b_old[cover_count[b_old] == 1]
            
            if len(critical_points) == 0:
                # Redundant - should have been caught above
                cover_count[b_old] -= 1
                codewords.pop(idx)
                improved = True
                no_improvement_count = 0
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                continue
            
            if len(critical_points) > 5:
                # Too many critical points, unlikely to find good swap
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    rand_cw = random.randint(0, N - 1)
                    if rand_cw not in codewords:
                        codewords.append(rand_cw)
                        cover_count[neighbors_np[rand_cw]] += 1
                        no_improvement_count = 0
                continue
            
            # Find candidates that cover ALL critical points
            # Use numpy for intersection
            candidate_mask = np.ones(N, dtype=bool)
            for p in critical_points:
                candidate_mask[coverers_np[p]] = True
                # Actually need AND, not OR. Let me fix this.
            
            # Better: start with first critical point's coverers, intersect with others
            if len(critical_points) > 0:
                candidate_set = set(coverers[critical_points[0]])
                for p in critical_points[1:]:
                    candidate_set.intersection_update(coverers[p])
            else:
                candidate_set = set()
            
            candidate_set.discard(old_cw)
            existing_set = set(codewords)
            candidate_set -= existing_set
            
            if candidate_set:
                candidates_list = list(candidate_set)
                
                # Score candidates: prefer those with low total cover count in their ball
                # and high coverage of uncovered-ish points
                scored_candidates = []
                for new_cw in candidates_list:
                    b_new = neighbors_np[new_cw]
                    # Cost: sum of (cover_count - 1) for points in b_new
                    # We want to minimize increase in redundancy
                    cost = np.sum(np.maximum(cover_count[b_new] - 1, 0))
                    # Bonus: cover points that are barely covered
                    bonus = np.sum(cover_count[b_new] == 1)
                    score = cost - 0.5 * bonus
                    scored_candidates.append((score, new_cw))
                
                scored_candidates.sort(key=lambda x: x[0])
                
                # Try top candidates
                for _, best_new_cw in scored_candidates[:10]:
                    b_new = neighbors_np[best_new_cw]
                    # Verify this swap is valid: after removing old_cw and adding new_cw,
                    # all points must still be covered
                    new_cover = cover_count.copy()
                    new_cover[b_old] -= 1
                    new_cover[b_new] += 1
                    if np.min(new_cover) > 0:
                        # Valid swap
                        cover_count = new_cover
                        codewords[idx] = best_new_cw
                        improved = True
                        no_improvement_count = 0
                        # Try to remove redundancies after swap
                        try_remove_redundant(codewords, cover_count, deadline)
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                        break
                
                if improved:
                    continue
        
        # 3. Try 2-for-1 move: remove two codewords, add one that covers their union
        if len(codewords) >= 2:
            if time.time() < deadline - 5:
                i1 = random.randint(0, len(codewords) - 1)
                i2 = random.randint(0, len(codewords) - 2)
                if i2 >= i1:
                    i2 += 1
                
                cw1 = codewords[i1]
                cw2 = codewords[i2]
                
                # Union of their balls
                union_mask = np.zeros(N, dtype=bool)
                union_mask[neighbors_np[cw1]] = True
                union_mask[neighbors_np[cw2]] = True
                union_points = np.where(union_mask)[0]
                
                # Find a codeword whose ball covers all union_points
                # and is not cw1 or cw2
                candidate_set = set(coverers[union_points[0]])
                for p in union_points[1:]:
                    candidate_set.intersection_update(coverers[p])
                
                candidate_set.discard(cw1)
                candidate_set.discard(cw2)
                existing_set = set(codewords)
                candidate_set -= existing_set
                
                if candidate_set:
                    # Try to find a candidate
                    for new_cw in list(candidate_set)[:5]:
                        b_new = neighbors_np[new_cw]
                        new_cover = cover_count.copy()
                        new_cover[neighbors_np[cw1]] -= 1
                        new_cover[neighbors_np[cw2]] -= 1
                        new_cover[b_new] += 1
                        if np.min(new_cover) > 0:
                            cover_count = new_cover
                            # Remove the two (higher index first)
                            if i1 > i2:
                                codewords.pop(i1)
                                codewords.pop(i2)
                            else:
                                codewords.pop(i2)
                                codewords.pop(i1)
                            codewords.append(new_cw)
                            improved = True
                            no_improvement_count = 0
                            try_remove_redundant(codewords, cover_count, deadline)
                            if len(codewords) < best_size:
                                best_size = len(codewords)
                                best_codewords = list(codewords)
                            break
                    
                    if improved:
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
    if np.min(final_cover_count) > 0:
        try_remove_redundant(final_codewords, final_cover_count, deadline)
        
    # Verify final validity
    if np.min(final_cover_count) > 0:
        return [list(all_words[cw]) for cw in final_codewords]
    else:
        return [list(all_words[cw]) for cw in codewords]
