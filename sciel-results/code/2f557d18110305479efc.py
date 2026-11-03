import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s margin before 150s kill
    
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors
    neighbors = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        neighbors.append(idxs)
    
    # Precompute cover bitsets
    cover_bitsets = []
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
            reverse_cover[j].append(i)
        cover_bitsets.append(bs)
    
    full_mask = (1 << total_words) - 1
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code

    def reduce_code(code):
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                for w in neighbors[c]:
                    counts[w] -= 1
                
                can_remove = True
                for w in neighbors[c]:
                    if counts[w] == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    changed = True
                else:
                    for w in neighbors[c]:
                        counts[w] += 1
                    new_code.append(c)
            code = new_code
        return code

    def local_search(code, dl):
        code = list(code)
        if not code:
            return code
            
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= dl:
                    break
                
                c_remove = code[i]
                
                # Try removal
                for w in neighbors[c_remove]:
                    counts[w] -= 1
                
                holes = []
                for w in neighbors[c_remove]:
                    if counts[w] == 0:
                        holes.append(w)
                
                if not holes:
                    code.pop(i)
                    improved = True
                    break
                else:
                    for w in neighbors[c_remove]:
                        counts[w] += 1
                    
                    # Try swap: find candidates covering all holes
                    holes.sort(key=lambda x: len(reverse_cover[x]))
                    candidates = set(reverse_cover[holes[0]])
                    
                    for h in holes[1:]:
                        candidates &= set(reverse_cover[h])
                        if not candidates:
                            break
                    
                    candidates.discard(c_remove)
                    
                    if candidates:
                        # Score: minimize increase in total "excess" coverage
                        # excess(w) = max(0, counts[w] - 1)
                        # New excess = sum over w in neighbors[c_new] of max(0, counts[w]+1-1) 
                        #             minus sum over w in neighbors[c_remove] of max(0, counts[w]-1-1)
                        # Simplified: prefer c_new whose neighbors have lower current counts
                        best_cand = None
                        best_score = float('inf')
                        cand_list = list(candidates)
                        
                        # If too many candidates, sample
                        if len(cand_list) > 20:
                            cand_list = random.sample(cand_list, 20)
                        
                        for cand in cand_list:
                            score = 0
                            for w in neighbors[cand]:
                                # After swap, w gets +1 from cand, -1 from c_remove if w in neighbors[c_remove]
                                if w in neighbors[c_remove]:
                                    # count stays same
                                    pass
                                else:
                                    # count increases by 1
                                    score += max(0, counts[w])
                            if score < best_score:
                                best_score = score
                                best_cand = cand
                        
                        if best_cand is not None:
                            for w in neighbors[c_remove]:
                                counts[w] -= 1
                            for w in neighbors[best_cand]:
                                counts[w] += 1
                            code[i] = best_cand
                            improved = True
                            break
        return code

    def perturb_repair(code, k, dl):
        code = list(code)
        if len(code) <= k:
            return code
        
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        critical_scores = []
        for i, c in enumerate(code):
            crit = 0
            for w in neighbors[c]:
                if counts[w] == 1:
                    crit += 1
            critical_scores.append(crit)
        
        indices = list(range(len(code)))
        indices.sort(key=lambda x: critical_scores[x])
        remove_indices = indices[:k]
        
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        covered_mask = 0
        for c in code:
            covered_mask |= cover_bitsets[c]
        uncovered_mask = full_mask & ~covered_mask
        
        while uncovered_mask != 0 and time.time() < dl:
            best_gain = 0
            best_candidates = []
            
            for i in range(total_words):
                gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
            
        return code

    best_code = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy + reduce (fast)
    t_phase1_end = start_time + 25.0
    seed = 0
    while time.time() < t_phase1_end:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        seed += 1
    
    # Phase 2: Local search on best solutions
    t_phase2_end = start_time + 75.0
    while time.time() < t_phase2_end and best_code is not None:
        code = list(best_code)
        code = local_search(code, t_phase2_end - 1.0)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        # Also try from slightly perturbed versions
        if time.time() < t_phase2_end - 10:
            k = random.randint(1, 3)
            perturbed = perturb_repair(best_code, k, t_phase2_end - 5)
            perturbed = reduce_code(perturbed)
            perturbed = local_search(perturbed, t_phase2_end - 2)
            perturbed = reduce_code(perturbed)
            
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = list(perturbed)
    
    # Phase 3: Final polish - aggressive local search
    if time.time() < deadline - 3 and best_code is not None:
        code = list(best_code)
        code = local_search(code, deadline - 2)
        code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    if best_code is None:
        best_code = greedy_construct(0)
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
