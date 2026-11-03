import time
import random
import sys

def construct(q, n, radius):
    total_words = q ** n
    
    # Precomputation
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

    neighbors = []
    cover_bitsets = []
    reverse_cover = [[] for _ in range(total_words)]
    
    # Generate neighbors and bitsets
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
        
        bs = 0
        for j in idxs:
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

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # 1. Try Removal
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
                    # Restore counts
                    for w in neighbors[c_remove]:
                        counts[w] += 1
                    
                    # 2. Try Swap
                    # Find candidates covering all holes
                    if len(holes) > 0:
                        # Sort holes by reverse cover size to speed up intersection
                        holes.sort(key=lambda x: len(reverse_cover[x]))
                        
                        # Start with the smallest set
                        candidates = set(reverse_cover[holes[0]])
                        
                        for h in holes[1:]:
                            candidates &= set(reverse_cover[h])
                            if not candidates:
                                break
                        
                        candidates.discard(c_remove)
                        
                        if candidates:
                            # Score candidates
                            # We want to minimize the number of "critical" words (count==1) 
                            # that are in neighbors[c_new] but not in holes.
                            # These words will become count 2, losing criticality.
                            
                            best_score = float('inf')
                            best_cand = None
                            # To avoid O(N) scoring for every candidate if there are many,
                            # we can just pick a random one if there are too many, 
                            # or score a random sample.
                            # But usually candidates are few.
                            
                            cand_list = list(candidates)
                            # If too many candidates, random pick is faster and often good enough
                            if len(cand_list) > 20:
                                best_cand = random.choice(cand_list)
                            else:
                                for c_new in cand_list:
                                    score = 0
                                    for w in neighbors[c_new]:
                                        if w not in holes and counts[w] == 1:
                                            score += 1
                                    if score < best_score:
                                        best_score = score
                                        best_cand = c_new
                            
                            if best_cand is not None:
                                # Perform Swap
                                for w in neighbors[c_remove]:
                                    counts[w] -= 1
                                for w in neighbors[best_cand]:
                                    counts[w] += 1
                                code[i] = best_cand
                                improved = True
                                break
        return code

    def perturb_repair(code, k, deadline):
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
        
        # Repair
        covered_mask = 0
        for c in code:
            covered_mask |= cover_bitsets[c]
        uncovered_mask = full_mask & ~covered_mask
        
        while uncovered_mask != 0 and time.time() < deadline:
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

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if time.time() < deadline - 15 and best_code is not None:
            k = random.randint(2, 5)
            perturbed = perturb_repair(best_code, k, deadline - 10)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 5:
                perturbed = local_search(perturbed, deadline - 2)
                perturbed = reduce_code(perturbed)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
        
    if best_code is None:
        best_code = greedy_construct(0)
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
