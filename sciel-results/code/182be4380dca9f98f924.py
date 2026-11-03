import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
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

    # Precompute neighbors efficiently
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
    
    # Precompute cover as Python int bitmasks
    cover_bitsets = []
    for i in range(total_words):
        bs = 0
        for j in neighbors[i]:
            bs |= (1 << j)
        cover_bitsets.append(bs)
    
    # Reverse cover: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbors[i]:
            reverse_cover[j].append(i)
    
    full_mask = (1 << total_words) - 1
    
    def greedy_construct(random_seed, strategy="standard"):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered_mask = full_mask
        code = []
        
        while uncovered_mask != 0:
            best_gain = 0
            best_candidates = []
            
            if strategy == "standard":
                for i in range(total_words):
                    gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(i)
            elif strategy == "randomized":
                # Randomize the order to break ties differently
                indices = list(range(total_words))
                random.shuffle(indices)
                for i in indices:
                    gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(i)
            else:  # "weighted" - prefer codewords that cover more currently-uncovered
                for i in range(total_words):
                    gain = (uncovered_mask & cover_bitsets[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            if strategy == "randomized" and len(best_candidates) > 1:
                # Pick randomly among top candidates
                chosen = random.choice(best_candidates[:min(10, len(best_candidates))])
            else:
                chosen = random.choice(best_candidates)
            
            code.append(chosen)
            uncovered_mask &= ~cover_bitsets[chosen]
        
        return code

    def reduce_code(code):
        if not code:
            return code
        
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

    def local_search(code, dl, max_iterations=100000):
        code = list(code)
        if not code:
            return code
            
        counts = [0] * total_words
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        improved = True
        iteration = 0
        while improved and time.time() < dl and iteration < max_iterations:
            improved = False
            iteration += 1
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
                    
                    # Try swap
                    if len(holes) <= 5:  # Only try swap if few holes
                        # Find candidates that cover all holes
                        candidates = set(reverse_cover[holes[0]])
                        for h in holes[1:]:
                            candidates &= set(reverse_cover[h])
                            if not candidates:
                                break
                        
                        candidates.discard(c_remove)
                        
                        if candidates:
                            # Score candidates: prefer those with lower excess coverage
                            best_cand = None
                            best_score = float('inf')
                            cand_list = list(candidates)
                            
                            if len(cand_list) > 30:
                                cand_list = random.sample(cand_list, 30)
                            
                            # Precompute which words are in neighbors[c_remove]
                            rem_set = set(neighbors[c_remove])
                            
                            for cand in cand_list:
                                score = 0
                                for w in neighbors[cand]:
                                    if w not in rem_set:
                                        # This word gets +1 coverage
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
        
        # Find least critical codewords (fewest uniquely covered words)
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
        
        # Greedily fill holes
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

    def two_opt_local_search(code, dl):
        """More aggressive: try moving a codeword to a completely different position"""
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
                
                # Remove it
                for w in neighbors[c_remove]:
                    counts[w] -= 1
                
                holes = [w for w in neighbors[c_remove] if counts[w] == 0]
                
                if not holes:
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for w in neighbors[c_remove]:
                        counts[w] += 1
                    
                    # Try to find a replacement that covers all holes
                    # and has minimal additional coverage
                    candidates = set(reverse_cover[holes[0]])
                    for h in holes[1:]:
                        candidates &= set(reverse_cover[h])
                        if not candidates:
                            break
                    
                    candidates.discard(c_remove)
                    
                    if candidates:
                        rem_set = set(neighbors[c_remove])
                        best_cand = None
                        best_score = float('inf')
                        cand_list = list(candidates)
                        
                        if len(cand_list) > 50:
                            cand_list = random.sample(cand_list, 50)
                        
                        for cand in cand_list:
                            score = 0
                            for w in neighbors[cand]:
                                if w not in rem_set:
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

    best_code = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy with different strategies
    t_phase1_end = start_time + 30.0
    seed = 0
    strategies = ["standard", "randomized", "weighted"]
    
    while time.time() < t_phase1_end:
        strategy = strategies[seed % len(strategies)]
        code = greedy_construct(seed, strategy)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        seed += 1
    
    # Phase 2: Local search on best solutions
    t_phase2_end = start_time + 80.0
    while time.time() < t_phase2_end and best_code is not None:
        code = list(best_code)
        code = local_search(code, t_phase2_end - 1.0)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        # Try perturbation
        if time.time() < t_phase2_end - 15:
            k = random.randint(1, 4)
            perturbed = perturb_repair(best_code, k, t_phase2_end - 5)
            perturbed = reduce_code(perturbed)
            perturbed = local_search(perturbed, t_phase2_end - 2)
            perturbed = reduce_code(perturbed)
            
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = list(perturbed)
    
    # Phase 3: Final aggressive polish
    if time.time() < deadline - 5 and best_code is not None:
        code = list(best_code)
        code = two_opt_local_search(code, deadline - 3)
        code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        
        # One more round of local search
        code = list(best_code)
        code = local_search(code, deadline - 1)
        code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
    
    if best_code is None:
        best_code = greedy_construct(0, "standard")
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
