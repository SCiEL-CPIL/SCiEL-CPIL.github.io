import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    total_words = q ** n
    cover_count = 1 + n * (q - 1)
    
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
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
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
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    # Precompute reverse mapping: for each word, which codewords cover it
    # Since the graph is symmetric (Hamming ball radius 1), reverse neighbors = forward neighbors
    # But we need the list of indices c such that word w is in neighbors_arr[c]
    # This is equivalent to: neighbors_arr[w] gives us the words covered by w
    # The reverse is: for word w, which c have w in their neighbor set?
    # For radius 1 Hamming, if w is in neighbors_arr[c], then c is in neighbors_arr[w]
    # So reverse_cover[w] = neighbors_arr[w]
    # Wait, that's only true if the relation is symmetric, which it is for Hamming distance.
    # dist(c, w) <= 1 iff dist(w, c) <= 1
    # So the set of c that cover w is exactly neighbors_arr[w].
    
    # However, for local search, we often need to find a replacement for a removed codeword c
    # such that the new codeword c' covers all "critical" words (words that were only covered by c).
    # Critical words = {w : w in neighbors_arr[c] and count[w] == 1}
    # We need c' such that critical_words is a subset of neighbors_arr[c']
    # Using reverse_cover: for each critical word w, the candidates are neighbors_arr[w].
    # Intersection of neighbors_arr[w] for all w in critical_words gives valid replacements.

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            if len(best_candidates) > 1:
                # Tie-break: random selection for diversity
                chosen = random.choice(best_candidates.tolist())
            else:
                chosen = best_candidates[0]
            
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                
                if np.any(counts == 0):
                    # Identify critical words: those that are now uncovered (count == 0)
                    # Wait, counts == 0 means uncovered.
                    # We need to find a replacement c' such that all uncovered words are covered by c'.
                    # Let U = {w : counts[w] == 0}
                    # We need c' such that U is a subset of neighbors_arr[c']
                    
                    holes = np.where(counts == 0)[0]
                    
                    if len(holes) > 10:
                        # Too many holes to efficiently intersect neighbor sets
                        counts[neighbors_c] += 1
                        continue
                    
                    # Find candidates that cover all holes
                    # Candidate c' covers hole h iff h is in neighbors_arr[c']
                    # Equivalently, c' is in neighbors_arr[h]
                    # So valid candidates are in the intersection of neighbors_arr[h] for all h in holes
                    
                    candidate_set = None
                    for h in holes:
                        rc = neighbors_arr[h, :]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        counts[neighbors_c] += 1
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        counts[neighbors_c] += 1
                        continue
                    
                    # Score candidates: prefer those that maximize overlap with existing coverage
                    # i.e., minimize the number of "new" words they cover (words with count 0 after removal, but not in holes? No, holes are all count 0)
                    # Actually, we want to minimize the "waste": words covered by c' that are already covered by others (count > 0)
                    # But wait, counts currently reflects state AFTER removing c_remove.
                    # So counts[w] > 0 means w is covered by at least one other codeword.
                    # We want to maximize the number of words in neighbors_arr[c'] that have counts > 0.
                    
                    cand_list = list(candidate_set)
                    if len(cand_list) > 20:
                        cand_list = random.sample(cand_list, 20)
                    
                    best_cand = None
                    best_score = -1
                    
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        redundant = np.sum(counts[cand_neigh] > 0)
                        # Penalize if cand is already in code (to avoid duplicates, though they should be distinct indices)
                        # Actually, code is a list of indices, so duplicates are possible if not careful.
                        # But greedy_construct picks distinct indices? No, it picks from 0..total_words-1.
                        # If we pick the same index twice, it's redundant.
                        if cand in code:
                            redundant -= 100
                        if redundant > best_score:
                            best_score = redundant
                            best_cand = cand
                    
                    if best_cand is not None:
                        code[i] = best_cand
                        counts[neighbors_arr[best_cand, :]] += 1
                        improved = True
                        break
                    else:
                        counts[neighbors_c] += 1
                else:
                    # Can remove c_remove
                    code.pop(i)
                    improved = True
                    break
        return code

    def swap_reduce(code, deadline):
        """Try pairwise swaps to reduce code size."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                if i >= len(code):
                    continue
                
                c1 = code[i]
                neighbors_c1 = neighbors_arr[c1, :]
                counts[neighbors_c1] -= 1
                
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    
                    # Try to find a single codeword that covers all holes
                    # Among existing codewords (excluding c1)
                    if len(holes) <= 5:
                        found_swap = False
                        for j in range(len(code)):
                            if j == i:
                                continue
                            c2 = code[j]
                            neighbors_c2 = neighbors_arr[c2, :]
                            if np.all(np.isin(holes, neighbors_c2)):
                                # Swap c1 with c2
                                code[i] = c2
                                code[j] = c1
                                counts[neighbors_c1] += 1
                                counts[neighbors_c2] -= 1
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    counts[neighbors_c1] += 1
                else:
                    # Can remove c1
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def perturb_repair(code, k, deadline):
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        uncovered = counts == 0
        
        while uncovered.any() and time.time() < deadline:
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            counts[neighbors_arr[chosen, :]] += 1
            uncovered = counts == 0
            
        return code

    def multi_perturb(code, deadline, num_perturb=3):
        """Try multiple perturbations and keep the best."""
        best = list(code)
        best_size = len(code)
        
        for _ in range(num_perturb):
            if time.time() >= deadline:
                break
            k = random.randint(1, min(4, len(code) - 1))
            perturbed = perturb_repair(code, k, deadline - 5)
            if len(perturbed) < best_size:
                best = perturbed
                best_size = len(perturbed)
        
        return best

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 20:
        # 1. Greedy
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        # 2. Local Search
        if time.time() < deadline - 10:
            code = local_search(code, deadline - 5)
            code = reduce_code(code)
        
        # 3. Swap reduction
        if time.time() < deadline - 8:
            code = swap_reduce(code, deadline - 5)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # 4. Perturbation Strategy
        if time.time() < deadline - 15 and best_code is not None:
            # Try multiple perturbations
            perturbed = multi_perturb(best_code, deadline - 10, num_perturb=2)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 8:
                perturbed = local_search(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
                perturbed = swap_reduce(perturbed, deadline - 3)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        seed += 1
    
    # Final polish: try to improve the best code found
    if best_code is not None and time.time() < deadline - 5:
        polished = local_search(best_code, deadline - 2)
        polished = reduce_code(polished)
        polished = swap_reduce(polished, deadline - 1)
        if len(polished) < best_size:
            best_code = polished
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
