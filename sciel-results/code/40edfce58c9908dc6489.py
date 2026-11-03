import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
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
    
    # Reverse neighbors (symmetric for radius 1)
    reverse_neighbors_arr = neighbors_arr

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        counts = np.zeros(total_words, dtype=np.int16)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            gains_primary = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains_primary > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains_primary[valid])
            best_candidates = np.where((gains_primary == max_gain) & valid)[0]
            
            if len(best_candidates) > 1:
                # Tie-break: prefer candidates that cover fewer already-singly-covered words
                singly_covered = (counts == 1)
                redundancy_penalty = np.sum(singly_covered[neighbors_arr[best_candidates]], axis=1)
                random_scores = np.random.random(len(best_candidates)) * 0.1
                # Minimize redundancy penalty with small random noise
                best_idx = np.argmin(redundancy_penalty + random_scores * 100)
                chosen = best_candidates[best_idx]
            else:
                chosen = best_candidates[0]
            
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
            counts[neighbors_arr[chosen, :]] += 1
        
        return code

    def reduce_code(code):
        """Remove redundant codewords."""
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
        """Local search: try removing each codeword and replacing if needed."""
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
                    # Identify holes
                    holes = np.where(counts == 0)[0]
                    
                    if len(holes) > 12:
                        counts[neighbors_c] += 1
                        continue
                    
                    # Find candidates covering all holes
                    candidate_set = None
                    for h in holes:
                        h_candidates = set(reverse_neighbors_arr[h, :].tolist())
                        if candidate_set is None:
                            candidate_set = h_candidates
                        else:
                            candidate_set &= h_candidates
                    
                    if not candidate_set:
                        counts[neighbors_c] += 1
                        continue
                    
                    candidate_set.discard(c_remove)
                    
                    if not candidate_set:
                        counts[neighbors_c] += 1
                        continue
                    
                    # Score candidates
                    cand_list = list(candidate_set)
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                    
                    best_cand = None
                    best_score = -1
                    code_set = set(code)
                    
                    for cand in cand_list:
                        cand_neigh = neighbors_arr[cand, :]
                        redundant = np.sum(counts[cand_neigh] > 0)
                        if cand in code_set:
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
                    
                    candidate_set = None
                    for h in holes:
                        h_candidates = set(reverse_neighbors_arr[h, :].tolist())
                        if candidate_set is None:
                            candidate_set = h_candidates
                        else:
                            candidate_set &= h_candidates
                    
                    if candidate_set:
                        candidate_set.discard(c1)
                        
                        if candidate_set:
                            code_set = set(code)
                            valid_candidates = candidate_set & code_set
                            
                            if valid_candidates:
                                best_cand = None
                                best_score = -1
                                for cand in valid_candidates:
                                    cand_neigh = neighbors_arr[cand, :]
                                    redundant = np.sum(counts[cand_neigh] > 0)
                                    if redundant > best_score:
                                        best_score = redundant
                                        best_cand = cand
                                
                                if best_cand is not None:
                                    j = code.index(best_cand)
                                    code[i] = best_cand
                                    code[j] = c1
                                    counts[neighbors_c1] += 1
                                    counts[neighbors_arr[best_cand, :]] -= 1
                                    improved = True
                                    break
                    
                    counts[neighbors_c1] += 1
                else:
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def perturb_repair(code, k, deadline):
        """Perturb by removing k codewords, then repair."""
        code = list(code)
        if len(code) <= k:
            return code
        
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
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

    def simulated_annealing(code, deadline, T0=10.0, T_min=0.1, cooling=0.95):
        """Simulated annealing to escape local minima."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        T = T0
        best_code = list(code)
        best_size = len(code)
        
        while T > T_min and time.time() < deadline:
            i = random.randint(0, len(code) - 1)
            c_old = code[i]
            
            # Try removing
            neighbors_c = neighbors_arr[c_old, :]
            counts[neighbors_c] -= 1
            
            if np.any(counts == 0):
                holes = np.where(counts == 0)[0]
                
                if len(holes) <= 10:
                    candidate_set = None
                    for h in holes:
                        h_candidates = set(reverse_neighbors_arr[h, :].tolist())
                        if candidate_set is None:
                            candidate_set = h_candidates
                        else:
                            candidate_set &= h_candidates
                    
                    if candidate_set:
                        candidate_set.discard(c_old)
                        if candidate_set:
                            cand_list = list(candidate_set)
                            if len(cand_list) > 30:
                                cand_list = random.sample(cand_list, 30)
                            
                            best_cand = None
                            best_score = -1
                            code_set = set(code)
                            
                            for cand in cand_list:
                                cand_neigh = neighbors_arr[cand, :]
                                redundant = np.sum(counts[cand_neigh] > 0)
                                if cand in code_set:
                                    redundant -= 100
                                if redundant > best_score:
                                    best_score = redundant
                                    best_cand = cand
                            
                            if best_cand is not None:
                                # Accept with probability based on score improvement
                                old_score = len(code)
                                new_score = len(code)  # same size, just swap
                                # Accept if score is better or with probability exp(-delta/T)
                                if random.random() < min(1.0, T):
                                    code[i] = best_cand
                                    counts[neighbors_arr[best_cand, :]] += 1
                                else:
                                    counts[neighbors_c] += 1
                                T *= cooling
                                continue
                        else:
                            counts[neighbors_c] += 1
                        T *= cooling
                        continue
                    else:
                        counts[neighbors_c] += 1
                        T *= cooling
                        continue
                else:
                    counts[neighbors_c] += 1
                    T *= cooling
                    continue
            else:
                # Can remove
                code.pop(i)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                T *= cooling
                continue
        
        return best_code

    # Main search loop
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 30:
        # 1. Greedy
        code = greedy_construct(seed)
        code = reduce_code(code)
        
        # 2. Local Search
        if time.time() < deadline - 15:
            code = local_search(code, deadline - 10)
            code = reduce_code(code)
        
        # 3. Swap reduction
        if time.time() < deadline - 10:
            code = swap_reduce(code, deadline - 5)
            code = reduce_code(code)
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # 4. Perturbation Strategy
        if time.time() < deadline - 25 and best_code is not None:
            perturbed = multi_perturb(best_code, deadline - 15, num_perturb=2)
            perturbed = reduce_code(perturbed)
            
            if time.time() < deadline - 10:
                perturbed = local_search(perturbed, deadline - 5)
                perturbed = reduce_code(perturbed)
                perturbed = swap_reduce(perturbed, deadline - 3)
                
            if len(perturbed) < best_size:
                best_size = len(perturbed)
                best_code = perturbed
        
        # 5. Simulated Annealing
        if time.time() < deadline - 20 and best_code is not None:
            sa_code = simulated_annealing(best_code, deadline - 10)
            sa_code = reduce_code(sa_code)
            
            if len(sa_code) < best_size:
                best_size = len(sa_code)
                best_code = sa_code
        
        seed += 1
    
    # Final polish
    if best_code is not None and time.time() < deadline - 5:
        polished = local_search(best_code, deadline - 3)
        polished = reduce_code(polished)
        polished = swap_reduce(polished, deadline - 2)
        polished = reduce_code(polished)
        if len(polished) < best_size:
            best_code = polished
    
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
