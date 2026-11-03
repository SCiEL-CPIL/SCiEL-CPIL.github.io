import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
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
    cover_count = 1 + n * (q - 1)
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

    # Reverse cover: for each word, list of codewords that cover it
    # Since the graph is regular and symmetric, the number of neighbors is constant.
    # But the list of codewords covering a word is the set of its neighbors in the graph.
    # We need this for repair: "Which codewords cover this specific uncovered word?"
    # This is exactly the neighbors of that word.
    # So reverse_cover is just neighbors_arr transposed?
    # neighbors_arr[i, j] is the j-th neighbor of i.
    # We want for a word w, the list of i such that w is a neighbor of i.
    # This is the same as the list of neighbors of w, because the graph is undirected.
    # So neighbors_arr[w, :] gives the list of codewords that cover w.
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation
            # gains[i] = number of uncovered words covered by codeword i
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            chosen = random.choice(best_candidates.tolist())
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

    def local_search(code, deadline, allow_same_size=False):
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
                
                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1
                
                if np.any(counts == 0):
                    # Cannot remove, try to swap or restore
                    counts[neighbors_arr[c_remove, :]] += 1
                    
                    # Find critical words
                    critical_mask = (counts == 1)
                    # Only consider critical words covered by c_remove
                    # Actually, if counts[w] == 1 and c_remove covers w, then w is critical.
                    # neighbors_arr[c_remove] gives the words covered by c_remove.
                    # We need to check which of these have count 1.
                    covered_by_c = neighbors_arr[c_remove, :]
                    critical_words = covered_by_c[critical_mask[covered_by_c]]
                    
                    if len(critical_words) == 0:
                        continue
                    
                    # Find candidates that cover all critical words
                    # Intersection of neighbors of each critical word
                    candidate_set = None
                    for cw in critical_words:
                        # neighbors_arr[cw] are the codewords covering cw
                        rc = neighbors_arr[cw, :]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())
                    
                    if candidate_set is None or len(candidate_set) == 0:
                        continue
                    
                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue
                    
                    # If allow_same_size, we can swap even if size doesn't change
                    # Here size doesn't change (remove 1, add 1)
                    best_w = random.choice(list(candidate_set))
                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                    break
                else:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
        
        return code

    def repair_greedy(code, deadline):
        """
        Greedily add codewords to cover any uncovered words.
        Returns the new code.
        """
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        uncovered = counts == 0
        
        while uncovered.any() and time.time() < deadline:
            # Find best codeword to add
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

    def perturb_repair(code, k, deadline):
        """
        Remove k random codewords and repair.
        """
        code = list(code)
        if len(code) <= k:
            return code
        
        # Remove k random distinct codewords
        remove_indices = random.sample(range(len(code)), k)
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        # Repair
        code = repair_greedy(code, deadline)
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = local_search(code, deadline - 10, allow_same_size=True)
    code = reduce_code(code)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search
    seed = 1
    while time.time() < deadline - 5:
        # Perturb the best code
        # Increase k as we go? Or keep small?
        # Small k (1-3) is safer.
        k = random.randint(1, 3)
        perturbed_code = perturb_repair(best_code, k, deadline - 5)
        
        # Local search on perturbed code
        perturbed_code = local_search(perturbed_code, deadline - 5, allow_same_size=True)
        perturbed_code = reduce_code(perturbed_code)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # If we have time, do a fresh restart occasionally
        if seed % 10 == 0 and time.time() < deadline - 20:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = local_search(fresh_code, deadline - 10, allow_same_size=True)
            fresh_code = reduce_code(fresh_code)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
