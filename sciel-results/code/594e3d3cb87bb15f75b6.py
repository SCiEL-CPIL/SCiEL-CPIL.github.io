import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    cover_size = 1 + n * (q - 1)
    
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

    # Precompute neighbors as list of lists
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
    
    # Precompute neighbor sets for fast intersection
    neighbor_sets = [frozenset(nb) for nb in neighbors]
    
    # Bitset representation using numpy uint64 arrays for vectorized gain computation
    # Each codeword's coverage is stored as an array of uint64 words
    num_uint64 = (total_words + 63) // 64
    cover_np = np.zeros((total_words, num_uint64), dtype=np.uint64)
    for i in range(total_words):
        for j in neighbors[i]:
            cover_np[i, j // 64] |= (1 << (j % 64))
    
    # Reverse: for each word, which codewords cover it (as a list)
    reverse_cover = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in neighbors[i]:
            reverse_cover[j].append(i)
    
    # Precompute reverse_cover as sets for fast intersection
    reverse_cover_sets = [frozenset(rc) for rc in reverse_cover]
    
    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        
        # Use numpy for vectorized gain computation
        # uncovered as numpy array of uint64
        uncovered_np = np.zeros(num_uint64, dtype=np.uint64)
        for i in range(total_words):
            uncovered_np[i // 64] |= (1 << (i % 64))
        
        code = []
        
        while True:
            # Check if all covered
            if not uncovered_np.any():
                break
            
            # Vectorized gain computation: for each candidate, count bits in (uncovered & cover[i])
            # We need to count set bits in each row of (uncovered_np[:, None] & cover_np)
            # This is expensive in pure numpy, so use a hybrid approach
            # Compute gains using Python int bit_count for speed
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                # Fast bit count using Python int
                gain = 0
                for k in range(num_uint64):
                    gain += (uncovered_np[k] & cover_np[i, k]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            for k in range(num_uint64):
                uncovered_np[k] &= ~cover_np[chosen, k]
        
        return code

    def reduce_code(code):
        # Use counts array for efficiency
        counts = np.zeros(total_words, dtype=np.int32)
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
                if np.any(counts[neighbors[c]] == 0):
                    for w in neighbors[c]:
                        counts[w] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_remove_swap(code, deadline):
        """Try to remove each codeword; if it creates uncovered words, try to swap with one that covers all of them."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
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
                for w in neighbors[c_remove]:
                    counts[w] -= 1
                
                # Check if any word becomes uncovered
                uncovered_words = [w for w in neighbors[c_remove] if counts[w] == 0]
                
                if not uncovered_words:
                    # Can remove
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for w in neighbors[c_remove]:
                        counts[w] += 1
                    
                    # Try to find a swap: a codeword that covers all uncovered_words
                    # Use precomputed frozensets for fast intersection
                    candidate_set = reverse_cover_sets[uncovered_words[0]]
                    for w in uncovered_words[1:]:
                        candidate_set = candidate_set & reverse_cover_sets[w]
                    
                    if c_remove in candidate_set:
                        candidate_set = candidate_set - {c_remove}
                    
                    if candidate_set:
                        best_w = random.choice(list(candidate_set))
                        # Remove c_remove, add best_w
                        for w in neighbors[c_remove]:
                            counts[w] -= 1
                        for w in neighbors[best_w]:
                            counts[w] += 1
                        code[i] = best_w
                        improved = True
                        break
        
        return code

    def local_search_critical(code, deadline):
        """Targeted local search: find words with count==1, try to reassign their coverage."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        # Build a mapping from word to list of codewords in code that cover it
        word_to_codes = {}
        for c in code:
            for w in neighbors[c]:
                if w not in word_to_codes:
                    word_to_codes[w] = []
                word_to_codes[w].append(c)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Find all words with count == 1
            critical_words = np.where(counts == 1)[0].tolist()
            
            if not critical_words:
                break
            
            random.shuffle(critical_words)
            
            for w in critical_words:
                if time.time() >= deadline:
                    break
                
                # Find which codeword covers w (should be exactly one)
                covering_code = word_to_codes.get(w, [None])[0]
                
                if covering_code is None:
                    continue
                
                # Try to find another codeword that also covers w and is in the code
                alternatives = [c for c in word_to_codes.get(w, []) if c != covering_code]
                
                if alternatives:
                    # Check if removing covering_code leaves any word uncovered
                    can_remove = True
                    for w2 in neighbors[covering_code]:
                        if w2 == w:
                            continue
                        if counts[w2] == 1:
                            can_remove = False
                            break
                    
                    if can_remove:
                        # Remove covering_code
                        for w2 in neighbors[covering_code]:
                            counts[w2] -= 1
                        code.remove(covering_code)
                        # Update word_to_codes
                        for w2 in neighbors[covering_code]:
                            if w2 in word_to_codes:
                                word_to_codes[w2].remove(covering_code)
                                if not word_to_codes[w2]:
                                    del word_to_codes[w2]
                        improved = True
                        break
        
        return code

    def repair_greedy(code, deadline):
        """Greedily add codewords to cover any uncovered words."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        while time.time() < deadline:
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Find best codeword to add - vectorized gain
            best_gain = 0
            best_candidates = []
            for i in range(total_words):
                gain = np.sum(counts[neighbors[i]] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
            
            chosen = random.choice(best_candidates)
            code.append(chosen)
            for w in neighbors[chosen]:
                counts[w] += 1
        
        return code

    def perturb_repair(code, k, deadline):
        """Remove k random codewords and repair."""
        code = list(code)
        if len(code) <= k:
            return code
        
        # Prefer removing codewords with low unique coverage
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for w in neighbors[c]:
                counts[w] += 1
        
        # Score each codeword by how many words it uniquely covers
        unique_scores = []
        for c in code:
            unique = sum(1 for w in neighbors[c] if counts[w] == 1)
            unique_scores.append(unique)
        
        # Remove codewords with lowest unique scores first
        order = sorted(range(len(code)), key=lambda i: unique_scores[i])
        remove_indices = order[:k]
        for idx in sorted(remove_indices, reverse=True):
            code.pop(idx)
        
        code = repair_greedy(code, deadline)
        return code

    def full_local_search(code, deadline):
        """Combine multiple local search strategies."""
        code = local_search_remove_swap(code, deadline)
        code = local_search_critical(code, deadline)
        code = local_search_remove_swap(code, deadline)
        code = reduce_code(code)
        # Additional pass if time permits
        if time.time() < deadline - 2:
            code = local_search_remove_swap(code, deadline)
        return code

    start_time = time.time()
    deadline = start_time + 115
    
    best_code = None
    best_size = float('inf')
    
    # Initial construction
    code = greedy_construct(0)
    code = reduce_code(code)
    code = full_local_search(code, deadline - 15)
    
    best_code = code
    best_size = len(code)
    
    # Iterated Local Search with varied perturbations
    seed = 1
    while time.time() < deadline - 8:
        # Different perturbation strategies
        strategy = seed % 4
        
        if strategy == 0:
            # Small targeted removal + repair
            k = random.randint(1, 2)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 1:
            # Medium targeted removal + repair
            k = random.randint(3, 4)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        elif strategy == 2:
            # Larger targeted removal + repair
            k = random.randint(5, 7)
            perturbed_code = perturb_repair(best_code, k, deadline - 8)
        else:
            # Swap-based perturbation: remove 2 low-unique, add 2 random, then repair
            perturbed_code = list(best_code)
            if len(perturbed_code) >= 2:
                # Find low-unique codewords to remove
                counts_tmp = np.zeros(total_words, dtype=np.int32)
                for c in perturbed_code:
                    for w in neighbors[c]:
                        counts_tmp[w] += 1
                unique_scores = []
                for c in perturbed_code:
                    unique = sum(1 for w in neighbors[c] if counts_tmp[w] == 1)
                    unique_scores.append(unique)
                order = sorted(range(len(perturbed_code)), key=lambda i: unique_scores[i])
                remove_idx = order[:2]
                for idx in sorted(remove_idx, reverse=True):
                    perturbed_code.pop(idx)
                # Add 2 random codewords
                for _ in range(2):
                    perturbed_code.append(random.randint(0, total_words - 1))
            perturbed_code = repair_greedy(perturbed_code, deadline - 8)
        
        # Local search on perturbed code
        perturbed_code = full_local_search(perturbed_code, deadline - 8)
        
        if len(perturbed_code) < best_size:
            best_size = len(perturbed_code)
            best_code = perturbed_code
        
        seed += 1
        
        # Fresh restarts with different seeds
        if seed % 6 == 0 and time.time() < deadline - 25:
            fresh_code = greedy_construct(seed)
            fresh_code = reduce_code(fresh_code)
            fresh_code = full_local_search(fresh_code, deadline - 25)
            if len(fresh_code) < best_size:
                best_size = len(fresh_code)
                best_code = fresh_code

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result
