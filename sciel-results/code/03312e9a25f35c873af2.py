import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute distance matrix
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # coverage_bool[i, j] is True if dist[i, j] <= radius
        coverage_bool = dist <= radius
        
        # Precompute coverage sets for faster intersection if needed, 
        # but numpy masks are fast for counting.
        # Let's stick to numpy arrays for counts.
        
        def greedy_set_cover(uncovered, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered_mask = np.zeros(total_words, dtype=bool)
            for i in uncovered:
                uncovered_mask[i] = True
                
            code_set = set(code_indices)
            code_list = list(code_set)
            
            # Initial counts
            counts = np.zeros(total_words, dtype=np.int8)
            for c in code_list:
                counts[coverage_bool[c]] += 1
            
            # Greedy step
            while not np.all(counts > 0): # While there are uncovered words (count 0)
                # Find best candidate
                # We want to maximize new coverage
                # new_cover = sum(coverage_bool[i] & (counts == 0))
                # Vectorized:
                uncovered_indices = np.where(counts == 0)[0]
                if len(uncovered_indices) == 0:
                    break
                
                # For each i, count how many of the currently uncovered words it covers
                # sub_cov = coverage_bool[:, uncovered_indices]
                # This is (total_words, len(uncovered_indices))
                # Sum along axis 1
                # This might be slow if uncovered is large. 
                # But total_words is 729. 729x729 is fine.
                
                cov_counts = coverage_bool[:, uncovered_indices].sum(axis=1)
                
                # Exclude already chosen
                for c in code_list:
                    cov_counts[c] = 0
                    
                best_idx = np.argmax(cov_counts)
                if cov_counts[best_idx] == 0:
                    break # No improvement possible? Should not happen if uncovered exists.
                
                code_list.append(best_idx)
                code_set.add(best_idx)
                counts[coverage_bool[best_idx]] += 1
                
            return code_list

        def local_search(code_list):
            code_set = set(code_list)
            counts = np.zeros(total_words, dtype=np.int8)
            for c in code_set:
                counts[coverage_bool[c]] += 1
                
            improved = True
            max_iter = 200
            iter_count = 0
            
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                
                # 1. Pruning
                removable = None
                for c in list(code_set):
                    if np.min(counts[coverage_bool[c]]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[coverage_bool[removable]] -= 1
                    improved = True
                    continue
                
                # 2. 1-1 Swap
                current_list = list(code_set)
                random.shuffle(current_list)
                swapped = False
                for c in current_list:
                    private_mask = (counts == 1) & coverage_bool[c]
                    private_words = np.where(private_mask)[0]
                    
                    if len(private_words) == 0:
                        # Can remove c entirely (redundant)
                        code_set.remove(c)
                        counts[coverage_bool[c]] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Find candidate covering all private_words
                    not_in_code = np.ones(total_words, dtype=bool)
                    for cc in code_set:
                        not_in_code[cc] = False
                        
                    sub_cov = coverage_bool[:, private_words]
                    covers_count = sub_cov.sum(axis=1)
                    
                    valid_candidates = np.where((covers_count == len(private_words)) & not_in_code)[0]
                    
                    if len(valid_candidates) > 0:
                        chosen = valid_candidates[random.randint(0, len(valid_candidates)-1)]
                        code_set.remove(c)
                        counts[coverage_bool[c]] -= 1
                        code_set.add(chosen)
                        counts[coverage_bool[chosen]] += 1
                        improved = True
                        swapped = True
                        break
                        
                if swapped:
                    continue
                    
                # 3. 2-1 Swap
                # Try removing two codewords c1, c2 and adding one c3
                # This is O(K^2 * N) which is 73^2 * 729 ~ 4M ops. Might be slow in Python loops.
                # We can sample pairs.
                
                # Let's try a limited number of random pairs
                if len(code_set) >= 2:
                    pairs = list(itertools.combinations(list(code_set), 2))
                    random.shuffle(pairs)
                    # Limit pairs to check to keep time low
                    for c1, c2 in pairs[:200]: 
                        # Private words of c1 and c2
                        p1_mask = (counts == 1) & coverage_bool[c1]
                        p2_mask = (counts == 1) & coverage_bool[c2]
                        
                        # Words that become uncovered if we remove c1 and c2
                        # These are words where count==1 and covered by c1 OR count==1 and covered by c2
                        # Note: A word cannot be private to both c1 and c2 if they are distinct?
                        # If a word is covered by c1 and c2, count >= 2. So private sets are disjoint.
                        
                        private_union_mask = p1_mask | p2_mask
                        private_words = np.where(private_union_mask)[0]
                        
                        if len(private_words) == 0:
                            # Both are redundant? Should have been pruned.
                            code_set.remove(c1)
                            counts[coverage_bool[c1]] -= 1
                            code_set.remove(c2)
                            counts[coverage_bool[c2]] -= 1
                            improved = True
                            swapped = True
                            break
                        
                        not_in_code = np.ones(total_words, dtype=bool)
                        for cc in code_set:
                            not_in_code[cc] = False
                        not_in_code[c1] = False
                        not_in_code[c2] = False
                        
                        sub_cov = coverage_bool[:, private_words]
                        covers_count = sub_cov.sum(axis=1)
                        
                        valid_candidates = np.where((covers_count == len(private_words)) & not_in_code)[0]
                        
                        if len(valid_candidates) > 0:
                            chosen = valid_candidates[random.randint(0, len(valid_candidates)-1)]
                            code_set.remove(c1)
                            counts[coverage_bool[c1]] -= 1
                            code_set.remove(c2)
                            counts[coverage_bool[c2]] -= 1
                            code_set.add(chosen)
                            counts[coverage_bool[chosen]] += 1
                            improved = True
                            swapped = True
                            break
                            
            return code_set

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110
        
        # Multi-start
        for seed in range(50):
            if time.time() - start_time > time_limit:
                break
                
            initial_code = greedy_set_cover(set(range(total_words)), [], random_seed=seed)
            ls_code = local_search(initial_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
                
        # Iterated Local Search on best
        if best_code is not None:
            while time.time() - start_time < time_limit:
                perturbed_code = list(best_code)
                k_remove = random.randint(1, 3)
                if len(perturbed_code) > k_remove:
                    to_remove = random.sample(perturbed_code, k_remove)
                    for r in to_remove:
                        perturbed_code.remove(r)
                
                covered_mask = np.zeros(total_words, dtype=bool)
                for c in perturbed_code:
                    covered_mask[coverage_bool[c]] = True
                uncovered = set(np.where(~covered_mask)[0])
                
                new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=random.randint(0, 10000))
                ls_code = local_search(new_code)
                
                if len(ls_code) < best_size:
                    best_size = len(ls_code)
                    best_code = ls_code

        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
