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
        
        # Precompute coverage sets for faster greedy
        # coverage_sets[i] is a set of indices j such that i covers j
        coverage_sets = [set(np.where(coverage_bool[i])[0]) for i in range(total_words)]
        
        def greedy_set_cover(uncovered, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = set(uncovered)
            code_indices = list(code_indices)
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                candidates = []
                
                # Vectorized-ish approach for speed:
                # We want to find i that maximizes |coverage_sets[i] & uncovered|
                # Since N=729, a simple loop is fast enough if sets are precomputed.
                for i in range(total_words):
                    if i in code_indices:
                        continue
                    new_cover = len(coverage_sets[i] & uncovered)
                    if new_cover > best_cover_count:
                        best_cover_count = new_cover
                        best_idx = i
                        candidates = [i]
                    elif new_cover == best_cover_count and new_cover > 0:
                        candidates.append(i)
                    
                if best_idx == -1:
                    break
                
                chosen = rng.choice(candidates)
                code_indices.append(chosen)
                uncovered -= coverage_sets[chosen]
                
            return code_indices

        def local_search(code_set):
            code_set = set(code_set)
            # counts[w] is number of codewords covering w
            counts = np.zeros(total_words, dtype=np.int8)
            for c in code_set:
                counts[coverage_bool[c]] += 1
            
            improved = True
            iterations = 0
            max_iterations = 2000
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning
                removable = None
                for c in list(code_set):
                    # If min count in coverage[c] > 1, c is redundant
                    if np.min(counts[coverage_bool[c]]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[coverage_bool[removable]] -= 1
                    improved = True
                    continue
                
                # 2. 1-Swap
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    # Find private words of c
                    private_mask = (counts == 1) & coverage_bool[c]
                    private_words = np.where(private_mask)[0]
                    
                    if len(private_words) == 0:
                        code_set.remove(c)
                        counts[coverage_bool[c]] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Find any w not in code_set that covers all private_words
                    not_in_code = np.ones(total_words, dtype=bool)
                    for cc in code_set:
                        not_in_code[cc] = False
                        
                    # Check if any candidate covers all private words
                    # sub_cov shape: (total_words, len(private_words))
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
                        
                if not swapped:
                    break
                    
            return code_set

        best_code = None
        best_size = float('inf')
        
        start_time = time.time()
        time_limit = 110
        
        # Initial Greedy
        initial_code = greedy_set_cover(set(range(total_words)), [])
        ls_code = local_search(initial_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code
            
        # Iterated Local Search
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            perturbed_code = list(best_code)
            
            # Perturbation: remove k random codewords
            k_remove = random.randint(1, 2)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            # Repair: find uncovered words and greedily add
            covered = np.zeros(total_words, dtype=bool)
            for c in perturbed_code:
                covered[coverage_bool[c]] = True
            
            uncovered = np.where(~covered)[0]
            uncovered_set = set(uncovered)
            
            # Greedy repair
            new_code = greedy_set_cover(uncovered_set, perturbed_code, random_seed=iterations)
            
            # Local Search
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
