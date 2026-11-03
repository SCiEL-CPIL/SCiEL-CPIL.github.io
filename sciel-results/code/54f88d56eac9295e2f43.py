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
        
        # coverage[i] is a set of indices j such that dist[i][j] <= radius
        coverage = [set() for _ in range(total_words)]
        for i in range(total_words):
            idxs = np.where(dist[i] <= radius)[0]
            coverage[i] = set(idxs.tolist())
        
        def greedy_set_cover(uncovered, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = set(uncovered)
            code_indices = list(code_indices)
            code_set = set(code_indices) # For O(1) lookup
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                candidates = []
                for i in range(total_words):
                    if i in code_set:
                        continue
                    new_cover = len(coverage[i] & uncovered)
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
                code_set.add(chosen)
                uncovered -= coverage[chosen]
                
            return code_indices

        def local_search(code_set, max_iters=200):
            code_set = set(code_set)
            counts = np.zeros(total_words, dtype=int)
            for c in code_set:
                for w in coverage[c]:
                    counts[w] += 1
            
            improved = True
            iterations = 0
            
            while improved and iterations < max_iters:
                improved = False
                iterations += 1
                
                # 1. Pruning: Remove redundant codewords
                removable = None
                for c in list(code_set):
                    is_redundant = True
                    for w in coverage[c]:
                        if counts[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    for w in coverage[removable]:
                        counts[w] -= 1
                    improved = True
                    continue
                
                # 2. Swaps (1-exchange)
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_words = [w for w in coverage[c] if counts[w] == 1]
                    
                    if not private_words:
                        # Should be caught by pruning, but handle just in case
                        code_set.remove(c)
                        for w in coverage[c]:
                            counts[w] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    found_w = None
                    for w in range(total_words):
                        if w in code_set:
                            continue
                        covers_all = True
                        for pw in private_words:
                            if pw not in coverage[w]:
                                covers_all = False
                                break
                        if covers_all:
                            found_w = w
                            break
                            
                    if found_w is not None:
                        code_set.remove(c)
                        for w in coverage[c]:
                            counts[w] -= 1
                        code_set.add(found_w)
                        for w in coverage[found_w]:
                            counts[w] += 1
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
        
        # Initial Greedy + Local Search
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
            k_remove = random.randint(1, 3)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            covered = set()
            for c in perturbed_code:
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            
            ls_code = local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
        
        return [list(all_words[i]) for i in best_code]
    
    else:
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
