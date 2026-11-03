import itertools
import random
import numpy as np
import time
import sys

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute coverage: for each word i, which words are within Hamming distance radius
        # Using vectorized operations for speed
        coverage = []
        for i in range(total_words):
            dist = np.abs(all_words - all_words[i]).sum(axis=1)
            idxs = np.where(dist <= radius)[0]
            coverage.append(idxs)
            
        # Precompute coverers: for each word w, which codewords cover it
        coverers = [[] for _ in range(total_words)]
        for i in range(total_words):
            for w in coverage[i]:
                coverers[w].append(i)
        
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
                for i in range(total_words):
                    if i in code_indices:
                        continue
                    new_cover = len(set(coverage[i]) & uncovered)
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
                uncovered -= set(coverage[chosen])
                
            return code_indices

        def local_search(code_set):
            code_set = set(code_set)
            if not code_set:
                return code_set
            
            def get_counts(codes):
                counts = np.zeros(total_words, dtype=np.int32)
                for c in codes:
                    counts[coverage[c]] += 1
                return counts
                
            counts = get_counts(code_set)
            max_iterations = 300
            iterations = 0
            improved = True
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning: remove codewords that don't have any private words
                for c in list(code_set):
                    private = np.where(counts[coverage[c]] == 1)[0]
                    if len(private) == 0:
                        code_set.remove(c)
                        counts[coverage[c]] -= 1
                        improved = True
                        break
                
                if improved:
                    continue
                
                # 2. Swap: try to replace a codeword with one that covers its private words
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_words = np.where(counts[coverage[c]] == 1)[0]
                    if len(private_words) == 0:
                        continue
                    
                    private_set = set(private_words.tolist())
                    
                    # Try to find a replacement codeword that covers all private words
                    best_replacement = -1
                    best_gain = 0
                    
                    for w in range(total_words):
                        if w in code_set:
                            continue
                        # Check if w covers all private words of c
                        covers_all = all(p in set(coverage[w]) for p in private_words)
                        if not covers_all:
                            continue
                        
                        # Calculate gain: how many additional words does w cover that c doesn't
                        covered_by_w = set(coverage[w])
                        covered_by_c = set(coverage[c])
                        gain = len(covered_by_w - covered_by_c)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_replacement = w
                    
                    if best_replacement != -1:
                        code_set.remove(c)
                        counts[coverage[c]] -= 1
                        code_set.add(best_replacement)
                        counts[coverage[best_replacement]] += 1
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
            best_code = list(ls_code)
        
        # Iterated Local Search
        iterations = 0
        while time.time() - start_time < time_limit:
            iterations += 1
            
            # Perturb the best code
            perturbed_code = list(best_code)
            k_remove = random.randint(1, 3)
            if len(perturbed_code) > k_remove:
                to_remove = random.sample(perturbed_code, k_remove)
                for r in to_remove:
                    perturbed_code.remove(r)
            
            # Find uncovered words
            covered = set()
            for c in perturbed_code:
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            
            # Re-cover greedily
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            
            # Local Search
            ls_code = local_search(new_code)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = list(ls_code)
                
        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback for other parameters
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = []
        for i, c in enumerate(all_words):
            idxs = []
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    idxs.append(j)
            coverage.append(idxs)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                if i in code_indices:
                    continue
                count = len(set(coverage[i]) & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered -= set(coverage[best_idx])
            
        return [list(all_words[i]) for i in code_indices]
