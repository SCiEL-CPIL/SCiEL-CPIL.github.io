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
        
        # Precompute coverage sets for faster intersection in greedy
        coverage_sets = [set(np.where(coverage_bool[i])[0]) for i in range(total_words)]
        
        # Precompute "coverers" for each word w: set of codewords that cover w
        # coverers[w] = {c | coverage_bool[c, w] is True}
        coverers = [set() for _ in range(total_words)]
        for c in range(total_words):
            for w in coverage_sets[c]:
                coverers[w].add(c)

        def greedy_set_cover(uncovered, code_indices, random_seed=None):
            if random_seed is not None:
                rng = random.Random(random_seed)
            else:
                rng = random.Random()
                
            uncovered = set(uncovered)
            code_indices = set(code_indices)
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                candidates = []
                
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
                code_indices.add(chosen)
                uncovered -= coverage_sets[chosen]
                
            return list(code_indices)

        def fast_local_search(code_set):
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
                
                # 1. Pruning (1-for-0)
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
                
                # 2. 1-for-1 Swaps
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_mask = (counts == 1) & coverage_bool[c]
                    private_words = np.where(private_mask)[0]
                    
                    if len(private_words) == 0:
                        code_set.remove(c)
                        counts[coverage_bool[c]] -= 1
                        improved = True
                        swapped = True
                        break
                    
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

                # 3. 2-for-1 Swaps (The Key Improvement)
                # Try to remove 2 codewords c1, c2 and add 1 codeword w
                # such that all words uniquely covered by c1 or c2 are covered by w.
                
                # We only need to check pairs if we haven't improved in a while or randomly
                # To save time, we sample a few pairs per iteration if 1-for-1 failed
                if len(code_set) >= 2:
                    # Sample a few pairs
                    pairs_checked = 0
                    max_pairs = 50
                    while pairs_checked < max_pairs:
                        c1, c2 = random.sample(list(code_set), 2)
                        pairs_checked += 1
                        
                        # Find words that become uncovered if we remove c1 and c2
                        # A word w is "private to the pair" if count(w)==1 and w in N(c1)
                        # OR count(w)==1 and w in N(c2)
                        # Note: If w in N(c1) and N(c2), count(w) >= 2, so it's not private to either.
                        
                        private_c1 = np.where((counts == 1) & coverage_bool[c1])[0]
                        private_c2 = np.where((counts == 1) & coverage_bool[c2])[0]
                        
                        # Combine private words
                        private_words = np.concatenate([private_c1, private_c2])
                        
                        if len(private_words) == 0:
                            # Both are redundant? Should have been caught by pruning
                            # But maybe one is redundant?
                            # If private is empty, removing them doesn't uncover anything.
                            # We can just remove one? No, if count(c1) is all >1, pruning catches it.
                            # If private is empty, it means every word in N(c1) is covered by >1 codeword.
                            # So c1 is redundant. Pruning should have caught it.
                            # However, pruning only checks one at a time.
                            # If we remove c1, counts change.
                            # If private_words is empty, we can just remove c1?
                            # Let's just skip this pair if no private words, as it implies redundancy.
                            continue

                        # We need a word w_new such that N(w_new) contains ALL private_words
                        # And w_new is not in code_set.
                        
                        # Find intersection of coverers of all private_words
                        # Start with coverers of the first private word
                        if len(private_words) > 0:
                            common_coverers = set(coverers[private_words[0]])
                            for p in private_words[1:]:
                                common_coverers.intersection_update(coverers[p])
                                if not common_coverers:
                                    break
                            
                            # Remove c1 and c2 from common_coverers (they are in code_set, and we are removing them)
                            # Also remove any other codeword in code_set
                            valid_new = [w for w in common_coverers if w not in code_set]
                            
                            if valid_new:
                                chosen = random.choice(valid_new)
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(chosen)
                                
                                # Update counts
                                counts[coverage_bool[c1]] -= 1
                                counts[coverage_bool[c2]] -= 1
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
        ls_code = fast_local_search(initial_code)
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
                covered.update(coverage_sets[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_set_cover(uncovered, perturbed_code, random_seed=iterations)
            ls_code = fast_local_search(new_code)
            
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
