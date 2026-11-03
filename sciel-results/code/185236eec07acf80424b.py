import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute coverage sets
        # coverage[i] is a set of indices j such that dist(i, j) <= radius
        # For q=3, n=6, radius=1, each word covers 1 + 6*(q-1) = 19 words.
        # Total words = 729.
        # We can precompute neighbors efficiently.
        
        coverage = [set() for _ in range(total_words)]
        for i in range(total_words):
            w = all_words[i]
            # Self
            coverage[i].add(i)
            # Radius 1 neighbors
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    # Find index of new_w
                    # Since we have all_words, we can use a dictionary for O(1) lookup
                    # But building a dict of 729 items is fast.
                    pass
        
        # Let's use a dictionary for fast index lookup
        word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
        
        for i in range(total_words):
            w = all_words[i]
            coverage[i].add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = w.copy()
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    coverage[i].add(idx)
                    
        # Also, coverage is symmetric? No, but dist is symmetric.
        # If i covers j, j covers i.
        # So we can just compute for i < j or just fill both.
        # The above loop fills coverage[i] with all j such that dist(i,j)<=1.
        
        def get_uncovered(code_set):
            covered = set()
            for c in code_set:
                covered.update(coverage[c])
            return set(range(total_words)) - covered

        def greedy_cover(uncovered, weights=None, rng=None):
            if rng is None:
                rng = random.Random()
            if weights is None:
                weights = np.ones(total_words)
                
            code = []
            uncovered = set(uncovered)
            
            while uncovered:
                best_idx = -1
                best_score = -1
                # To speed up, we can pre-compute potential gains, but uncovered changes.
                # For 729 words, iterating all is fine.
                for i in range(total_words):
                    if i in code: # Optimization: skip already chosen? Not strictly necessary if we just add to set
                        continue
                    gain = len(coverage[i] & uncovered)
                    # Lagrangian-like score: gain / weight? Or gain - weight?
                    # Standard greedy: max gain.
                    # Weighted greedy: max gain / weight[i] (if weight represents cost)
                    # Here we want to minimize code size, so cost of adding i is 1.
                    # Let's stick to max gain, but use weights to break ties or bias.
                    score = gain
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        # Random tie-breaking
                        if best_score > 0:
                             # We could collect candidates, but for simplicity:
                             pass
                            
                if best_idx == -1:
                    break
                    
                code.append(best_idx)
                uncovered -= coverage[best_idx]
                
            return code

        def local_search(code_set, rng=None):
            if rng is None:
                rng = random.Random()
            code_set = set(code_set)
            
            # Precompute counts for each word: how many codewords cover it
            counts = np.zeros(total_words, dtype=int)
            for c in code_set:
                for w in coverage[c]:
                    counts[w] += 1
            
            improved = True
            iterations = 0
            max_iterations = 100
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                # 1. Pruning: Remove any codeword that is not the unique cover for any word
                removable = None
                for c in list(code_set):
                    # Check if there is any word w in coverage[c] such that counts[w] == 1
                    # If no such word exists, c is redundant.
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
                
                # 2. Swaps: Try to replace a codeword with another that covers its private words
                current_list = list(code_set)
                rng.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_words = [w for w in coverage[c] if counts[w] == 1]
                    
                    if not private_words:
                        # Should have been caught by pruning, but just in case
                        code_set.remove(c)
                        for w in coverage[c]:
                            counts[w] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Find a replacement c' not in code_set such that coverage[c'] covers all private_words
                    # Optimization: Only check c' that cover at least one private word?
                    # Actually, we need c' to cover ALL private words.
                    # Private words are unique to c.
                    # So c' must be a neighbor of all private words.
                    
                    candidates = []
                    # Intersect coverage of all private words?
                    # If private_words is large, intersection is small.
                    # If private_words is 1, candidates are all neighbors of that word.
                    
                    # Let's just iterate all potential replacements (not in code_set)
                    # This is O(N^2) per swap attempt, which is 729^2 ~ 500k.
                    # With 73 codewords, 73 * 500k = 36M operations per LS iteration.
                    # Might be slow.
                    
                    # Better: 
                    # For each private word w, the replacement must be in coverage[w].
                    # So candidates = intersection of coverage[w] for w in private_words.
                    if len(private_words) == 1:
                        cand_set = coverage[private_words[0]]
                    else:
                        cand_set = coverage[private_words[0]]
                        for w in private_words[1:]:
                            cand_set &= coverage[w]
                            if not cand_set:
                                break
                    
                    if not cand_set:
                        continue
                        
                    # Filter out existing codewords
                    valid_candidates = [c2 for c2 in cand_set if c2 not in code_set]
                    
                    if not valid_candidates:
                        continue
                        
                    # Pick one randomly or best (by coverage of remaining uncovered? No, we just need to maintain coverage)
                    # Any valid candidate works for a valid swap.
                    # To improve "quality", we might prefer one that covers more "shared" words?
                    # For now, random pick.
                    c_new = rng.choice(valid_candidates)
                    
                    # Perform swap
                    code_set.remove(c)
                    for w in coverage[c]:
                        counts[w] -= 1
                    code_set.add(c_new)
                    for w in coverage[c_new]:
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
        time_limit = 115
        
        # Initial Solution
        initial_code = greedy_cover(set(range(total_words)))
        ls_code = local_search(initial_code)
        best_code = ls_code
        best_size = len(ls_code)
        
        # Iterated Local Search
        rng = random.Random(42)
        
        while time.time() - start_time < time_limit:
            # Perturb
            perturbed = list(best_code)
            # Remove 1-3 random codewords
            k_remove = rng.randint(1, 3)
            if len(perturbed) > k_remove:
                to_remove = rng.sample(perturbed, k_remove)
                for r in to_remove:
                    perturbed.remove(r)
            
            # Re-cover
            covered = set()
            for c in perturbed:
                covered.update(coverage[c])
            uncovered = set(range(total_words)) - covered
            
            new_code = greedy_cover(uncovered)
            # Combine and clean up
            new_code_set = set(perturbed) | set(new_code)
            
            # Local Search
            ls_code = local_search(new_code_set, rng)
            
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
                
            # Optional: Accept worse solutions with some probability (Simulated Annealing style)
            # But for now, hill climbing is robust.
            
        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback for other parameters
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
