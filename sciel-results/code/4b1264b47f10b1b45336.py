import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute distances
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # Precompute coverage sets
        coverage = [set() for _ in range(total_words)]
        for i in range(total_words):
            idxs = np.where(dist[i] <= radius)[0]
            coverage[i] = set(idxs.tolist())
            
        # Greedy Set Cover
        uncovered = set(range(total_words))
        code_set = set()
        
        while uncovered:
            best_idx = -1
            best_cover_count = -1
            for i in range(total_words):
                new_cover = len(coverage[i] & uncovered)
                if new_cover > best_cover_count:
                    best_cover_count = new_cover
                    best_idx = i
                    if best_cover_count == len(uncovered):
                        break
            
            if best_idx == -1:
                break
                
            code_set.add(best_idx)
            uncovered -= coverage[best_idx]
            
        # Local Search
        counts = np.zeros(total_words, dtype=int)
        for c in code_set:
            counts[list(coverage[c])] += 1
            
        time_limit = 110
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Pruning: Remove redundant codewords
            for c in list(code_set):
                if np.min(counts[list(coverage[c])]) > 1:
                    code_set.remove(c)
                    counts[list(coverage[c])] -= 1
                    improved = True
                    break # Restart loop to update counts
            
            if improved:
                continue
                
            # 2. Swap: Replace c with w if valid
            # Shuffle to randomize
            code_list = list(code_set)
            random.shuffle(code_list)
            
            swapped = False
            for c in code_list:
                private_words = [w for w in coverage[c] if counts[w] == 1]
                
                # If no private words, it's removable (should have been caught by pruning, but check anyway)
                if not private_words:
                    code_set.remove(c)
                    counts[list(coverage[c])] -= 1
                    improved = True
                    swapped = True
                    break
                
                # Find a replacement w that covers all private words
                # w must not be in code_set
                # We iterate through all non-codewords
                # Optimization: Only consider w that covers at least one private word?
                # Actually, w must cover ALL private words.
                
                found_w = None
                for w in range(total_words):
                    if w in code_set:
                        continue
                    # Check if w covers all private words
                    # private_words is small
                    if all(pw in coverage[w] for pw in private_words):
                        # Check if swap is valid (it is if w covers private words)
                        # But we should also ensure w doesn't break other things?
                        # Removing c only affects private words (others have count >= 2).
                        # Adding w only adds coverage.
                        # So swap is valid.
                        found_w = w
                        break
                        
                if found_w is not None:
                    code_set.remove(c)
                    counts[list(coverage[c])] -= 1
                    code_set.add(found_w)
                    counts[list(coverage[found_w])] += 1
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                continue
                
            # 3. Merge: Add w, remove c1, c2 if valid
            # This reduces size by 1.
            # Try random w not in code_set
            candidates = [w for w in range(total_words) if w not in code_set]
            if candidates:
                w = random.choice(candidates)
                
                # Find c1, c2 in code_set such that removing them and adding w keeps validity
                # This is complex. Simplified: 
                # If w covers all private words of c1 AND c2, and c1, c2 have no other private words?
                # Actually, if we remove c1 and c2, words with count 1 in c1 or c2 lose coverage.
                # w must cover all such words.
                
                # Let's try a simpler heuristic:
                # If we add w, can we remove two codewords?
                # This is expensive to check for all pairs.
                # We'll skip this for now and rely on Swaps + Pruning.
                # If we are stuck, we can try random restarts of the Greedy phase.
                
            # If no improvement, break (stuck in local minimum)
            # To avoid infinite loop, we break.
            break
            
        return [list(all_words[i]) for i in code_set]
    
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
