import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Total space size
    space_size = q ** n
    
    # Precompute all words as tuples for hashing, and map to indices
    # For q=3, n=7, space_size=2187. This is small enough to handle explicitly.
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Generate all words
    all_words = [tuple(idx_to_word(i)) for i in range(space_size)]
    
    # Precompute coverage sets for all possible codewords
    # coverage[codeword_idx] = set of word indices covered
    # To save memory and time, we can generate neighbors on the fly or precompute for all 2187 candidates.
    # Precomputing for all 2187 candidates is fast (2187 * 15 ops).
    
    candidate_indices = list(range(space_size))
    cover_sets = [None] * space_size
    
    for cw_idx in candidate_indices:
        cw = all_words[cw_idx]
        covered = set()
        covered.add(cw_idx)
        for pos in range(n):
            for val in range(q):
                if val == cw[pos]:
                    continue
                new_word = list(cw)
                new_word[pos] = val
                covered.add(word_to_idx(tuple(new_word)))
        cover_sets[cw_idx] = covered

    best_solution = None
    best_size = space_size + 1
    
    # Greedy Set Cover with Local Search
    # We'll try a few random seeds
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        
        # Start with a random set of codewords
        # Heuristic: start with size close to lower bound
        # Lower bound is 156. Let's start with 160.
        initial_size = 160 + random.randint(0, 10)
        
        # Randomly select initial codewords
        current_codewords = set(random.sample(candidate_indices, initial_size))
        
        # Track coverage counts
        # cover_count[word_idx] = number of codewords in current_codewords that cover word_idx
        cover_count = np.zeros(space_size, dtype=np.int32)
        for cw in current_codewords:
            for w in cover_sets[cw]:
                cover_count[w] += 1
        
        # Check if covered
        if np.all(cover_count > 0):
            # Try to reduce
            # Local search: remove redundant, then swap
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                
                # 1. Remove redundant codewords
                # A codeword is redundant if all words it covers are covered by >= 2 codewords
                # (i.e., removing it leaves cover_count >= 1 for all its words)
                redundant = []
                for cw in list(current_codewords):
                    is_redundant = True
                    for w in cover_sets[cw]:
                        if cover_count[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        redundant.append(cw)
                
                if redundant:
                    # Remove one at a time to update counts
                    for cw in redundant:
                        if time.time() > deadline - 2:
                            break
                        for w in cover_sets[cw]:
                            cover_count[w] -= 1
                        current_codewords.remove(cw)
                        improved = True
                    # If we removed any, restart the loop to check for more
                    if time.time() > deadline - 2:
                        break
                    continue # Restart loop after removals
                
                # 2. Swap codewords to potentially reduce size or enable removal
                # This is expensive. We might just rely on random restarts and greedy fill.
                # Let's try a simple greedy fill first if not fully covered.
        
        if not np.all(cover_count > 0):
            # Greedy fill: add codewords that cover the most uncovered words
            while not np.all(cover_count > 0) and time.time() < deadline - 2:
                uncovered = np.where(cover_count == 0)[0]
                if len(uncovered) == 0:
                    break
                
                best_cw = -1
                best_gain = -1
                # Sample a subset of candidates to speed up if needed, but 2187 is small
                # Iterate through all candidates not in current set
                for cw in candidate_indices:
                    if cw in current_codewords:
                        continue
                    gain = 0
                    for w in cover_sets[cw]:
                        if cover_count[w] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_cw = cw
                    if best_gain >= len(uncovered):
                        break
                
                if best_cw == -1:
                    break
                    
                current_codewords.add(best_cw)
                for w in cover_sets[best_cw]:
                    cover_count[w] += 1
                
                # After adding, try to remove redundancies immediately
                # This can lead to a smaller set than pure greedy
                removed_any = False
                for cw in list(current_codewords):
                    if time.time() > deadline - 2:
                        break
                    is_redundant = True
                    for w in cover_sets[cw]:
                        if cover_count[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for w in cover_sets[cw]:
                            cover_count[w] -= 1
                        current_codewords.remove(cw)
                        removed_any = True
                        break # Remove one and re-evaluate
                
                if removed_any:
                    continue # Re-check coverage and redundancies
            
            if not np.all(cover_count > 0):
                continue
        
        # Final polish: try to remove any remaining redundancies
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for cw in list(current_codewords):
                is_redundant = True
                for w in cover_sets[cw]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in cover_sets[cw]:
                        cover_count[w] -= 1
                    current_codewords.remove(cw)
                    improved = True
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_solution = list(current_codewords)
            
            # If we hit the theoretical lower bound, we can stop
            if best_size <= 156:
                break

    if best_solution is None:
        # Fallback to a valid construction (e.g., the seed one or a simple greedy)
        # The seed program returns 216. Let's just return a valid greedy solution.
        # Greedy from scratch
        current_codewords = set()
        cover_count = np.zeros(space_size, dtype=np.int32)
        while not np.all(cover_count > 0):
            uncovered = np.where(cover_count == 0)[0]
            best_cw = -1
            best_gain = -1
            for cw in candidate_indices:
                if cw in current_codewords:
                    continue
                gain = 0
                for w in cover_sets[cw]:
                    if cover_count[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cw = cw
            if best_cw == -1:
                break
            current_codewords.add(best_cw)
            for w in cover_sets[best_cw]:
                cover_count[w] += 1
        best_solution = list(current_codewords)

    # Convert indices back to word lists
    return [list(all_words[i]) for i in best_solution]
