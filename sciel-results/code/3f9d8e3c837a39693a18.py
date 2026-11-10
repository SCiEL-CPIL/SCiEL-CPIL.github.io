import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code using a greedy set-cover heuristic.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Specific to the task: q=3, n=6, radius=1
    # We handle the general case but optimize for the known task parameters if they match.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters to ensure validity, 
        # though the task specifies these parameters.
        import itertools
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # General fallback: trivial code (all words)
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Optimized path for q=3, n=6, radius=1
    total_words = 3 ** 6
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(6):
            idx = idx * 3 + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * 6
        for i in range(5, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute spheres for all 729 words
    # sphere[i] is a list of indices covered by word i
    spheres = []
    for i in range(total_words):
        w = idx_to_word(i)
        covered_indices = [i]
        for pos in range(6):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                covered_indices.append(word_to_idx(w))
        w[pos] = orig_val # restore, though w is local
        spheres.append(covered_indices)
    
    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    chosen_indices = []
    
    # Optimization: Keep track of uncovered count for each candidate
    # Initially, every candidate covers 13 words.
    # We can just scan all candidates each time to find the one covering the most uncovered words.
    # 729 candidates * 729 max iterations is ~500k checks, each check involves summing 13 bools.
    # This is fast enough.
    
    while not np.all(covered):
        best_idx = -1
        best_count = -1
        
        for i in range(total_words):
            # Count how many neighbors of i are currently uncovered
            # Using numpy slicing might be faster if we precompute sphere as array
            # But list comprehension is fine for 13 elements
            count = sum(1 for j in spheres[i] if not covered[j])
            
            if count > best_count:
                best_count = count
                best_idx = i
                # Early exit if we found a perfect cover (13 uncovered)
                if best_count == 13:
                    break
        
        if best_idx == -1:
            break # Should not happen
            
        chosen_indices.append(best_idx)
        
        # Update covered status
        for j in spheres[best_idx]:
            covered[j] = True
            
        # Deadline check
        if time.time() > deadline:
            break

    # Convert indices back to words
    result = [idx_to_word(i) for i in chosen_indices]
    
    # Safety check: Ensure distinctness (greedy adds unique indices, so they are distinct)
    # And validity (checked by construction, but let's be sure if time permits)
    # The prompt says "Valid when every word... is within Hamming distance radius".
    # Our greedy ensures this.
    
    return result
