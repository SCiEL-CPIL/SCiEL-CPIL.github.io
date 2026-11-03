import itertools
import random
import time
import numpy as np

# Precompute neighbors for q=3, n=7
# This is specific to the task, but the function signature takes q, n, radius.
# We should handle general q, n, radius but optimize for the specific case if possible?
# The task says: "The evaluator passes this task's alphabet size q, length n and covering radius."
# It implies we should write a general solution, but we can specialize if we detect q=3, n=7, radius=1.
# However, the prompt says "Keep the function names and signatures".
# I will write a general solution that is fast for small n, but specifically tuned for the 3^7 case if it matches.

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute neighbors
    # For efficiency in the specific case, we can generate neighbors on the fly or precompute.
    # 2187 is small enough to precompute for any reasonable q,n.
    
    words = []
    for i in range(total_words):
        # Convert i to base-q digits
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
        
    # Build neighbor list
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = words[i]
        # Self
        neighbors[i].append(i)
        # Hamming distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                # Convert back to index
                idx = 0
                power = 1
                for j in range(n):
                    idx += new_w[j] * power
                    power *= q
                neighbors[i].append(idx)
                
    # Helper to check if a set of codewords covers all words
    def is_covering(code_set):
        covered = np.zeros(total_words, dtype=bool)
        for c in code_set:
            for nb in neighbors[c]:
                covered[nb] = True
        return np.all(covered)
        
    # Helper to get coverage counts
    def get_counts(code_set):
        counts = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                counts[nb] += 1
        return counts

    # Greedy construction
    def greedy_cover(seed_val):
        rng = random.Random(seed_val)
        uncovered = set(range(total_words))
        code = set()
        
        # Precompute coverage of each potential codeword on the *initial* set? 
        # No, uncovered changes.
        
        while uncovered:
            best_c = -1
            best_cov_count = -1
            # To speed up, we can sample or check all. 
            # For 2187 words, checking all 2187 candidates each step is 2187*2187 ~ 4.7M ops per step?
            # No, we need to count how many *uncovered* words each candidate covers.
            # Candidate c covers neighbors[c].
            # Intersection with uncovered.
            
            # Optimization: 
            # If we pick c from uncovered, it covers itself.
            # Let's just iterate through all possible c in range(total_words)
            # This is O(N^2) per step? No, O(N * |neighbors|).
            # |neighbors| = 1 + n*(q-1). For 3,7: 1 + 7*2 = 15.
            # So 2187 * 15 ~ 32k ops per step.
            # Max steps ~ 200. Total ~ 6M ops. Very fast.
            
            for c in range(total_words):
                # Count how many neighbors of c are in uncovered
                # We can do this faster if we maintain a boolean array for uncovered
                # But set lookup is O(1).
                cov_count = 0
                for nb in neighbors[c]:
                    if nb in uncovered:
                        cov_count += 1
                if cov_count > best_cov_count:
                    best_cov_count = cov_count
                    best_c = c
                # Random tie breaking
                elif cov_count == best_cov_count and cov_count > 0:
                    if rng.random() < 0.5:
                        best_c = c
                        
            if best_c == -1:
                break # Should not happen if uncovered is not empty
                
            code.add(best_c)
            # Remove covered words from uncovered
            for nb in neighbors[best_c]:
                uncovered.discard(nb)
                
        return code

    # Local Search
    def local_search(initial_code, seed_val):
        rng = random.Random(seed_val)
        code = set(initial_code)
        counts = get_counts(code)
        
        # Verify validity
        if np.any(counts == 0):
             return None, 999999 # Invalid
        
        best_size = len(code)
        
        while time.time() - start_time < time_limit:
            # Try to remove
            removed = False
            # Find a codeword c such that all its neighbors have count > 1
            # We can iterate through code
            for c in list(code):
                removable = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove c
                    code.discard(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    best_size = len(code)
                    removed = True
                    break # Restart loop to check for new removals
            
            if not removed:
                # Try to swap
                # Pick a random c_old
                if not code:
                    break
                c_old = rng.choice(list(code))
                # Pick a random c_new
                c_new = rng.randint(0, total_words - 1)
                
                if c_new in code:
                    continue
                    
                # Check if swap is valid
                # New counts = counts - neighbors[c_old] + neighbors[c_new]
                # We need min(new_counts) >= 1
                
                # Efficient check:
                # Words affected are neighbors[c_old] and neighbors[c_new]
                # Let A = neighbors[c_old], B = neighbors[c_new]
                # For w in A \ B: new_count = counts[w] - 1. Must be >= 1 => counts[w] >= 2
                # For w in B \ A: new_count = counts[w] + 1. Always >= 1
                # For w in A \cap B: new_count = counts[w]. Must be >= 1 (already true)
                
                valid_swap = True
                for nb in neighbors[c_old]:
                    if nb not in neighbors[c_new]: # This check is slow
                        if counts[nb] < 2:
                            valid_swap = False
                            break
                
                if valid_swap:
                    # Perform swap
                    code.discard(c_old)
                    code.add(c_new)
                    for nb in neighbors[c_old]:
                        counts[nb] -= 1
                    for nb in neighbors[c_new]:
                        counts[nb] += 1
                    # Update best size? Size is same.
                    # But structure changed.
                    # We might be able to remove now.
                    continue
                
                # If too many failed swaps, maybe restart?
                # For now, just keep trying.
                
        return code, best_size

    best_code = None
    best_size = 999999
    
    # Run multiple random seeds
    for i in range(50): # Max 50 restarts, but time limited
        if time.time() - start_time > time_limit:
            break
            
        # Generate greedy cover
        code = greedy_cover(i)
        
        # Optimize
        opt_code, size = local_search(code, i)
        
        if opt_code is not None:
            if size < best_size:
                best_size = size
                best_code = opt_code
                
        # If we hit the known optimum, break?
        if best_size == 186:
            break

    if best_code is None:
        # Fallback to seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to words
    result = []
    for idx in best_code:
        result.append(list(words[idx]))
        
    return result
