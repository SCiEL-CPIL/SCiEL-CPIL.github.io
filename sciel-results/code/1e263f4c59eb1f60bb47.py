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
    time_limit = 115.0  # Safety margin
    
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
        
    # Build neighbor list and intersection masks
    neighbors = [[] for _ in range(total_words)]
    # Precompute intersection masks for swap optimization
    # intersect_mask[i] is a set of indices in neighbors[i] that are NOT in neighbors[j]?
    # Actually, for swap (old, new), we need to know if any word w has count[w] == 1 and w is in neighbors[old] but not neighbors[new].
    # Let A = neighbors[old], B = neighbors[new].
    # Danger set D = A \ B.
    # If any w in D has count[w] == 1, swap is invalid.
    # Precomputing A\B for all pairs is O(N^2 * |A|) which is 2187^2 * 15 ~ 72M ops. Might be too slow to precompute fully at start?
    # 72M simple integer ops in Python might take ~5-10s. Acceptable?
    # Alternatively, we can compute on the fly but use a precomputed "in_neighbors" lookup?
    # Let's try to precompute the "danger" check more efficiently.
    # We can precompute for each word w, the set of codewords that cover it.
    # Let's stick to the current on-the-fly but optimize the inner loop.
    
    # Actually, let's just precompute the neighbor lists as sets for faster intersection/difference?
    # neighbors_set[i] = set(neighbors[i])
    neighbors_set = [set() for _ in range(total_words)]
    
    for i in range(total_words):
        w = words[i]
        # Self
        neighbors[i].append(i)
        neighbors_set[i].add(i)
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
                neighbors_set[i].add(idx)
                
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
        
        # Precompute "only covered by c" logic?
        # For swap (c_old, c_new):
        # Invalid if exists w in (N(c_old) \ N(c_new)) such that counts[w] == 1.
        
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
                # Optimization: Use set difference
                # A_set = neighbors_set[c_old]
                # B_set = neighbors_set[c_new]
                # danger_set = A_set - B_set
                # if any(counts[w] == 1 for w in danger_set): invalid
                
                # To make this faster, we can precompute the set difference?
                # Or just iterate A and check if in B.
                # Since |A| is small (15), this is fast.
                for nb in neighbors[c_old]:
                    if nb not in neighbors_set[c_new]:
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

    # Polish phase: Try to find a smaller code by random restarts of local search with random perturbations
    def polish(code, seed_val):
        rng = random.Random(seed_val)
        current_code = set(code)
        counts = get_counts(current_code)
        best_code = set(current_code)
        best_size = len(current_code)
        
        # Try to remove more aggressively by trying all pairs? No, too slow.
        # Instead, do a few rounds of "shake" and "remove"
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            # Try to remove any removable
            removed_any = False
            for c in list(current_code):
                removable = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        removable = False
                        break
                if removable:
                    current_code.discard(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    best_size = len(current_code)
                    best_code = set(current_code)
                    removed_any = True
                    break
            if removed_any:
                continue
            
            # If no removal, try a random swap that might enable removal later
            # Or just accept current best
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = set(current_code)
                
        return best_code, best_size

    best_code = None
    best_size = 999999
    
    # Run multiple random seeds
    for i in range(100): # Increased restarts, time limited
        if time.time() - start_time > time_limit:
            break
            
        # Generate greedy cover
        code = greedy_cover(i)
        
        # Optimize
        opt_code, size = local_search(code, i)
        
        if opt_code is not None:
            # Polish
            pol_code, pol_size = polish(opt_code, i + 1000)
            if pol_size < best_size:
                best_size = pol_size
                best_code = pol_code
            elif size < best_size:
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
