# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    # Use a peeling heuristic based on the dual distance-2 packing.
    # For q=3, n=8, radius=1:
    # 1. Generate a valid covering code (initially the direct sum construction).
    # 2. Iteratively remove codewords whose 17-neighborhood is fully covered by other selected codewords.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use the direct sum construction
        # For general case, we'll use a simple greedy or the tetracode direct sum if applicable
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specific optimization for q=3, n=8, radius=1
    # Start with the direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Convert to numpy array for efficient operations
    code_arr = np.array(code, dtype=np.int8)
    m = len(code_arr)
    
    if m == 0:
        return code
    
    # Map from word tuple to index in 0..6560
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    # Precompute neighborhoods for each codeword
    # neighborhood[i] is a list of indices in 0..6560
    neighborhoods = []
    for c in code:
        neigh = [word_to_idx(c)]
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_c = c[:]
                    new_c[pos] = val
                    neigh.append(word_to_idx(new_c))
        neighborhoods.append(neigh)
    
    # Coverage count: how many codewords cover each word
    coverage = np.zeros(6561, dtype=np.int32)
    for neigh in neighborhoods:
        for idx in neigh:
            coverage[idx] += 1
    
    # Peeling loop: Batch removal of all removable codewords
    # A codeword is removable if all words in its neighborhood are covered >= 2 times
    # (meaning they are covered by at least one other codeword)
    
    changed = True
    while changed:
        changed = False
        to_remove = []
        
        # Identify all removable codewords in current set
        for i in range(m):
            # Check if codeword i is removable
            # We need coverage[idx] >= 2 for all idx in neighborhoods[i]
            # Using numpy min for speed
            if np.min(coverage[neighborhoods[i]]) >= 2:
                to_remove.append(i)
        
        if to_remove:
            changed = True
            # Remove all identified codewords
            # We must be careful: removing one might affect the removability of others,
            # but since we checked they are removable based on the *current* coverage,
            # and we are removing them all, we just subtract their contributions.
            # Note: If two removable codewords overlap, removing both is still valid 
            # because each word in their union was covered by at least 2 codewords 
            # (including the one being removed and at least one other).
            # Wait, is it possible that A and B are both removable, but a word W is 
            # covered only by A and B? 
            # If W is covered by A and B, coverage[W] = 2. 
            # When checking A, W is in A's neighborhood, coverage[W]=2 >= 2, so A is removable.
            # When checking B, W is in B's neighborhood, coverage[W]=2 >= 2, so B is removable.
            # If we remove both A and B, coverage[W] becomes 0. This is INVALID.
            
            # Therefore, we CANNOT remove all removable codewords in a single batch blindly.
            # We must ensure that for every word, at least one covering codeword remains.
            # The condition "coverage[idx] >= 2" ensures that if we remove *one* codeword, 
            # the word is still covered. But if we remove *multiple* codewords that are the 
            # only two covers for a word, we break it.
            
            # Strategy: Remove one by one, but optimize the selection.
            # Or, use a matching/set cover approach on the "critical" overlaps?
            # Given the time limit, a simple "remove one, update, repeat" is safer but slower.
            # Let's try a slightly smarter "greedy peeling":
            # Pick the codeword with the maximum redundancy (sum of coverage in neighborhood) 
            # among those that are removable. Remove it. Update. Repeat.
            # This was the original approach. Let's stick to removing one at a time 
            # but optimize the inner loop.
            
            # To speed up the original approach, we can just pick the first removable one found.
            # Randomized peeling might escape local minima? No, this is a monotonic removal.
            # Let's just pick the first removable one to minimize overhead.
            
            # Re-evaluating: The original code picked the one with max redundancy.
            # Picking the first one is O(1) selection after O(M*N) scan.
            # Let's just take the first one found in `to_remove`.
            
            idx_to_remove = to_remove[0]
            
            # Update coverage
            for idx in neighborhoods[idx_to_remove]:
                coverage[idx] -= 1
            
            # Remove from code_arr and neighborhoods
            # np.delete is O(M) which is fine for M~500.
            code_arr = np.delete(code_arr, idx_to_remove, axis=0)
            neighborhoods.pop(idx_to_remove)
            m -= 1

    return [list(row) for row in code_arr]
# EVOLVE-BLOCK-END
