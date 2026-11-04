import itertools
import time
import random

def construct(q, n, radius):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin from 120s limit
    
    # This function is specialized for the ternary n=8 radius=1 case
    # but should handle general inputs by falling back to a simple method
    # if parameters don't match the optimized path.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback: Direct sum with Tetracode-like logic or just all words?
        # For non-optimized cases, we return a valid but potentially large code.
        # Simplest valid code: All words.
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Optimized path for q=3, n=8, radius=1
    
    # 1. Generate the initial 729-word code (Tetracode x All)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # 2. Precompute indices for all words in {0,1,2}^8
    # Index = w[0]*3^7 + ... + w[7]*3^0
    # We can use a dictionary or just calculate index on the fly.
    # Calculating index is fast: sum(w[i] * (3**i) for i in range(8))
    
    total_words = 3**8
    coverage = [0] * total_words
    
    def get_index(word):
        idx = 0
        for i in range(8):
            idx += word[i] * (3**i)
        return idx

    def get_neighbors(word):
        # Returns list of 17 words (self + 16 neighbors)
        res = [word]
        for i in range(8):
            # Change position i to the other 2 values
            for delta in [1, 2]:
                new_val = (word[i] + delta) % 3
                new_word = list(word)
                new_word[i] = new_val
                res.append(tuple(new_word))
        return res

    # 3. Initialize coverage
    for c in initial_code:
        idx_c = get_index(c)
        # We need to cover c and its neighbors
        # Actually, the sphere of c covers c and its neighbors.
        # So we increment coverage for all w in sphere(c).
        for w in get_neighbors(c):
            coverage[get_index(w)] += 1
            
    current_code = initial_code[:]
    
    # 4. Greedy Removal
    # We iterate through the code and try to remove each word.
    # If removal doesn't leave any word with coverage 0, we remove it.
    
    # To make it faster, we can precompute the spheres of all initial codewords.
    # But spheres depend only on the word, not the code.
    # However, we might generate new codewords? No, we only remove.
    # So we can precompute spheres for all 729 words.
    
    spheres = {}
    for c in initial_code:
        spheres[c] = get_neighbors(c)
        
    # Precompute indices for all words in spheres to avoid repeated get_index calls
    sphere_indices = {}
    for c in initial_code:
        idx_list = []
        for w in spheres[c]:
            idx_list.append(get_index(w))
        sphere_indices[c] = idx_list
        
    changed = True
    while changed:
        changed = False
        # Iterate over a copy of the current code to allow removal during iteration
        # Or iterate by index and skip removed ones.
        # Since we remove from a list, indices shift.
        # Better: iterate over indices in reverse or use a set of active codewords.
        
        # Let's use a list of active codewords.
        # We try to remove each one.
        
        # Optimization: Randomize order?
        # For now, fixed order.
        
        to_remove = []
        
        # We need to check each codeword in current_code.
        # If we remove one, coverage changes.
        # So we must check one by one, updating coverage immediately if successful.
        
        for i in range(len(current_code)):
            if time.time() > deadline:
                break
                
            c = current_code[i]
            
            # Check if we can remove c
            # Decrement coverage for all w in sphere(c)
            can_remove = True
            affected_indices = sphere_indices[c]
            
            for idx in affected_indices:
                coverage[idx] -= 1
                if coverage[idx] == 0:
                    can_remove = False
                    break
                    
            if can_remove:
                # Successfully removed c
                # coverage is already decremented
                current_code.pop(i)
                changed = True
            else:
                # Revert coverage
                for idx in affected_indices:
                    coverage[idx] += 1
                    if coverage[idx] > 0 and can_remove == False:
                         # We broke early, so we only need to revert up to the point of break?
                         # No, we decremented ALL of them?
                         # In the loop above, if we break, we only decremented SOME.
                         # We need to revert only the ones we decremented.
                         pass
                # Let's re-implement the check to be safer
                # Decrement all, check min, revert all if min < 1.
                
                # Actually, the logic above:
                # for idx in affected_indices:
                #    coverage[idx] -= 1
                #    if coverage[idx] == 0: break
                # If we break, we haven't decremented the rest.
                # So we must revert the ones we DID decrement.
                
                # Let's restart the revert logic properly.
                # We need to know how many we decremented.
                
                # Better approach:
                # 1. Decrement all.
                # 2. Check if any is 0.
                # 3. If any is 0, revert all.
                
                # Let's rewrite the inner loop for correctness.
                
                pass # Placeholder for corrected logic below

    # Corrected Greedy Removal Logic
    # Re-initialize coverage if we messed up, or just write it right the first time.
    
    # Let's restart the greedy part with correct logic.
    
    # Re-initialize coverage
    coverage = [0] * total_words
    for c in initial_code:
        for idx in sphere_indices[c]:
            coverage[idx] += 1
            
    current_code = initial_code[:]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(current_code)):
            if time.time() > deadline:
                break
                
            c = current_code[i]
            affected_indices = sphere_indices[c]
            
            # Decrement all
            for idx in affected_indices:
                coverage[idx] -= 1
                
            # Check if any is 0
            min_cov = min(coverage[idx] for idx in affected_indices)
            
            if min_cov >= 1:
                # Safe to remove
                current_code.pop(i)
                changed = True
            else:
                # Revert
                for idx in affected_indices:
                    coverage[idx] += 1
                    
    return [list(c) for c in current_code]
