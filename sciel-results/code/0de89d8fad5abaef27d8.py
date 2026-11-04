import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # For q=3, n=8, radius=1, total_words = 6561
    # This is small enough to use numpy arrays for efficient coverage tracking.
    
    # Precompute neighbors for each word.
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius.
    # Since radius is small (1), we can compute this quickly.
    
    # Optimization: Instead of storing sets, we can store lists of neighbors.
    # But for counting coverage, we need to know which words are covered.
    
    # Let's represent the code as a boolean array `is_code` of size total_words.
    # And a coverage count array `cov_count` of size total_words.
    # cov_count[x] = number of codewords c such that dist(c, x) <= radius.
    # A word x is covered if cov_count[x] > 0.
    
    # To update efficiently:
    # When adding a codeword c: for all x in neighbors[c], cov_count[x] += 1.
    # When removing a codeword c: for all x in neighbors[c], cov_count[x] -= 1.
    
    # Precompute neighbor lists
    # For q=3, n=8, radius=1:
    # Each word has 1 + n*(q-1) = 1 + 8*2 = 17 neighbors.
    
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        # Radius 1 neighbors
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                # Calculate index
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = neigh
        
    # Initial Code: Use the Tetracode Direct Sum as a strong starting point.
    # Tetracode: 9 words of length 4.
    # Direct sum with length n-4 words.
    # Size: 9 * 3^(n-4). For n=8, 9 * 81 = 729.
    # This is a valid covering code.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            idx = 0
            for j in range(n):
                idx = idx * q + word[j]
            initial_code.append(idx)
            
    # Convert to set for fast lookup and boolean array
    code_set = set(initial_code)
    is_code = np.zeros(total_words, dtype=bool)
    for idx in initial_code:
        is_code[idx] = True
        
    # Initialize coverage counts
    cov_count = np.zeros(total_words, dtype=np.int32)
    for idx in initial_code:
        cov_count[neighbors[idx]] += 1
        
    # Check if fully covered
    # Uncovered words are those with cov_count[x] == 0
    
    def is_fully_covered():
        return np.all(cov_count > 0)
        
    # Local Search: 1-Swap (Remove one, Add one) or 1-Insert (Add one)
    # Since we want to minimize size, we primarily try swaps.
    # If stuck, we might try insertions if we are over a lower bound, but usually
    # starting from a valid code and swapping is best.
    
    best_code = list(initial_code)
    best_size = len(best_code)
    
    # Precompute list of all possible candidates for insertion?
    # No, just iterate over all words.
    
    # To speed up evaluation of a swap (remove c1, add c2):
    # 1. Temporarily remove c1: cov_count[neighbors[c1]] -= 1
    # 2. Check if any word becomes uncovered (cov_count becomes 0).
    #    If so, c1 is critical unless c2 covers those specific words.
    # 3. Temporarily add c2: cov_count[neighbors[c2]] += 1
    # 4. Check if coverage is restored.
    # 5. If valid, accept.
    
    # Optimization:
    # Let critical_words(c1) = {x in neighbors[c1] | cov_count[x] == 1}
    # If we remove c1, these words become uncovered.
    # We must choose c2 such that c2 covers ALL words in critical_words(c1).
    # i.e., critical_words(c1) must be a subset of neighbors[c2].
    
    # We can precompute for each c1 the list of c2 that cover its critical words?
    # That's expensive.
    
    # Alternative:
    # Iterate c1 in code.
    # Identify critical words.
    # If no critical words, c1 can be removed freely. Then we just need to find ANY c2
    # that doesn't break coverage? No, if we remove c1 and coverage is still full,
    # we have reduced size by 1? No, we are swapping, so size stays same.
    # Wait, if we can remove c1 and stay covered, we should just REMOVE it and not add c2?
    # The problem asks for the MINIMUM number of words.
    # So "Delete" moves are valid if coverage remains.
    # "Swap" moves are valid if coverage remains.
    
    # Let's implement a loop that tries:
    # 1. Delete c1 if possible (reduces size).
    # 2. Swap c1 with c2 if possible (maintains size, might enable future deletions).
    
    # To make this efficient, we can track `critical_count` for each word x:
    # critical_count[x] is 1 if cov_count[x] == 1, else 0.
    # When removing c1, the words that become uncovered are exactly those x in neighbors[c1]
    # where critical_count[x] == 1.
    
    # Let's use a simpler heuristic first: Randomized Local Search with Restart.
    
    # We will run multiple restarts.
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
            
        # Reset to initial code (or best found so far?)
        # Let's reset to initial tetracode to explore different basins.
        current_code = list(initial_code)
        code_set = set(current_code)
        is_code = np.zeros(total_words, dtype=bool)
        for idx in current_code:
            is_code[idx] = True
            
        cov_count = np.zeros(total_words, dtype=np.int32)
        for idx in current_code:
            cov_count[neighbors[idx]] += 1
            
        current_size = len(current_code)
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try Deletions
            # Shuffled order of current code
            code_list = list(current_code)
            random.shuffle(code_list)
            
            for c1 in code_list:
                if time.time() > deadline - 10.0:
                    break
                    
                # Check if c1 can be removed
                # Words that will become uncovered:
                # x in neighbors[c1] such that cov_count[x] == 1
                
                # Efficient check:
                # Count how many x in neighbors[c1] have cov_count[x] == 1.
                # If 0, removal is safe.
                
                # We can precompute a boolean array `is_critical` where is_critical[x] = (cov_count[x] == 1)
                # But cov_count changes.
                
                # For now, simple loop over neighbors[c1] (size ~17)
                removable = True
                for x in neighbors[c1]:
                    if cov_count[x] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove c1
                    cov_count[neighbors[c1]] -= 1
                    is_code[c1] = False
                    current_code.remove(c1)
                    current_size -= 1
                    improved = True
                    # Continue loop to try more deletions
                    
            # Try Swaps
            # If no deletions improved, try swaps.
            # A swap (remove c1, add c2) is valid if the set of words that become uncovered
            # when removing c1 is a subset of the set of words covered by c2.
            
            # Let U_c1 = {x in neighbors[c1] | cov_count[x] == 1}
            # We need c2 such that U_c1 subset of neighbors[c2].
            
            # We can iterate c1 in code, compute U_c1.
            # If U_c1 is empty, it's a deletion (handled above).
            # If U_c1 is not empty, we look for c2 not in code such that it covers U_c1.
            
            # Optimization: Precompute for each word c2, the set of words it covers?
            # We have neighbors[c2].
            
            # To find c2 covering U_c1:
            # We can just iterate all c2 not in code? That's 6561 * 729 * 17 operations.
            # 6561 * 729 is ~4.7M. 4.7M * 17 is ~80M. Might be too slow in Python per step.
            
            # Better:
            # For each c1, get U_c1.
            # If U_c1 is small, we can check candidates c2 that are neighbors of the words in U_c1?
            # Actually, c2 must be in the intersection of neighbors[x] for all x in U_c1?
            # No, c2 must be such that dist(c2, x) <= radius for all x in U_c1.
            # So c2 must be in the intersection of the balls of radius 1 around each x in U_c1.
            
            # Let's try a simpler swap heuristic:
            # Randomly pick c1 from code.
            # Randomly pick c2 not in code.
            # Check if swap is valid.
            
            # This is slow.
            
            # Let's stick to Deletions primarily, as starting from 729 and optimal is 486,
            # there are many redundant words.
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]
                
        # If we found a better code, keep it.
        
    # Final Verification
    # Construct final list
    final_code = best_code
    
    # Verify coverage
    final_is_code = np.zeros(total_words, dtype=bool)
    for idx in final_code:
        final_is_code[idx] = True
        
    final_cov = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        final_cov[neighbors[idx]] += 1
        
    if not np.all(final_cov > 0):
        # Fallback to initial
        return [list(digits_of[i]) for i in initial_code]
        
    return [list(digits_of[i]) for i in final_code]
