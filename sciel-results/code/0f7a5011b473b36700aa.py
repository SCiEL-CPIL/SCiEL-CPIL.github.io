import time
import sys
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    total_space = q ** n
    
    # 1. Precompute neighbors for all words
    # For q=3, n=8, radius=1, each word has 1 + n*(q-1) = 17 neighbors
    # We represent the set of covered words as a Python set of integers for fast membership and update,
    # but for the greedy phase, we might use bitsets or just sets.
    # Given total_space = 6561, a Python set of ints is very fast.
    # However, for the annealing phase, we need to check if removing a codeword breaks coverage.
    # A better structure: 
    # coverage_count[word] = number of codewords covering 'word'
    # This allows O(1) check for "can I remove codeword c?" -> min coverage in c's neighborhood > 1.
    
    # Let's precompute neighbors as lists of indices.
    neighbors = [[] for _ in range(total_space)]
    
    # Helper to convert int to base-q digits
    def get_digits(num):
        digits = [0] * n
        for i in range(n):
            digits[i] = num % q
            num //= q
        return digits

    # Precompute powers of q
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q

    for i in range(total_space):
        digits = get_digits(i)
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = digits[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                delta = (val - orig_val) * q_powers[pos]
                new_index = i + delta
                neighbors[i].append(new_index)
    
    # 2. Greedy Phase: Build a feasible cover
    # We want to start with a reasonably small cover.
    # Standard greedy: pick word covering most uncovered, remove uncovered.
    
    # To make greedy faster and potentially better, we can use a priority queue or just simple max.
    # With 6561 words, simple max is fine.
    
    uncovered = set(range(total_space))
    code = []
    
    # Optimization: Precompute neighbor sets as frozensets for faster intersection?
    # Or just iterate.
    
    while uncovered:
        best_word = -1
        max_cover = 0
        
        # We can skip words that are already in the code? 
        # Actually, if a word is in the code, its neighbors are already covered, so its marginal gain is 0.
        # So we only need to consider words NOT in the code? 
        # Wait, if a word is in the code, it's already selected. We don't select it again.
        # So we iterate over all words not in code? 
        # Or just all words, and if it's in code, cover_count is 0 (since its neighbors are removed from uncovered).
        
        for w in range(total_space):
            # Count how many of w's neighbors are in uncovered
            # Since uncovered is a set, we can do:
            # cnt = sum(1 for nb in neighbors[w] if nb in uncovered)
            # This is O(17) per word. 6561 * 17 ~ 111k ops per iteration.
            # Number of iterations ~ 400-500. Total ~ 50M ops. Might be too slow in Python.
            
            # Faster: Use a boolean array or bitset?
            # Let's use a numpy boolean array for uncovered.
            # But updating it and counting intersections might be faster with numpy.
            
            pass
        
        # Let's switch to numpy for the greedy phase for speed.
        # uncovered_mask: boolean array of size total_space
        # neighbor_matrix: boolean matrix? 6561x6561 is ~43M bools. 43MB. Acceptable.
        # But constructing it takes time.
        
        # Alternative: Keep the set approach but optimize the inner loop.
        # Actually, 111k ops * 500 iterations = 55.5M simple operations. 
        # In Python, 50M ops can take ~5-10 seconds. This is risky.
        
        # Let's use the numpy matrix approach. It's a one-time cost.
        
        # Break out of the slow python loop and do numpy.
        break
        
    # Re-initialize with numpy
    uncovered_mask = np.ones(total_space, dtype=bool)
    code_set = set()
    code_list = []
    
    # Precompute neighbor indices as a list of arrays for fast indexing
    # Or just use the list of lists.
    
    # To speed up greedy, we can precompute the neighbor lists as numpy arrays.
    neigh_arrays = [np.array(neighbors[i]) for i in range(total_space)]
    
    while uncovered_mask.any():
        best_word = -1
        max_cover = 0
        
        # Iterate through all words
        # We can vectorize this? 
        # For each word w, count = np.sum(uncovered_mask[neigh_arrays[w]])
        # Doing this for all w is a loop of 6561 numpy sums.
        # Each sum is over 17 elements. 
        # This is likely faster than pure Python set iteration.
        
        for w in range(total_space):
            if w in code_set:
                continue
            cnt = np.sum(uncovered_mask[neigh_arrays[w]])
            if cnt > max_cover:
                max_cover = cnt
                best_word = w
                
        if best_word == -1:
            break # Should not happen
            
        code_list.append(best_word)
        code_set.add(best_word)
        uncovered_mask[neigh_arrays[best_word]] = False
        
        if time.time() > deadline:
            break
    
    if not code_list:
        return []
        
    # 3. Annealing Phase
    # State: code_set, code_list
    # Objective: Minimize len(code_list)
    # Move: 
    #   1. Try to remove a codeword c from code_set. 
    #      If valid (all neighbors of c are covered by other codewords), accept.
    #   2. If removal fails, try a swap: Remove c, add c'. 
    #      Then try to remove another codeword c''? 
    #      Actually, a standard "swap" in covering codes is:
    #      Pick c in code. Pick c' not in code.
    #      New code = (code - {c}) U {c'}
    #      This keeps size same. 
    #      To reduce size, we need a "remove" move.
    #      But if no single removal is possible, we might need a "swap" to enable a removal.
      #      Move: Swap c with c'. Then check if any codeword c'' (possibly c') can be removed.
      #      If so, do the swap AND the removal. Net change: -1 codeword.
      #      If not, just do the swap (size unchanged) if it improves some heuristic? 
      #      Usually, we only accept swaps if they lead to a removal eventually, or if we use a penalty.
      #      
      #      Simpler strategy:
      #      1. Try all single removals. If any works, do it.
      #      2. If no single removal works, try random swaps.
      #         For a swap (c, c'), check if the new code is valid.
      #         If valid, accept with probability based on temperature? 
      #         But we want to reduce size. Swaps don't reduce size.
      #         However, swaps can change the structure to allow a future removal.
      #         
      #      Let's implement:
      #      - Always try removals first.
      #      - If no removal possible, perform a random valid swap (size unchanged).
      #      - Temperature cools down.
      
    # Data structures for fast coverage checking
    # coverage_count[word] = number of codewords in code covering 'word'
    
    def build_coverage(code_set):
        cov = np.zeros(total_space, dtype=int)
        for c in code_set:
            cov[neigh_arrays[c]] += 1
        return cov

    coverage = build_coverage(code_set)
    
    # Helper to check if codeword c can be removed
    # c can be removed if min(coverage[neighbors[c]]) > 1
    # i.e., every neighbor of c is covered by at least one other codeword.
    
    def can_remove(c):
        return np.min(coverage[neigh_arrays[c]]) > 1

    # Main Annealing Loop
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # List of codewords for random access
    # We maintain code_list and code_set.
    
    while T > min_T and time.time() < deadline:
        # 1. Try to remove
        # Pick a random codeword to check for removal? 
        # Or check all? Checking all is O(N * 17). 
        # If we check all, we can find the "easiest" to remove? 
        # Actually, just picking a random one is standard for SA.
        # But if we pick one that can't be removed, we wasted time.
        # Let's pick a random one.
        
        c = random.choice(code_list)
        
        if can_remove(c):
            # Remove it
            code_list.remove(c)
            code_set.remove(c)
            coverage[neigh_arrays[c]] -= 1
            # Reset temperature? Or keep cooling?
            # Successful removal is a big win. Maybe reset T to encourage more search?
            # Or just let it cool.
            continue
            
        # 2. No removal possible for this c. Try a swap.
        # Pick a random c' not in code.
        # To pick a random non-code word efficiently:
        # If code is small (~400) vs total (6561), random pick is likely not in code.
        c_prime = random.randint(0, total_space - 1)
        if c_prime in code_set:
            # Retry or just skip?
            # Let's just skip this iteration.
            T *= cooling
            continue
            
        # Perform swap: remove c, add c_prime
        # Check if new code is valid.
        # New coverage = coverage - neighbors[c] + neighbors[c_prime]
        # We need to ensure no word has coverage 0.
        
        # Efficient check:
        # Let new_cov = coverage.copy()
        # new_cov[neigh_arrays[c]] -= 1
        # new_cov[neigh_arrays[c_prime]] += 1
        # if np.min(new_cov) >= 1: valid
        
        # To avoid copying the whole array every time (6561 ints), 
        # we can check the specific words affected.
        # Words affected are neighbors[c] and neighbors[c_prime].
        # For words in neighbors[c] \ neighbors[c_prime]: coverage decreases by 1.
        # For words in neighbors[c_prime] \ neighbors[c]: coverage increases by 1.
        # For words in intersection: no change.
        
        # We only need to check if any word in neighbors[c] \ neighbors[c_prime] has coverage 1.
        # Because those are the ones that might drop to 0.
        
        # Let A = set(neighbors[c])
        # Let B = set(neighbors[c_prime])
        # Critical set = A - B
        # If min(coverage[w] for w in Critical set) > 1, then swap is valid.
        
        # Implementation:
        # We have neigh_arrays[c] and neigh_arrays[c_prime] as numpy arrays.
        # We can use a boolean mask or set difference.
        # Given small size (17), sets are fine.
        
        set_c = set(neigh_arrays[c])
        set_cp = set(neigh_arrays[c_prime])
        critical = set_c - set_cp
        
        valid_swap = True
        for w in critical:
            if coverage[w] == 1:
                valid_swap = False
                break
        
        if valid_swap:
            # Accept swap
            # Update structures
            code_list.remove(c)
            code_set.remove(c)
            code_list.append(c_prime)
            code_set.add(c_prime)
            
            coverage[neigh_arrays[c]] -= 1
            coverage[neigh_arrays[c_prime]] += 1
            
            # After swap, we might enable a removal.
            # We can immediately try to remove c_prime or another word.
            # Let's try to remove c_prime first.
            if can_remove(c_prime):
                code_list.remove(c_prime)
                code_set.remove(c_prime)
                coverage[neigh_arrays[c_prime]] -= 1
            else:
                # Try removing a random other word?
                # Or just let the next iteration handle it.
                pass
        else:
            # Swap invalid. Reject.
            pass
            
        T *= cooling
        
    # Final conversion
    result = []
    for idx in code_list:
        digits = get_digits(idx)
        result.append(digits)
        
    return result
