import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    total_words = q ** n
    
    # Optimization for the specific known case q=3, n=6, radius=1
    # Using bitmask representation for covers.
    # With 729 words, we need 729 bits. Python integers handle this natively and efficiently.
    
    def encode(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def decode(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute covers
    # covers[c] is a bitmask of words covered by codeword c (distance <= radius)
    # We generate the mask by iterating over all words and checking distance.
    # For n=6, q=3, total_words=729. 
    # Distance calculation: sum(1 for a,b in zip(w1, w2) if a!=b)
    # To speed up cover precomputation, we can precompute all words as tuples.
    
    all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Precompute covers using bitmasks
    covers = [0] * total_words
    
    # Optimization: For radius 1, we can construct the mask faster.
    # But general radius is required by interface.
    # Let's stick to the general method but optimized with numpy if possible?
    # Actually, pure python loop for 729*729 is ~500k iterations.
    # Each iteration is a zip and sum. This is fast enough (< 0.5s).
    
    for c_idx, c_word in enumerate(all_words):
        mask = 0
        # We can iterate over v_idx and check distance
        for v_idx, v_word in enumerate(all_words):
            # Calculate Hamming distance
            dist = 0
            for i in range(n):
                if c_word[i] != v_word[i]:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                mask |= (1 << v_idx)
        covers[c_idx] = mask

    # Greedy Set Cover
    # Initialize uncovered mask
    uncovered = (1 << total_words) - 1
    solution = []
    
    # To speed up greedy, we can pre-sort candidates or just iterate.
    # Iterating 729 items is cheap.
    # We need to pick c that maximizes popcount(covers[c] & uncovered).
    
    # Helper to count bits
    # bin(x).count('1') is okay, but int.bit_count() is faster in Py3.10+
    # Assuming Py3.8+, we might need a fallback or just use bin.
    # Let's use a helper.
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy Loop
    while uncovered:
        best_c = -1
        best_gain = 0
        # Iterate through all possible codewords
        for c in range(total_words):
            if c in solution: # Skip already chosen (optimization, but solution is list)
                continue
            gain = popcount(covers[c] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_c = c
                if gain == popcount(uncovered):
                    break
        if best_c == -1:
            break
        solution.append(best_c)
        uncovered &= ~covers[best_c]
        
    # Local Search: 1-swap and removal
    sol_set = set(solution)
    
    # Precompute union of covers for fast redundancy check?
    # Union changes with every swap.
    
    def is_redundant(c, current_set):
        # Check if covers[c] is subset of union of covers of others
        union_mask = 0
        for other in current_set:
            if other != c:
                union_mask |= covers[other]
        return (covers[c] & ~union_mask) == 0
        
    def try_reduce():
        # Try to remove any redundant codeword
        for c in list(sol_set):
            if is_redundant(c, sol_set):
                sol_set.remove(c)
                return True
        return False
        
    def try_swap_and_reduce():
        # Try to swap one codeword for another to enable a removal
        # Iterate over c_old in sol_set
        for c_old in list(sol_set):
            # Calculate words only covered by c_old
            union_rest = 0
            for c in sol_set:
                if c != c_old:
                    union_rest |= covers[c]
            lost = covers[c_old] & ~union_rest
            
            if lost == 0:
                # Redundant, handled by try_reduce
                continue
                
            # Find c_new not in sol_set such that lost is subset of covers[c_new]
            # To speed up, we can precompute for each 'lost' mask, which c_new covers it?
            # 'lost' varies. 
            # Just iterate c_new.
            for c_new in range(total_words):
                if c_new in sol_set:
                    continue
                if (lost & ~covers[c_new]) == 0:
                    # Valid swap candidate
                    # Perform swap
                    new_set = sol_set.copy()
                    new_set.remove(c_old)
                    new_set.add(c_new)
                    
                    # Check if any codeword in new_set is now redundant
                    # We only need to check codewords that might have become redundant.
                    # Actually, adding c_new can only make others MORE redundant, not less.
                    # Removing c_old can only make others LESS redundant (or same).
                    # So we only need to check if any c in new_set is redundant.
                    # Specifically, c_new might make some old c redundant.
                    # And since we removed c_old, some c might have been relying on c_old? No, c_old was covering 'lost' which c_new now covers.
                    # So coverage is maintained.
                    # We check all c in new_set for redundancy.
                    
                    found_redundant = False
                    for c in new_set:
                        if is_redundant(c, new_set):
                            # Accept swap and removal
                            sol_set = new_set
                            sol_set.remove(c)
                            found_redundant = True
                            break
                            
                    if found_redundant:
                        return True
                            
        return False

    # Main optimization loop
    # We run until no improvements or deadline.
    # To avoid getting stuck in a local minimum where 1-swap doesn't work,
    # we might need 2-swaps or randomized restarts.
    # However, for K_3(6,1), 1-swap is often sufficient to reach 73 from a good greedy start.
    # Let's add a randomized component if stuck.
    
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline:
        improved = False
        
        if try_reduce():
            improved = True
            no_improve_count = 0
            continue
            
        if try_swap_and_reduce():
            improved = True
            no_improve_count = 0
            continue
            
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Try a randomized swap to escape local minimum
                # Pick a random c_old and a random c_new that covers 'lost'
                # Or just break if we are confident.
                # Let's try one random swap attempt.
                random_c_old = random.choice(list(sol_set))
                union_rest = 0
                for c in sol_set:
                    if c != random_c_old:
                        union_rest |= covers[c]
                lost = covers[random_c_old] & ~union_rest
                
                if lost:
                    # Find random c_new covering lost
                    candidates = [c for c in range(total_words) if c not in sol_set and (lost & ~covers[c]) == 0]
                    if candidates:
                        random_c_new = random.choice(candidates)
                        new_set = sol_set.copy()
                        new_set.remove(random_c_old)
                        new_set.add(random_c_new)
                        
                        # Check for redundancy
                        for c in new_set:
                            if is_redundant(c, new_set):
                                sol_set = new_set
                                sol_set.remove(c)
                                no_improve_count = 0
                                improved = True
                                break
                        
                if not improved:
                    # If still stuck, break to save time, assuming we are at a local optimum
                    # For this specific problem, greedy + 1-swap usually gets to 73 or 74.
                    # If we are at 74, we might not reach 73 without deeper search.
                    # But 73 is the target.
                    break
                else:
                    continue
            
    return [decode(c) for c in sorted(sol_set)]
