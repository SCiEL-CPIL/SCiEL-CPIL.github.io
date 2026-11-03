import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 145.0

    total_space = q ** n

    # 1. Precompute powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # 2. Precompute neighbor masks for each word
    # masks[j] is a Python int representing the set of words covered by codeword j
    # This is used for fast gain calculation and coverage checking in SA and Greedy
    masks = [0] * total_space
    for j in range(total_space):
        m = 0
        # j covers itself
        m |= (1 << j)
        
        if radius >= 1:
            for pos in range(n):
                orig_val = (j // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    i = j + (new_val - orig_val) * powers[pos]
                    m |= (1 << i)
                    
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = (j // powers[pos1]) % q
                    v2 = (j // powers[pos2]) % q
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            i = j + (nv1 - v1) * powers[pos1] + (nv2 - v2) * powers[pos2]
                            m |= (1 << i)
                            
        masks[j] = m

    full_mask = (1 << total_space) - 1
    
    # 3. Build MILP matrix for potential exact solve or initial bound
    # We only build this if we have time, as it's expensive for large N, but N=2187 is small.
    # However, the SA approach doesn't strictly need the matrix, only the masks.
    # We will skip the full MILP matrix construction to save time for SA, 
    # unless we want to use MILP as a fallback or initializer.
    # Given the directive to try SA, we will focus on SA.
    
    # 4. Simulated Annealing
    # State: set of selected codewords (represented as a boolean array or set of indices)
    # Energy: number of uncovered words (primary) + lambda * number of selected codewords (secondary)
    # We want to minimize Energy.
    # Start with a greedy solution to have a valid cover.
    
    def greedy_cover():
        remaining = full_mask
        selected = set()
        # To speed up greedy, we can use a max-heap or just scan.
        # Scanning 2187 items is fast.
        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1:
                break
            selected.add(best_word)
            remaining &= ~masks[best_word]
        return selected, remaining

    # Run greedy to get initial solution
    initial_selected, initial_remaining = greedy_cover()
    
    # If greedy didn't cover everything, add all remaining words (shouldn't happen)
    if initial_remaining:
        # This is a safety net. In practice, greedy should cover all.
        # If it doesn't, we have a bug or time issue.
        # For now, just add everything to ensure validity.
        for i in range(total_space):
            if i not in initial_selected:
                initial_selected.add(i)
        initial_remaining = 0

    best_selected = set(initial_selected)
    best_energy = len(best_selected)
    current_selected = set(initial_selected)
    current_remaining = 0 # Greedy ensures valid cover
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    T = T_start
    steps_per_temp = 500
    
    # Precompute list of all words for random selection
    all_words = list(range(total_space))
    random.shuffle(all_words)
    
    while T > T_end and time.time() < deadline - 5:
        # Perform steps_per_temp moves
        for _ in range(steps_per_temp):
            if time.time() > deadline - 5:
                break
                
            # Choose a random move: Add, Remove, or Swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random word
                w = random.choice(all_words)
                if w in current_selected:
                    continue
                # Calculate delta energy
                # New uncovered = current_remaining & ~masks[w]
                # Delta uncovered = current_remaining.bit_count() - (current_remaining & ~masks[w]).bit_count()
                # This is equal to (current_remaining & masks[w]).bit_count()
                # Delta size = +1
                # We track current_remaining explicitly for speed? 
                # Recalculating current_remaining from scratch is O(N/word_size) which is fast for N=2187.
                # But maintaining it is better.
                
                # Let's maintain current_remaining
                # Actually, recalculating is safer and fast enough for 2187 bits.
                # 2187 bits is ~35 64-bit words. Bit operations are very fast.
                
                new_remaining = current_remaining & ~masks[w]
                delta_uncovered = current_remaining.bit_count() - new_remaining.bit_count()
                delta_size = 1
                # Energy change = delta_uncovered (negative is good) + lambda * delta_size
                # We want to minimize. 
                # Let's define Energy = uncovered_count + 0.1 * selected_count
                # delta_E = -delta_uncovered + 0.1 * delta_size
                # Wait, uncovered decreases, so delta_uncovered is positive reduction.
                # E_new = (U - delta_uncovered) + 0.1 * (S + 1)
                # delta_E = -delta_uncovered + 0.1
                delta_E = -delta_uncovered + 0.1
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_selected.add(w)
                    current_remaining = new_remaining
                    
            elif move_type < 0.6:
                # Remove a random word
                if not current_selected:
                    continue
                w = random.choice(list(current_selected))
                # Check if w is redundant
                # If we remove w, do we uncover any words?
                # A word i is covered by w if i is in masks[w].
                # i remains covered if it is in masks[x] for some x in current_selected \ {w}.
                # This is hard to check incrementally without coverage counts.
                # Simple check: 
                # new_remaining = current_remaining | (masks[w] & ~(union of others))
                # This is expensive.
                # Alternative: Just try the move and check validity?
                # If we remove w, we might uncover words.
                # Let's approximate: assume removing w uncovers (masks[w] & current_remaining) ? No.
                # Let's use a simpler heuristic for removal:
                # Only remove if w covers no unique words? 
                # To check unique coverage: 
                # unique = masks[w]
                # for x in current_selected:
                #    if x != w: unique &= ~masks[x]
                # This is O(S * N/word_size). S~200, N=2187. 200*35 = 7000 ops. Very fast.
                
                unique_to_w = masks[w]
                for x in current_selected:
                    if x != w:
                        unique_to_w &= ~masks[x]
                        
                if unique_to_w:
                    # w covers some unique words. Removing it will uncover them.
                    # delta_uncovered = unique_to_w.bit_count()
                    # delta_size = -1
                    # delta_E = delta_uncovered - 0.1
                    delta_uncovered = unique_to_w.bit_count()
                    delta_E = delta_uncovered - 0.1
                else:
                    # w is redundant
                    delta_uncovered = 0
                    delta_E = -0.1
                    
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    current_selected.remove(w)
                    if unique_to_w:
                        current_remaining |= unique_to_w
                    # If redundant, current_remaining unchanged
                    
            else:
                # Swap: Remove one, Add one
                if not current_selected:
                    continue
                w_out = random.choice(list(current_selected))
                w_in = random.choice(all_words)
                if w_in in current_selected:
                    continue
                    
                # Calculate delta
                # Remove w_out, Add w_in
                # This is complex. Let's just try it and calculate new state?
                # Or approximate delta.
                # Let's skip swap for now to keep it simple and robust. 
                # Add and Remove are sufficient for SA.
                pass

            # Update best
            if current_remaining == 0:
                current_size = len(current_selected)
                # Energy for valid cover is just size (since uncovered=0)
                # But we used energy = uncovered + 0.1 * size for moves.
                # For comparison of valid covers, size is the metric.
                if current_size < best_energy:
                    best_energy = current_size
                    best_selected = set(current_selected)
                    
        T *= cooling_rate

    # 5. Polish the best solution found
    # Try to remove any redundant words from best_selected
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for w in list(best_selected):
            # Check if w is redundant in best_selected
            unique_to_w = masks[w]
            for x in best_selected:
                if x != w:
                    unique_to_w &= ~masks[x]
            if not unique_to_w:
                best_selected.remove(w)
                improved = True
                break # Restart loop as set changed

    # 6. Final verification
    if not best_selected:
        # Fallback to greedy if SA failed completely
        best_selected, _ = greedy_cover()

    # Convert to list of lists
    best_words = []
    for idx in best_selected:
        word = [0] * n
        val = idx
        for pos in range(n):
            word[pos] = val % q
            val //= q
        best_words.append(word)
        
    # Ensure distinctness (set ensures this)
    # Ensure validity (polish ensures no redundant removals, greedy ensures coverage)
    # Double check validity just in case
    covered = 0
    for w in best_words:
        idx = 0
        val = 0
        for pos in range(n):
            val += w[pos] * powers[pos]
        covered |= masks[val]
        
    if covered != full_mask:
        # Invalid, fallback to greedy
        best_words = []
        selected, rem = greedy_cover()
        for idx in selected:
            word = [0] * n
            val = idx
            for pos in range(n):
                word[pos] = val % q
                val //= q
            best_words.append(word)
            
    return best_words
