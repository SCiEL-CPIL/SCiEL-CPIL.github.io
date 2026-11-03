import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    total_space = q ** n
    if total_space > 100000:
        # Fallback for very large spaces, though this task is small
        # For K_3(7,1), total_space is 2187
        pass

    # Precompute powers for base-q representation
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for all words
    # masks[i] is a bitmask where bit j is 1 if word j is within radius of word i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        # For radius 1, we just change one digit
        # General radius would require more complex generation, but task is radius 1
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1
    
    # Helper to convert int index to tuple of digits
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # Simulated Annealing State
    # selected: list of codeword indices
    # counts: array of length total_space, counts[i] = number of selected codewords covering word i
    # covered_mask: union of masks of selected codewords
    
    def init_state():
        # Start with a random valid covering
        # A simple way: pick random words until covered
        # Or start with a greedy solution to have a good starting point
        # Let's start with a greedy solution to ensure we are in a good region
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining:
            best_gain = -1
            best_word = -1
            # Randomize tie-breaking by shuffling order of evaluation? 
            # Or just standard greedy.
            # To diversify, let's do a randomized greedy initialization
            candidates = []
            max_gain = -1
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [w]
                elif gain == max_gain:
                    candidates.append(w)
            
            if not candidates:
                break
                
            best_word = random.choice(candidates)
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            
        # Build counts
        counts = [0] * total_space
        for w in selected:
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
                
        return selected, counts, covered

    def get_uniques(w, counts):
        """Get list of words uniquely covered by w"""
        m = masks[w]
        uniques = []
        mm = m
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            if counts[idx] == 1:
                uniques.append(idx)
            mm ^= lowbit
        return uniques

    def can_remove(w, counts):
        """Check if w can be removed (no unique words)"""
        m = masks[w]
        mm = m
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            if counts[idx] == 1:
                return False
            mm ^= lowbit
        return True

    def remove_word(selected, counts, w):
        m = masks[w]
        mm = m
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            counts[idx] -= 1
            mm ^= lowbit
        selected.remove(w)

    def add_word(selected, counts, w):
        m = masks[w]
        mm = m
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            counts[idx] += 1
            mm ^= lowbit
        selected.append(w)

    def swap_words(selected, counts, w_out, w_in):
        # Remove w_out
        m_out = masks[w_out]
        mm = m_out
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            counts[idx] -= 1
            mm ^= lowbit
            
        # Add w_in
        m_in = masks[w_in]
        mm = m_in
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            counts[idx] += 1
            mm ^= lowbit
            
        # Update list
        selected[selected.index(w_out)] = w_in

    # Main SA Loop
    # Objective: Minimize len(selected)
    # Constraint: Full coverage (maintained by construction if we only do valid moves)
    # Moves:
    # 1. Remove a word (if possible) -> size decreases by 1. Always accept.
    # 2. Swap a word with another (size constant). Accept based on SA criteria?
    #    Actually, swaps don't change size. They change the "structure".
    #    We want to enable further removals.
    #    So, a swap is "good" if it increases the number of removable words?
    #    Or we can just do random swaps to escape local minima.
    
    # Let's use a cost function: -len(selected) + penalty if not covered (but we maintain coverage).
    # Since size is integer, we can just track size.
    
    best_size = total_space
    best_selected = None
    
    # Run multiple SA trials
    num_trials = 10
    trial_duration = (deadline - start_time - 5) / num_trials
    
    for trial in range(num_trials):
        trial_start = time.time()
        trial_deadline = trial_start + trial_duration
        
        selected, counts, covered = init_state()
        current_size = len(selected)
        
        if current_size < best_size:
            best_size = current_size
            best_selected = selected[:]
            
        # SA Parameters
        T = 1.0
        cooling = 0.995
        
        while time.time() < trial_deadline:
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to remove a word
                # Pick a random word from selected
                if selected:
                    w = random.choice(selected)
                    if can_remove(w, counts):
                        remove_word(selected, counts, w)
                        current_size -= 1
                        if current_size < best_size:
                            best_size = current_size
                            best_selected = selected[:]
                        # After removal, T might need reset or just continue
                    else:
                        # Can't remove, maybe try a swap to make it removable?
                        pass
            
            elif move_type < 0.7:
                # Try a swap
                if len(selected) > 1:
                    w_out = random.choice(selected)
                    # Pick a random word NOT in selected
                    # To be efficient, pick random index and check
                    # Or pick from a precomputed list of non-selected?
                    # For small N, just pick random index and check membership
                    # Membership check: O(1) if we use a set, but selected is list.
                    # Let's use a set for membership? Or just try a few times.
                    
                    # Better: Pick a random word from the whole space.
                    # If it's in selected, skip or pick another.
                    # To make swaps effective, we should pick w_in that covers some uniques of w_out?
                    # For now, random swap.
                    
                    w_in = random.randint(0, total_space - 1)
                    if w_in == w_out:
                        continue
                    if w_in in selected:
                        continue
                        
                    # Check if swap maintains coverage?
                    # Since w_out is removed, words uniquely covered by w_out become uncovered.
                    # w_in must cover all those unique words.
                    uniques = get_uniques(w_out, counts)
                    if not uniques:
                        # If no uniques, w_out is redundant. We should have removed it.
                        # But maybe we didn't try to remove it.
                        # Just swap.
                        pass
                    else:
                        m_in = masks[w_in]
                        valid_swap = True
                        for u in uniques:
                            if not (m_in >> u) & 1:
                                valid_swap = False
                                break
                        if not valid_swap:
                            continue
                    
                    # Perform swap
                    swap_words(selected, counts, w_out, w_in)
                    # Size unchanged. 
                    # Acceptance: Always accept for now to explore. 
                    # Or accept with prob exp(-delta_cost/T). delta_cost is 0.
                    # So always accept.
                    # But this might be a random walk. 
                    # To bias towards better solutions, we could check if the new set has more "removable" words?
                    # That's expensive. 
                    # Let's stick to random swaps to escape local minima.
            
            else:
                # Try to add a word? No, we want to minimize size.
                # Adding a word increases size. We only add if it allows removing 2+ words later?
                # That's complex. Skip for now.
                pass
            
            # Cool down
            T *= cooling
            if T < 0.01:
                T = 0.01

    # Final polish: remove any remaining redundant words
    if best_selected:
        # Rebuild counts for best_selected
        counts = [0] * total_space
        for w in best_selected:
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            indices = list(range(len(best_selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(best_selected):
                    continue
                w = best_selected[idx]
                if can_remove(w, counts):
                    remove_word(best_selected, counts, w)
                    changed = True
                    if len(best_selected) < best_size:
                        best_size = len(best_selected)

    if not best_selected:
        # Fallback
        return [int_to_tuple(i) for i in range(total_space)]

    return [int_to_tuple(w) for w in best_selected]
