"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius radius.
    Uses bitmask-based greedy construction and local search (removal/exchange).
    """
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin for 120s limit
    
    # Handle trivial cases
    if radius == 0:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    if n == 0:
        return [[]]
    
    # Special case: ternary tetracode construction for q=3, n=4, radius=1
    if q == 3 and n == 4 and radius == 1:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    # Precompute balls as bitmasks
    # ball_mask[i] is an integer where bit j is 1 if word j is within radius of word i
    ball_masks = [0] * num_words
    for i, w in enumerate(all_words):
        mask = 0
        for r in range(radius + 1):
            for positions in itertools.combinations(range(n), r):
                if r == 0:
                    mask |= (1 << word_to_idx[w])
                else:
                    for deltas in itertools.product(range(1, q), repeat=r):
                        new_word = list(w)
                        for pos, delta in zip(positions, deltas):
                            new_word[pos] = (new_word[pos] + delta) % q
                        idx = word_to_idx[tuple(new_word)]
                        mask |= (1 << idx)
        ball_masks[i] = mask
    
    full_mask = (1 << num_words) - 1
    
    # 1. Greedy Construction
    uncovered = full_mask
    code_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = -1
        # Iterate through all possible codewords
        # Optimization: We can skip words that are already covered? No, we want the one covering most *uncovered*.
        # We can pre-filter candidates? For now, simple loop.
        for i in range(num_words):
            count = bin(ball_masks[i] & uncovered).count("1")
            if count > best_count:
                best_count = count
                best_idx = i
                if best_count == bin(uncovered).count("1"):
                    break # Perfect cover found for this step
        
        if best_idx == -1 or best_count == 0:
            break
            
        code_indices.append(best_idx)
        uncovered &= ~ball_masks[best_idx]
        
    # Check if greedy finished
    if uncovered != 0:
        # Fallback: add random words until covered (should not happen with greedy)
        while uncovered:
            i = random.randint(0, num_words - 1)
            code_indices.append(i)
            uncovered &= ~ball_masks[i]
            if time.time() > deadline:
                break

    best_code = list(code_indices)
    best_size = len(best_code)
    
    # 2. Local Search: Removal and Exchange
    # We maintain a set of active indices
    active_set = set(best_code)
    
    # Helper to calculate union mask of active set
    def get_union_mask(active_list):
        union = 0
        for i in active_list:
            union |= ball_masks[i]
        return union
    
    current_union = get_union_mask(best_code)
    
    rng = random.Random(42)
    
    # Removal Phase
    while time.time() < deadline:
        removed_any = False
        # Try to remove each codeword
        # To make it faster, we can shuffle the order
        indices_to_try = list(best_code)
        rng.shuffle(indices_to_try)
        
        for idx in indices_to_try:
            if idx not in active_set:
                continue
            
            # Calculate union without this idx
            new_union = 0
            for j in best_code:
                if j != idx:
                    new_union |= ball_masks[j]
            
            if new_union == full_mask:
                # Safe to remove
                best_code.remove(idx)
                active_set.remove(idx)
                current_union = new_union
                removed_any = True
                if len(best_code) < best_size:
                    best_size = len(best_code)
        
        if not removed_any:
            break # No more removals possible
    
    # Exchange Phase
    # Try to replace one codeword with another to potentially allow further removals or just improve structure
    # A simple exchange: remove i, add j, check if coverage is maintained.
    # This alone doesn't reduce size, but might enable future removals.
    # Better: Try to replace i with j such that the new set has smaller "waste" or enables removal.
    # Given time constraints, we stick to aggressive removal and random restarts if time permits.
    
    # Random Restart / Perturbation if time remains
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Start from best_code
            current_code = list(best_code)
            current_union = get_union_mask(current_code)
            
            # Randomly remove a few and try to re-add optimally?
            # Or just run removal again on a perturbed version?
            
            # Let's try a simple local search: random exchange
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                if len(current_code) <= 1:
                    break
                
                # Pick a random codeword to remove
                rem_idx = rng.choice(current_code)
                # Pick a random word to add
                add_idx = rng.randint(0, num_words - 1)
                
                if add_idx in current_code:
                    continue
                
                # Check if (current_code - rem_idx + add_idx) covers everything
                # Efficient check:
                # Union of (current_code - rem_idx) is current_union ^ ball_masks[rem_idx] ? No, intersection matters.
                # We need union of (current_code - {rem_idx}) | ball_masks[add_idx] == full_mask
                
                # Calculate union of current_code without rem_idx
                union_without_rem = 0
                for j in current_code:
                    if j != rem_idx:
                        union_without_rem |= ball_masks[j]
                
                new_union = union_without_rem | ball_masks[add_idx]
                
                if new_union == full_mask:
                    # Valid exchange
                    current_code.remove(rem_idx)
                    current_code.append(add_idx)
                    current_union = new_union
            
            # Try removals on the new current_code
            made_progress = True
            while made_progress:
                made_progress = False
                indices_to_try = list(current_code)
                rng.shuffle(indices_to_try)
                for idx in indices_to_try:
                    new_union = 0
                    for j in current_code:
                        if j != idx:
                            new_union |= ball_masks[j]
                    if new_union == full_mask:
                        current_code.remove(idx)
                        current_union = new_union
                        made_progress = True
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
            
    return [list(all_words[i]) for i in best_code]

# EVOLVE-BLOCK-END
