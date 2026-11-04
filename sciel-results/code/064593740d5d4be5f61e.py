"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    For the canonical task q=3, n=6, radius=1, we aim for ~73 words.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _greedy_cover(q, n, radius, deadline)

    # Precompute all words as integers 0..728
    all_ints = list(range(3 ** n))
    
    def word_to_int(w):
        val = 0
        for x in w:
            val = val * 3 + x
        return val

    def int_to_word(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % 3
            val //= 3
        return w

    # Precompute neighbor lists (balls of radius 1) for all 729 words
    neighbor_lists = []
    for i in all_ints:
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
            w[pos] = orig
        neighbor_lists.append(ball)

    # Precompute the tetracode (perfect code of length 4, radius 1)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_tails = list(itertools.product(range(3), repeat=n - 4))
    
    # Initial code: direct product of tetracode and all tails
    initial_code = [h + list(t) for h in tetracode for t in base_tails]
    
    # Convert to integer representation for speed
    initial_ints = [word_to_int(c) for c in initial_code]
    
    # counts[i] = number of codewords covering word i
    counts = np.zeros(3 ** n, dtype=np.int32)
    
    for ci in initial_ints:
        for nb in neighbor_lists[ci]:
            counts[nb] += 1

    active_ints = list(initial_ints)
    
    # Helper to check if a codeword can be removed
    def can_remove(ci):
        ball = neighbor_lists[ci]
        # Need counts[nb] >= 2 for all nb in ball
        for nb in ball:
            if counts[nb] < 2:
                return False
        return True

    # Greedy removal phase
    def greedy_remove(active_ints, counts, neighbor_lists, deadline):
        removed = 0
        while time.time() < deadline - 10.0:
            found = False
            # Try to find a removable codeword
            # Randomize order to escape local minima? Or just sequential.
            # Let's try sequential first, it's faster.
            for idx in range(len(active_ints)):
                ci = active_ints[idx]
                if can_remove(ci):
                    # Remove it
                    ball = neighbor_lists[ci]
                    for nb in ball:
                        counts[nb] -= 1
                    active_ints.pop(idx)
                    removed += 1
                    found = True
                    break
            if not found:
                break
        return active_ints, counts, removed

    active_ints, counts, _ = greedy_remove(active_ints, counts, neighbor_lists, deadline)
    
    # Local search: try to swap codewords to enable further removals
    # A swap: remove c1, add c2.
    # Valid if:
    # 1. c1 is "almost" removable, but some neighbors have count 1.
    # 2. Adding c2 covers those neighbors (and doesn't break anything else, i.e., counts don't go negative, which is impossible since we add).
    # Actually, the condition is:
    # Let D = {w in Ball(c1) | counts[w] == 1}.
    # If D is empty, c1 is removable (handled above).
    # If D is not empty, we need to add c2 such that D subset Ball(c2).
    # Also, we must ensure that after removing c1 and adding c2, all words are still covered.
    # Since c1 covered Ball(c1), and c2 covers Ball(c2).
    # Words in Ball(c1) \ Ball(c2) lose c1's coverage. If any had count 1, they become 0.
    # So we need D subset Ball(c2).
    # Words outside Ball(c1) gain c2's coverage, so they are safe (count increases).
    # So the condition is simply: D subset Ball(c2).
    
    # We can try this for a limited number of steps.
    
    def local_search_swap(active_ints, counts, neighbor_lists, deadline):
        # Precompute set of active codewords for fast lookup
        active_set = set(active_ints)
        
        # For each active codeword, compute its "deficit" set D (neighbors with count 1)
        # We want to find c1, c2 such that D(c1) subset Ball(c2).
        
        # To speed up, we can iterate over c1 in active_ints.
        # For each c1, compute D.
        # If D is empty, remove c1 (should have been handled, but maybe counts changed).
        # If not, look for c2 in all_ints (or active_ints? No, c2 can be any word, even not in code, but usually we swap within code or add new. 
        # Wait, the problem is minimizing size. So we remove one and add one? No, that keeps size same.
        # We want to reduce size. So we remove c1 and DON'T add c2? That's the greedy removal.
        # The swap is: Remove c1, Add c2. Size remains same. But it might enable a subsequent removal.
        # So we do a "swap" to change the configuration, then try greedy removal again.
        
        # Let's implement a "move" that swaps c1 with c2, then tries to remove something.
        # If the swap + removal leads to a smaller code, accept.
        # If the swap leads to a same size code but different config, accept if it allows future removals (hard to know).
        # Let's just do: Try all swaps (c1, c2) where c1 is in code, c2 is in code (or all space?).
        # Swapping with all space is expensive. Let's restrict c2 to active_ints first.
        # Actually, adding a new word c2 not in code is also a valid move (increase size, then remove more?). No, we want to minimize.
        # So we only consider moves that don't increase size.
        # Move 1: Remove c1 (size -1). Handled by greedy.
        # Move 2: Swap c1 for c2 (size 0).
        
        # Let's try a few random swaps to see if they unlock removals.
        
        best_active = list(active_ints)
        best_counts = counts.copy()
        
        # Limit number of swap attempts
        max_swaps = 5000
        swap_count = 0
        
        while time.time() < deadline - 5.0 and swap_count < max_swaps:
            # Pick random c1 from active
            if not active_ints:
                break
            c1 = random.choice(active_ints)
            
            # Compute D(c1)
            D = []
            for nb in neighbor_lists[c1]:
                if counts[nb] == 1:
                    D.append(nb)
            
            if not D:
                # Can remove c1
                for nb in neighbor_lists[c1]:
                    counts[nb] -= 1
                active_ints.remove(c1)
                continue
            
            # Need c2 such that D subset Ball(c2)
            # c2 can be any word in the space.
            # Let's pick a random c2 from the space? Or from active?
            # If c2 is in active, it's a swap. If not, it's a replace.
            # Let's try c2 from active first (cheaper to check? No, same cost).
            # Actually, we can just pick a random c2 from all_ints.
            c2 = random.choice(all_ints)
            
            # Check if D subset Ball(c2)
            ball_c2 = set(neighbor_lists[c2])
            if all(nb in ball_c2 for nb in D):
                # Perform swap: remove c1, add c2
                # Update counts
                for nb in neighbor_lists[c1]:
                    counts[nb] -= 1
                for nb in neighbor_lists[c2]:
                    counts[nb] += 1
                
                # Update active list
                active_ints.remove(c1)
                if c2 not in active_ints:
                    active_ints.append(c2)
                
                swap_count += 1
                
                # Try greedy removal again
                active_ints, counts, _ = greedy_remove(active_ints, counts, neighbor_lists, deadline)
                
                # If we found a smaller code, great.
                # If not, we might want to revert if this swap didn't help?
                # For now, we keep the state. It's a random walk.
                
                # To avoid getting stuck in a bad configuration, we could track best size.
                # But greedy_remove already minimizes locally.
        
        return active_ints, counts

    active_ints, counts = local_search_swap(active_ints, counts, neighbor_lists, deadline)
    
    # Final greedy removal pass
    active_ints, counts, _ = greedy_remove(active_ints, counts, neighbor_lists, deadline)
    
    # Convert back to list of lists
    result = [int_to_word(ci) for ci in active_ints]
    
    # Verify validity (optional, but good for debugging)
    # For production, we trust the counts.
    
    return result


def _greedy_cover(q, n, radius, deadline):
    """Simple greedy fallback for non-canonical parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    while uncovered and time.time() < deadline:
        # Pick a word that covers the most uncovered words
        best_word = None
        best_count = -1
        for w in uncovered:
            count = 0
            for u in uncovered:
                if sum(1 for i in range(n) if u[i] != w[i]) <= radius:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(list(best_word))
        # Remove covered words
        to_remove = []
        for u in uncovered:
            if sum(1 for i in range(n) if u[i] != best_word[i]) <= radius:
                to_remove.append(u)
        for u in to_remove:
            uncovered.remove(u)
            
    return code
# EVOLVE-BLOCK-END
