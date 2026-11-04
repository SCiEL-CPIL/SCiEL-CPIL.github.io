import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # The evaluator kills at 150s. We keep a strict margin to ensure return.
    deadline = start_time + 145.0
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighborhoods) as bitmasks
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    def get_cover_mask(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask

    def is_covered(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask == all_mask

    def greedy_prune(initial_set):
        """
        Greedily remove redundant codewords.
        A codeword is redundant if its ball is entirely covered by the union of the balls of the other codewords.
        """
        code = list(initial_set)
        if not code:
            return code
            
        # To make it deterministic and potentially faster, we can sort by some heuristic?
        # For now, just iterate. To avoid O(N^2 * BitOps) being too slow, 
        # we can maintain the cover mask dynamically? 
        # Actually, checking redundancy requires: (CoverWithoutC) == AllMask?
        # CoverWithoutC = CurrentCover & ~balls[c]
        # This is O(1) bit ops per check if we have CurrentCover.
        # But CurrentCover changes as we remove.
        
        # Strategy:
        # 1. Compute initial cover mask.
        # 2. Iterate through the list. For each c, check if removing it keeps the cover valid.
        #    If yes, remove it and update the cover mask.
        #    If no, keep it.
        # 3. Repeat until no more changes.
        
        changed = True
        while changed:
            changed = False
            # Recompute cover mask at start of pass? 
            # Or maintain it? 
            # If we remove c, new_cover = old_cover & ~balls[c].
            # If we don't remove c, cover stays same.
            # So we can maintain it.
            
            # However, the "validity" of a later word might change if we removed an earlier word?
            # No, removing a word only shrinks the cover. If a word was redundant before, 
            # it might become non-redundant if we remove another word that covered part of its unique contribution?
            # Wait. If c1 is redundant, removing it doesn't change the covered set.
            # If c2 was redundant with respect to the full set, is it redundant with respect to the set minus c1?
            # Yes, because the covered set is identical.
            # So a single pass in any order is sufficient to find *a* minimal subset?
            # Not necessarily minimal size, but minimal in the sense that no single element can be removed.
            # To get a smaller set, we might need to restart or use a smarter order.
            
            # Let's use a randomized restart approach for pruning if time permits, 
            # but for now, a single pass with random shuffle is a good heuristic.
            
            random.shuffle(code)
            
            # We need the current cover mask.
            # It's expensive to recompute from scratch every pass.
            # Let's just do one pass. If it improves, we might do another.
            
            # Actually, let's just do a simple loop.
            current_cover = get_cover_mask(code)
            
            i = 0
            while i < len(code):
                if time.time() > deadline - 10.0:
                    break
                c = code[i]
                # Check if c is redundant
                if (current_cover & ~balls[c]) == all_mask:
                    # Remove c
                    code.pop(i)
                    current_cover &= ~balls[c]
                    changed = True
                    # Do NOT increment i, because the element at i is now the next one
                else:
                    i += 1
                    
            if not changed:
                break
                
        return code

    # Strategy:
    # 1. Generate a "structured" initial set using the product property.
    #    For q=3, n=8, we can view the space as 4+4.
    #    Let A be a covering code for (q, 4, 1) and B be a covering code for (q, 4, 1).
    #    The product A x B is a covering code for (q, 8, 2)? 
    #    No, the radius adds? Or max?
    #    Distance in product space is sum of distances.
    #    If we pick (a, b) with a in A, b in B.
    #    For any (x, y), there exists a in A with d(x,a)<=1 and b in B with d(y,b)<=1.
    #    Then d((x,y), (a,b)) = d(x,a) + d(y,b) <= 2.
    #    So A x B covers radius 2. We need radius 1.
    #    
    #    So direct product doesn't give radius 1 directly.
    #    
    #    Alternative: Use a "grid" based on a known small code.
    #    The best known code for K_3(8,1) is 486.
    #    K_3(4,1) is 9? (3^4 = 81, ball size 1+4*2=9. 81/9=9. Yes, perfect code exists? 
    #    Hamming code H(3, 4, 1)? 3^4 / (1 + 4*2) = 81/9 = 9. Yes.
    #    So there is a perfect code of size 9 for n=4.
    #    
    #    If we take the product of two perfect codes of size 9, we get a code of size 81.
    #    Does it cover radius 1?
    #    As established, d((x,y), (a,b)) = d(x,a) + d(y,b).
    #    If x is covered by a (d<=1) and y is covered by b (d<=1), the sum can be 2.
    #    So (a,b) might be at distance 2 from (x,y).
    #    However, if d(x,a)=0 and d(y,b)=1, distance is 1.
    #    If d(x,a)=1 and d(y,b)=0, distance is 1.
    #    If d(x,a)=1 and d(y,b)=1, distance is 2.
    #    So the product code covers all points where at least one component is a center?
    #    No, it covers points where the sum of distances is <= 1? No, it's a specific set of centers.
    #    
    #    Actually, let's look at the "Direct Sum" construction mentioned in inspirations.
    #    Maybe we can just use a dense random set and prune?
    #    The previous program used Greedy Set Cover to *build* the set.
    #    This new strategy uses a *dense* initial set (e.g., all words where sum of digits % 3 == 0?) 
    #    and then prunes.
    
    # Let's try a "Lattice" approach.
    # Select all words where sum(digits) % q == 0.
    # Size: 3^8 / 3 = 2187.
    # Is it a covering code for radius 1?
    # For any word w, can we change one digit to make the sum divisible by 3?
    # Yes. If sum(w) % 3 == r, we can change a digit d to d' such that (sum - d + d') % 3 == 0.
    # d' = (d - r) % 3. Since q=3, there is always a valid d' in {0,1,2}.
    # If r=0, we don't need to change (distance 0).
    # If r=1, we change a digit d to (d-1)%3. Distance 1.
    # If r=2, we change a digit d to (d-2)%3. Distance 1.
    # So the set of all words with sum % 3 == 0 is a covering code of radius 1!
    # Size is 3^8 / 3 = 2187.
    # This is a valid starting point. 2187 is much larger than 486, but pruning might work well.
    
    initial_set = []
    for i in range(total_words):
        s = sum(digits_of[i])
        if s % q == 0:
            initial_set.append(i)
            
    # Prune this large set
    pruned_code = greedy_prune(initial_set)
    
    # If we have time, try to improve with a few random restarts of pruning on perturbed sets?
    # Or just use the pruned result.
    # 2187 -> Pruning might get it down significantly.
    # Let's see how fast greedy_prune is.
    # 2187 items. Each check is O(1) big-int op.
    # Pass 1: 2187 checks.
    # It should be very fast (< 1s).
    
    # If we still have time, we can try a "Local Search" on the pruned set to swap words?
    # But the prompt asks for a STRUCTURAL change. The "Lattice + Prune" is a different family 
    # from "Greedy Set Cover + SA".
    
    # Let's add a second phase: Randomized Pruning with different orders?
    # Or just return the pruned lattice code.
    
    # To be safe and potentially better, let's do a few iterations of:
    # 1. Take current best.
    # 2. Randomly swap a few words with others from the lattice (or whole space)?
    # 3. Prune.
    # 4. If better, keep.
    
    best_code = pruned_code
    best_size = len(best_code)
    
    # If time permits, try to reduce further via "Swap and Prune"
    # We can pick a random word from the current code, replace it with a random word from the 
    # "Lattice" (or whole space), and prune. If the resulting size is smaller, accept.
    
    # Precompute lattice indices for fast access? 
    # We have initial_set which is the lattice.
    
    time_limit_for_search = deadline - 15.0
    
    while time.time() < time_limit_for_search:
        if len(best_code) <= 486: # Known best
            break
            
        # Create a copy
        candidate = best_code[:]
        
        # Perform a few random swaps
        num_swaps = random.randint(1, 3)
        for _ in range(num_swaps):
            if not candidate:
                break
            idx_to_swap = random.randint(0, len(candidate) - 1)
            # Pick a new word from the lattice (initial_set) or whole space?
            # Lattice is safer to maintain "structure".
            new_word = random.choice(initial_set)
            if new_word not in candidate:
                candidate[idx_to_swap] = new_word
        
        # Prune the candidate
        pruned_candidate = greedy_prune(candidate)
        
        if len(pruned_candidate) < len(best_code):
            best_code = pruned_candidate
            best_size = len(best_code)
            
    # Final cleanup is already done by greedy_prune (it runs until no change).
    
    # Convert indices back to words
    return [list(digits_of[i]) for i in best_code]
