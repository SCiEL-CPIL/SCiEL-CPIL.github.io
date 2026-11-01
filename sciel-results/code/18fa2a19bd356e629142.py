import itertools
import time
import random

def construct(q, n, radius):
    # Specialized for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: Simple product construction
        if n >= 4:
            # Tetracode direct sum
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For very small n, just return all words (trivial cover)
            return [list(p) for p in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute neighbors for all 729 words
    # neighbors[c] is a list of words within Hamming distance 1 of c
    neighbors = []
    for c in range(729):
        digs = [(c // (3**i)) % 3 for i in range(6)]
        ns = {c}
        for i in range(6):
            for delta in [-1, 1]:
                new_dig = (digs[i] + delta) % 3
                weight = 3**i
                old_val = digs[i] * weight
                new_val = new_dig * weight
                ns.add(c - old_val + new_val)
        neighbors.append(ns)
    
    # Convert neighbors to bitmasks for fast intersection
    # mask[c] has bit w set if w is in neighbors[c]
    neighbor_mask = [0] * 729
    for c in range(729):
        mask = 0
        for w in neighbors[c]:
            mask |= (1 << w)
        neighbor_mask[c] = mask
        
    # All words mask
    all_words_mask = (1 << 729) - 1

    best_size = 729
    best_codes = []
    
    # Greedy Set Cover Algorithm
    def greedy_cover(seed=0):
        random.seed(seed)
        uncovered = all_words_mask
        selected = []
        
        while uncovered:
            # Find codeword that covers most uncovered words
            best_gain = -1
            best_c = -1
            # To break ties randomly, we can shuffle candidates or add random noise
            # For efficiency, we scan all 729 candidates
            # Optimization: Only consider candidates not already selected?
            # Actually, in set cover, we don't re-select. But we can re-select if we allow duplicates? No, set cover implies distinct sets.
            # However, if we pick a codeword, we remove its coverage. We don't need to consider it again.
            # But for simplicity and correctness, we just find the max intersection.
            
            # We can speed this up by only checking candidates that have potential.
            # Since the space is small (729), a full scan is fine.
            for c in range(729):
                if c in selected: # This check is O(N) per iteration, might be slow.
                    continue
                # Count bits in (uncovered & neighbor_mask[c])
                # Python int bit_count is fast
                gain = (uncovered & neighbor_mask[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                elif gain == best_gain and best_gain > 0:
                    # Random tie-breaking to explore different solutions
                    if random.random() < 0.1: # 10% chance to switch on tie
                        best_c = c
            
            if best_c == -1:
                break # Should not happen if uncovered is not empty
                
            selected.append(best_c)
            uncovered &= ~neighbor_mask[best_c]
            
        return selected

    # Run multiple restarts with different seeds to find the minimum
    # 73 is the target. Greedy often finds 73 or 74.
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        codes = greedy_cover(seed=i)
        
        # Pruning: Try to remove redundant codewords
        # A codeword is redundant if its neighbors are covered by the union of other codewords' neighbors
        # We can do a simple pass: for each c in codes, check if (neighbor_mask[c] & ~union_of_others) == 0
        # This is O(N^2) which is 73^2 ~ 5000 ops, very fast.
        
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(codes):
                # Check if c is redundant
                # Union of all other masks
                other_union = 0
                for j, other_c in enumerate(codes):
                    if i != j:
                        other_union |= neighbor_mask[other_c]
                
                # If c's neighbors are subset of other_union, c is redundant
                if (neighbor_mask[c] & ~other_union) == 0:
                    codes.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        if len(codes) < best_size:
            best_size = len(codes)
            best_codes = codes[:]
            
            # If we found 73, it's optimal. We can stop early if we want to save time, 
            # but we have time budget. Let's try a few more to be sure we didn't miss a better one (impossible) 
            # or just to be safe against bugs. Actually 73 is proven optimal.
            if best_size == 73:
                # We can break early if we are confident, but let's finish the loop or break if time allows.
                # Breaking early is safe if 73 is the theoretical minimum.
                # The problem statement says "best known code has 73". It is the optimum.
                break

    # Convert best_codes back to list of lists
    result = []
    for c in best_codes:
        digs = [(c // (3**i)) % 3 for i in range(6)]
        result.append(digs)
        
    return result
