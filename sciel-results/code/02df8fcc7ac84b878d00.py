import time
import random
import itertools

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute masks by size for fast alpha calculation
MASKS_BY_SIZE = [[] for _ in range(N_POINTS + 1)]
for m in range(N_MASKS):
    MASKS_BY_SIZE[POP_COUNT[m]].append(m)

# Generate all possible triples as bitmasks
ALL_TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append(mask)

# Precompute pairs for each triple to check linearity
# triple_pairs[triple_idx] = list of (point1, point2) pairs? 
# Or better, for each triple, the 3 pairs of points.
# To check if a new triple conflicts with existing ones, we check if any of its 3 pairs are already "used".
# We maintain a set of used pairs.

def get_alpha(triples_masks):
    """
    Calculate the independence number of the hypergraph defined by triples_masks.
    A set of points is independent if it contains no triple.
    """
    # blocked_count[m] = number of triples contained in mask m
    # If blocked_count[m] == 0, m is independent.
    blocked_count = [0] * N_MASKS
    
    for t in triples_masks:
        # Update supersets of t
        comp = ((1 << N_POINTS) - 1) ^ t
        sub = comp
        while True:
            blocked_count[t | sub] += 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    # Find largest m with blocked_count[m] == 0
    for k in range(N_POINTS, 0, -1):
        for m in MASKS_BY_SIZE[k]:
            if blocked_count[m] == 0:
                return k
    return 0

def construct():
    deadline = time.time() + 110.0
    _random = random.Random(42)
    
    best_alpha = N_POINTS
    best_triples = []
    
    # Greedy construction of a maximal linear triple system
    # We want to maximize the number of triples to minimize alpha.
    
    def greedy_build(seed):
        rng = random.Random(seed)
        used_pairs = set()
        triples = []
        
        # Randomly shuffle all triples
        candidates = ALL_TRIPLES[:]
        rng.shuffle(candidates)
        
        for t in candidates:
            # Check if t is compatible with current set
            a, b, c = [], [], []
            # Extract points
            pts = []
            mask = t
            while mask:
                lsb = mask & -mask
                idx = (lsb.bit_length() - 1)
                pts.append(idx)
                mask ^= lsb
            
            # Check pairs
            p1, p2, p3 = pts
            if (p1, p2) in used_pairs or (p2, p1) in used_pairs: continue
            if (p1, p3) in used_pairs or (p3, p1) in used_pairs: continue
            if (p2, p3) in used_pairs or (p3, p2) in used_pairs: continue
            
            # Add triple
            triples.append(t)
            used_pairs.add((p1, p2))
            used_pairs.add((p1, p3))
            used_pairs.add((p2, p3))
            
        return triples

    # Initial greedy build
    current_triples = greedy_build(42)
    current_alpha = get_alpha(current_triples)
    best_alpha = current_alpha
    best_triples = current_triples[:]
    
    # Local Search: Try to improve alpha by swapping triples
    # A move is valid if the new set of triples is still linear.
    # Swapping one triple for another is complex to check linearity for all.
    # Simpler: Randomly remove a triple, then try to add a different one?
    # Or: Simulated Annealing style?
    
    # Let's try a simple hill climbing: 
    # 1. Pick a random triple in the system.
    # 2. Remove it.
    # 3. Try to add a random valid triple from the remaining candidates.
    # 4. If alpha decreases or stays same, accept.
    
    # Precompute validity check helper
    def is_linear(triples_list):
        used_pairs = set()
        for t in triples_list:
            pts = []
            mask = t
            while mask:
                lsb = mask & -mask
                idx = (lsb.bit_length() - 1)
                pts.append(idx)
                mask ^= lsb
            p1, p2, p3 = pts
            if (p1, p2) in used_pairs: return False
            if (p1, p3) in used_pairs: return False
            if (p2, p3) in used_pairs: return False
            used_pairs.add((p1, p2))
            used_pairs.add((p1, p3))
            used_pairs.add((p2, p3))
        return True

    # Optimization: Maintain used_pairs incrementally? 
    # For simplicity and safety within time limits, we can just re-check or use a faster check.
    # Given 16 points, max 37 triples, checking linearity is fast.
    
    # Better local search: 
    # We have a set of triples. 
    # We can try to "rotate" triples.
    
    # Let's do a simple restart-based search with greedy + local tweak
    for seed in range(42, 100):
        if time.time() > deadline - 10:
            break
            
        t_triples = greedy_build(seed)
        if len(t_triples) < len(best_triples):
            continue
            
        t_alpha = get_alpha(t_triples)
        
        if t_alpha < best_alpha:
            best_alpha = t_alpha
            best_triples = t_triples[:]
            
        # Local search on this candidate
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try swapping each triple with a random compatible one
            for i in range(len(t_triples)):
                if time.time() > deadline - 5:
                    break
                old_t = t_triples[i]
                # Remove old_t
                t_triples.pop(i)
                
                # Find a new triple that is compatible with the rest
                # To be fast, just pick a random candidate from ALL_TRIPLES
                # and check if it's compatible with the current set.
                # This is O(N) per check.
                
                # To make it more effective, pick candidates that cover "heavy" points?
                # For now, random.
                for _ in range(50): # Try 50 random replacements
                    if time.time() > deadline - 5:
                        break
                    new_t = _random.choice(ALL_TRIPLES)
                    if new_t == old_t:
                        continue
                    
                    # Check if new_t is compatible with t_triples
                    # Extract points of new_t
                    pts_new = []
                    mask = new_t
                    while mask:
                        lsb = mask & -mask
                        idx = (lsb.bit_length() - 1)
                        pts_new.append(idx)
                        mask ^= lsb
                    p1, p2, p3 = pts_new
                    
                    compatible = True
                    # Check against all triples in t_triples
                    for j in range(len(t_triples)):
                        mask_j = t_triples[j]
                        # Intersection size > 1 means conflict
                        if (mask_j & new_t).bit_count() > 1:
                            compatible = False
                            break
                    
                    if compatible:
                        # Accept swap
                        t_triples.append(new_t)
                        new_alpha = get_alpha(t_triples)
                        if new_alpha <= best_alpha:
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = t_triples[:]
                            improved = True
                            break # Move to next i
                        else:
                            # Revert
                            t_triples.pop() # Remove new_t
                            # Re-insert old_t at position i? 
                            # For simplicity, just keep the set unordered for now?
                            # The order doesn't matter for alpha.
                            # But we popped old_t. We need to put it back or keep the new one if it's not worse?
                            # If new_alpha > best_alpha, we should revert to old state.
                            t_triples.append(old_t)
                            # Actually, if we don't accept, we should restore the list to exactly what it was before the pop?
                            # The list is now missing old_t and missing new_t.
                            # We appended old_t. So it's back to original.
                        # If compatible but alpha didn't improve, we might still accept to explore?
                        # Let's stick to strict improvement or equal for stability.
                        # If we accept equal, we might cycle.
                        # Let's try accepting equal with probability 0.5?
                        # For now, strict improvement.
                
                # If we didn't find a replacement, old_t is back in the list (appended).
                # Wait, if we break after accepting, old_t is gone, new_t is in.
                # If we don't break, old_t is back.
                
                # Note: The loop `for i in range(len(t_triples))` is problematic if we modify the list length.
                # We should iterate over a copy or indices carefully.
                # Since we are doing simple swaps, length stays constant.
                # But we pop and append.
                # If we accept, we break the inner loop, so we don't process remaining candidates for this i.
                # Then we move to next i.
                # This is fine.

    # Final check
    if not best_triples:
        # Fallback: return a small valid system
        # Just one triple
        return [[0, 1, 2]]
        
    # Convert bitmasks to lists
    result = []
    for t in best_triples:
        pts = []
        mask = t
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            pts.append(idx)
            mask ^= lsb
        result.append(pts)
        
    return result
