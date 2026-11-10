import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 12
    m = 40
    
    # The current program uses a greedy approach. 
    # A materially different strategy is to pre-compute the optimal structure analytically 
    # or via a much faster exhaustive search over a restricted, high-probability space, 
    # rather than iteratively adding sets.
    
    # Insight: To maximize comparable pairs in a poset of subsets, we want a "chain-like" structure.
    # A maximal chain has length n+1 = 13 sets. 
    # If we take 3 disjoint maximal chains, we have 39 sets. The 40th set can be connected to many.
    # However, a single long chain is not enough. We need to pack as many sets as possible into 
    # structures where comparability is high.
    
    # Let's try a different approach: Local Search / Simulated Annealing on the set of 40 sets.
    # But 40 sets is too many for direct optimization.
    
    # Alternative: Construct the set family based on "layers" or "ranks".
    # The number of comparable pairs is sum over all pairs {A, B} of 1 if A < B.
    
    # Let's try a specific constructive heuristic that is different from greedy:
    # Start with a maximal chain (13 sets).
    # Then add sets that are comparable to the most existing sets, but use a different metric or order.
    
    # Actually, let's look at the structure of the current solution.
    # It picks: Empty, 12 singletons, Full, 12 size-11 sets. That's 26 sets.
    # Then it greedily adds 14 more.
    
    # A different strategy: Exhaustive search over a small parameterized family of sets.
    # Or, use a randomized restart local search where we swap sets.
    
    # Let's implement a fast local search:
    # 1. Initialize with a good heuristic (e.g., the greedy one, or a random one).
    # 2. Define a neighborhood: replace one set with another.
    # 3. Evaluate the change in score.
    # 4. Accept if better, or with probability p (Simulated Annealing).
    
    # However, evaluating the score for 40 sets takes O(40^2) = 1600 comparisons.
    # Each comparison is O(n) or O(1) with bitmasks.
    # 1600 * 12 = 19200 ops per evaluation.
    # In Python, this is fast. We can do many iterations.
    
    # Let's use bitmasks for speed.
    
    def score_sets(sets):
        count = 0
        N = len(sets)
        for i in range(N):
            for j in range(i+1, N):
                if sets[i] & sets[j] == sets[i]:
                    if sets[i] != sets[j]:
                        count += 1
                elif sets[i] & sets[j] == sets[j]:
                    if sets[i] != sets[j]:
                        count += 1
        return count
    
    # Precompute all subsets as bitmasks? 2^12 = 4096.
    # This is very small. We can store all 4096 masks.
    # But we only need 40.
    
    # Let's try a "Best First" search over the space of 40-set families? No, too big.
    
    # Let's try a different constructive idea:
    # Partition the 12 elements into groups.
    # Or, use a "balanced" structure.
    
    # Actually, the greedy algorithm is already quite good (1.68).
    # Can we improve it by looking ahead?
    # Or by using a different initial set?
    
    # Let's try a randomized hill climbing approach starting from the greedy solution.
    
    # 1. Generate the greedy solution (fast).
    # 2. Convert to bitmasks.
    # 3. Local search: try swapping one set with a random other set (not in the family).
    #    If the score increases, keep it.
    #    If not, revert.
    #    Repeat until time limit.
    
    # This is a different design family: Local Search vs. Greedy Construction.
    
    # Step 1: Greedy Construction (same as before, but optimized)
    base_sets = []
    base_sets.append(0) # Empty
    for i in range(n):
        base_sets.append(1 << i) # Singletons
    base_sets.append((1 << n) - 1) # Full
    for i in range(n):
        s = (1 << n) - 1 - (1 << i) # Size 11
        base_sets.append(s)
        
    current_masks = base_sets[:]
    
    # Candidate pool: all other subsets
    all_subsets = list(range(1 << n))
    # Filter out current ones
    current_set = set(current_masks)
    candidates = [s for s in all_subsets if s not in current_set]
    
    # Greedy fill
    while len(current_masks) < m:
        if time.time() > deadline:
            break
        best_gain = -1
        best_mask = None
        for c in candidates:
            if c in current_set:
                continue
            gain = 0
            for s in current_masks:
                if (c & s) == c:
                    if c != s:
                        gain += 1
                elif (c & s) == s:
                    if c != s:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is None:
            break
        current_masks.append(best_mask)
        current_set.add(best_mask)
        
    # Step 2: Local Search
    # We have 40 masks.
    # Score calculation:
    def calc_score(masks):
        count = 0
        N = len(masks)
        for i in range(N):
            for j in range(i+1, N):
                if (masks[i] & masks[j]) == masks[i]:
                    if masks[i] != masks[j]:
                        count += 1
                elif (masks[i] & masks[j]) == masks[j]:
                    if masks[i] != masks[j]:
                        count += 1
        return count
        
    current_score = calc_score(current_masks)
    
    # Precompute comparability matrix?
    # Or just recompute score. 40^2/2 = 800 pairs. Very fast.
    
    # Randomly select a mask to swap.
    # Choose a random index in current_masks.
    # Choose a random mask from candidates (or all_subsets not in current_set).
    # Calculate new score.
    # If new_score > current_score, update.
    
    # To speed up, we can calculate the delta in score.
    # Delta = (new pairs with new mask) - (old pairs with old mask)
    # Pairs with old mask: compare old_mask with all other 39 masks.
    # Pairs with new mask: compare new_mask with all other 39 masks.
    
    def count_comparables(mask, others):
        count = 0
        for o in others:
            if o == mask:
                continue
            if (mask & o) == mask:
                count += 1
            elif (mask & o) == o:
                count += 1
        return count
        
    # Local Search Loop
    last_improvement = time.time()
    temperature = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        if time.time() - last_improvement > 5.0:
            # Restart or cool down?
            # Just keep going, maybe randomize more.
            pass
            
        # Pick a random index to replace
        idx = random.randint(0, m - 1)
        old_mask = current_masks[idx]
        
        # Pick a random replacement
        # Sample from all_subsets
        new_mask = random.choice(all_subsets)
        
        if new_mask == old_mask:
            continue
            
        if new_mask in current_set:
            continue
            
        # Calculate delta
        others = [m for i, m in enumerate(current_masks) if i != idx]
        
        old_contrib = count_comparables(old_mask, others)
        new_contrib = count_comparables(new_mask, others)
        
        delta = new_contrib - old_contrib
        
        if delta > 0:
            # Accept
            current_masks[idx] = new_mask
            current_set.remove(old_mask)
            current_set.add(new_mask)
            current_score += delta
            last_improvement = time.time()
        else:
            # Reject (Hill Climbing)
            # Could use Simulated Annealing here, but HC is simpler and often sufficient for small landscapes.
            pass
            
    # Convert back to lists
    result = []
    for mask in current_masks:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
