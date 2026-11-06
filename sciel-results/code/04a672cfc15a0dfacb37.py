import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin before 150s kill
    
    N = 32
    MAX_VAL = 512
    
    # Precompute all possible 3-term APs (a, a+d, a+2d) for d in 1..511
    # We represent the set of values as a bitset (Python int)
    # A value x is in the set if (mask >> x) & 1 == 1
    
    # Helper to get score (number of distinct d) from a mask
    def get_score(mask):
        # For each d, check if there exists a such that a, a+d, a+2d are in mask
        # We can iterate d from 1 to MAX_VAL//2 (since a+2d <= 512 implies d <= 256 if a>=1, but a can be small)
        # Actually max d is such that 1, 1+d, 1+2d <= 512 => 2d <= 511 => d <= 255.5 => 255.
        # Wait, if a=1, d=255, 1, 256, 511. All <= 512.
        # If a=2, d=255, 2, 257, 512. All <= 512.
        # So max d is 255.
        score = 0
        # Optimization: iterate d, check if any AP with diff d exists
        # An AP with diff d exists if there is a '111' pattern in the bitset shifted by d?
        # Let M be the mask.
        # AP exists for d if (M & (M >> d) & (M >> 2*d)) != 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
                score += 1
        return score

    # Helper to get the set of covered differences as a bitmask or set
    def get_covered_diffs(mask):
        covered = set()
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
                covered.add(d)
        return covered

    # Initial set: Random or Greedy?
    # Let's start with a greedy construction that prioritizes covering new differences.
    # Start with empty set? No, need 32 elements.
    # Let's start with a random set and improve.
    
    # Better initial: Greedy add
    # Start with 1..32? No, that covers few diffs.
    # Let's try a random start and hill climb.
    
    def random_initial():
        return set(random.sample(range(1, MAX_VAL + 1), N))
    
    current_set = random_initial()
    current_mask = 0
    for x in current_set:
        current_mask |= (1 << x)
        
    current_score = get_score(current_mask)
    best_score = current_score
    best_set = set(current_set)
    
    # Greedy Phase: Try to add elements that maximize new coverage? 
    # But size is fixed. So we swap.
    
    # Simulated Annealing / Hill Climbing with Bitsets
    # To speed up, we can precompute the "contribution" of a new element x to the score?
    # That's hard because it depends on existing elements.
    
    # Instead, we do fast local search using bitsets.
    # Each move: remove x, add y.
    # We can speed up the score calculation by only checking diffs that might change.
    # However, with N=32 and MAX=512, a full score calc is ~255 iterations of bit ops.
    # Bit ops on 512-bit ints are very fast in Python (single machine word or two).
    # 255 * 3 ops = ~765 ops per score calc.
    # 100k iterations * 765 = 76.5M ops. Should be well under 10s.
    
    # Let's implement a fast swap loop.
    
    t_start = time.time()
    
    # Precompute list of all values
    all_values = list(range(1, MAX_VAL + 1))
    
    # To make it smarter, we can bias the random choice of 'y' towards values that
    # are "far" from the current set to encourage larger d, or "close" to encourage small d?
    # Actually, we want to cover ALL d.
    
    # Let's use a "targeted" search:
    # 1. Identify uncovered diffs.
    # 2. For each uncovered diff d, find a triplet (a, a+d, a+2d) where 2 elements are in S and 1 is not.
    # 3. Propose swapping an element NOT in that triplet (or one that is redundant) with the missing element.
    
    # This is complex to implement efficiently in pure Python without heavy overhead.
    # Let's stick to fast random swaps with a bitset score, but add a "restart" mechanism
    # and a "greedy initialization" to get a good start.
    
    # Greedy Initialization:
    # Start with empty set.
    # Add elements one by one to maximize score.
    # This is O(N * MAX_VAL * ScoreCost). 32 * 512 * 255 ~ 4M ops. Fast.
    
    def greedy_initial():
        S = set()
        mask = 0
        for _ in range(N):
            best_val = -1
            best_score = -1
            # Sample a subset of candidates to speed up greedy?
            # Or iterate all? 512 is small.
            for v in all_values:
                if v in S:
                    continue
                new_mask = mask | (1 << v)
                sc = get_score(new_mask)
                if sc > best_score:
                    best_score = sc
                    best_val = v
            S.add(best_val)
            mask |= (1 << best_val)
        return S, mask, best_score

    current_set, current_mask, current_score = greedy_initial()
    best_score = current_score
    best_set = set(current_set)
    
    # Local Search
    # We will perform random swaps.
    # To improve, we can look at "uncovered" diffs and try to fix them.
    
    while time.time() < deadline:
        # Get current uncovered diffs
        covered = get_covered_diffs(current_mask)
        uncovered = [d for d in range(1, 256) if d not in covered]
        
        if not uncovered:
            # All diffs covered? Max possible is 255.
            # Score is 255.
            break
            
        # Strategy: Pick a random uncovered diff d.
        # Find a triplet for d that is "almost" in the set (2 in, 1 out).
        # If found, swap the "out" element with a random element from the set that is not part of the triplet.
        
        move_made = False
        d_target = random.choice(uncovered)
        
        # Find candidates for d_target
        # a, a+d, a+2d
        # We want 2 in S, 1 not in S.
        best_swap_out = None
        best_swap_in = None
        
        # Iterate possible a
        max_a = MAX_VAL - 2 * d_target
        for a in range(1, max_a + 1):
            x, y, z = a, a + d_target, a + 2 * d_target
            in_set = [x in current_set, y in current_set, z in current_set]
            count_in = sum(in_set)
            
            if count_in == 2:
                # Identify the one not in set
                if not in_set[0]:
                    missing = x
                    present = [y, z]
                elif not in_set[1]:
                    missing = y
                    present = [x, z]
                else:
                    missing = z
                    present = [x, y]
                
                # We need to add 'missing' and remove one element.
                # Ideally, remove an element that is NOT in 'present' (so we don't break this AP).
                # If we remove an element from 'present', we break this AP, so we must ensure we gain more elsewhere.
                # Simple heuristic: Try to remove an element not in 'present'.
                
                candidates_to_remove = [v for v in current_set if v not in present]
                if candidates_to_remove:
                    # Pick a random one to remove
                    rem = random.choice(candidates_to_remove)
                    # Check if this swap improves score
                    new_mask = (current_mask & ~(1 << rem)) | (1 << missing)
                    new_score = get_score(new_mask)
                    
                    if new_score > current_score:
                        # Accept
                        current_set.remove(rem)
                        current_set.add(missing)
                        current_mask = new_mask
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                        
                        move_made = True
                        break # Restart search for next uncovered diff
        
        if not move_made:
            # Fallback: Random swap
            rem = random.choice(list(current_set))
            while True:
                add = random.randint(1, MAX_VAL)
                if add not in current_set:
                    break
            
            new_mask = (current_mask & ~(1 << rem)) | (1 << add)
            new_score = get_score(new_mask)
            
            if new_score >= current_score:
                current_set.remove(rem)
                current_set.add(add)
                current_mask = new_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                # Sometimes accept worse moves to escape local optima (Simulated Annealing light)
                # But keep it simple: only accept if better or equal to avoid degradation.
                # If stuck, restart with a perturbation?
                # Let's just continue random swaps.
                pass

    return sorted(best_set)
