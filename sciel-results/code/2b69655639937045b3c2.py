import numpy as np
from itertools import combinations
import time

def construct():
    """
    Find the smallest n (5 <= n <= 90) for which a tournament exists where every
    set of 4 vertices has a common dominator.
    """
    start_time = time.time()
    time_limit = 115  # Safety margin before 120s hard limit

    # Helper: Generate Paley tournament orientation for prime p
    def get_paley_orientation(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0) # i -> j
                else:
                    orientation.append(1) # j -> i
        return orientation

    # Helper: Validate if a Paley tournament is valid
    # Returns True if valid, False otherwise.
    # Uses bitmask intersection of in-neighbors.
    def is_paley_valid(p):
        if p < 5:
            return False
        
        # 1. Generate in_masks
        # in_mask[u] has bit v set if v -> u
        in_mask = [0] * p
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    # i -> j
                    in_mask[j] |= (1 << i)
        
        # 2. Check all 4-subsets
        # For subset {a,b,c,d}, dominators = in_mask[a] & in_mask[b] & in_mask[c] & in_mask[d]
        # If non-zero, valid.
        
        # Optimization: Pre-compute pairs? No, just iterate.
        # For p up to 67, this is ~6.4M iterations.
        # Python loops are slow, but 6.4M * ~0.5us (optimistic) = 3s.
        # Realistically 6.4M * 1us = 6.4s.
        # We can try to speed this up with numpy?
        # Numpy vectorization of combinations is tricky.
        # Let's stick to Python loops but optimize the inner part.
        
        in_m = in_mask
        # Using local variable for list lookup
        for a, b, c, d in combinations(range(p), 4):
            if (in_m[a] & in_m[b] & in_m[c] & in_m[d]) == 0:
                return False
        return True

    # Helper: Random Local Search for a valid tournament
    # Starts with a random tournament, then flips edges to fix violations.
    # Returns orientation list if found, else None.
    def local_search(n, time_budget):
        end_time = start_time + time_budget
        
        # Generate random orientation
        # 0: i->j, 1: j->i
        orientation = [1 if np.random.rand() > 0.5 else 0 for _ in range(n*(n-1)//2)]
        
        # Helper to get index
        def idx(i, j):
            if i > j: i, j = j, i
            return i*n - i*(i+1)//2 + (j - i - 1)
            # Formula check:
            # Pairs for i=0: (0,1)..(0,n-1) -> indices 0 .. n-2
            # Pairs for i=1: (1,2)..(1,n-1) -> indices n-1 .. 2n-3
            # General index for (i,j) i<j:
            # Count of pairs with first vertex < i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
            # Offset in row i: j - i - 1
            # Total: i*(n-1) - i*(i-1)/2 + j - i - 1
            # Let's re-verify.
            # n=3. Pairs: (0,1), (0,2), (1,2).
            # idx(0,1): 0*(2) - 0 + 0 = 0. Correct.
            # idx(0,2): 0*(2) - 0 + 1 = 1. Correct.
            # idx(1,2): 1*(2) - 1*0/2 + 0 = 2. Correct.
            # My formula above: i*n - i*(i+1)//2 + (j - i - 1)
            # idx(0,1): 0 - 0 + 0 = 0.
            # idx(0,2): 0 - 0 + 1 = 1.
            # idx(1,2): 3 - 1 + 0 = 2.
            # Both work. Let's use the simpler one if I can derive it cleanly.
            # Standard: index = i*n - i*(i+1)//2 + (j - i - 1)
            
        # Re-implement idx for clarity and speed
        def get_idx(i, j):
            if i > j:
                i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        # Helper to check validity and return list of violations
        # A violation is a 4-set with no dominator.
        # We need to know which edges to flip.
        # Flipping an edge (u,v) changes dominator status of sets containing u or v.
        # This is complex.
        
        # Simpler heuristic:
        # 1. Check if valid. If yes, return.
        # 2. Find a violating 4-set {a,b,c,d}.
        # 3. Pick a random edge within or connecting to this set and flip it?
        # Actually, if {a,b,c,d} has no dominator, it means for every vertex v, v does NOT dominate all 4.
        # i.e., for every v, there is some u in {a,b,c,d} such that NOT (v->u).
        # We want to create a dominator.
        # Let's pick a random vertex v.
        # Check which of a,b,c,d are NOT dominated by v.
        # If we can flip edges to make v dominate them all, we fix this violation.
        # But flipping an edge might break other violations.
        
        # Let's try a simpler approach:
        # Iterate through 4-sets.
        # If invalid, pick a random vertex v.
        # For each u in the 4-set, if v does not dominate u, flip the edge (v,u) so that v->u.
        # This ensures v dominates the 4-set.
        # This might introduce new violations, but we hope to converge.
        
        # To make this efficient, we need to maintain the tournament state and be able to check validity quickly.
        # Checking validity is O(n^4).
        # If we do this every iteration, it's slow.
        
        # Alternative:
        # Just do a random restart with a limited number of steps?
        # For n < 67, the space is huge.
        
        # Given the time constraint, local search is risky.
        # The most reliable improvement is to ensure we find 67 quickly and return it.
        # The current program already does this.
        # Is there a known smaller tournament?
        # "The smallest known has 67 vertices".
        # So 67 is the target.
        
        # What if we just return 67 immediately?
        # We need to generate the orientation for 67.
        # This is fast.
        # Then we can spend the rest of the time trying to find < 67.
        # But if we don't find < 67, we return 67.
        
        # The main bottleneck in the current program is the validation of 67 (and potentially 59, 47, etc).
        # If we trust that 67 is valid, we can skip validation and just return it.
        # But we must be sure.
        # The Paley tournament for p=67 is a standard object.
        # I will assume it is valid based on the problem statement "smallest known has 67".
        
        # So, strategy:
        # 1. Immediately construct Paley(67) and return it?
        # No, we should still try to find smaller ones if we have time.
        # But if we return 67 immediately, we get a valid score of 1.2388.
        # If we spend 100s searching for 31-66 and fail, we still return 67.
        # The score is the same.
        # However, if we find 66, we get a better score.
        
        # So, let's try to optimize the search for smaller n.
        # But the search is very hard.
        
        # Let's stick to the Paley candidates.
        # 31, 43, 47, 59, 67.
        # We will check them in order.
        # If we find a valid one, return it.
        # If we run out of time, return 67 (constructed, not validated, but trusted).
        
        # To save time, we can validate smaller ones first (fast).
        # 31: 31k checks.
        # 43: 123k checks.
        # 47: 178k checks.
        # 59: 450k checks.
        # 67: 6.4M checks.
        
        # Total checks if all fail: ~7M.
        # 7M * 1us = 7s.
        # This is well within the time limit.
        
        # So the current strategy is fine.
        # The only improvement is to ensure we don't waste time on invalid code paths.
        
        # Let's refine the loop.
        pass

    # Candidates to check
    candidates = [31, 43, 47, 59, 67]
    
    # We will check them in ascending order.
    # If valid, return.
    # If invalid, continue.
    # If time runs out, return 67.
    
    for p in candidates:
        if time.time() - start_time > time_limit - 5:
            break
            
        if is_paley_valid(p):
            orientation = get_paley_orientation(p)
            return {'n': p, 'orientation': orientation}
            
    # Fallback: Return Paley(67)
    # We assume 67 is valid.
    p = 67
    orientation = get_paley_orientation(p)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-START
# (The code above is the implementation of construct())
# EVOLVE-BLOCK-END
