"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import sys

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Strategy: Local Search on Set Membership
    # 1. Start with a large valid Sidon set (greedy construction).
    # 2. Iteratively try to REMOVE an element from the set.
    # 3. If removing an element breaks maximality (i.e., the set is no longer maximal),
    #    try to REPAIR maximality by adding a different element that was previously blocked.
    # 4. If the repair succeeds and the set size is smaller (or equal but different structure), keep it.
    # 5. This effectively searches for "holes" in the set that can be filled by "shifting" elements,
    #    potentially reducing the total count if the new element blocks more others than the removed one did.
    #    Actually, simply removing an element reduces size by 1. But it breaks maximality.
    #    To restore maximality, we must add SOME element. If we can add an element that was previously blocked by the removed one, 
    #    we are back to the same size. 
    #    Wait, if we remove x, and then add y, size is same. 
    #    How do we get SMALLER?
    #    
    #    Alternative View:
    #    A maximal Sidon set S is minimal if no element can be removed without violating maximality? No, that's not right.
    #    A minimal maximal Sidon set is one where no proper subset is also maximal.
    #    If we remove x from S, S' = S \ {x}. S' is NOT maximal (by definition of S being maximal, x was needed to block something? No, x blocks something, but maybe y blocks it too?).
    #    Actually, if S is maximal, S' is not maximal. This means there exists some z not in S' such that S' U {z} is Sidon.
    #    If we can find such a z, and if S' U {z} is "better" in some way, we might iterate.
    #    But S' U {z} has the same size as S.
    #    
    #    To reduce size, we need a different approach.
    #    Consider the "blocking" relationship.
    #    If we have a set S, and we remove x, we might be able to add MULTIPLE elements? No, we add one to restore maximality?
    #    No, we don't HAVE to restore maximality immediately. We can just keep S' (which is not maximal) and try to remove more?
    #    No, the final result MUST be maximal.
    #    
    #    Let's use a "Swap" move.
    #    Remove x, Add y. Size constant.
    #    This explores the space of maximal sets of fixed size.
    #    To change size, we need a "Remove" move that results in a maximal set?
    #    If S is maximal, S\{x} is never maximal.
    #    So we can never just remove.
    #    
    #    Wait, is it possible that a SMALLER maximal set exists?
    #    Yes.
    #    How to find it?
    #    Start with a large maximal set.
    #    Try to find a subset of S that is ALSO maximal?
    #    If S is maximal, any subset T of S is NOT maximal (unless T=S).
    #    So we cannot find a smaller maximal set as a subset of a larger one.
    #    The smaller maximal sets are DISJOINT from the larger ones? Or at least not subsets.
    #    
    #    So Local Search on "Remove" is impossible if we require maximality at every step.
    #    We must allow intermediate states to be NON-MAXIMAL?
    #    If we allow non-maximal states, we can remove elements freely.
    #    Then we must "repair" to maximality.
    #    Repairing to maximality usually adds elements.
    #    So the cycle is: Remove (size -1), Repair (size +k).
    #    If k=0, we found a smaller maximal set!
    #    When can k=0?
    #    If S\{x} is already maximal? Impossible if S is maximal.
    #    So k >= 1.
    #    So this approach doesn't reduce size.
    
    #    Let's rethink.
    #    The current best score is 1.2308, which corresponds to a size of 20 (25/20 = 1.25? No. Seed is 25? 25/20 = 1.25. 25/21 = 1.19. 25/20.3?
    #    Score = Seed / Value.
    #    If Score = 1.2308 and Seed = 25, Value = 25 / 1.2308 = 20.3. So Size = 20 or 21?
    #    If Size = 20, Score = 1.25.
    #    If Size = 21, Score = 1.19.
    #    Maybe Seed is different?
    #    "search score is seed/value, so 1.0 equals the seed program".
    #    If the seed program produces a set of size 25, then Seed=25.
    #    If current score is 1.2308, Value = 25/1.2308 = 20.31.
    #    This implies the current program finds a set of size 20? Or 21?
    #    If it finds 20, score is 1.25.
    #    If it finds 21, score is 1.19.
    #    1.2308 is between 1.19 and 1.25.
    #    Maybe the seed is not 25?
    #    Or maybe the score is averaged? No, "raw value".
    #    Let's assume the current program finds size 20.
    #    Wait, 25/20 = 1.25.
    #    25/21 = 1.1904.
    #    1.2308 is closer to 1.25.
    #    Maybe the seed is 24.6? Unlikely.
    #    Maybe the current program finds size 20 sometimes and 21 others?
    #    Or maybe the "seed" is the size of the initial greedy set?
    #    
    #    Let's focus on finding size 19 or lower.
    #    Known lower bound for N=256 is ~sqrt(256) = 16.
    #    So 19 is possible.
    
    #    New Strategy: Randomized Restart with "Greedy Removal" from a Random Maximal Set?
    #    No, we established that subsets of maximal sets are not maximal.
    
    #    How about "Greedy Construction" with a different selection rule?
    #    The current program uses "Max Blocking".
    #    What if we use "Min Blocking"?
    #    If we pick an element that blocks FEW others, we keep more options open.
    #    This might lead to a LARGER set?
    #    We want a SMALLER set.
    #    So "Max Blocking" is the right heuristic for minimization.
    
    #    What if we use a "Lookahead" heuristic?
    #    Instead of 1-step blocking, estimate the final set size.
    
    #    What if we use a different representation?
    #    Represent the set by its sums?
    
    #    Let's try a "Simulated Annealing" on the set of elements, allowing invalid states?
    #    Energy = |S| * Penalty + (0 if Sidon else Large_Penalty) + (0 if Maximal else Large_Penalty).
    #    This is complex to evaluate.
    
    #    Let's try a simpler structural change:
    #    "Greedy with Randomization and Backtracking".
    #    1. Pick a random element.
    #    2. Add it.
    #    3. Recursively pick the next best element.
    #    4. If stuck (no valid element), backtrack.
    #    5. This is a DFS.
    #    To minimize size, we want to branch on elements that block the most?
    #    This is essentially what the current program does, but without backtracking.
    #    Adding backtracking might find a smaller set if the greedy choice was suboptimal.
    
    #    Let's implement a DFS with pruning.
    #    State: (current_set, available_set, sums_set).
    #    Goal: Find a maximal set with size < current_best.
    #    Pruning: If len(current_set) >= current_best, prune.
    #    Ordering: Sort available elements by "blocking power" descending.
    
    #    This could be exponential, but with pruning and N=256, it might work if the optimal size is small (e.g., 18).
    #    The branching factor is high initially, but "blocking" reduces the available set quickly.
    
    def is_valid_addition(v, selected, sums):
        if 2 * v in sums:
            return False
        for x in selected:
            if (v + x) in sums:
                return False
        return True

    def get_blocking_power(v, selected, sums, available):
        """Calculate how many elements in available become invalid if v is added."""
        new_sums = set(sums)
        new_sums.add(2 * v)
        for x in selected:
            new_sums.add(v + x)
        
        block_count = 0
        for u in available:
            if u == v:
                continue
            # Check if u is valid with new_sums and selected+[v]
            # u is valid if is_valid_addition(u, selected+[v], new_sums) is False
            # Wait, we want to count how many are INVALID.
            # So if is_valid_addition returns False, it's blocked.
            if not is_valid_addition(u, selected + [v], new_sums):
                block_count += 1
        return block_count

    best_size = 25 # Initial guess, will be updated by a quick greedy
    best_set = None
    
    # Quick greedy to get a baseline
    selected = []
    sums = set()
    available = set(range(1, N + 1))
    while available:
        valid_candidates = [v for v in available if is_valid_addition(v, selected, sums)]
        if not valid_candidates:
            break
        # Pick max blocking
        best_v = max(valid_candidates, key=lambda v: get_blocking_power(v, selected, sums, available))
        selected.append(best_v)
        sums.add(2 * best_v)
        for x in selected:
            if x != best_v:
                sums.add(best_v + x)
        available.remove(best_v)
    
    best_size = len(selected)
    best_set = list(selected)
    
    # Now try DFS to find smaller
    # We need to be careful with recursion depth and time.
    
    def dfs(selected, sums, available, target_size):
        nonlocal best_size, best_set
        
        if len(selected) >= target_size:
            return False # Prune
        
        # Check if selected is maximal
        # If no valid candidate exists, it's maximal
        valid_candidates = [v for v in available if is_valid_addition(v, selected, sums)]
        
        if not valid_candidates:
            # Maximal set found
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = list(selected)
                return True
            else:
                return False
        
        # Order valid_candidates by blocking power
        # To save time, only calculate for top K?
        # For now, calculate all if list is small, else sample?
        # N=256, valid_candidates might be ~100.
        # Sorting 100 items with O(N) cost each is 100*100 = 10k ops.
        # This is fine.
        
        # Optimization: If time is tight, skip sorting and just iterate?
        # Sorting helps find small sets faster.
        
        powers = []
        for v in valid_candidates:
            # Only calculate blocking if we have time?
            # Let's just use a simpler heuristic for ordering: random or value?
            # Actually, "Max Blocking" is the key.
            # But calculating it is expensive.
            # Let's approximate: Pick v that is "central"?
            # Or just iterate in random order?
            # If we iterate in random order, we might find a small set by chance.
            
            # Let's try sorting by value first, then blocking?
            # No, let's stick to blocking but limit the calculation.
            # If len(valid_candidates) > 20, just pick randomly?
            # Or pick the one with max "degree" in a precomputed graph?
            
            # For this DFS, let's use a simpler ordering:
            # Sort by v (ascending) or random.
            # Actually, let's just pick one randomly from valid_candidates.
            # This turns DFS into Randomized DFS.
            
            pass
        
        # Randomized DFS
        random.shuffle(valid_candidates)
        
        for v in valid_candidates:
            if time.time() > deadline:
                return False
            
            # Add v
            new_selected = selected + [v]
            new_sums = set(sums)
            new_sums.add(2 * v)
            for x in selected:
                new_sums.add(v + x)
            
            new_available = set(available)
            new_available.remove(v)
            
            if dfs(new_selected, new_sums, new_available, target_size):
                # Found a better solution, we can stop this branch
                # But we want to find the BEST, so we should continue?
                # If we found size K, we can prune branches with size >= K.
                # The nonlocal best_size is updated.
                # The prune condition `len(selected) >= target_size` uses the global target.
                # We should update target_size to best_size + 1 to prune more aggressively.
                # But target_size is passed by value.
                # Let's use a nonlocal variable for the current best.
                pass
        
        return False

    # The above DFS is a bit messy with the target_size.
    # Let's refactor to use a nonlocal `current_best` for pruning.
    
    current_best_size = best_size
    current_best_set = best_set
    
    def dfs_optimized(selected, sums, available):
        nonlocal current_best_size, current_best_set
        
        if len(selected) >= current_best_size:
            return
        
        valid_candidates = [v for v in available if is_valid_addition(v, selected, sums)]
        
        if not valid_candidates:
            # Maximal
            if len(selected) < current_best_size:
                current_best_size = len(selected)
                current_best_set = list(selected)
            return
        
        # Heuristic: Try candidates that block the most first?
        # To avoid expensive calculation, let's just shuffle.
        # Or sort by "value" (small values block large sums? No).
        # Let's try sorting by "blocking power" but only for the top 10 candidates?
        # No, let's just shuffle. Randomness helps escape local minima.
        
        random.shuffle(valid_candidates)
        
        for v in valid_candidates:
            if time.time() > deadline:
                return
            
            new_selected = selected + [v]
            new_sums = set(sums)
            new_sums.add(2 * v)
            for x in selected:
                new_sums.add(v + x)
            
            new_available = set(available)
            new_available.remove(v)
            
            dfs_optimized(new_selected, new_sums, new_available)
            
            # Early exit if we found the theoretical lower bound?
            if current_best_size <= 16:
                return

    # Run DFS
    # Start with empty set?
    # The greedy found a set of size ~20.
    # DFS from empty set might be too slow if branching factor is high.
    # But "blocking" reduces the pool quickly.
    # Let's try it.
    
    dfs_optimized([], set(), set(range(1, N + 1)))
    
    # If DFS didn't improve, try some random restarts of the greedy?
    # The current program already does that.
    # Let's just return the best found.
    
    if current_best_set:
        return current_best_set
    else:
        return best_set

# EVOLVE-BLOCK-END
