"""
Structural change: Deterministic iterative deepening backtracking solver.
Instead of stochastic local search, we use a CSP approach:
1. Fix the set size (start small, increase if stuck).
2. Use iterative deepening on the maximum element (diameter).
3. Backtracking search with constraint propagation (difference bitmask).
4. Branch and bound: prune if current max exceeds bound.
"""
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We need to find a set S containing 0, with |S| <= 32,
    # such that all differences 1..30 are present and all differences are distinct.
    # We want to minimize max(S).
    
    # A perfect difference set of order k has size k^2+k+1.
    # For differences 1..30, we need at least enough elements to generate 30 distinct differences.
    # Max differences for size n is n(n-1)/2.
    # n=9 -> 36 diffs. So n=9 is the minimum size.
    # However, we don't need a *perfect* difference set (which covers 1..v-1),
    # just a *truncated* one covering 1..30.
    
    # Let's try to find a solution with size n, starting from n=9.
    # If no solution found with small diameter, we increase diameter.
    # Actually, the diameter is the bottleneck. Let's iterate on diameter D.
    
    # Lower bound on diameter:
    # We need 30 distinct differences.
    # If we have n elements, max differences is n(n-1)/2.
    # Also, the largest difference is max(S) - min(S) = max(S) since 0 is in S.
    # So D >= 30.
    
    # Let's try D from 30 upwards.
    # For a fixed D, we search for a Sidon set in [0, D] containing 0,
    # with all diffs 1..30 present.
    
    # Optimization:
    # Since 0 is in the set, the differences are just the elements of the set (d = x - 0 = x)
    # AND differences between other pairs.
    # Wait, if 0 is in the set, then every element x in S is a difference (x-0).
    # So S must contain 1, 2, ..., 30?
    # NO. The condition is "every d in 1..30 occurring as a difference".
    # If 0 is in S, then for any x in S, x is a difference.
    # So if 1 is a difference, either 1 is in S, or there exist a,b in S such that a-b=1.
    # But since 0 is in S, if 1 is NOT in S, we need a,b in S\{0} with a-b=1.
    # However, if 1 is in S, that's fine.
    # Crucially, if 1, 2, ..., 30 are ALL in S, then we have 31 elements (0..30).
    # Differences of 0..30:
    # 1-0=1, 2-0=2, ..., 30-0=30.
    # But also 2-1=1 (duplicate!), 3-2=1 (duplicate!).
    # So S cannot contain consecutive integers if we want distinct differences.
    # Thus, we CANNOT have {0, 1, ..., 30} as a subset.
    
    # This implies we need "longer" elements to create the differences 1..30 without collisions.
    # Example: To get diff 1, we can have 1 in S (then 1-0=1). If 1 is in S, we cannot have 2 in S (2-1=1).
    # So if 1 is in S, 2 is not. How do we get diff 2?
    # 2-0=2 (so 2 in S? No, 2-1=1 collision).
    # If 1 in S, 2 not in S.
    # We need diff 2. Candidates: 2-0 (requires 2 in S, conflict), 3-1 (requires 3 in S).
    # So if 1 in S, 3 in S, then 3-1=2.
    # Check diffs so far: {1, 3}. Diffs: 1-0=1, 3-0=3, 3-1=2. All distinct.
    # Next diff 3. We have 3-0=3. So diff 3 is covered.
    # Next diff 4. Candidates: 4-0 (4 in S?), 5-1 (5 in S?), 4-?
    # If 4 in S: 4-0=4, 4-1=3 (collision with 3-0=3). So 4 cannot be in S if 1,3 are.
    # So we need 5-1=4? Requires 5 in S.
    # Let's check {0, 1, 3, 5}.
    # Diffs:
    # 1-0=1
    # 3-0=3, 3-1=2
    # 5-0=5, 5-1=4, 5-3=2 (Collision! 3-1=2 and 5-3=2).
    # So {0, 1, 3, 5} is invalid.
    
    # This is a classic Sidon set problem.
    # We will use backtracking.
    
    def solve(max_elem, size_limit=32):
        """
        Find a Sidon set in [0, max_elem] containing 0,
        with all differences 1..30 present.
        Returns the set or None.
        """
        # Bitmask for differences 1..30. Bit i is set if diff i is found.
        # We also need to track ALL differences to ensure Sidon property.
        # Since max_elem can be up to ~100, differences can be up to 100.
        # We'll use a set for all diffs, and a bitmask for 1..30.
        
        # Heuristic: Order of adding elements.
        # We must include 0.
        # The set must cover diffs 1..30.
        
        # State:
        # current_set: list of elements
        # used_diffs: set of all differences generated so far
        # target_mask: bitmask of diffs 1..30 found so far
        
        current_set = [0]
        used_diffs = set()
        target_mask = 0  # 30 bits
        
        # We need to select `size_limit - 1` more elements.
        # But we don't know the exact size. We can stop early if we have a valid set.
        # Actually, we want to MINIMIZE max_elem.
        # The backtracking will naturally find a solution if one exists within the bound.
        
        # To speed up, we can use a greedy hint or ordering.
        # Let's try adding elements in increasing order? No, that's restrictive.
        # Let's try adding elements that cover "missing" diffs.
        
        # However, pure backtracking on 32 elements is too slow without strong pruning.
        # Pruning:
        # 1. If a new element x creates a duplicate diff with any y in current_set, prune.
        # 2. If we can't cover remaining diffs with remaining slots, prune.
        
        # Let's implement a recursive search.
        # To avoid deep recursion limits, we'll use a stack or increase limit.
        sys.setrecursionlimit(10000)
        
        def backtrack(idx, current_max):
            # idx: number of elements selected so far (including 0)
            # current_max: max element in current_set
            
            # Check if we have a valid solution
            if (target_mask == (1 << 30) - 1):
                return current_set[:]
            
            # If we've reached size limit, fail
            if idx >= size_limit:
                return None
            
            # Determine candidates for the next element
            # We can pick any integer from 1 to max_elem not in current_set.
            # To be efficient, we should prioritize candidates that cover missing diffs.
            
            # Missing diffs:
            missing = [(1 << 30) - 1] & ~target_mask
            if missing == 0:
                return current_set[:]
            
            # Generate candidates
            # For each missing diff d, candidates are y+d or y-d for y in current_set.
            # Also, we can just pick any new number.
            
            # Let's collect all potential candidates that might help.
            candidates = set()
            for d in range(1, 31):
                if not (target_mask & (1 << (d - 1))):
                    for y in current_set:
                        c1 = y + d
                        if c1 <= max_elem and c1 not in current_set:
                            candidates.add(c1)
                        c2 = y - d
                        if c2 >= 0 and c2 not in current_set:
                            candidates.add(c2)
            
            # If no candidates help with missing diffs, we might need to add "filler" elements?
            # No, filler elements don't help cover missing diffs directly, but they might enable future diffs?
            # Actually, if we add x, it creates diffs x-y. If x-y is not in 1..30, it doesn't help target_mask.
            # But it might be a duplicate? No, if it's not in 1..30, it's just a "waste" diff.
            # However, we are limited by size. Adding waste elements is bad.
            # So we should ONLY add elements that create at least one new diff in 1..30.
            
            if not candidates:
                return None
            
            # Sort candidates to try most promising first?
            # Maybe smallest first to keep max_elem small?
            # Or candidates that cover more missing diffs?
            
            # Let's try smallest first to minimize max_elem quickly.
            for x in sorted(candidates):
                # Check if x is valid (no duplicate diffs)
                valid = True
                new_mask = target_mask
                for y in current_set:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                if not valid:
                    continue
                
                # Pruning: Can we still finish?
                # Remaining slots: size_limit - idx - 1
                # Remaining missing diffs: popcount((1<<30)-1 & ~new_mask)
                # Max new diffs we can generate with k elements?
                # Each new element x generates at most k diffs with existing elements.
                # But they must be distinct and not in used_diffs.
                # This bound is weak. Let's rely on the search.
                
                # Recurse
                current_set.append(x)
                used_diffs_copy = set(used_diffs)
                for y in current_set[:-1]:
                    used_diffs_copy.add(abs(x - y))
                
                res = backtrack(idx + 1, max(x, current_max))
                if res is not None:
                    return res
                
                current_set.pop()
                # Restore used_diffs? We passed a copy, so no need to restore if we use local vars.
                # But my backtrack signature doesn't take used_diffs as arg, it uses closure.
                # This is a bug in my draft. I need to pass state or copy it.
                
            return None
        
        # The closure approach is messy for backtracking with state changes.
        # Let's rewrite with explicit state passing or careful restoration.
        
        # Better approach: Pass state as arguments.
        
        def bt(current_set, used_diffs, target_mask, idx):
            if (target_mask == (1 << 30) - 1):
                return current_set[:]
            
            if idx >= size_limit:
                return None
            
            # Candidates
            candidates = set()
            for d in range(1, 31):
                if not (target_mask & (1 << (d - 1))):
                    for y in current_set:
                        c1 = y + d
                        if c1 <= max_elem and c1 not in current_set:
                            candidates.add(c1)
                        c2 = y - d
                        if c2 >= 0 and c2 not in current_set:
                            candidates.add(c2)
            
            if not candidates:
                return None
            
            # Heuristic: Try candidates that cover the most missing diffs?
            # Or just sorted?
            # Let's try sorted to find small solutions first.
            for x in sorted(candidates):
                valid = True
                new_mask = target_mask
                new_used = set(used_diffs)
                
                for y in current_set:
                    d = abs(x - y)
                    if d in new_used:
                        valid = False
                        break
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                if not valid:
                    continue
                
                current_set.append(x)
                res = bt(current_set, new_used, new_mask, idx + 1)
                current_set.pop()
                
                if res is not None:
                    return res
            
            return None
        
        # Initial state
        initial_set = [0]
        initial_diffs = set()
        initial_mask = 0
        
        return bt(initial_set, initial_diffs, initial_mask, 1)
    
    # Iterative deepening on diameter
    # Start from lower bound.
    # Lower bound:
    # We need 30 diffs.
    # Smallest Sidon set covering 1..30?
    # Known Sidon sets:
    # {0, 1, 4, 10, ...}
    # Let's start D from 30.
    
    best_solution = None
    for D in range(30, 1001):  # Upper bound 1000 should be safe
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a solution with max element <= D
        # Note: The set must be in [0, D].
        sol = solve(D)
        
        if sol is not None:
            # Verify
            diffs = set()
            for i in range(len(sol)):
                for j in range(i + 1, len(sol)):
                    d = sol[j] - sol[i]
                    if d in diffs:
                        break
                    diffs.add(d)
                else:
                    continue
                break
            else:
                if all(d in diffs for d in range(1, 31)):
                    best_solution = sol
                    # Since we are increasing D, the first valid solution is optimal for this D.
                    # But we want the GLOBAL minimum D.
                    # So we can return immediately?
                    # Yes, because we are iterating D from low to high.
                    # The first D for which a solution exists is the minimal diameter.
                    # Wait, is it possible that a solution exists for D=35 but not D=30?
                    # Yes.
                    # Is it possible that a solution exists for D=30 but not D=35?
                    # No, if a set is in [0,30], it is also in [0,35].
                    # So the set of valid diameters is [D_min, infinity).
                    # So the first D we find is the minimum.
                    return sol
    
    # If no solution found in loop, we should have a fallback?
    # The problem guarantees a solution exists (implied by "construct()").
    # But if we timed out, we return None?
    # The contract says "return the best result found".
    # If we found nothing, we must return something valid?
    # "invalid result ... scores 0".
    # So we MUST return a valid set.
    # If backtracking fails to find one in time, we need a greedy fallback.
    
    # Greedy Fallback
    def greedy_fallback():
        a = [0]
        used_diffs = set()
        target_mask = 0
        # Try to add elements that cover missing diffs
        # We don't have a strict D limit here, just try to keep it small.
        # This is a heuristic.
        
        # Actually, if the backtracking failed, it's likely because the search space was too big
        # or we hit the time limit.
        # Let's try a simple greedy:
        # For d in 1..30:
        #   if d not covered:
        #      try to add x such that x-y=d or y-x=d for some y in a.
        #      pick smallest valid x.
        
        for _ in range(50): # max iterations
            if target_mask == (1 << 30) - 1:
                break
            # Find missing diffs
            missing = [d for d in range(1, 31) if not (target_mask & (1 << (d - 1)))]
            if not missing:
                break
            
            # Try to cover the smallest missing diff
            d = missing[0]
            found = False
            # Try adding x
            # Candidates: y+d, y-d
            candidates = set()
            for y in a:
                candidates.add(y + d)
                candidates.add(y - d)
            
            for x in sorted(candidates):
                if x < 0 or x in a:
                    continue
                # Check validity
                valid = True
                new_mask = target_mask
                new_used = set(used_diffs)
                for y in a:
                    dd = abs(x - y)
                    if dd in new_used:
                        valid = False
                        break
                    new_used.add(dd)
                    if 1 <= dd <= 30:
                        new_mask |= (1 << (dd - 1))
                
                if valid:
                    a.append(x)
                    used_diffs = new_used
                    target_mask = new_mask
                    found = True
                    break
            
            if not found:
                # Try larger jumps?
                # Just add a large number to avoid collisions?
                # This might break the "minimize max" goal but ensures validity.
                x = max(a) + 1
                while True:
                    valid = True
                    new_mask = target_mask
                    new_used = set(used_diffs)
                    for y in a:
                        dd = abs(x - y)
                        if dd in new_used:
                            valid = False
                            break
                        new_used.add(dd)
                        if 1 <= dd <= 30:
                            new_mask |= (1 << (dd - 1))
                    if valid:
                        a.append(x)
                        used_diffs = new_used
                        target_mask = new_mask
                        break
                    x += 1
        
        # Verify final set
        diffs = set()
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                d = a[j] - a[i]
                if d in diffs:
                    return None # Should not happen if greedy was careful
                diffs.add(d)
        
        if all(d in diffs for d in range(1, 31)):
            return a
        return None
    
    if best_solution is None:
        best_solution = greedy_fallback()
    
    if best_solution is None:
        # Absolute fallback: return a known valid set?
        # We can hardcode a small Sidon set if we know one.
        # But we don't.
        # Let's return an empty list? No, must contain 0.
        # Return [0]? No, must cover 1..30.
        # This path should rarely be taken.
        return [0]
    
    return best_solution

# EVOLVE-BLOCK-END
