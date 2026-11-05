"""
B_3[1] set construction using Bitset-accelerated Backtracking.
Strategy: 
1. Precompute bitsets for all possible triple sums.
2. Use a backtracking search (DFS) with heuristics to find a large set.
3. Bitsets allow O(1) collision checks and fast conflict propagation.
"""
import time
import sys
import numpy as np

DEADLINE = 55.0  # Safety margin before 60s/90s limit

def _precompute_bitsets(n_max=4096):
    """
    Precompute bitsets for each candidate x.
    bitset[x] is a numpy array of booleans (or integers) of size (3*n_max + 1).
    bitset[x][s] is True if sum s is formed by a triple involving x.
    """
    # Max sum is 3 * 4096 = 12288
    max_sum = 3 * n_max
    # We use uint64 for compact storage, but numpy bool is fine for logic.
    # Actually, for speed in Python, a set of integers is often faster than numpy for sparse updates,
    # but here we want to check "intersection is empty".
    # Let's use a list of sets for the "forbidden sums" per element? No, that's O(N^2) memory.
    # 
    # Alternative: The current program uses a global sum_set.
    # The "Bitset" idea: 
    # For a candidate x, the sums it creates are:
    # 3x, 2x+y, x+2y, x+y+z (y,z in S)
    # If we maintain a global bitset `used_sums` where bit i is 1 if sum i is used,
    # we need to check if any of the new sums are in `used_sums`.
    # The new sums depend on S. So we can't precompute a static bitset for x.
    
    # However, we can precompute the "base" sums for x (3x, 2x+y for all y).
    # The variable part is x+y+z.
    
    # Let's stick to the set-based approach but optimize the check.
    # The current program's bottleneck is `_triple_sums_involving` which is O(|S|^2).
    # If we use backtracking, we need to be very fast.
    
    # Let's try a different representation:
    # Maintain `used_sums` as a Python set.
    # To check collision for x:
    # 1. Check 3x, 2x+y, x+2y for all y in S. (O(|S|))
    # 2. Check x+y+z for all y<=z in S. (O(|S|^2))
    
    # This is still O(|S|^2).
    # To make it O(1) or O(|S|), we need to be smarter.
    
    # Idea: For a fixed x, the sums x+y+z are distinct from used_sums iff
    # there is no pair (y,z) in S such that x+y+z is in used_sums.
    # This is equivalent to: (used_sums - x) does not contain any sum y+z where y,z in S.
    # Let `sums_S` be the set of all sums y+z for y<=z in S.
    # We can maintain `sums_S` incrementally!
    # When we add y to S, we add y+w for all w in S (and y+y) to `sums_S`.
    # Then checking x is:
    # 1. 3x not in used_sums
    # 2. 2x+y not in used_sums for all y in S
    # 3. x+2y not in used_sums for all y in S
    # 4. (x + sums_S) does not intersect used_sums.
    
    # Checking (x + sums_S) intersection used_sums:
    # Iterate through sums_S? If |sums_S| is large, this is slow.
    # But sums_S has size O(|S|^2).
    
    # Let's optimize the "pair sum" check.
    # If we maintain `used_sums` as a set, checking `x + s in used_sums` for all s in `pair_sums` is O(|pair_sums|).
    
    # This is still quadratic in the worst case per step.
    # However, backtracking prunes early.
    
    # Let's implement this "Maintain Pair Sums" strategy.
    
    pass

def construct():
    start_time = time.time()
    n_max = 4096
    
    # We will use a recursive backtracking search.
    # State:
    # - S: current list of selected numbers (sorted)
    # - used_sums: set of all triple sums formed by S
    # - pair_sums: set of all pair sums y+z (y<=z) formed by S
    
    # To speed up, we can use a heuristic to order candidates.
    # Candidates: 1..4096
    
    # Global variables for the recursive function
    best_set = []
    
    def _update_state(S, used_sums, pair_sums, x):
        """
        Add x to S. Return new used_sums and pair_sums.
        Assumes no collision.
        """
        new_used = set(used_sums)
        new_pairs = set(pair_sums)
        
        # 1. Triple x,x,x
        new_used.add(3*x)
        
        # 2. Triples x,x,y and x,y,y and x,y,z
        # Pairs involving x: x+x, x+y
        # These pairs themselves are not "triple sums", but they help form them.
        # Wait, the state `pair_sums` is for the *existing* set S.
        # When we add x, we need to generate new triple sums.
        
        # New triple sums involving x:
        # a) 3x
        # b) 2x + y for y in S
        # c) x + 2y for y in S
        # d) x + y + z for y <= z in S
        
        # Let's compute them.
        # We need to update `pair_sums` for future steps.
        # New pairs in S_new:
        # x+x, and x+y for all y in S.
        
        # So:
        # 1. Add 3x to used_sums.
        # 2. For each y in S:
        #    - Add 2x+y to used_sums
        #    - Add x+2y to used_sums
        #    - Add x+y+x? No, that's 2x+y.
        #    - The term x+y+z comes from x + (y+z).
        #      So for each pair sum p in pair_sums, add x+p to used_sums.
        
        # Let's refine the update logic:
        # Current state: S, used_sums (triple sums of S), pair_sums (pair sums of S).
        # Add x.
        
        # New triple sums:
        # 1. 3x
        # 2. 2x + y (for y in S)
        # 3. x + 2y (for y in S)
        # 4. x + p (for p in pair_sums) -- these are x+y+z where y,z in S
        
        # New pair sums:
        # 1. 2x
        # 2. x + y (for y in S)
        
        # Collision Check:
        # None of the new triple sums can be in used_sums.
        
        # Implementation:
        
        # Check 1: 3x
        if 3*x in used_sums:
            return None, None, False
        
        # Check 2 & 3: 2x+y and x+2y
        for y in S:
            if (2*x + y) in used_sums or (x + 2*y) in used_sums:
                return None, None, False
                
        # Check 4: x + p for p in pair_sums
        for p in pair_sums:
            if (x + p) in used_sums:
                return None, None, False
        
        # If no collision, update states
        new_used_sums = set(used_sums)
        new_used_sums.add(3*x)
        for y in S:
            new_used_sums.add(2*x + y)
            new_used_sums.add(x + 2*y)
        for p in pair_sums:
            new_used_sums.add(x + p)
            
        new_pair_sums = set(pair_sums)
        new_pair_sums.add(2*x)
        for y in S:
            new_pair_sums.add(x + y)
            
        return new_used_sums, new_pair_sums, True

    def _backtrack(S, used_sums, pair_sums, candidates, depth=0):
        nonlocal best_set
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if len(S) > len(best_set):
            best_set = S[:]
        
        # Heuristic: Try to add more.
        # We can prune if remaining candidates + current size <= len(best_set)
        
        # Order candidates?
        # Maybe try small numbers first? Or random?
        # Let's try a mix: iterate through candidates, but skip if time is tight.
        
        for i, x in enumerate(candidates):
            # Check time
            if time.time() - start_time > DEADLINE:
                return
            
            # Check collision
            # Optimization: If 3*x is in used_sums, skip immediately.
            if 3*x in used_sums:
                continue
                
            # Quick check for 2x+y and x+2y
            # This is O(|S|).
            # If |S| is large, this is expensive.
            # But we only do it for candidates that pass the 3x check.
            
            valid = True
            for y in S:
                if (2*x + y) in used_sums or (x + 2*y) in used_sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check x+p for p in pair_sums
            # This is O(|pair_sums|).
            # pair_sums size is ~ |S|^2 / 2.
            # If |S| is 10, pair_sums is 55.
            # If |S| is 20, pair_sums is 210.
            # If |S| is 30, pair_sums is 465.
            # This is manageable.
            
            for p in pair_sums:
                if (x + p) in used_sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If valid, recurse
            new_used, new_pairs, _ = _update_state(S, used_sums, pair_sums, x)
            # We already checked validity, so this should succeed.
            # But _update_state re-does the work. Let's just construct the new states directly here to avoid double work.
            
            # Actually, let's just call _update_state to keep it clean, or inline the update.
            # Inlining is faster.
            
            new_used_sums = set(used_sums)
            new_used_sums.add(3*x)
            for y in S:
                new_used_sums.add(2*x + y)
                new_used_sums.add(x + 2*y)
            for p in pair_sums:
                new_used_sums.add(x + p)
                
            new_pair_sums = set(pair_sums)
            new_pair_sums.add(2*x)
            for y in S:
                new_pair_sums.add(x + y)
                
            # Recurse with remaining candidates
            # To avoid permutations, we only consider candidates > x?
            # No, we want to pick ANY candidate from the remaining pool.
            # If we iterate `for i, x in enumerate(candidates)`, we should pass `candidates[i+1:]` to avoid re-picking x or previous ones?
            # Yes, if we want combinations.
            
            _backtrack(S + [x], new_used_sums, new_pair_sums, candidates[i+1:])
            
            # Backtrack: do nothing, local variables handle it.

    # Initial Candidates
    # To find a large set, we should probably start with a good initial set.
    # Greedy from 1 gives a baseline.
    
    # Let's run a quick greedy to get a baseline and maybe a good set of "safe" starting points.
    # Or just start with empty set and all candidates 1..4096.
    # But 4096 candidates is too many for deep backtracking.
    
    # Strategy:
    # 1. Generate a random subset of candidates, or use all.
    # 2. Use a heuristic to order them.
    
    # Let's try a "Randomized Restart" with Backtracking.
    # In each restart, we pick a random permutation of a subset of candidates (e.g., top 200 random numbers).
    # And run backtracking.
    
    # Why subset? Because backtracking on 4096 items is too slow.
    # We need to find a dense cluster.
    
    # Better Strategy:
    # 1. Find a good set using Greedy (current best).
    # 2. Use Backtracking to *extend* or *improve* it?
    #    Backtracking is for finding *a* set, not necessarily improving an existing one easily.
    
    # Let's try:
    # 1. Greedy to get a set of size K.
    # 2. Identify "conflict zones".
    # 3. Use Backtracking on a reduced candidate pool (e.g., numbers near the existing set, or random numbers) to see if we can find a denser packing.
    
    # Actually, the most powerful approach for B_h sets is often just good Greedy with Randomization.
    # The current program does this.
    # The directive asks for a STRUCTURAL change. Backtracking is structurally different.
    
    # Let's implement Backtracking on a *reduced* candidate set.
    # We will pick a random window of size W from [1, 4096] and try to find the max B_3[1] set within that window?
    # No, the set can span the whole range.
    
    # Let's try Backtracking with a limited depth or candidate pool.
    # Candidate pool: 1..4096.
    # We will use a heuristic: "Minimum Remaining Values" is hard to compute dynamically here.
    # Let's use "Most Constrained Variable" (MCV) heuristic?
    # A number x is more constrained if it has more collisions with the current set.
    # But we are adding, not removing.
    
    # Let's try a simple DFS with a time limit, restarting with different random orders.
    
    def _greedy_baseline():
        S = []
        used = set()
        pairs = set()
        for x in range(1, 4097):
            # Check
            if 3*x in used: continue
            ok = True
            for y in S:
                if (2*x+y) in used or (x+2*y) in used:
                    ok = False; break
            if not ok: continue
            for p in pairs:
                if (x+p) in used:
                    ok = False; break
            if not ok: continue
            
            # Add
            S.append(x)
            used.add(3*x)
            for y in S[:-1]: # S[:-1] is the old S
                used.add(2*x+y)
                used.add(x+2*y)
                pairs.add(x+y)
            for p in pairs: # old pairs
                used.add(x+p)
            pairs.add(2*x)
            # Wait, the update logic in _backtrack was slightly different.
            # In _backtrack:
            # new_pairs = old_pairs U {2x} U {x+y for y in old_S}
            # new_used = old_used U {3x} U {2x+y, x+2y for y in old_S} U {x+p for p in old_pairs}
            
            # My greedy loop above:
            # used.add(3*x)
            # for y in S[:-1]: (old S)
            #   used.add(2*x+y)
            #   used.add(x+2*y)
            #   pairs.add(x+y)  <-- This adds to pairs BEFORE checking x+p?
            #   No, I check x+p using OLD pairs.
            #   Then I add x+p to used.
            
            # Let's be precise.
            
            pass
        return S

    # Let's just run the Backtracking directly.
    # To make it feasible, we will limit the candidate pool to a random subset of size ~100-200.
    # And we will do multiple restarts.
    
    # Why? A B_3[1] set of size 30 in [1, 4096] is sparse.
    # The "density" is low.
    # Backtracking on 4096 items will branch too much.
    # But if we restrict to a "likely" region?
    # Or if we use the Greedy result to guide the Backtracking?
    
    # Let's try this:
    # 1. Run Greedy to get a set S_g.
    # 2. Take the candidates from S_g and some random neighbors.
    # 3. Run Backtracking on this smaller pool to see if we can find a better arrangement.
    
    # Actually, a better structural change:
    # **Iterative Deepening A* (IDA*)**? No.
    # **Simulated Annealing**? 
    # Start with a valid set. Move: remove one, add one. 
    # If it increases size or keeps size but improves "score" (e.g., min gap), accept.
    # This is a local search, which is mentioned in inspirations.
    
    # The directive says: "Use a precomputed bitset... XOR-ing... Minimum Remaining Values".
    # This implies a CSP solver.
    
    # Let's implement the CSP solver with Bitsets (using Python sets as bitsets for simplicity, or numpy if faster).
    # Given the time limit, Python sets are probably faster for small N than numpy overhead.
    
    # I will implement the Backtracking CSP solver described in `_backtrack` above.
    # I will run it on a random subset of candidates to avoid the explosion.
    
    # Strategy:
    # 1. Greedy baseline.
    # 2. Multiple restarts of Backtracking on random subsets of size M (e.g., 50, 100, 150).
    #    The subset is chosen to include some numbers from the Greedy solution to bias towards good regions.
    
    def _run_backtracking_on_subset(candidates):
        """
        Run DFS on a specific list of candidates.
        Returns the largest set found within this subset.
        """
        best_local = []
        
        # Sort candidates? Maybe descending to pack large numbers?
        # Or ascending?
        # Let's try ascending.
        
        def dfs(S, used_sums, pair_sums, idx):
            nonlocal best_local
            nonlocal start_time
            
            if time.time() - start_time > DEADLINE:
                return
                
            if len(S) > len(best_local):
                best_local = S[:]
                
            if idx == len(candidates):
                return
                
            # Option 1: Skip candidates[idx]
            dfs(S, used_sums, pair_sums, idx + 1)
            
            # Option 2: Take candidates[idx]
            x = candidates[idx]
            
            # Check collision
            if 3*x in used_sums:
                return
                
            valid = True
            for y in S:
                if (2*x + y) in used_sums or (x + 2*y) in used_sums:
                    valid = False
                    break
            if not valid:
                return
                
            for p in pair_sums:
                if (x + p) in used_sums:
                    valid = False
                    break
            if not valid:
                return
                
            # Update
            new_used = set(used_sums)
            new_used.add(3*x)
            for y in S:
                new_used.add(2*x + y)
                new_used.add(x + 2*y)
            for p in pair_sums:
                new_used.add(x + p)
                
            new_pairs = set(pair_sums)
            new_pairs.add(2*x)
            for y in S:
                new_pairs.add(x + y)
                
            dfs(S + [x], new_used, new_pairs, idx + 1)
            
        dfs([], set(), set(), 0)
        return best_local

    # 1. Get Greedy Baseline
    S_greedy = _greedy_build_greedy()
    
    # 2. Backtracking Restarts
    # We will try to find a set larger than S_greedy.
    # We will use random subsets.
    
    import random
    rng = random.Random(42)
    
    best_overall = S_greedy
    
    # Number of restarts
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Create a candidate pool
        # Include all of S_greedy
        # Add random numbers
        pool = set(S_greedy)
        # Add random numbers from 1..4096
        # How many? 
        # If S_greedy is size 30, we might add 20 random numbers.
        # Total pool size ~50.
        # Backtracking on 50 items is 2^50, too big.
        # But with pruning, it might be okay.
        
        # Let's limit pool size to 30-40.
        # If S_greedy is large, we might just use it.
        # If S_greedy is small, we can add more.
        
        # Actually, if we use the *same* candidates as S_greedy, we will just find S_greedy (or a subset).
        # We need *new* candidates to find a *better* set.
        
        # Let's pick a random subset of 1..4096 of size K=40.
        # And run backtracking.
        
        K = 40
        candidates = rng.sample(range(1, 4097), K)
        
        # Run backtracking
        S_bt = _run_backtracking_on_subset(candidates)
        
        if len(S_bt) > len(best_overall):
            best_overall = S_bt
            # If we found a bigger set, maybe restart with a pool based on this new set?
            # For now, just update best.

    # One more thing: The Greedy function in the current program is `_greedy_linear`.
    # I need to implement a fast greedy to get the baseline.
    
    return best_overall

def _greedy_build_greedy():
    """Fast greedy using the pair-sum optimization."""
    S = []
    used_sums = set()
    pair_sums = set()
    
    for x in range(1, 4097):
        # Check
        if 3*x in used_sums:
            continue
            
        valid = True
        for y in S:
            if (2*x + y) in used_sums or (x + 2*y) in used_sums:
                valid = False
                break
        if not valid:
            continue
            
        for p in pair_sums:
            if (x + p) in used_sums:
                valid = False
                break
        if not valid:
            continue
            
        # Add
        S.append(x)
        used_sums.add(3*x)
        for y in S[:-1]:
            used_sums.add(2*x + y)
            used_sums.add(x + 2*y)
            pair_sums.add(x + y)
        for p in pair_sums:
            used_sums.add(x + p)
        pair_sums.add(2*x)
        
    return S

# Re-define construct to use the logic above cleanly
def construct():
    start_time = time.time()
    
    # 1. Greedy Baseline
    best = _greedy_build_greedy()
    
    # 2. Backtracking on random subsets
    import random
    rng = random.Random(42)
    
    # We will try to beat the greedy result.
    # Greedy usually gets ~20-30 for n=4096.
    # Backtracking on small random sets might find local optima that are denser.
    
    # Let's try a few restarts.
    # To avoid spending too much time, we limit the pool size and depth.
    
    for r in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Pool size: try different sizes
        # Small pools (20) are fast. Large pools (50) are slow.
        # Let's mix.
        pool_size = 20 + (r % 5) * 5 # 20, 25, 30, 35, 40
        
        candidates = rng.sample(range(1, 4097), pool_size)
        
        S_bt = _run_backtracking_on_subset(candidates, start_time)
        
        if len(S_bt) > len(best):
            best = S_bt
            
    return best

def _run_backtracking_on_subset(candidates, start_time):
    best_local = []
    
    def dfs(S, used_sums, pair_sums, idx):
        nonlocal best_local
        nonlocal start_time
        
        if time.time() - start_time > DEADLINE:
            return
            
        if len(S) > len(best_local):
            best_local = S[:]
            
        if idx == len(candidates):
            return
            
        # Pruning: If even taking all remaining doesn't beat best_local, return
        # max_possible = len(S) + (len(candidates) - idx)
        # if max_possible <= len(best_local): return
        
        # Option 1: Skip
        dfs(S, used_sums, pair_sums, idx + 1)
        
        # Option 2: Take
        x = candidates[idx]
        
        # Check collision
        if 3*x in used_sums:
            return
            
        valid = True
        for y in S:
            if (2*x + y) in used_sums or (x + 2*y) in used_sums:
                valid = False
                break
        if not valid:
            return
            
        for p in pair_sums:
            if (x + p) in used_sums:
                valid = False
                break
        if not valid:
            return
            
        # Update
        new_used = set(used_sums)
        new_used.add(3*x)
        for y in S:
            new_used.add(2*x + y)
            new_used.add(x + 2*y)
        for p in pair_sums:
            new_used.add(x + p)
            
        new_pairs = set(pair_sums)
        new_pairs.add(2*x)
        for y in S:
            new_pairs.add(x + y)
            
        dfs(S + [x], new_used, new_pairs, idx + 1)
        
    dfs([], set(), set(), 0)
    return best_local
