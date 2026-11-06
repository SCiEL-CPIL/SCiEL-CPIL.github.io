"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit

    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    def get_extendable(selected, sums):
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            new_sums = {2 * x}
            new_sums.update(x + y for y in selected)
            
            if new_sums & sums:
                continue
            
            extendable.append(x)
        return extendable

    # Build conflict graph
    # Vertices: 1..N
    # Edge (u, v) exists if u and v cannot coexist in a Sidon set.
    # Two numbers u, v (u < v) conflict if:
    # 1. 2u == 2v (impossible for distinct u,v)
    # 2. 2u == u + v => u == v (impossible)
    # 3. 2v == u + v => u == v (impossible)
    # 4. u + v == u + v (trivial, but we need to check against other pairs)
    # Wait, the conflict is not just pairwise. 
    # A Sidon set requires ALL pair sums to be distinct.
    # So a set S is a Sidon set if for all a,b,c,d in S with a<=b, c<=d, 
    # a+b = c+d implies (a,b) == (c,d) or (a,b) == (d,c).
    # 
    # The "conflict graph" for MIS formulation of Sidon sets is tricky because 
    # Sidon property is a 4-uniform hypergraph constraint (or 3-uniform if we fix one).
    # However, the prompt suggests: "edges connect elements that cannot coexist in a Sidon set".
    # This implies a pairwise conflict definition. 
    # If we define conflict as "u and v form a sum that collides with another sum formed by u or v", 
    # that's not quite right because the collision depends on the rest of the set.
    #
    # Actually, a standard reduction for Sidon sets to Independent Set is not direct pairwise.
    # But let's look at the "Greedy max-degree vertex selection" inspiration.
    # The cell finding says: "Greedy max-degree vertex selection minimizes the final set size by aggressively blocking incompatible candidates".
    # This suggests the conflict graph is built such that if we pick a vertex, we remove its neighbors.
    # 
    # Let's reconsider the conflict definition. 
    # If we have a set S, and we want to add x, it fails if there exist a,b in S such that x+a = x+b (impossible) 
    # or x+a = a+b (implies x=b, impossible) 
    # or 2x = a+b.
    # So x conflicts with a,b if a+b = 2x.
    # This is a 3-way constraint: {x, a, b} cannot all be in the set.
    #
    # The prompt's strategy: "Model the problem as a Maximum Independent Set (MIS) on a conflict graph where vertices are numbers 1–256 and edges connect elements that cannot coexist in a Sidon set".
    # This description is slightly mathematically loose for Sidon sets (which are 4-uniform), 
    # but we can interpret "cannot coexist" as "if both are selected, they create a high risk of collision" 
    # or we can use a different heuristic:
    # 
    # Alternative Interpretation:
    # The "Maximal Sidon Set" problem can be approached by constructing a set that is "dense" in terms of blocking.
    # If we pick a number x, it blocks all pairs (a,b) such that a+b = 2x.
    # 
    # Let's try a different structural approach: 
    # Greedy construction with "Max Blocking" or "Max Conflict" removal.
    # 
    # The cell finding mentions: "Greedy max-degree vertex selection minimizes the final set size... Measured: 1 -> 1.231".
    # This implies the current score 1.2308 is already using or close to this.
    # Wait, the current program uses Simulated Annealing.
    # The cell finding says "strategy of a program that improved island 4" measured 1 -> 1.231.
    # This suggests the SA program (current) is the one that scored 1.231.
    # 
    # The directive says: "Make a STRUCTURAL change... different design or modelling family".
    # The suggested strategy is MIS on conflict graph.
    # 
    # How to build the conflict graph for Sidon sets?
    # A valid Sidon set is an independent set in the 4-uniform hypergraph.
    # Approximating this with a graph is hard.
    # 
    # However, we can use a "Pairwise Collision Graph" for a specific purpose:
    # If we fix a subset S, x is blocked if there is a pair in S summing to 2x.
    # 
    # Let's try a "Backtracking with Conflict-Driven Heuristics" or "Greedy with Lookahead".
    # 
    # Structural Change: 
    # Instead of SA on the set, use a Deterministic Greedy Algorithm with a specific ordering 
    # based on "Conflict Density".
    # 
    # Algorithm:
    # 1. Start with all numbers 1..N.
    # 2. While we haven't formed a maximal Sidon set:
    #    - Calculate for each remaining candidate x, how many "potential collisions" it has with the current set.
    #    - But we want to MINIMIZE size. 
    #    - To minimize size while staying maximal, we should pick elements that "block" many others.
    #    - An element x blocks y if y cannot be added after x.
    #    - This depends on the rest of the set.
    # 
    # Let's try "Reverse Greedy":
    # Start with an empty set.
    # At each step, pick the element x that, if added, would result in the "largest" number of future elements being blocked?
    # This is hard to compute exactly.
    # 
    # Alternative: 
    # Use the MIS heuristic on a graph where edges represent "strong conflicts".
    # Define a graph G where u~v if u+v is "popular" (many pairs sum to it)? No.
    # 
    # Let's stick to the prompt's suggestion: "greedy vertex elimination heuristic that iteratively removes the vertex with the highest degree (most conflicts) to force the remaining set into a small, inclusion-maximal independent set."
    # 
    # This sounds like:
    # 1. Build a graph where edges represent conflicts.
    # 2. Repeatedly remove the vertex with the highest degree.
    # 3. The remaining vertices form the set?
    # 
    # If we remove high-degree vertices, we are removing the "most conflicting" ones.
    # The remaining vertices are those with few conflicts.
    # This would likely yield a LARGE set (low conflict = easy to coexist).
    # We want a SMALL set.
    # 
    # So maybe we should ADD vertices?
    # If we add the vertex with the highest degree, we block its neighbors.
    # This is a standard Greedy MIS heuristic: Pick a vertex, add to set, remove it and its neighbors.
    # This yields a maximal independent set.
    # Does it yield a small one?
    # Greedy MIS usually yields a set of size O(N log N / d). 
    # If we pick high-degree vertices first, we remove many neighbors, so the set size should be smaller.
    # 
    # So the strategy is:
    # 1. Define a conflict graph G.
    # 2. Greedy MIS:
    #    - Available = {1..N}
    #    - Set = []
    #    - While Available is not empty:
    #        - Pick v in Available with highest degree in G (restricted to Available? or global?)
    #        - Add v to Set
    #        - Remove v and all its neighbors in G from Available
    # 3. Check if Set is Sidon.
    # 4. Check if Set is Maximal.
    # 
    # The key is: What is G?
    # For Sidon sets, a pair (u,v) doesn't inherently conflict unless we consider the sums.
    # However, if we assume a "random" Sidon set, most pairs are compatible.
    # 
    # Maybe the conflict graph should be based on the sums?
    # Two numbers u and v are "conflicting" if they are part of a "bad" configuration?
    # 
    # Let's try a different graph:
    # Vertices: 1..N
    # Edge (u,v) if u+v is in a set of "frequent sums"?
    # 
    # Actually, let's look at the "Cell Finding" again.
    # "Greedy max-degree vertex selection minimizes the final set size... Measured: 1 -> 1.231".
    # This suggests the method works.
    # 
    # Let's implement a Greedy MIS on a graph where edges are defined by "potential collision".
    # We can define u~v if there exists w such that u+w = 2v or similar?
    # 
    # Let's try a simpler heuristic:
    # Greedy construction where we pick the element that "blocks" the most other elements.
    # 
    # Algorithm:
    # 1. Candidate set C = {1..N}
    # 2. Set S = []
    # 3. Sums = set()
    # 4. While C is not empty:
    #    - For each x in C, calculate "blocking power".
    #      Blocking power = number of y in C\{x} such that y cannot be added to S ∪ {x}.
    #      y cannot be added if:
    #         - 2y is in Sums ∪ {2x}
    #         - y+s is in Sums ∪ {2x, x+s for s in S} for some s in S ∪ {x}
    #    - Pick x with max blocking power.
    #    - Add x to S, update Sums.
    #    - Remove x from C.
    #    - Remove all y from C that are now blocked (cannot be added to S).
    # 
    # This ensures S is a Sidon set (by construction, we only add valid elements).
    # It ensures S is Maximal (by construction, we remove all blocked elements, so no unblocked element remains).
    # It aims to Minimize Size (by picking elements that block many others).
    # 
    # This is a structural change from SA to Greedy Blocking.
    # 
    # Complexity:
    # N=256.
    # Loop runs at most N times (likely much fewer).
    # In each loop, we check up to N candidates.
    # For each candidate, we check up to N others.
    # Checking if y is blocked involves checking sums.
    # Sums size is O(|S|^2).
    # Total complexity ~ N^3 * |S|^2.
    # 256^3 * 256^2 is too high?
    # 256^3 = 16M. 16M * 100 (approx sums) = 1.6B. Might be too slow in Python.
    # 
    # Optimization:
    # We can precompute or update blocking status incrementally.
    # 
    # Let's refine the "Blocking" check.
    # y is blocked by S ∪ {x} if:
    # 1. 2y is in Sums ∪ {2x}
    # 2. There exists s in S ∪ {x} such that y+s is in Sums ∪ {2x, x+s' for s' in S}
    #    Actually, if we add x, the new sums are {2x} ∪ {x+s for s in S}.
    #    y is blocked if {2y} ∪ {y+s for s in S ∪ {x}} intersects {Sums} ∪ {2x} ∪ {x+s for s in S}.
    # 
    # This is still O(|S|) per check.
    # 
    # Let's try a simpler approximation:
    # Pick x that maximizes the number of y such that 2y = x + s for some s in S, or 2y = 2x (y=x), or y+s = 2x.
    # 
    # Let's implement the Greedy Blocking with optimizations.
    # 
    # To speed up:
    # - Maintain a list of "alive" candidates.
    # - When we add x, we can quickly identify which candidates are blocked.
    # 
    # Let's code this.

    def greedy_blocking():
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        
        while alive:
            best_x = None
            max_blocked = -1
            
            # Heuristic: Pick x that blocks the most
            # To save time, we might not check all x if time is tight.
            # But N=256 is small.
            
            # We can also add a bit of randomness to escape local optima?
            # The prompt asks for a structural change. Greedy is deterministic.
            # We can try multiple random orderings or perturbations.
            
            # For now, deterministic greedy.
            # We can iterate over a random permutation of alive to break ties?
            # Or just pick the first one with max blocked.
            
            candidates = list(alive)
            # Optional: Shuffle to add variety
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added to S
                # x is in alive, so it should be compatible with S.
                # But we should verify.
                new_sums_x = {2 * x}
                new_sums_x.update(x + s for s in S)
                
                if new_sums_x & Sums:
                    continue # Should not happen if alive is maintained correctly
                
                # Calculate how many other elements in alive are blocked by adding x
                # An element y in alive (y != x) is blocked if:
                # It cannot be added to S ∪ {x}.
                # This happens if {2y} ∪ {y+s for s in S ∪ {x}} intersects {Sums} ∪ {2x} ∪ {x+s for s in S}
                
                blocked_count = 0
                # We only care about y in alive, y != x
                # To speed up, we can just count how many y in alive are "conflicting" with the new set.
                
                # Approximation: Count y such that 2y is in new_sums_x or y+s is in new_sums_x for some s in S.
                # Actually, if y is in alive, it is compatible with S.
                # So y is blocked by x if:
                # 1. 2y is in {2x} ∪ {x+s for s in S}
                # 2. y+x is in Sums (since y+x is a new sum, if it's in Sums, collision)
                # 3. y+s is in {2x} ∪ {x+s' for s' in S} for some s in S?
                #    If y+s = 2x, then y = 2x-s.
                #    If y+s = x+s', then y = x+s'-s.
                
                # Let's just do the full check for y in alive, y != x.
                # But this is O(N) per x, so O(N^2) per step.
                # Total O(N^3). 256^3 = 16M.
                # In Python, 16M operations is ~1-2 seconds.
                # We have 58 seconds. So we can do many restarts or more complex checks.
                
                for y in alive:
                    if y == x:
                        continue
                    # Check if y is blocked by S ∪ {x}
                    # New sums introduced by x: 2x, x+s for s in S
                    # Sums introduced by y: 2y, y+s for s in S ∪ {x}
                    
                    # Collision if any sum from y is in Sums or new_sums_x
                    # 1. 2y in Sums? No, y is alive, so compatible with S.
                    # 2. 2y in new_sums_x?
                    if 2 * y in new_sums_x:
                        blocked_count += 1
                        continue
                    
                    # 3. y+s in Sums? No, y is alive.
                    # 4. y+s in new_sums_x?
                    #    y+s = 2x  => s = 2x - y
                    #    y+s = x+s' => s = x + s' - y
                    #    We need to check if for any s in S, y+s is in new_sums_x.
                    #    new_sums_x = {2x} ∪ {x+s' for s' in S}
                    #    So we check if (y+s) == 2x OR (y+s) == x+s' for some s' in S.
                    #    (y+s) == 2x => s = 2x - y. If 2x-y in S, then blocked.
                    #    (y+s) == x+s' => s' = y+s-x. If y+s-x in S, then blocked.
                    
                    # Efficient check:
                    # Is (2x - y) in S?
                    if (2 * x - y) in S:
                        blocked_count += 1
                        continue
                    
                    # Is there s in S such that (y+s - x) in S?
                    # i.e., is there s in S such that y+s-x is in S?
                    # Let s2 = y+s-x. We need s in S and s2 in S.
                    # This is equivalent to: is there s in S such that (y-x+s) in S?
                    # Let d = y-x. Is there s in S such that s+d in S?
                    
                    # Precompute differences? 
                    # For now, simple loop.
                    blocked = False
                    for s in S:
                        if (y + s - x) in S:
                            blocked = True
                            break
                    if blocked:
                        blocked_count += 1
                        continue
                        
                    # Also check y+x in Sums?
                    # y+x is a new sum. If y+x is in Sums, collision.
                    # But y is alive, so y+x should not be in Sums?
                    # Wait, if y is alive, it is compatible with S.
                    # Does that mean y+s not in Sums for all s in S? Yes.
                    # But y+x is not in S.
                    # So we must check if y+x is in Sums.
                    if (y + x) in Sums:
                        blocked_count += 1
                        continue

                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_x = x
                    
            if best_x is None:
                # Fallback: pick any
                best_x = next(iter(alive))
                
            # Add best_x to S
            S.append(best_x)
            new_sums_best = {2 * best_x}
            new_sums_best.update(best_x + s for s in S[:-1]) # S without best_x
            Sums.update(new_sums_best)
            
            # Remove best_x from alive
            alive.remove(best_x)
            
            # Remove all y in alive that are now blocked
            # This is important to maintain the invariant that alive contains only valid candidates.
            to_remove = []
            for y in alive:
                # Check if y is compatible with S (which now includes best_x)
                # New sums: 2y, y+s for s in S
                # Check against Sums
                # 1. 2y in Sums?
                if 2 * y in Sums:
                    to_remove.append(y)
                    continue
                # 2. y+s in Sums for any s in S?
                blocked = False
                for s in S:
                    if (y + s) in Sums:
                        blocked = True
                        break
                if blocked:
                    to_remove.append(y)
                    continue
                    
            for y in to_remove:
                alive.remove(y)

        return S

    # Run greedy blocking
    best_set = greedy_blocking()
    
    # Verify
    if not is_sidon(best_set):
        # Fallback
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    # Check maximality
    sums = get_sums(best_set)
    extendable = get_extendable(best_set, sums)
    
    if extendable:
        # It's not maximal? 
        # My greedy algorithm should ensure maximality by removing all blocked elements.
        # If it's not maximal, it means my blocking check was incomplete.
        # Let's try to add the extendable elements? No, that increases size.
        # We want to minimize size.
        # If it's not maximal, it's invalid.
        # We need to fix it.
        # 
        # Perhaps we should run a local search to fix non-maximality?
        # Or just return the greedy result and hope the verifier accepts it?
        # No, it must be maximal.
        
        # Let's try a simple repair:
        # If there are extendable elements, it means we didn't block them.
        # This implies our "blocked" check missed some collisions.
        # 
        # Let's try a different strategy: 
        # Start with the greedy set.
        # While not maximal:
        #    Find an extendable element x.
        #    To make it non-extendable, we must add an element that blocks x.
        #    But adding an element increases size.
        #    We want to minimize size.
        #    
        #    Alternative: Remove an element from S that is "redundant" for blocking x?
        #    No, removing an element might unblock other things.
        
        # Given the time, let's assume the greedy blocking is correct and 
        # the issue is with my implementation of the blocking check.
        # 
        # Let's refine the blocking check in the main loop.
        # 
        # Actually, let's just use the SA program but with a different initialization 
        # based on the greedy blocking result.
        # 
        # Or, let's just return the greedy set if it's valid and maximal.
        # If not, we need a fallback.
        
        # Let's try to repair by adding the extendable elements? 
        # No, that violates minimization.
        
        # Let's try to find a smaller maximal set by swapping?
        
        # For now, if it's not maximal, we'll use the SA fallback from the original program 
        # but initialized with this greedy set.
        
        pass

    # If we made it here, we return best_set
    # But we need to ensure it's maximal.
    # 
    # Let's add a repair step:
    # If not maximal, we can try to swap elements to block the extendable ones.
    
    if extendable:
        # Try to repair
        # This is complex. Let's just use the SA approach from the original 
        # but start from this greedy set.
        
        current = list(best_set)
        cur_sums = get_sums(current)
        
        def objective(selected):
            if not selected:
                return float('inf')
            sums = get_sums(selected)
            expected = len(selected) * (len(selected) + 1) // 2
            violations = expected - len(sums)
            
            if violations > 0:
                return violations * 1000 + len(selected)
            
            extendable = get_extendable(selected, sums)
            non_max_count = len(extendable)
            
            if non_max_count > 0:
                return non_max_count * 100 + len(selected)
            
            return len(selected)

        cur_obj = objective(current)
        best_obj = cur_obj
        best_set_sa = list(current)
        
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.995
        
        while time.time() < deadline:
            if T < T_min:
                break
                
            move_type = random.random()
            new_set = list(current)
            
            if move_type < 0.3 and len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                new_set.pop(idx)
            elif move_type < 0.6:
                missing = [x for x in range(1, N+1) if x not in current]
                if missing:
                    add_val = random.choice(missing)
                    new_set.append(add_val)
                    new_set.sort()
            else:
                if len(current) > 1:
                    idx_rem = random.randint(0, len(current) - 1)
                    new_set.pop(idx_rem)
                    missing = [x for x in range(1, N+1) if x not in current]
                    if missing:
                        add_val = random.choice(missing)
                        new_set.append(add_val)
                        new_set.sort()
            
            new_obj = objective(new_set)
            
            if new_obj < cur_obj:
                current = new_set
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_set_sa = list(current)
            else:
                delta = new_obj - cur_obj
                if T > 0:
                    accept_prob = math.exp(-delta / T)
                    if random.random() < accept_prob:
                        current = new_set
                        cur_obj = new_obj
                        
            T *= cooling_rate

        return best_set_sa

    return best_set

# EVOLVE-BLOCK-END
