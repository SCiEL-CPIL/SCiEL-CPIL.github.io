def construct():
    import time
    import random
    from random import Random

    # Safety margin: 150s kill limit, aim for 145s
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j in range(i, len(S_list)):
                b = S_list[j]
                if a + b in S:
                    return False
        return True

    def solve_instance(vals, time_limit):
        n = len(vals)
        if n == 0:
            return set()
        
        # 1. Build Conflict Graph
        # Node i conflicts with Node j if vals[i] + vals[j] is in vals.
        # We want Maximum Independent Set (MIS).
        # adj[i] is a bitset of nodes j such that i and j conflict.
        # Note: If vals[i] + vals[i] is in vals, i conflicts with itself? 
        # In MIS, self-loops mean the node cannot be in the set.
        # But standard MIS assumes simple graphs. If 2*vals[i] is in vals, 
        # then we cannot pick vals[i] at all? 
        # Wait, if a+b=c and a=b, then 2a=c. If we pick a, we cannot pick c? 
        # No, the condition is "no a, b, c in S such that a+b=c".
        # If we pick a and c, and 2a=c, then a+a=c is a violation.
        # So a and c conflict.
        # Does a conflict with itself? 
        # If a is in S, is a+a in S? If yes, violation.
        # So if 2a in vals, we cannot have both a and 2a.
        # This is an edge between a and 2a.
        # There are no self-loops in the sense of "a conflicts with a" unless we define the graph such that 
        # picking a forces something. But here, the constraint is on triples.
        # The conflict graph G has vertices V. Edge (u,v) exists if u+v is in V.
        # An independent set in G is a set S such that for any u,v in S, u+v is NOT in S.
        # This is exactly the sum-free condition.
        
        val_to_idx = {v: i for i, v in enumerate(vals)}
        adj = [0] * n
        
        for i in range(n):
            vi = vals[i]
            for j in range(n):
                vj = vals[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Edge between i and j? 
                    # No, the constraint is: if i and j are in S, then k must NOT be in S.
                    # This is a 3-uniform hypergraph constraint, not a graph constraint directly.
                    # Wait. 
                    # Sum-free set: no a,b,c in S with a+b=c.
                    # This is equivalent to: S is an independent set in the graph where 
                    # edges connect a and b if a+b is in V? 
                    # No. If we pick a and b, we forbid c.
                    # If we pick a and c, we forbid b.
                    # If we pick b and c, we forbid a.
                    # This is a 3-uniform hypergraph independent set problem.
                    # However, for integers, we can map this to a graph problem?
                    # Actually, the standard reduction for sum-free sets in integers is:
                    # Sort the set. If we pick a large element, it can't be the sum of two smaller ones?
                    # No, that's not right either.
                    
                    # Let's re-evaluate.
                    # The current program treats it as: Start with odds. 
                    # Why are odds sum-free? Odd+Odd=Even. If S contains only odds, a+b is even, so not in S.
                    # This works because the set is "parity-separated".
                    
                    # Can we use a general graph approach?
                    # If we define a graph G where u~v if u+v is in V, then an independent set S in G 
                    # means for any u,v in S, u+v is NOT in V.
                    # But the sum-free condition requires u+v not in S.
                    # If u+v is in V but not in S, that's fine.
                    # So, if S is an independent set in G, then for any u,v in S, u+v is not in V, 
                    # which implies u+v is not in S.
                    # So ANY independent set in G is sum-free.
                    # Is the converse true? No. A sum-free set might have u,v in S such that u+v is in V 
                    # but u+v is not in S. In that case, u and v are adjacent in G, so S is NOT independent in G.
                    # So MIS on G gives a valid sum-free set, but might not be optimal for sum-free size.
                    # However, it's a strong lower bound.
                    
                    # Let's check the parity argument again.
                    # If S is all odds, no edges in G? 
                    # If u,v are odd, u+v is even. If even numbers are in V, then u~v in G.
                    # So odds are NOT necessarily an independent set in G.
                    # Example: V={1, 3, 4}. 1+3=4. 1~3 in G. 
                    # S={1,3} is sum-free? 1+3=4 not in S. Yes.
                    # But S={1,3} is not independent in G because 1~3.
                    # So MIS on G would exclude {1,3} if it's the max independent set?
                    # Max independent set in G for {1,3,4}: Edges (1,3). Max IS is {1,4} or {3,4} or {1,3}? 
                    # {1,3} is not independent. {1,4} is independent (1+4=5 not in V). 
                    # Size 2. 
                    # Sum-free sets: {1,3} (size 2), {1,4} (size 2), {3,4} (size 2).
                    # So MIS on G finds a sum-free set of optimal size here.
                    
                    # Is it always true that Max Sum-Free Set size >= Max Independent Set in G?
                    # Yes, because any Independent Set in G is Sum-Free.
                    # So solving MIS on G gives a valid solution.
                    # Can we do better? 
                    # The "conflict" is only if the sum is IN S.
                    # So if u+v is in V but we choose NOT to include u+v in S, then u and v can both be in S.
                    # This is a "3-uniform" constraint.
                    # However, for random sets, the "parity" heuristic is very strong.
                    # The current program gets 1.45.
                    
                    # Let's try a different angle: Local Search on the Sum-Free property directly.
                    # Start with a greedy sum-free set.
                    # Then try to swap elements to increase size.
        
        # Strategy:
        # 1. Generate multiple initial candidates using different heuristics (Odds, Evens, Random, Greedy by abs value).
        # 2. For each candidate, perform local search:
        #    - Try adding any element from V\S. If it violates sum-free, identify the "blocking" pairs.
        #    - If adding x requires removing k elements, and k < 1 (impossible) or k=0 (add it), do it.
        #    - If k=1, it's a swap. Replace the blocker with x? Or just add x and remove blocker? Size stays same.
        #    - If we allow size to stay same, we might find a path to a larger set.
        #    - Better: Try to add x. If conflicts exist, try to remove a MINIMUM set of elements from S to make x valid.
        #    - This is the "cost" approach. If cost < 1, add. (Impossible for integers? Cost is integer).
        #    - If cost == 1, size is neutral. We can accept neutral moves to escape local optima (Simulated Annealing style).
        #    - If cost > 1, reject.
        
        # Let's implement a Simulated Annealing / Randomized Local Search.
        # State: S (set of indices or values).
        # Objective: Maximize |S| subject to Sum-Free.
        # Moves:
        # 1. Add x in V\S. If valid, do it.
        # 2. If invalid, find min removal set R. If |R| == 1, consider swap (remove R, add x). 
        #    This keeps size same. Accept with probability p.
        # 3. If |R| > 1, reject.
        # 4. Also try removing an element to see if it allows adding two others? (Complex).
        
        # Simpler: 
        # Start with Greedy.
        # Then iterate:
        #   For each x in V\S:
        #      Check if x can be added.
        #      If yes, add it.
        #      If no, check if removing ONE element y from S allows adding x.
        #      If yes, and random < p, do the swap (remove y, add x).
        #   Also, try removing random elements to "unblock" others?
        
        # Let's refine the "Min Removal" calculation.
        # To add x, we need to remove a set R from S such that (S\R) U {x} is sum-free.
        # Conflicts involving x:
        # 1. x+x=2x in S -> remove 2x.
        # 2. x+y=z in S -> remove x (no, x is new), remove y, or remove z.
        #    Since x is new, we can't remove x. We must remove y or z.
        #    This creates a graph on S where edges are (y,z) if x+y=z.
        #    We need a Vertex Cover of this graph?
        #    Wait. If we remove y, the triple (x,y,z) is gone.
        #    If we remove z, the triple is gone.
        #    So yes, for each pair (y,z) with x+y=z, we must remove y or z.
        #    Also for y+z=x, we must remove y or z.
        #    So we have a graph H on S. Find Min Vertex Cover of H.
        #    If Min VC size is 0, add x.
        #    If Min VC size is 1, we can swap (remove 1, add x). Size neutral.
        #    If Min VC size > 1, cost is too high (size decreases).
        
        # So the search space is:
        # - Moves with gain +1 (VC size 0)
        # - Moves with gain 0 (VC size 1)
        
        # We will use a hill-climbing algorithm that accepts +1 moves always, 
        # and 0 moves with some probability to escape plateaus.
        
        best_S = set()
        best_size = -1
        
        # Initial Candidates
        candidates = []
        
        # 1. Odds
        candidates.append(set(x for x in vals if x % 2 != 0))
        # 2. Evens
        candidates.append(set(x for x in vals if x % 2 == 0))
        # 3. Positive numbers
        candidates.append(set(x for x in vals if x > 0))
        # 4. Negative numbers
        candidates.append(set(x for x in vals if x < 0))
        # 5. Large absolute values
        sorted_vals = sorted(vals, key=lambda x: -abs(x))
        candidates.append(set(sorted_vals[:n//2]))
        # 6. Small absolute values
        candidates.append(set(sorted_vals[:n//2])) # Wait, sorted_vals is sorted by -abs, so first n//2 are largest.
        # Let's make one with smallest abs
        candidates.append(set(sorted(vals, key=lambda x: abs(x))[:n//2]))
        
        # 7. Random
        rng = Random(42)
        for _ in range(5):
            random_set = set(rng.sample(vals, n//2))
            candidates.append(random_set)
            
        # Ensure candidates are sum-free by pruning? 
        # No, the local search will fix them if we start with valid ones.
        # But if we start with invalid ones, we need to make them valid first.
        # Let's enforce validity in the initial step by removing conflicts.
        
        def make_valid(S):
            # Remove elements that cause conflicts.
            # Simple: iterate, if a+b=c in S, remove c (or a, or b).
            # To maximize size, we should remove the one that causes most future conflicts?
            # For now, just remove c (the sum) as it's often "larger" or derived.
            # Actually, removing any one makes that triple valid.
            # Let's just return a valid subset by checking and removing.
            S = set(S)
            changed = True
            while changed:
                changed = False
                S_list = list(S)
                for i in range(len(S_list)):
                    a = S_list[i]
                    for j in range(i, len(S_list)):
                        b = S_list[j]
                        if a + b in S:
                            # Remove a+b
                            S.remove(a+b)
                            changed = True
                            break
                    if changed:
                        break
            return S

        # Refine candidates to be valid
        valid_candidates = []
        for c in candidates:
            vc = make_valid(c)
            if len(vc) > len(best_S):
                best_S = vc
                best_size = len(vc)
            valid_candidates.append(vc)

        # Local Search
        # We will run local search on each candidate, or just the best one?
        # Running on all is safer.
        
        # Time allocation per candidate
        t_per_candidate = time_limit / len(valid_candidates)
        
        for S in valid_candidates:
            if time.time() > deadline:
                break
                
            # Local Search Loop
            # Accept +1 moves always. Accept 0 moves with prob 0.1 (or annealing).
            # Stop if no +1 moves found in last K iterations?
            
            # To speed up VC calculation, we precompute neighbors?
            # For a fixed x, the graph H depends on x.
            # H nodes are S. Edges (y,z) if x+y=z or y+z=x.
            
            # Optimization: 
            # Instead of finding Min VC from scratch (which is NP-hard but small here),
            # we can use a greedy approximation for VC to estimate cost.
            # If Greedy VC size is 1, cost is likely 1.
            # If Greedy VC size is 0, cost is 0.
            # If Greedy VC size > 1, cost is likely > 1.
            # Since we only care about cost 0 and 1, we can just check:
            # 1. Are there NO conflicts? -> Cost 0.
            # 2. Are there conflicts, but they can be resolved by removing 1 element?
            #    This means all conflict pairs share a common element? 
            #    Or there is only 1 conflict pair?
            #    If there are multiple disjoint conflict pairs, cost >= 2.
            #    So:
            #    - Collect all conflict pairs P.
            #    - If P is empty: Cost 0.
            #    - Else, find the element in S that appears in the most pairs?
            #      If removing it covers all pairs, Cost 1.
            #      Else Cost > 1.
            
            # This is much faster than exact VC.
            
            def get_cost_and_remover(x, S):
                # Returns (cost, remover)
                # cost: min number of elements to remove from S to allow x
                # remover: the element to remove if cost==1, None if cost==0
                S_list = list(S)
                pairs = []
                
                # Check 2x
                if 2*x in S:
                    # Must remove 2x.
                    # But wait, if 2x in S, and we remove 2x, is it enough?
                    # We must also check other conflicts.
                    # So 2x is a "forced" removal candidate.
                    # Let's just add it to pairs as a "forced" node?
                    # Actually, if 2x in S, we MUST remove 2x.
                    # So if we remove 2x, we then check if other conflicts remain.
                    # This complicates the "single remover" logic.
                    
                    # Let's stick to the pair logic.
                    # Conflict: x+x=2x. Pair (x,x)? No, x is not in S.
                    # The constraint is: 2x not in S.
                    # So if 2x in S, we must remove 2x.
                    # This is a "singleton" constraint.
                    
                    # Let's handle singletons separately.
                    must_remove = set()
                    if 2*x in S:
                        must_remove.add(2*x)
                    
                    # Pairs
                    for u in S_list:
                        # u + x = v
                        v = u + x
                        if v in S and v != u: # v!=u handled by 2x case? No, u+x=v. If u=x, 2x=v.
                            # If u=x, then 2x=v. We already handled 2x.
                            # So assume u != x.
                            # Add pair (u, v)
                            if u < v:
                                pairs.append((u, v))
                            else:
                                pairs.append((v, u))
                        
                        # u + v = x
                        v = x - u
                        if v in S:
                            if v == u:
                                # 2u = x. Must remove u.
                                must_remove.add(u)
                            else:
                                if u < v:
                                    pairs.append((u, v))
                                else:
                                    pairs.append((v, u))
                    
                    # Remove duplicates
                    pairs = list(set(pairs))
                    
                    # If must_remove is not empty, we must remove those.
                    # If |must_remove| > 1, cost > 1.
                    if len(must_remove) > 1:
                        return 2, None
                    if len(must_remove) == 1:
                        mr = next(iter(must_remove))
                        # Check if removing mr covers all pairs.
                        # A pair (a,b) is covered if a==mr or b==mr.
                        for p in pairs:
                            if p[0] != mr and p[1] != mr:
                                return 2, None
                        # If all pairs covered, cost is 1 (remove mr).
                        # But wait, are there other conflicts not involving mr?
                        # We checked all pairs. If all involve mr, then removing mr fixes them.
                        return 1, mr
                    
                    # must_remove is empty.
                    if not pairs:
                        return 0, None
                    
                    # Find if a single element covers all pairs.
                    # Count frequency of each node in pairs.
                    freq = {}
                    for p in pairs:
                        freq[p[0]] = freq.get(p[0], 0) + 1
                        freq[p[1]] = freq.get(p[1], 0) + 1
                    
                    # If max freq == len(pairs), that node covers all.
                    max_count = max(freq.values())
                    if max_count == len(pairs):
                        # Find the node
                        for node, cnt in freq.items():
                            if cnt == max_count:
                                return 1, node
                    return 2, None

            # Local Search Loop
            current_S = set(S)
            current_size = len(current_S)
            no_improve_count = 0
            max_no_improve = 100
            p_swap = 0.2
            
            while time.time() < deadline and no_improve_count < max_no_improve:
                improved = False
                # Shuffle V\S to randomize order
                to_try = [x for x in vals if x not in current_S]
                random.shuffle(to_try)
                
                for x in to_try:
                    if time.time() > deadline:
                        break
                    
                    cost, remover = get_cost_and_remover(x, current_S)
                    
                    if cost == 0:
                        current_S.add(x)
                        current_size += 1
                        improved = True
                        no_improve_count = 0
                        # After adding, continue trying to add more? 
                        # Yes, the loop continues.
                    elif cost == 1:
                        # Swap
                        if random.random() < p_swap:
                            current_S.remove(remover)
                            current_S.add(x)
                            # Size unchanged
                            # Don't reset no_improve_count for swaps? 
                            # Or reset partially?
                            # Let's not reset, to ensure we don't loop forever on swaps.
                            improved = True # Mark as activity
                            
                if not improved:
                    no_improve_count += 1
                else:
                    # If we only swapped, no_improve_count should maybe increase?
                    # If we added, it's a real improvement.
                    # Let's distinguish.
                    pass

            if current_size > best_size:
                best_size = current_size
                best_S = set(current_S)

        return best_S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback: just odds
            vals = instance(size, seed)
            results.append(sorted(set(x for x in vals if x % 2 != 0)))
            continue
            
        vals = instance(size, seed)
        # Allocate remaining time
        remaining = deadline - time.time()
        # Give 1/3 of remaining to each? No, sequential.
        # Let's give 40% of remaining to first, 30% to second, 30% to third?
        # Or just a fixed chunk?
        # 120s total available.
        # Let's use 40s for each instance.
        t_limit = min(remaining, 40.0)
        
        S = solve_instance(vals, t_limit)
        
        # Verify
        if not is_sum_free(S):
            S = set(x for x in vals if x % 2 != 0)
            
        results.append(sorted(S))

    return results

