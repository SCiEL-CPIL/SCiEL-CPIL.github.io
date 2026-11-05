import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        
        # Map values to indices for faster lookup if needed, but sets are fine for n=240
        # Build Hyperedges: list of tuples (a, b, c) such that a+b=c
        # We only care about distinct elements? No, a=b is allowed.
        # So (a, a, 2a) is a hyperedge if 2a in V.
        # (a, b, a+b) is a hyperedge if a!=b and a+b in V.
        
        hyperedges = []
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                hyperedges.append((a, a, 2 * a))
            for j in range(i + 1, len(V_list)):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    hyperedges.append((a, b, c))
        
        # Precompute for each vertex, the list of hyperedges it is part of
        # And for each vertex v, and each other vertex u, do they conflict?
        # Conflict depends on S. 
        # Let's stick to the "Blockers" approach.
        
        # For fast blocker calculation:
        # For a given v, which pairs (u, w) in S would form a violation with v?
        # If v is the sum: v = u + w. Need u, w in S.
        # If v is an addend: v + u = w. Need u, w in S.
        
        # Let's precompute neighbors for each v in the hypergraph sense.
        # adj[v] = list of (u, w) such that {v, u, w} is a hyperedge.
        # Note: (a,a,2a) has u=a, w=2a. But a is v. So (v, v, 2v).
        
        adj = {v: [] for v in V_set}
        for (a, b, c) in hyperedges:
            # a, b, c are in V_set
            # Add to adj of a, b, c
            if a == b:
                # (a, a, 2a)
                # If v=a, others are a, 2a. But a is v.
                # So if we add a, we need to check if 2a is in S? 
                # No, the violation is a+a=2a. If a in S and 2a in S, violation.
                # So if we try to add v=a, and 2a in S, we have a conflict?
                # Yes, because a (new) + a (old) = 2a (old).
                # Wait, if a is already in S, we don't "add" it.
                # We only consider v not in S.
                # So if v=a, and 2a in S, then adding a creates violation (a, a, 2a).
                # The "blocker" is 2a? No, the violation involves a (new), a (old? no, a is new), 2a (old).
                # Actually, if a is NOT in S, and we add it, we need another a in S to form a+a.
                # But sets don't have duplicates. So a+a=2a requires TWO copies of a?
                # The problem says "a, b, c in it". Sets have unique elements.
                # So a+b=c with a=b implies 2a=c.
                # If a is in S, and 2a is in S, then we have a, a, 2a?
                # No, we have element a and element 2a.
                # The condition is "no a, b, c in S such that a+b=c".
                # If S={a, 2a}, is it sum-free?
                # Check: a+a=2a. Are a, a, 2a in S?
                # "a, b, c in it" usually implies elements of the set.
                # Does "a, b, c" mean distinct elements?
                # "a = b allowed" is in the prompt.
                # So if S={1, 2}, 1+1=2. 1 is in S, 1 is in S, 2 is in S.
                # So {1, 2} is NOT sum-free.
                # So yes, 2a=c is a violation if a, 2a in S.
                
                # So for v=a, if 2a in S, adding a creates violation.
                # Blocker is 2a.
                adj[a].append((2 * a, 2 * a)) # Just store the other elements?
                # Actually, let's store the other elements in the hyperedge.
                # For (a, a, 2a):
                # If v=a, others are a, 2a. But a is v. So we need to know if 2a is in S?
                # And do we need another a? No, a is the one being added.
                # The violation is a(new) + a(old?) = 2a.
                # Wait, if a is NOT in S, there is no "old a".
                # So we cannot form a+a unless a is already in S?
                # No, if we ADD a, then a is in S.
                # But we need TWO a's to sum to 2a?
                # The definition is: no a, b, c in S such that a+b=c.
                # If S={a, 2a}, we pick a=a, b=a, c=2a.
                # Are a, b, c in S?
                # a is in S. b is in S (same element). c is in S.
                # So YES, it's a violation.
                # So if we add v=a, and 2a is in S, we have a violation.
                # The "blocker" is 2a?
                # If we remove 2a, the violation is gone.
                # So blocker is 2a.
                
            else:
                # a, b, c distinct? Not necessarily, but a!=b here.
                # If c == a? a+b=a => b=0. 0 not in V.
                # If c == b? a+b=b => a=0.
                # So a,b,c are distinct if a!=b and a,b != 0.
                
                # For v=a: others are b, c.
                adj[a].append((b, c))
                # For v=b: others are a, c.
                adj[b].append((a, c))
                # For v=c: others are a, b.
                adj[c].append((a, b))
            
            # Handle a==b case properly in adj
            # For (a, a, 2a):
            # v=a: others are a, 2a.
            # But we only care about elements in S \ {v}.
            # If we add v=a, we need to check if there is ANOTHER a in S? No, sets.
            # So the "other a" is v itself?
            # The violation is v + v = 2v.
            # This requires v in S and 2v in S.
            # Since v is being added, v will be in S.
            # So we only need to check if 2v is in S.
            # So for v=a, the blocker is 2a.
            # My adj[a].append((2*a, 2*a)) is confusing.
            # Let's just handle the 2a case separately or store "partner" logic.
            
            # Let's restart adj construction cleanly.
            # For each v, store a list of "conflict pairs" (u, w).
            # If v is added, and BOTH u and w are in S, violation.
            # Exception: if u == v, then we only need w in S?
            # Case a=a: v=a. Violation a+a=2a.
            # Needs: a in S (yes, adding it) and 2a in S.
            # So if 2a in S, conflict.
            # This is a "single blocker" case?
            # No, it's a condition: if 2a in S, block.
            
            # Let's simplify:
            # For each v, precompute `blocks[v]` = set of u such that {v, u} is part of a hyperedge?
            # No, hyperedges are size 3.
            
            # Let's just iterate hyperedges on the fly for blocker calculation.
            # N is small.
            
        def get_blockers(v, S):
            """Return set of u in S that must be removed to add v."""
            blockers = set()
            # Check all hyperedges involving v
            # We can iterate all hyperedges, but that's slow.
            # Better: iterate adj[v]
            # But adj construction above was messy.
            
            # Let's build a clean adjacency:
            # neigh[v] = list of (u, w) such that {v, u, w} is a hyperedge.
            # If u == v, then we only need w in S.
            # If u != v, we need both u and w in S.
            
            # Rebuild adj correctly
            # (a, a, 2a):
            # v=a: (a, 2a). u=a (self), w=2a.
            # v=2a: (a, a). u=a, w=a.
            
            # (a, b, c) with a!=b:
            # v=a: (b, c)
            # v=b: (a, c)
            # v=c: (a, b)
            
            # Let's assume adj is built correctly.
            # I will rebuild it in the function for clarity.
            
            for (u, w) in adj[v]:
                if u == v:
                    # Self-loop in logic: v+v=w.
                    # Need w in S.
                    if w in S:
                        blockers.add(w)
                else:
                    # Need u in S AND w in S.
                    # But wait, if u == w?
                    # (a, b, a+b). If a+b == b => a=0.
                    # If a+b == a => b=0.
                    # So u, w distinct from v, and u != w unless 0 involved.
                    if u in S and w in S:
                        # Both u and w are blockers?
                        # If we remove u, violation gone.
                        # If we remove w, violation gone.
                        # So we only need to remove ONE of them?
                        # No, the violation is u + w = v (or similar).
                        # If we keep both, violation exists.
                        # So we must remove at least one.
                        # Which one?
                        # To maximize size, we should remove the one that causes fewer OTHER violations?
                        # For now, add both to blockers? No, that's too aggressive.
                        # If we add both to blockers, we remove 2, gain 1. Net -1.
                        # But maybe we only need to remove 1.
                        # This makes "get_blockers" complex.
                        
                        # Simplification: 
                        # If we identify a violation (v, u, w), we can choose to remove u OR w.
                        # Let's add both to a "conflict set" and resolve later?
                        # Or, conservatively, assume we must remove both?
                        # That would underestimate the gain.
                        
                        # Better: Return the list of conflicts, and let the solver decide.
                        # For this heuristic, let's say:
                        # If (u, w) are both in S, we have a conflict.
                        # We must remove u OR w.
                        # Let's add u to blockers. (Arbitrary choice)
                        # Or, better, add the one with higher degree?
                        
                        # For simplicity in this fast loop:
                        # Add both? No.
                        # Let's just add u.
                        blockers.add(u)
                        # And w?
                        # If we remove u, is the conflict gone? Yes.
                        # So we don't need to remove w.
                        # So blockers.add(u) is sufficient to "fix" this specific hyperedge.
                        # But what if u is in another conflict?
                        
                        # Let's refine: 
                        # We want to find a set R of vertices to remove such that S \ R + {v} is independent.
                        # This is a hitting set problem on the conflicts.
                        # Given small N, we can just try to remove the "most conflicting" vertex.
                        
                        pass # Logic handled in main loop

            return blockers

        # The above get_blockers is flawed for the "remove one of two" case.
        # Let's implement a proper local search step.
        
        def try_add(v, S):
            """
            Attempts to add v to S.
            Returns (new_S, gain) if successful, else (None, 0).
            """
            if v in S:
                return S, 0
            
            # Find all hyperedges that would be violated if v is added.
            # A hyperedge (a,b,c) is violated if v is one of a,b,c and the other two are in S.
            # Let Conflicts = list of (u, w) pairs such that {v, u, w} is a hyperedge and u, w in S.
            # Note: if u == v, then we only need w in S.
            
            conflicts = []
            for (u, w) in adj[v]:
                if u == v:
                    if w in S:
                        conflicts.append((w, w)) # Special case: remove w
                else:
                    if u in S and w in S:
                        conflicts.append((u, w))
            
            if not conflicts:
                S.add(v)
                return S, 1
            
            # We have conflicts.
            # We need to remove a set R of vertices from S such that for every (u, w) in conflicts:
            # if u == w: u in R.
            # if u != w: u in R or w in R.
            # We want to minimize |R|.
            # Since N is small, we can try to remove the vertex that appears in the most conflicts.
            
            # Count frequency of each vertex in conflicts
            freq = {}
            for (u, w) in conflicts:
                if u == w:
                    freq[u] = freq.get(u, 0) + 1
                else:
                    freq[u] = freq.get(u, 0) + 1
                    freq[w] = freq.get(w, 0) + 1
            
            # Try removing the vertex with highest frequency
            if not freq:
                return S, 0
                
            max_freq = max(freq.values())
            candidates = [k for k, v in freq.items() if v == max_freq]
            
            # Try each candidate
            best_S = S
            best_gain = 0
            
            for rem in candidates:
                trial_S = set(S)
                trial_S.discard(rem)
                trial_S.add(v)
                
                # Check if trial_S is valid (no conflicts remaining)
                # We only removed one vertex. Did it resolve all conflicts?
                # If a conflict was (u, w) with u != w, and we removed u, it's resolved.
                # If we removed w, it's resolved.
                # So if we pick a vertex that is in ALL conflicts, it resolves all.
                # If not, some conflicts remain.
                
                # Let's verify trial_S is sum-free.
                # Fast check:
                valid = True
                for (u, w) in conflicts:
                    if u == w:
                        if u in trial_S:
                            valid = False
                            break
                    else:
                        if u in trial_S and w in trial_S:
                            valid = False
                            break
                
                if valid:
                    gain = 1 - 1 # Added v, removed rem. Net 0.
                    # But wait, if we removed rem, we might have freed up space?
                    # No, size is same.
                    # However, if gain > 0, we prefer it.
                    # Here gain is 0.
                    # We only accept if gain > 0 OR (gain == 0 and random accept).
                    # For now, let's only accept gain > 0.
                    # If gain == 0, we might cycle.
                    
                    # Actually, if we can add v and remove rem, and the new set has FEWER total conflicts,
                    # it might be better. But we don't track that easily.
                    
                    # Let's look for Gain > 0 cases first.
                    # Gain > 0 implies we added v and removed 0 vertices.
                    # That means conflicts was empty.
                    # So this branch (conflicts not empty) always has gain <= 0.
                    
                    # So we should only enter this branch if we are willing to do 0-gain moves?
                    # Or if we can remove 1 vertex and add 2? (2-opt)
                    
                    # Let's stick to 1-opt (Add 1, Remove 0) for positive gain.
                    # And 1-swap (Add 1, Remove 1) for neutral moves (with probability).
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_S = trial_S
            
            # If best_gain > 0, return it.
            # If best_gain == 0, return with probability p.
            
            return best_S, best_gain

        # The logic above is a bit tangled. Let's simplify the Local Search.
        
        # 1. Initialize S with a greedy fill.
        # 2. Loop:
        #    a. Try to add a random v not in S.
        #    b. If it conflicts, try to remove one blocker.
        #    c. If size increases, accept.
        #    d. If size same, accept with prob 0.1.
        
        def local_search(S, V_set, budget, rng):
            t_start = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            
            # Precompute adj for this instance
            # (Already done in outer scope? No, adj depends on V_set)
            # I need to build adj inside solve_instance.
            
            while time.time() - t_start < budget:
                # Pick random v not in S
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break
                v = rng.choice(remaining)
                
                # Get conflicts
                conflicts = []
                for (u, w) in adj[v]:
                    if u == v:
                        if w in S:
                            conflicts.append((w, w))
                    else:
                        if u in S and w in S:
                            conflicts.append((u, w))
                
                if not conflicts:
                    S.add(v)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    continue
                
                # Try to resolve by removing one vertex
                # Find vertex in S that is in the most conflicts
                freq = {}
                for (u, w) in conflicts:
                    if u == w:
                        freq[u] = freq.get(u, 0) + 1
                    else:
                        freq[u] = freq.get(u, 0) + 1
                        freq[w] = freq.get(w, 0) + 1
                
                if not freq:
                    continue
                    
                max_f = max(freq.values())
                candidates = [k for k, c in freq.items() if c == max_f]
                rem = rng.choice(candidates)
                
                # Try removing rem
                trial_S = set(S)
                trial_S.discard(rem)
                trial_S.add(v)
                
                # Check if valid
                valid = True
                for (u, w) in conflicts:
                    if u == w:
                        if u in trial_S:
                            valid = False
                            break
                    else:
                        if u in trial_S and w in trial_S:
                            valid = False
                            break
                
                if valid:
                    # Size is same (added 1, removed 1)
                    # Accept with probability 0.2 to explore
                    if rng.random() < 0.2:
                        S = trial_S
                    # Update best if size is larger (not possible here)
                    
            return best_S

        # Build adj
        adj = {v: [] for v in V_set}
        for (a, b, c) in hyperedges:
            if a == b:
                # (a, a, 2a)
                # v=a: (a, 2a) -> self, 2a
                adj[a].append((a, 2 * a))
                # v=2a: (a, a)
                adj[2 * a].append((a, a))
            else:
                adj[a].append((b, c))
                adj[b].append((a, c))
                adj[c].append((a, b))
        
        # Initial Solutions
        best_S = set()
        best_size = 0
        rng = random.Random(seed_off)
        
        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        if len(S_odds) > best_size:
            best_size = len(S_odds)
            best_S = S_odds
            
        # 2. Greedy by Degree (fewest conflicts first)
        # Conflict count: number of hyperedges containing v
        deg = {v: 0 for v in V_set}
        for (a, b, c) in hyperedges:
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1
            
        S_greedy = set()
        for v in sorted(V_set, key=lambda x: (deg[x], abs(x))):
            # Check if v can be added
            can_add = True
            for (u, w) in adj[v]:
                if u == v:
                    if w in S_greedy:
                        can_add = False
                        break
                else:
                    if u in S_greedy and w in S_greedy:
                        can_add = False
                        break
            if can_add:
                S_greedy.add(v)
                
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_S = S_greedy
            
        # 3. Local Search from best_S
        budget_ls = budget * 0.8
        best_S = local_search(best_S, V_set, budget_ls, rng)
        
        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results
