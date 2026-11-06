import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_mis(V, time_limit, rng):
        """
        Solves Maximum Independent Set on the conflict graph of V.
        Conflict graph: u ~ v if u + v in V.
        """
        n = len(V)
        if n == 0:
            return []
        
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(V)}
        
        # Build adjacency list
        # adj[i] is a set of indices j such that V[i] + V[j] in V
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            vi = V[i]
            for j in range(i, n):
                vj = V[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # i and j conflict because i+j is in V
                    adj[i].add(j)
                    adj[j].add(i)
                    # Note: i and k conflict? 
                    # If i+k in V, then edge (i,k).
                    # If j+k in V, then edge (j,k).
                    # The condition for sum-free S is: no a,b,c in S with a+b=c.
                    # This is equivalent to: for all a,b in S, a+b not in S.
                    # This is equivalent to: S is an independent set in graph G where (u,v) in E iff u+v in V.
                    # Proof:
                    # If S is independent, then for all u,v in S, (u,v) not in E, so u+v not in V.
                    # Since S subset V, u+v not in V implies u+v not in S.
                    # So no a,b,c in S with a+b=c.
                    # Conversely, if S is sum-free, then for all u,v in S, u+v not in S.
                    # Since S subset V, u+v not in S does NOT imply u+v not in V.
                    # Wait.
                    # If u+v in V but u+v not in S, is S sum-free?
                    # Yes. The condition is a+b=c with a,b,c IN S.
                    # So if u+v is in V but not in S, it's fine.
                    # My graph definition: (u,v) in E if u+v in V.
                    # If S is independent in this graph, then for all u,v in S, u+v not in V.
                    # This is a STRONGER condition than sum-free.
                    # Sum-free allows u+v to be in V, as long as u+v is NOT in S.
                    
                    # Let's re-read the definition carefully.
                    # "no a + b = c with a, b, c in it"
                    # So if u, v in S, and u+v in V, we just need u+v NOT in S.
                    
                    # So the problem is NOT Maximum Independent Set on the graph defined by u+v in V.
                    # It is: Select S subset V such that for all u,v in S, u+v not in S.
                    
                    # This is a 3-uniform hypergraph independent set problem?
                    # Or we can transform it.
                    
                    # Let's check the current program's logic.
                    # It uses `is_sum_free` which checks `a+b in S`.
                    # And `count_conflicts` counts conflicts based on `a+b in S`.
                    
                    # So my MIS reduction was flawed.
                    # The conflict is not between u and v directly, but the triple (u, v, u+v).
                    # We cannot have u, v, and u+v all in S.
                    
                    # This is a "3-SAT" like problem or Independent Set in a 3-uniform hypergraph.
                    # However, we can still use local search.
                    
                    pass

        # Correct Approach for Local Search:
        # State: S (set of values)
        # Objective: Maximize |S|
        # Constraint: For all a,b in S, a+b not in S.
        
        # We will use a penalty function:
        # Score = |S| - lambda * Conflicts
        # Conflict count: number of pairs (a,b) in S such that a+b in S.
        
        # Initialization: Greedy
        # Start with empty set. Add elements one by one if they don't create a conflict.
        # To be greedy, we might sort by something.
        
        S = set()
        S_list = []
        
        # Helper to check if x can be added to S
        def can_add(x, S_set, S_list):
            for a in S_list:
                if (x + a) in S_set or (x - a) in S_set:
                    return False
            return True

        # Greedy Init: Try all permutations? No.
        # Just iterate through V.
        # Maybe sort V to improve greedy?
        # Often, taking large absolute values first helps? Or small?
        # Let's try multiple greedy inits.
        
        best_S = set()
        
        for init_iter in range(5):
            curr_S = set()
            curr_list = []
            candidates = list(V)
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, curr_S, curr_list):
                    curr_S.add(x)
                    curr_list.append(x)
            
            if len(curr_S) > len(best_S):
                best_S = curr_S

        # Local Search (Simulated Annealing)
        S = best_S
        S_list = list(S)
        
        def count_conflicts(S_set, S_list):
            count = 0
            for i, a in enumerate(S_list):
                for b in S_list[i+1:]: # Check pairs a,b. If a+b in S, conflict.
                    if a + b in S_set:
                        count += 1
                # Also check a+a
                if 2*a in S_set:
                    count += 1
            return count

        # Initial penalty
        current_conflicts = count_conflicts(S, S_list)
        current_score = len(S) - 10.0 * current_conflicts
        
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        # Precompute neighbors for faster conflict check?
        # When adding x, we check against all a in S.
        # When removing x, we remove conflicts involving x.
        
        time_limit_abs = start_time + time_limit
        
        while time.time() < time_limit_abs:
            if T < T_min:
                break
                
            # Pick a random move
            # 1. Add a random element from V \ S
            # 2. Remove a random element from S
            # 3. Swap?
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add
                candidates = [x for x in V if x not in S]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                
                # Check conflicts if we add x
                new_conflicts_delta = 0
                for a in S_list:
                    if (x + a) in S or (x - a) in S:
                        # (x+a) in S means x,a,x+a form a conflict.
                        # (x-a) in S means a, x-a, x form a conflict.
                        new_conflicts_delta += 1
                if 2*x in S:
                    new_conflicts_delta += 1
                    
                delta_score = 1 - 10.0 * new_conflicts_delta
                
                if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                    S.add(x)
                    S_list.append(x)
                    current_score += delta_score
                    current_conflicts += new_conflicts_delta
            else:
                # Try to remove
                if not S_list:
                    continue
                x = rng.choice(S_list)
                
                # Count conflicts involving x
                conflicts_involved = 0
                for a in S_list:
                    if a == x:
                        continue
                    if (x + a) in S or (x - a) in S:
                        conflicts_involved += 1
                if 2*x in S:
                    conflicts_involved += 1
                    
                # Note: A conflict (a,b,c) is counted once in the global count?
                # My count_conflicts iterates pairs.
                # If I remove x, I remove all pairs involving x.
                # delta_conflicts = -conflicts_involved
                # delta_score = -1 - 10.0 * (-conflicts_involved) = -1 + 10.0 * conflicts_involved
                
                delta_score = -1 + 10.0 * conflicts_involved
                
                if delta_score > 0 or rng.random() < np.exp(delta_score / T):
                    S.remove(x)
                    S_list.remove(x)
                    current_score += delta_score
                    current_conflicts -= conflicts_involved
            
            T *= cooling
            
            # Optional: If we have a valid set, try to improve it specifically?
            # The SA should handle it.
            
        # Final polish: Greedy add
        improved = True
        while improved and time.time() < time_limit_abs:
            improved = False
            for x in V:
                if x not in S and can_add(x, S, S_list):
                    S.add(x)
                    S_list.append(x)
                    improved = True
        
        return list(S)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        best_size = 0
        best_set = []
        
        # Allocate time
        if size <= 60:
            base_time = 20.0
        elif size <= 120:
            base_time = 30.0
        else:
            base_time = 50.0
            
        # Run multiple restarts of SA
        # Since SA is stochastic, restarts help.
        # We will run it for a fixed time budget per restart.
        
        restarts = 5
        time_per_restart = base_time / restarts
        
        for r in range(restarts):
            if time.time() > start_time + DEADLINE - 5:
                break
                
            remaining = min(time_per_restart, DEADLINE - (time.time() - start_time))
            if remaining < 1.0:
                break
                
            rng = random.Random(seed + r * 12345)
            res = solve_mis(V, remaining, rng)
            
            if len(res) > best_size:
                best_size = len(res)
                best_set = res
                
        results.append(best_set)

    return results
