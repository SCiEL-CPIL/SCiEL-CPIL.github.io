"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit (evaluator kills at 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def solve_one(S, time_budget=30.0):
        n = len(S)
        S_set = set(S)
        elem_to_idx = {x: i for i, x in enumerate(S)}
        
        # Build conflict graph
        # i ~ j if S[i] + S[j] in S_set OR S[i] - S[j] in S_set OR S[j] - S[i] in S_set
        # Note: if S[i] == S[j], we handle via 2*S[i] check implicitly if i!=j?
        # No, i and j are distinct indices.
        # If S[i] + S[j] in S_set, they conflict.
        # If S[i] - S[j] in S_set, they conflict (since S[j] + (S[i]-S[j]) = S[i]).
        # If S[j] - S[i] in S_set, they conflict.
        
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = S[i], S[j]
                # Check sums and differences
                # a+b in S_set
                # a-b in S_set
                # b-a in S_set
                if a + b in S_set or a - b in S_set or b - a in S_set:
                    adj[i].add(j)
                    adj[j].add(i)
                
                # Handle self-conflict logic:
                # If 2*a in S_set, let k be index of 2*a.
                # Then i and k conflict.
                # This is covered if we check pairs (i, k).
                # Since we iterate all pairs, this is handled when i and k are the pair.
                
        # Local Search
        best_set = set()
        best_size = 0
        
        def try_greedy(order):
            current = set()
            for i in order:
                if not (adj[i] & current):
                    current.add(i)
            return current

        def local_search(start_set):
            current = set(start_set)
            current_size = len(current)
            improved = True
            while improved:
                improved = False
                # Try to add any vertex not in current
                # Sort candidates by degree in current? Or just iterate
                # To speed up, we can check if there's any vertex that can be added
                # If not, try swaps
                
                # 1. Try adding
                for i in range(n):
                    if i not in current and not (adj[i] & current):
                        current.add(i)
                        current_size += 1
                        improved = True
                        break # Restart loop to allow more adds
                
                if not improved:
                    # 2. Try swaps: remove u, add v
                    # For each v not in current, find neighbors in current
                    # If we remove one neighbor u, can we add v?
                    # Only if v has no other neighbors in current - {u}
                    
                    # Optimization: Only try swaps that might help?
                    # Since size is constant, we just want to change the set.
                    # We accept any valid swap to explore the space.
                    
                    # Limit swaps to avoid infinite loops? 
                    # Use a random order for swaps to break symmetry.
                    unselected = [i for i in range(n) if i not in current]
                    random.shuffle(unselected)
                    
                    for v in unselected:
                        if v in current: continue
                        neighbors_in_current = adj[v] & current
                        if not neighbors_in_current:
                            # Should have been added in step 1
                            continue
                        
                        # Try removing each neighbor
                        # Pick one neighbor randomly to try
                        u = random.choice(list(neighbors_in_current))
                        
                        # Check if v can be added if u is removed
                        # v conflicts with (neighbors_in_current - {u})?
                        if not (adj[v] & (current - {u})):
                            # Valid swap
                            current.remove(u)
                            current.add(v)
                            improved = True
                            break # Restart loop
                    
        # Multiple restarts with different orderings
        start_time = time.time()
        
        # Ordering 1: By degree ascending
        degrees = [len(adj[i]) for i in range(n)]
        order1 = sorted(range(n), key=lambda x: degrees[x])
        
        # Ordering 2: By degree descending
        order2 = sorted(range(n), key=lambda x: -degrees[x])
        
        # Ordering 3: Random
        order3 = list(range(n))
        random.shuffle(order3)
        
        # Ordering 4: By absolute value descending (heuristic for sum-free)
        order4 = sorted(range(n), key=lambda x: -abs(S[x]))
        
        best_set = set()
        
        for order in [order1, order2, order3, order4]:
            if time.time() - start_time > time_budget:
                break
                
            greedy_set = try_greedy(order)
            if len(greedy_set) > len(best_set):
                best_set = greedy_set
                
            # Local search on this greedy set
            ls_set = local_search(greedy_set)
            if len(ls_set) > len(best_set):
                best_set = ls_set
                
            # Try more random restarts if time permits
            restarts = 0
            while time.time() - start_time < time_budget and restarts < 50:
                random_order = list(range(n))
                random.shuffle(random_order)
                g_set = try_greedy(random_order)
                ls_set = local_search(g_set)
                if len(ls_set) > len(best_set):
                    best_set = ls_set
                restarts += 1

        return [S[i] for i in best_set]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            remaining = deadline - time.time()
            # Allocate more time to larger instances
            if size <= 60:
                budget = min(15.0, remaining - 5)
            elif size <= 120:
                budget = min(30.0, remaining - 5)
            else:
                budget = min(80.0, remaining - 5)
            
            result = solve_one(S, time_budget=max(5.0, budget))
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)

    return results
# EVOLVE-BLOCK-END
