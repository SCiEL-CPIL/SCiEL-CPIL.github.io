import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 120.0
    margin = 5.0 # Safety margin
    
    instances = [
        (30, 0.3, 201),
        (35, 0.25, 202),
        (40, 0.2, 203)
    ]
    
    results = []
    
    for n, p, seed in instances:
        # 1. Generate Graph
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        
        # 2. Enumerate Maximal Cliques (Bron-Kerbosch with pivot)
        maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                # Choose pivot u from P union X to minimize branching
                # Heuristic: pick u with max degree in P
                # For speed, iterate P and X to find best pivot
                best_u = -1
                max_count = -1
                # Check P first, then X
                # We want to maximize |P ∩ N(u)|
                for u in P:
                    count = len(P & adj[u])
                    if count > max_count:
                        max_count = count
                        best_u = u
                for u in X:
                    count = len(P & adj[u])
                    if count > max_count:
                        max_count = count
                        best_u = u
                
                # Branch on P \ N(pivot)
                if best_u != -1:
                    candidates = [v for v in P if v not in adj[best_u]]
                else:
                    candidates = list(P)
            else:
                candidates = list(P)
            
            for v in candidates:
                R_new = R | {v}
                P_new = P & adj[v]
                X_new = X & adj[v]
                bron_kerbosch(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
                
        # Start
        # Use sets for mutability
        P_initial = set(range(n))
        bron_kerbosch(set(), P_initial, set())
        
        # 3. Prepare Hitting Set Problem
        # Filter cliques of size >= 2
        cliques = [list(c) for c in maximal_cliques if len(c) >= 2]
        
        if not cliques:
            results.append([])
            continue
        
        # Precompute vertex to clique indices
        vertex_to_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        num_cliques = len(cliques)
        
        # 4. Greedy Initialization
        S = set()
        cover_count = [0] * num_cliques
        uncovered = num_cliques
        
        while uncovered > 0:
            best_v = -1
            best_gain = -1
            for v in range(n):
                if v in S:
                    continue
                gain = 0
                for c_idx in vertex_to_cliques[v]:
                    if cover_count[c_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v == -1:
                break # Should not happen if uncovered > 0
                
            S.add(best_v)
            for c_idx in vertex_to_cliques[best_v]:
                if cover_count[c_idx] == 0:
                    uncovered -= 1
                cover_count[c_idx] += 1
        
        # 5. Simulated Annealing
        # State: S (set), cover_count, uncovered
        # We will modify S in place and track best
        
        best_S = set(S)
        best_size = len(S)
        
        # Time allocation
        # Estimate time per instance: 120s / 3 = 40s
        # But we should check global time.
        current_time = time.time()
        remaining_time = time_limit - (current_time - start_time) - margin
        if remaining_time <= 0:
            results.append(sorted(best_S))
            return results # Exit early if no time left for other instances
        
        # SA Parameters
        # We want to spend most of the remaining time on SA
        # Let's run SA until time is up.
        
        T = 1.0
        T_min = 0.001
        # Estimate steps per second.
        # One step involves iterating over neighbors of v in clique graph.
        # Avg degree in clique hypergraph?
        # Let's just run a loop with time check.
        
        rng_sa = random.Random(seed + 1000)
        
        # Helper to check validity and cost
        # We maintain cover_count and uncovered.
        
        def try_move(v, is_add):
            # Returns delta_E, new_uncovered_delta
            # We don't actually apply yet, just calculate
            # But we need to know how many cliques change state.
            
            delta_uncovered = 0
            if is_add:
                # Adding v
                for c_idx in vertex_to_cliques[v]:
                    if cover_count[c_idx] == 0:
                        delta_uncovered -= 1 # Becomes covered
            else:
                # Removing v
                for c_idx in vertex_to_cliques[v]:
                    if cover_count[c_idx] == 1:
                        delta_uncovered += 1 # Becomes uncovered
            
            return delta_uncovered

        # Main SA Loop
        # Use a stricter deadline to ensure we have time to process remaining instances
        # and to avoid timeout in the evaluator
        sa_deadline = start_time + time_limit - margin - (2.0 / 3.0) * (len(instances) - instances.index((n,p,seed))) * (time_limit / len(instances))
        # Simpler: just check if we have enough time left for this instance + remaining
        # Let's just use a dynamic check in the loop
        while time.time() - start_time < time_limit - margin - 10.0: # Reserve 10s buffer for other instances and overhead
            # Pick random vertex
            v = rng_sa.randint(0, n - 1)
            
            if v in S:
                # Try to remove
                delta_uncov = try_move(v, False)
                if delta_uncov == 0:
                    # Valid move, cost decreases by 1
                    # Always accept
                    S.remove(v)
                    for c_idx in vertex_to_cliques[v]:
                        cover_count[c_idx] -= 1
                    uncovered -= 0 # delta_uncov is 0
                    # Update best
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = set(S)
                else:
                    # Invalid move
                    # Cost change: -1 (size) + M * delta_uncov
                    M = 100 # Penalty
                    delta_E = -1 + M * delta_uncov
                    if delta_E < 0 or rng_sa.random() < pow(2.718281828, -delta_E / T):
                        # Accept
                        S.remove(v)
                        for c_idx in vertex_to_cliques[v]:
                            cover_count[c_idx] -= 1
                        uncovered += delta_uncov
            else:
                # Try to add
                delta_uncov = try_move(v, True)
                # Cost change: +1 (size) + M * delta_uncov
                # Note: delta_uncov is negative if it covers something
                M = 100
                delta_E = 1 + M * delta_uncov
                if delta_E < 0 or rng_sa.random() < pow(2.718281828, -delta_E / T):
                    # Accept
                    S.add(v)
                    for c_idx in vertex_to_cliques[v]:
                        if cover_count[c_idx] == 0:
                            uncovered -= 1
                        cover_count[c_idx] += 1
                    # If we became valid and smaller? 
                    # We added a vertex, so size increased. 
                    # But if it was invalid before, and now valid, 
                    # the penalty drop might make it "better" in SA terms.
                    # We only update best if it is valid.
                    if uncovered == 0 and len(S) < best_size:
                        best_size = len(S)
                        best_S = set(S)
            
            # Cool down
            T *= 0.999
            if T < T_min:
                T = T_min
                
        results.append(sorted(best_S))
        
    return results
