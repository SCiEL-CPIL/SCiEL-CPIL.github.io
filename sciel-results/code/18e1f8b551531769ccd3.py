import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """Enumerate all maximal cliques of adj using Bron-Kerbosch with pivot."""
        n = len(adj)
        cliques = []
        
        # Convert to bitmasks for speed? n<=40 fits in 64-bit int.
        # But Python sets are okay for n=40.
        # Let's stick to sets for clarity and sufficient speed.
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(frozenset(R))
                return
                
            # Choose pivot u from P ∪ X to minimize branching
            # Pivot should be in P ∪ X
            P_union_X = P | X
            if not P_union_X:
                return
            
            # Simple pivot: pick any vertex from P_union_X
            # Better: pick vertex with max degree in P
            u = None
            max_deg = -1
            for v in P_union_X:
                deg = len(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    u = v
            
            # Branch on P \ N(u)
            candidates = P - adj[u]
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
                
        bk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        
        # Enumerate maximal cliques
        all_cliques = bron_kerbosch(adj)
        
        # Filter: size >= 2
        relevant_cliques = [c for c in all_cliques if len(c) >= 2]
        
        if not relevant_cliques:
            return []
        
        # Precompute incidence: vertex -> list of clique indices
        # This helps in checking validity quickly?
        # Actually, for local search, we need to check if S hits all cliques.
        # A clique C is hit by S if C & S is not empty.
        
        best_solution = None
        best_size = float('inf')
        
        # Multiple restarts with different random tie-breakings
        # We will run greedy with random priority shuffles
        num_restarts = 50 # Adjust based on time
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Greedy Hitting Set
            # 1. Map each vertex to the set of clique indices it covers
            #    Actually, we need to pick vertices to cover cliques.
            #    Let's maintain a list of uncovered cliques.
            
            # To allow random tie-breaking, we can shuffle the vertices or add random noise to counts.
            # Let's add random noise to the count of cliques covered by each vertex.
            
            # Initial state: all cliques uncovered.
            # We represent cliques as sets of vertices.
            
            # Efficient Greedy:
            # For each vertex v, count how many *currently uncovered* cliques it hits.
            # This is dynamic.
            
            # Simpler Greedy:
            # 1. Count total cliques hit by each vertex (static).
            # 2. Pick max.
            # 3. Remove covered cliques.
            # 4. Recount.
            
            # Given small n, recounting is fine.
            
            uncovered = list(relevant_cliques)
            solution = set()
            
            while uncovered:
                if time.time() > deadline - 2.0:
                    break
                
                # Calculate score for each vertex not in solution
                # Score = number of uncovered cliques containing v
                scores = {}
                for v in range(n):
                    if v in solution:
                        continue
                    count = 0
                    for c in uncovered:
                        if v in c:
                            count += 1
                    if count > 0:
                        # Add small random noise for tie-breaking diversity
                        scores[v] = count + random.random() * 0.01
                
                if not scores:
                    # Should not happen if uncovered is not empty and graph is valid
                    break
                
                # Pick vertex with highest score
                best_v = max(scores, key=scores.get)
                solution.add(best_v)
                
                # Remove covered cliques
                new_uncovered = []
                for c in uncovered:
                    if best_v not in c:
                        new_uncovered.append(c)
                uncovered = new_uncovered
            
            # Local Search on the greedy solution
            current_sol = set(solution)
            
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                
                # 1. Try to remove any vertex
                for v in list(current_sol):
                    if time.time() > deadline - 1.0:
                        break
                    # Check if current_sol - {v} is valid
                    # i.e., does every clique in relevant_cliques intersect current_sol - {v}?
                    # Equivalent: Is there a clique C such that C & current_sol == {v}?
                    # If such C exists, v is essential.
                    essential = False
                    for c in relevant_cliques:
                        if c & current_sol == {v}:
                            essential = True
                            break
                    if not essential:
                        current_sol.remove(v)
                        improved = True
                        # Restart loop to check for more removals
                        break 
                
                if improved:
                    continue
                
                # 2. Try 2-exchange: swap v in sol with u not in sol
                # Accept if it allows a subsequent removal
                sol_list = list(current_sol)
                for v in sol_list:
                    if time.time() > deadline - 1.0:
                        break
                    for u in range(n):
                        if u in current_sol:
                            continue
                        # Try swap
                        new_sol = set(current_sol)
                        new_sol.remove(v)
                        new_sol.add(u)
                        
                        # Check if new_sol is valid
                        valid = True
                        for c in relevant_cliques:
                            if not (c & new_sol):
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # If valid, check if we can remove any vertex from new_sol
                        # If yes, accept the swap and the removal
                        for w in list(new_sol):
                            if time.time() > deadline - 1.0:
                                break
                            # Check if new_sol - {w} is valid
                            # Equivalent: Is there a clique C such that C & new_sol == {w}?
                            essential = False
                            for c in relevant_cliques:
                                if c & new_sol == {w}:
                                    essential = True
                                    break
                            if not essential:
                                # Accept swap and removal
                                current_sol = new_sol
                                current_sol.remove(w)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                # 3. Try 3-exchange: remove v, add u, add w
                # This is expensive, so only do it if we have time
                if not improved and time.time() < deadline - 3.0:
                    sol_list = list(current_sol)
                    for v in sol_list:
                        if time.time() > deadline - 3.0:
                            break
                        for u in range(n):
                            if u in current_sol:
                                continue
                            for w in range(n):
                                if w in current_sol or w == u:
                                    continue
                                # Try remove v, add u, add w
                                new_sol = set(current_sol)
                                new_sol.remove(v)
                                new_sol.add(u)
                                new_sol.add(w)
                                
                                # Check if new_sol is valid
                                valid = True
                                for c in relevant_cliques:
                                    if not (c & new_sol):
                                        valid = False
                                        break
                                if not valid:
                                    continue
                                
                                # If valid, check if we can remove any vertex from new_sol
                                # If yes, accept the swap and the removal
                                for x in list(new_sol):
                                    if time.time() > deadline - 3.0:
                                        break
                                    # Check if new_sol - {x} is valid
                                    # Equivalent: Is there a clique C such that C & new_sol == {x}?
                                    essential = False
                                    for c in relevant_cliques:
                                        if c & new_sol == {x}:
                                            essential = True
                                            break
                                    if not essential:
                                        # Accept swap and removal
                                        current_sol = new_sol
                                        current_sol.remove(x)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
            
            if len(current_sol) < best_size:
                best_size = len(current_sol)
                best_solution = set(current_sol)
            
            # If we've found a small solution, we can stop early
            if best_size <= 3:
                break
        
        return sorted(best_solution) if best_solution else []

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Not enough time, return empty (invalid but better than timeout)
            out.append([])
            continue
        solution = solve_instance(n, p, seed)
        out.append(solution)
    
    return out
