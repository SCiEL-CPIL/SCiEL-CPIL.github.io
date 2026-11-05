import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj):
        """
        Enumerates all maximal cliques in the graph defined by adjacency lists.
        Uses Bron-Kerbosch with pivot.
        """
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X
            # Heuristic: choose u with max degree in P
            P_X = P.union(X)
            if not P_X:
                return
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(u)
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        # Initial call
        # P = all vertices, X = empty
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj, edges):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        # Convert to list of sets for easier manipulation
        cliques_list = [set(c) for c in cliques]
        n = len(adj)
        
        # Greedy Initialization
        def greedy_init():
            S = set()
            remaining = list(cliques_list)
            # Index cliques by vertices for speed? 
            # With small number of cliques, just filtering is fine.
            
            # Sort cliques by size descending? Or random?
            # Random restarts help.
            rng = random.Random()
            
            while remaining:
                # Pick a random uncovered clique to break ties or introduce randomness
                # Actually, standard greedy picks vertex covering most remaining.
                # Let's pick the vertex that covers the most remaining cliques.
                best_v = None
                max_covered = -1
                # Optimization: If number of remaining is small, check all vertices
                # Here n is small (30-40), cliques are small.
                for v in range(n):
                    if v in S: continue
                    count = 0
                    for c in remaining:
                        if v in c:
                            count += 1
                    if count > max_covered:
                        max_covered = count
                        best_v = v
                if best_v is None:
                    break
                S.add(best_v)
                # Remove covered cliques
                remaining = [c for c in remaining if best_v not in c]
                
            # Prune
            S = prune(S)
            return S

        def prune(S):
            S = set(S)
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    S_test = S - {v}
                    if is_valid(S_test):
                        S = S_test
                        changed = True
            return S

        def is_valid(S):
            for c in cliques_list:
                if not (S & c):
                    return False
            return True

        def uncovered_count(S):
            count = 0
            for c in cliques_list:
                if not (S & c):
                    count += 1
            return count

        def score(S):
            # Penalty for invalidity
            unc = uncovered_count(S)
            if unc > 0:
                return len(S) + 1000 * unc
            return len(S)

        # Multi-start Greedy + Local Search
        best_S = greedy_init()
        best_score = score(best_S)
        
        # Local Search: Simulated Annealing
        current_S = set(best_S)
        current_score = best_score
        
        T = 1.0
        min_T = 0.01
        cooling = 0.95
        
        # Estimate steps based on time
        # We have ~100s total. 3 instances. ~30s per instance.
        # n=40, cliques~100. Score calc is fast.
        
        while time.time() - start_time < deadline - 5:
            # Generate neighbor
            move_type = random.random()
            neighbor_S = set(current_S)
            
            if move_type < 0.33:
                # Add
                v = random.randint(0, n-1)
                neighbor_S.add(v)
            elif move_type < 0.66:
                # Remove
                if current_S:
                    v = random.choice(list(current_S))
                    neighbor_S.remove(v)
            else:
                # Swap
                if current_S:
                    v_in = random.choice(list(current_S))
                    v_out = random.randint(0, n-1)
                    if v_in != v_out:
                        neighbor_S.remove(v_in)
                        neighbor_S.add(v_out)
            
            new_score = score(neighbor_S)
            delta = new_score - current_score
            
            if delta < 0:
                current_S = neighbor_S
                current_score = new_score
                if current_score < best_score:
                    best_S = set(current_S)
                    best_score = current_score
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(2.71828, -delta / T):
                    current_S = neighbor_S
                    current_score = new_score
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return sorted(best_S)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            # Fallback: return empty or best effort? 
            # If we run out of time, we should return what we have or a valid simple set.
            # The greedy init is valid. Let's just run greedy if SA didn't start or if we are late.
            adj, edges = random_graph(n, p, seed)
            # Quick greedy
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            remaining = [set(c) for c in cliques]
            while remaining:
                best_v = None
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = sum(1 for c in remaining if v in c)
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v is None: break
                S.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            # Prune
            def is_valid_quick(S):
                for c in remaining: # Wait, remaining is updated. Need original cliques.
                    pass
            # Just return S, it's valid by construction (greedy adds until all covered)
            out.append(sorted(S))
            continue
            
        adj, edges = random_graph(n, p, seed)
        res = solve_instance(adj, edges)
        out.append(res)
        
    return out
