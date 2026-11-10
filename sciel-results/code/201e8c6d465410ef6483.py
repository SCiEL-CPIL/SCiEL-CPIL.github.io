import time
import random
import sys
from itertools import combinations

def construct():
    sys.setrecursionlimit(10000)

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def solve_instance(n, p, seed, time_budget):
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            return []

        m = len(cliques)
        
        # C[v] is a set of clique indices
        C = [set() for _ in range(n)]
        for i, c in enumerate(cliques):
            for v in c:
                C[v].add(i)
                
        # Precompute for speed
        C_list = [list(s) for s in C]
        C_size = [len(s) for s in C]
        
        # Build incidence matrix for cliques to vertices (for lower bounds)
        # We want to know for each clique, which vertices cover it.
        # Actually we have C[v] which is vertices -> cliques.
        # Let's build V[c] = set of vertices in clique c.
        V = [set(c) for c in cliques]
        
        def greedy_init(rng):
            uncovered = set(range(m))
            selected = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not C[v]:
                        continue
                    count = len(uncovered & C[v])
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and count > 0:
                        if rng.random() < 0.5:
                            best_v = v
                
                if best_v == -1:
                    break
                    
                selected.add(best_v)
                uncovered -= C[best_v]
                
            return selected

        def local_search(S, rng, time_limit):
            S = set(S)
            # cover_count[c] = number of selected vertices in clique c
            cover_count = [0] * m
            for v in S:
                for c in C[v]:
                    cover_count[c] += 1
            
            improved = True
            start = time.time()
            
            while improved and (time.time() - start) < time_limit:
                improved = False
                
                # 1. Removals
                for v in list(S):
                    can_remove = True
                    for c in C[v]:
                        if cover_count[c] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for c in C[v]:
                            cover_count[c] -= 1
                        S.remove(v)
                        improved = True
                        break
                
                if improved:
                    continue
                    
                # 2. Swaps
                found_swap = False
                # Try to find a swap that reduces "slack" or just maintains size while allowing future removals
                # We iterate v_out in S, and v_in not in S.
                # A swap is valid if for all c in C[v_out] where cover_count[c] == 1, c is in C[v_in].
                
                # To be more efficient, we can prioritize v_out that have many "critical" cliques (cover_count==1)
                # because moving them is risky but necessary to escape local optima.
                
                v_out_list = list(S)
                rng.shuffle(v_out_list)
                
                for v_out in v_out_list:
                    critical = [c for c in C[v_out] if cover_count[c] == 1]
                    if not critical:
                        continue # Should have been removed
                    
                    # Find best v_in
                    # v_in must cover all critical cliques.
                    # We want a v_in that also covers many other cliques to potentially enable removals.
                    best_v_in = -1
                    best_score = -1
                    
                    # Iterate v_in not in S
                    # To speed up, we can just iterate all v and check membership
                    for v_in in range(n):
                        if v_in in S:
                            continue
                        valid = True
                        for c in critical:
                            if c not in C[v_in]:
                                valid = False
                                break
                        if valid:
                            # Score: number of cliques covered by v_in
                            # Or maybe number of cliques that become "safe" to remove?
                            # Simple: total coverage
                            score = C_size[v_in]
                            if score > best_score:
                                best_score = score
                                best_v_in = v_in
                                if score == m:
                                    break
                    
                    if best_v_in != -1:
                        # Perform swap
                        for c in C[v_out]:
                            cover_count[c] -= 1
                        for c in C[best_v_in]:
                            cover_count[c] += 1
                        
                        S.remove(v_out)
                        S.add(best_v_in)
                        found_swap = True
                        break # Restart loop
                
                if found_swap:
                    improved = True

            return S

        # Branch and Bound with Lower Bound
        # State: (uncovered_cliques, current_set, current_size)
        # We want to minimize current_size.
        
        def lower_bound(uncovered):
            # Max number of disjoint cliques in uncovered
            # Greedy approximation:
            # Sort cliques by size (smallest first) to pack more?
            # Or just pick the one with most unique vertices?
            # Let's try a simple greedy: pick clique, remove it and all intersecting cliques.
            # This gives a set of disjoint cliques. The size of this set is a lower bound.
            # To get a *tighter* lower bound, we want the *maximum* such set.
            # Finding max set of disjoint cliques is NP-hard (packing), but n is small.
            # We can approximate by trying a few greedy orderings.
            
            if not uncovered:
                return 0
            
            # Greedy 1: Smallest clique first
            cliques_sorted = sorted(uncovered, key=lambda c: len(V[c]))
            count = 0
            remaining = set(uncovered)
            for c in cliques_sorted:
                if c not in remaining:
                    continue
                count += 1
                # Remove c and all cliques that share a vertex with c
                # Intersects if V[c] & V[c'] is not empty
                # We can precompute intersection graph?
                # For now, just check.
                to_remove = {c}
                for c2 in list(remaining):
                    if c2 == c:
                        continue
                    if V[c] & V[c2]:
                        to_remove.add(c2)
                remaining -= to_remove
            
            # Greedy 2: Largest clique first? (Usually worse for packing count)
            # Let's stick to one good heuristic or try 2-3.
            
            # Greedy 3: Clique with most "rare" vertices?
            # For now, return count.
            return count

        def branch_and_bound(uncovered, current_set, current_size, best_size, deadline):
            if current_size >= best_size:
                return None
            
            if not uncovered:
                return set(current_set)
            
            if time.time() > deadline:
                return None
            
            lb = lower_bound(uncovered)
            if current_size + lb >= best_size:
                return None
            
            # Choose a clique to hit
            # Heuristic: pick the clique with the fewest available vertices?
            # Or the one that appears in the most "conflicts"?
            # Let's pick the first one in sorted order (by size) to keep it simple but effective.
            # Actually, picking the one with smallest |V[c]| is standard for hitting set.
            c = min(uncovered, key=lambda x: len(V[x]))
            
            # Branch on vertices in V[c]
            # Order vertices by how many other cliques they cover (descending) to find good solutions early
            verts = list(V[c])
            verts.sort(key=lambda v: -len(C[v] & uncovered))
            
            for v in verts:
                new_uncovered = uncovered - C[v]
                new_set = current_set | {v}
                new_size = current_size + 1
                
                if new_size >= best_size:
                    continue
                
                res = branch_and_bound(new_uncovered, new_set, new_size, best_size, deadline)
                if res is not None:
                    best_size = new_size
                    return res
            
            return None

        best_set = None
        best_size = n # Upper bound
        
        start_time = time.time()
        
        # Phase 1: Greedy + Local Search to get a good initial upper bound
        for i in range(20):
            if time.time() - start_time > time_budget * 0.2:
                break
            rng = random.Random(1000 + i)
            cand = greedy_init(rng)
            cand = local_search(cand, rng, 0.05)
            if best_set is None or len(cand) < best_size:
                best_size = len(cand)
                best_set = set(cand)
        
        # Phase 2: Branch and Bound
        # We use the best found so far as the initial bound.
        # We need to be careful with recursion depth and time.
        
        uncovered_init = set(range(m))
        
        # To help B&B, we can order vertices globally?
        # The B&B function above handles ordering per branch.
        
        # Run B&B with a time limit
        bnb_deadline = time.time() + time_budget * 0.8
        
        # We might want to restart B&B with different vertex orderings if it doesn't finish?
        # For now, one run.
        
        # The B&B function modifies best_size internally? No, it returns a set.
        # We need to pass best_size and update it.
        # The current implementation returns the first solution found that is better?
        # No, it prunes if current_size + lb >= best_size.
        # It returns a solution of size < best_size.
        # But it only returns the *first* one found in the DFS.
        # We want the *best* one.
        # Since it's DFS, it will find a solution, update best_size, and continue?
        # No, the current implementation returns immediately upon finding a valid solution.
        # This is bad for optimization. We need to keep searching for better solutions.
        
        # Let's modify B&B to keep searching and track the global best.
        
        global_best = best_size
        global_best_set = best_set
        
        def branch_and_bound_opt(uncovered, current_set, current_size, deadline):
            nonlocal global_best, global_best_set
            
            if current_size >= global_best:
                return
            
            if not uncovered:
                if current_size < global_best:
                    global_best = current_size
                    global_best_set = set(current_set)
                return
            
            if time.time() > deadline:
                return
            
            lb = lower_bound(uncovered)
            if current_size + lb >= global_best:
                return
            
            c = min(uncovered, key=lambda x: len(V[x]))
            verts = list(V[c])
            verts.sort(key=lambda v: -len(C[v] & uncovered))
            
            for v in verts:
                new_uncovered = uncovered - C[v]
                new_set = current_set | {v}
                new_size = current_size + 1
                
                if new_size >= global_best:
                    continue
                
                branch_and_bound_opt(new_uncovered, new_set, new_size, deadline)
                
                # Optional: if we found a solution of size 1? (Impossible for these graphs likely)
                # If global_best is very small, we can prune more.
        
        branch_and_bound_opt(uncovered_init, set(), 0, bnb_deadline)
        
        if global_best_set is not None:
            return sorted(global_best_set)
        else:
            return sorted(best_set)

    out = []
    start_time = time.time()
    total_deadline = 140.0 # Leave 10s margin for safety
    
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining = total_deadline - (time.time() - start_time)
        if remaining <= 0:
            # Fallback: return all vertices (valid but large)
            out.append(sorted(range(n)))
            continue
        
        # Allocate time: give more time to larger/harder instances?
        # Simple: divide equally.
        inst_budget = remaining / (len(instances) - idx)
        
        res = solve_instance(n, p, seed, inst_budget)
        out.append(res)
        
    return out
