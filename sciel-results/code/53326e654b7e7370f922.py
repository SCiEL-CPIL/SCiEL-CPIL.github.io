import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (or 150s if strict)
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of bitmasks, each representing a maximal clique.
        """
        cliques = []
        
        def bk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                # We only care about cliques with at least 2 vertices
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            
            # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(adj, n):
        """
        Finds a minimum clique transversal.
        Strategy:
        1. Enumerate all maximal cliques of size >= 2.
        2. Formulate as Minimum Set Cover:
           Universe U = set of maximal cliques
           Sets S_v = { C in U | v in C }
           Find min |T| such that union_{v in T} S_v = U.
        3. Use Greedy Set Cover to get an initial solution.
        4. Use Local Search (removal and swap) to minimize size.
        """
        cliques = enumerate_maximal_cliques(adj, n)
        
        if not cliques:
            return []
        
        # Map vertex to list of clique indices it belongs to
        vertex_to_cliques = [[] for _ in range(n)]
        for idx, c in enumerate(cliques):
            m = c
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                vertex_to_cliques[v].append(idx)
                m ^= lsb

        # Precompute clique masks for quick intersection checks? 
        # Or just use a set of uncovered clique indices.
        
        # Greedy Set Cover
        uncovered = set(range(len(cliques)))
        selected = []
        
        # Optimization: Precompute coverage count for each vertex?
        # Since n is small, recalculating is fine, but we can be smarter.
        
        while uncovered:
            if time.time() > deadline:
                break
            best_v = -1
            max_covered = 0
            
            # Only consider vertices that are in at least one uncovered clique
            # We can iterate over all vertices, but skip those with 0 coverage
            for v in range(n):
                if not uncovered:
                    break
                # Count how many uncovered cliques this vertex covers
                count = 0
                for idx in vertex_to_cliques[v]:
                    if idx in uncovered:
                        count += 1
                
                if count > max_covered:
                    max_covered = count
                    best_v = v
            
            if best_v == -1 or max_covered == 0:
                # Should not happen if uncovered is not empty and graph is valid
                break
                
            selected.append(best_v)
            for idx in vertex_to_cliques[best_v]:
                if idx in uncovered:
                    uncovered.remove(idx)

        # Local Search: Try to remove vertices
        # Sort selected vertices by "importance" (fewest unique cliques covered by them)?
        # Actually, just iterate.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to remove any vertex
            for i in range(len(selected)):
                v = selected[i]
                temp_selected = selected[:i] + selected[i+1:]
                
                # Check if temp_selected still covers all cliques
                # Build set of cliques covered by temp_selected
                covered_cliques = set()
                for u in temp_selected:
                    covered_cliques.update(vertex_to_cliques[u])
                
                if covered_cliques == set(range(len(cliques))):
                    selected = temp_selected
                    changed = True
                    break # Restart loop as indices changed

            # Try to swap one vertex for another?
            # If removal fails, maybe we can swap a "bad" vertex for a "good" one?
            # For now, removal is the primary reduction.
            # A swap would be: remove v, add w, if size stays same but maybe enables future removals?
            # Or if size decreases? No, swap keeps size.
            # If we can't remove, we are likely at a local minimum for this neighborhood.
            # To escape, we could try 2-opt (remove 2, add 1) but that's complex.
            # Given small n, greedy + removal is often optimal or near-optimal.

        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5:
            # Fallback: return empty or heuristic if running out of time
            # But these should be fast.
            pass
        adj = random_graph(n, p, seed)
        vc = solve_instance(adj, n)
        results.append(vc)
        
    return results
