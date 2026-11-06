import time
import math
import random as _random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    def random_graph(n, p, seed):
        """Generate edge list using the same RNG as the verifier."""
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj_bitmask(edges, n):
        """Build adjacency as integer bitmasks."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def bron_kerbosch_bitmask(adj, n):
        """
        Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting,
        using integer bitmasks for speed.
        Returns a list of lists (each a list of vertex indices).
        """
        cliques = []
        full_mask = (1 << n) - 1

        def popcount(x):
            return bin(x).count('1')

        def bk(R_mask, P_mask, X_mask):
            if P_mask == 0 and X_mask == 0:
                if popcount(R_mask) >= 2:
                    # Convert bitmask to list of vertex indices
                    verts = [i for i in range(n) if (R_mask >> i) & 1]
                    cliques.append(verts)
                return

            P_union_X = P_mask | X_mask
            if P_union_X == 0:
                return

            # Choose pivot u in P ∪ X to maximize |P ∩ N(u)|
            max_inter = -1
            pivot = -1
            temp = P_union_X
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                inter = P_mask & adj[u]
                cnt = popcount(inter)
                if cnt > max_inter:
                    max_inter = cnt
                    pivot = u
                temp ^= lsb

            candidates = P_mask & ~adj[pivot] if pivot >= 0 else P_mask

            if candidates == 0:
                return

            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1

                new_R = R_mask | lsb
                new_P = P_mask & adj[v]
                new_X = X_mask & adj[v]

                bk(new_R, new_P, new_X)

                P_mask &= ~lsb
                X_mask |= lsb
                temp ^= lsb

        bk(0, full_mask, 0)
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        adj = build_adj_bitmask(edges, n)

        # Enumerate maximal cliques
        cliques = bron_kerbosch_bitmask(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # Build vertex -> clique coverage bitmask
        # vertex_clique_mask[v] = bitmask of cliques containing vertex v
        vertex_clique_mask = [0] * n
        for ci, c in enumerate(cliques):
            bit = 1 << ci
            for v in c:
                vertex_clique_mask[v] |= bit

        # Identify candidate vertices (those in at least one clique)
        candidates = [v for v in range(n) if vertex_clique_mask[v] != 0]

        if not candidates:
            return []

        # Dominance Pruning using bitmasks
        # If C_u ⊆ C_v (as bitmasks), then u is dominated by v
        active = sorted(candidates, key=lambda v: -bin(vertex_clique_mask[v]).count('1'))

        keep = set()
        for v in active:
            dominated = False
            for u in keep:
                # Check if vertex_clique_mask[v] is subset of vertex_clique_mask[u]
                if (vertex_clique_mask[v] & ~vertex_clique_mask[u]) == 0:
                    dominated = True
                    break
            if not dominated:
                keep.add(v)

        candidates = list(keep)
        num_cands = len(candidates)

        if num_cands == 0:
            return []

        # Precompute clique candidate lists
        clique_cands = []
        for ci, c in enumerate(cliques):
            cc = [v for v in c if v in keep]
            clique_cands.append(cc)

        # Precompute coverage counts for each vertex on uncovered sets
        # This helps with faster lower bound computation
        def compute_coverage(v, unc_mask):
            return bin(unc_mask & vertex_clique_mask[v]).count('1')

        # Maximum matching lower bound on the conflict graph
        # Two cliques conflict if they share no common candidate vertex
        # A matching in the conflict graph gives a lower bound on transversal size
        def max_matching_lower_bound(unc_mask):
            """Compute maximum matching in conflict graph of uncovered cliques."""
            # Get list of uncovered clique indices
            unc_cliques = []
            temp = unc_mask
            while temp:
                lsb = temp & (-temp)
                ci = lsb.bit_length() - 1
                unc_cliques.append(ci)
                temp ^= lsb
            
            if len(unc_cliques) <= 1:
                return len(unc_cliques)
            
            # Build conflict graph: two cliques conflict if no common candidate
            # Use greedy matching for speed
            matched = {}
            match_count = 0
            for i, ci in enumerate(unc_cliques):
                if ci in matched:
                    continue
                found = False
                for j in range(i + 1, len(unc_cliques)):
                    cj = unc_cliques[j]
                    if cj in matched:
                        continue
                    # Check if ci and cj have no common candidate
                    common = set(clique_cands[ci]) & set(clique_cands[cj])
                    if not common:
                        matched[ci] = cj
                        matched[cj] = ci
                        match_count += 1
                        found = True
                        break
                if not found:
                    # ci must be covered by a distinct vertex
                    pass
            
            return match_count + (len(unc_cliques) - 2 * match_count)

        # Greedy Set Cover using bitmasks with randomization
        def greedy_cover(randomize=False, seed_offset=0):
            selected = set()
            unc = (1 << m) - 1  # all cliques uncovered
            while unc:
                best_v = -1
                best_cov = 0
                best_tie_breaker = 0
                rng = _random.Random(seed + seed_offset) if randomize else None
                for v in candidates:
                    if v in selected:
                        continue
                    cov = compute_coverage(v, unc)
                    if cov > best_cov:
                        best_cov = cov
                        best_v = v
                        best_tie_breaker = rng.randint(0, 1000000) if randomize else 0
                    elif cov == best_cov and cov > 0:
                        if randomize:
                            tie_breaker = rng.randint(0, 1000000)
                            if tie_breaker > best_tie_breaker:
                                best_tie_breaker = tie_breaker
                                best_v = v
                        else:
                            # Prefer vertices with higher degree in original graph
                            if adj[v].bit_count() > adj[best_v].bit_count():
                                best_v = v
                if best_v == -1:
                    break
                selected.add(best_v)
                unc &= ~vertex_clique_mask[best_v]
            return selected

        # Try multiple greedy starts
        init_sol = greedy_cover(randomize=False)
        best_size = len(init_sol)
        best_solution = set(init_sol)
        
        # Try a few randomized greedy starts
        for restart in range(3):
            if time.time() - start_time > DEADLINE:
                break
            sol = greedy_cover(randomize=True, seed_offset=restart * 12345)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = set(sol)

        # Branch and Bound using bitmasks for uncovered
        def branch_and_bound(selected, uncovered_mask, depth=0):
            nonlocal best_size, best_solution

            if uncovered_mask == 0:
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_solution = set(selected)
                return

            if len(selected) >= best_size:
                return

            if time.time() - start_time > DEADLINE:
                return

            # Lower Bound 1: max coverage of any single remaining vertex
            max_cov = 0
            for v in candidates:
                if v in selected:
                    continue
                cov = compute_coverage(v, uncovered_mask)
                if cov > max_cov:
                    max_cov = cov

            if max_cov == 0:
                return

            lb1 = len(selected) + math.ceil(bin(uncovered_mask).count('1') / max_cov)
            if lb1 >= best_size:
                return

            # Lower Bound 2: Maximum matching on conflict graph
            lb2 = len(selected) + max_matching_lower_bound(uncovered_mask)
            lb = max(lb1, lb2)
            if lb >= best_size:
                return

            # Branching heuristic: choose uncovered clique with fewest available candidates
            target_ci = -1
            min_avail = float('inf')
            temp = uncovered_mask
            while temp:
                lsb = temp & (-temp)
                ci = lsb.bit_length() - 1
                avail = 0
                for v in clique_cands[ci]:
                    if v not in selected:
                        avail += 1
                if avail < min_avail:
                    min_avail = avail
                    target_ci = ci
                    if avail == 1:
                        break
                temp ^= lsb

            if target_ci == -1:
                return

            # Branch on vertices in this specific clique
            verts_in_target = [v for v in clique_cands[target_ci] if v not in selected]
            verts_in_target.sort(key=lambda v: -compute_coverage(v, uncovered_mask))

            for v in verts_in_target:
                new_selected = selected | {v}
                new_uncovered = uncovered_mask & ~vertex_clique_mask[v]
                branch_and_bound(new_selected, new_uncovered, depth + 1)

        initial_uncovered = (1 << m) - 1
        branch_and_bound(set(), initial_uncovered)

        # Local search polish: try to remove vertices and do 2-swap improvements
        def local_search_polish(sol):
            nonlocal best_size, best_solution
            sol = set(sol)
            improved = True
            while improved:
                improved = False
                # Try removing one vertex
                for v in list(sol):
                    if len(sol) - 1 >= best_size:
                        continue
                    new_sol = sol - {v}
                    covered = 0
                    for u in new_sol:
                        covered |= vertex_clique_mask[u]
                    if covered == (1 << m) - 1:
                        sol = new_sol
                        improved = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_solution = set(sol)
                        break
                # Try 2-swap: remove one, add another
                if len(sol) > 0:
                    for v in list(sol):
                        if len(sol) >= best_size:
                            continue
                        sol_without_v = sol - {v}
                        # Find vertices not in sol that could replace v
                        for w in candidates:
                            if w in sol_without_v:
                                continue
                            new_sol = sol_without_v | {w}
                            covered = 0
                            for u in new_sol:
                                covered |= vertex_clique_mask[u]
                            if covered == (1 << m) - 1:
                                sol = new_sol
                                improved = True
                                if len(sol) < best_size:
                                    best_size = len(sol)
                                    best_solution = set(sol)
                                break
                        if improved:
                            break
            return sol

        # Polish the best solution found
        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        if best_solution is None:
            return sorted(init_sol)

        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
