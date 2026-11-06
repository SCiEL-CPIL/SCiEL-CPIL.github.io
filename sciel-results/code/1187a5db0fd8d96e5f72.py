import time
import math
import random as _random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    sys.setrecursionlimit(10000)

    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(adj, n):
        cliques = []

        def bk(R, P, X):
            if time.time() - start_time > DEADLINE:
                return

            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return

            best_u = None
            best_count = -1
            for u in P:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            for u in X:
                cnt = len(P & adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u

            if best_u is None:
                candidates = list(P)
            else:
                candidates = list(P - adj[best_u])

            for v in candidates:
                if time.time() - start_time > DEADLINE:
                    return
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        cliques = bron_kerbosch(adj, n)

        if not cliques:
            return []

        m = len(cliques)

        # vertex -> bitmask of clique indices it belongs to
        vertex_cliques = [0] * n
        for ci, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v] |= (1 << ci)

        all_cliques_mask = (1 << m) - 1

        active_vertices = [v for v in range(n) if vertex_cliques[v]]
        if not active_vertices:
            return []

        # Dominance pruning: drop a vertex whose clique-set is a subset of another's
        active_sorted = sorted(active_vertices, key=lambda v: -bin(vertex_cliques[v]).count('1'))
        keep_vertices = set()
        for v in active_sorted:
            dominated = False
            for u in keep_vertices:
                if (vertex_cliques[v] & ~vertex_cliques[u]) == 0:
                    dominated = True
                    break
            if not dominated:
                keep_vertices.add(v)

        candidates = sorted(keep_vertices)
        if not candidates:
            return []

        num_cand = len(candidates)
        cand_covers = [vertex_cliques[v] for v in candidates]

        # For each clique, bitmask of candidates that cover it
        clique_cand_mask = [0] * m
        for i in range(num_cand):
            mask = cand_covers[i]
            for ci in range(m):
                if mask & (1 << ci):
                    clique_cand_mask[ci] |= (1 << i)

        # Try ILP for an exact / near-exact starting point
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np

            A = np.zeros((m, num_cand))
            for ci in range(m):
                for i in range(num_cand):
                    if cand_covers[i] & (1 << ci):
                        A[ci, i] = 1.0

            c = np.ones(num_cand)
            constraints = [LinearConstraint(A, lb=np.ones(m), ub=np.inf)]
            bounds = Bounds(lb=0, ub=1)
            result = milp(c=c, constraints=constraints, bounds=bounds,
                          integrality=np.ones(num_cand))
            if result.success and result.x is not None:
                sol_mask = 0
                for i in range(num_cand):
                    if result.x[i] > 0.5:
                        sol_mask |= (1 << i)
                covered = 0
                for i in range(num_cand):
                    if sol_mask & (1 << i):
                        covered |= cand_covers[i]
                if covered == all_cliques_mask:
                    best_vertices = [candidates[i] for i in range(num_cand) if sol_mask & (1 << i)]
                    return sorted(best_vertices)
        except Exception:
            pass

        def greedy_cover(uncovered_mask, rng=None):
            selected_mask = 0
            unc = uncovered_mask
            while unc:
                best_idx = -1
                best_cov = 0
                best_list = []
                for i in range(num_cand):
                    if selected_mask & (1 << i):
                        continue
                    cov = bin(unc & cand_covers[i]).count('1')
                    if cov > best_cov:
                        best_cov = cov
                        best_list = [i]
                    elif cov == best_cov and best_cov > 0:
                        best_list.append(i)
                if not best_list:
                    break
                if rng and len(best_list) > 1:
                    best_idx = rng.choice(best_list)
                else:
                    best_idx = best_list[0]
                selected_mask |= (1 << best_idx)
                unc &= ~cand_covers[best_idx]
            return selected_mask

        # Multi-start randomized greedy for a strong initial incumbent
        rng = _random.Random(42)
        init_sol = greedy_cover(all_cliques_mask)
        best_size = bin(init_sol).count('1')
        best_mask = init_sol
        for _ in range(40):
            sol = greedy_cover(all_cliques_mask, rng)
            sz = bin(sol).count('1')
            if sz < best_size:
                best_size = sz
                best_mask = sol
            elif sz == best_size:
                # Try to improve tie via local search
                improved = local_search(sol)
                if bin(improved).count('1') < best_size:
                    best_size = bin(improved).count('1')
                    best_mask = improved

        if best_size == 0:
            return []

        # Conflict (line) graph over cliques: edge if a candidate covers both
        # (i.e., the two cliques cannot both be forced by a single vertex).
        conflict = [0] * m
        for i in range(num_cand):
            cm = cand_covers[i]
            b = cm
            while b:
                lsb = b & (-b)
                ci = lsb.bit_length() - 1
                conflict[ci] |= (cm & ~lsb)
                b &= b - 1

        # Maximum matching size on a subgraph induced by a clique-mask (lower bound)
        def matching_lb(cmask):
            idx = []
            b = cmask
            while b:
                lsb = b & (-b)
                idx.append(lsb.bit_length() - 1)
                b &= b - 1
            k = len(idx)
            pos = {v: j for j, v in enumerate(idx)}
            match = [-1] * k
            def aug(u, seen):
                for w in idx:
                    if (conflict[u] >> w) & 1 and not seen[w]:
                        seen[w] = True
                        if match[w] == -1 or aug(match[w], seen):
                            match[w] = u
                            return True
                return False
            res = 0
            for u in idx:
                seen = [False] * k
                if aug(u, seen):
                    res += 1
            return res

        def compute_lb(uncovered_mask):
            if not uncovered_mask:
                return 0
            max_cov = 0
            for i in range(num_cand):
                cov = bin(uncovered_mask & cand_covers[i]).count('1')
                if cov > max_cov:
                    max_cov = cov
            if max_cov == 0:
                return float('inf')
            cov_lb = math.ceil(bin(uncovered_mask).count('1') / max_cov)
            match_lb = matching_lb(uncovered_mask)
            return max(cov_lb, match_lb)

        def branch_and_bound_iterative(selected_mask, uncovered_mask):
            nonlocal best_size, best_mask
            stack = [(selected_mask, uncovered_mask)]
            while stack:
                if time.time() - start_time > DEADLINE - 5:
                    return
                sel, unc = stack.pop()
                if not unc:
                    sz = bin(sel).count('1')
                    if sz < best_size:
                        best_size = sz
                        best_mask = sel
                    continue
                if bin(sel).count('1') >= best_size:
                    continue
                lb = bin(sel).count('1') + compute_lb(unc)
                if lb >= best_size:
                    continue
                # Branch on the uncovered clique with the fewest available candidates
                target_ci = -1
                min_avail = float('inf')
                for ci in range(m):
                    if unc & (1 << ci):
                        avail = bin(clique_cand_mask[ci] & ~sel).count('1')
                        if avail == 0:
                            continue
                        if avail < min_avail:
                            min_avail = avail
                            target_ci = ci
                            if min_avail == 1:
                                break
                if target_ci == -1:
                    continue
                avail_list = []
                for i in range(num_cand):
                    if (clique_cand_mask[target_ci] >> i) & 1 and not (sel >> i) & 1:
                        cov = bin(unc & cand_covers[i]).count('1')
                        avail_list.append((cov, i))
                avail_list.sort(reverse=True)
                for cov, i in avail_list:
                    new_sel = sel | (1 << i)
                    new_unc = unc & ~cand_covers[i]
                    stack.append((new_sel, new_unc))

        branch_and_bound_iterative(0, all_cliques_mask)

        # Local search polish: redundant removals then 1-for-1 swaps
        def local_search(sol_mask):
            improved = True
            while improved:
                improved = False
                for i in range(num_cand):
                    if sol_mask & (1 << i):
                        trial = sol_mask & ~(1 << i)
                        covered = 0
                        for j in range(num_cand):
                            if trial & (1 << j):
                                covered |= cand_covers[j]
                        if covered == all_cliques_mask:
                            sol_mask = trial
                            improved = True
                            break
                if time.time() - start_time > DEADLINE - 10:
                    break
                for i in range(num_cand):
                    if not (sol_mask & (1 << i)):
                        continue
                    for j in range(num_cand):
                        if sol_mask & (1 << j):
                            continue
                        trial = (sol_mask & ~(1 << i)) | (1 << j)
                        covered = 0
                        for k in range(num_cand):
                            if trial & (1 << k):
                                covered |= cand_covers[k]
                        if covered == all_cliques_mask and bin(trial).count('1') < bin(sol_mask).count('1'):
                            sol_mask = trial
                            improved = True
                            break
                    if improved:
                        break
                if time.time() - start_time > DEADLINE - 10:
                    break
            return sol_mask

        best_mask = local_search(best_mask)
        best_size = bin(best_mask).count('1')

        # Additional polish: try to find smaller solutions via more aggressive local search
        if time.time() - start_time < DEADLINE - 15:
            # Try multiple random restarts of local search from perturbed solutions
            for _ in range(20):
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Perturb current best
                perturbed = best_mask
                # Remove a random selected vertex
                sel_indices = [i for i in range(num_cand) if perturbed & (1 << i)]
                if not sel_indices:
                    break
                remove_idx = rng.choice(sel_indices)
                perturbed &= ~(1 << remove_idx)
                # Re-add random vertices until coverage is restored
                while True:
                    covered = 0
                    for j in range(num_cand):
                        if perturbed & (1 << j):
                            covered |= cand_covers[j]
                    if covered == all_cliques_mask:
                        break
                    # Find uncovered cliques
                    unc = all_cliques_mask & ~covered
                    # Find a random candidate that covers an uncovered clique
                    added = False
                    for i in range(num_cand):
                        if not (perturbed & (1 << i)) and (cand_covers[i] & unc):
                            perturbed |= (1 << i)
                            added = True
                            break
                    if not added:
                        break
                # Local search from perturbed
                polished = local_search(perturbed)
                if bin(polished).count('1') < best_size:
                    best_size = bin(polished).count('1')
                    best_mask = polished

        best_vertices = [candidates[i] for i in range(num_cand) if best_mask & (1 << i)]
        return sorted(best_vertices)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
