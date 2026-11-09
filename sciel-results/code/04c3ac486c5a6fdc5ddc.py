import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145  # safety margin from 150s

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def solve_instance(n, p, seed, time_budget):
        adj_orig = get_graph(n, p, seed)

        if not any(adj_orig):
            return []

        def count_edges(adj):
            total = 0
            for u in range(n):
                total += adj[u].bit_count()
            return total // 2

        def find_large_clique(adj, n, rng=None, max_iter=5000):
            """Find a large clique using greedy expansion from best seed edge."""
            if not any(adj):
                return []

            # Find edge with maximum common neighbors
            best_u, best_v = -1, -1
            best_common = -1
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in iter_bits(adj[u]):
                    if v > u:
                        common = (adj[u] & adj[v]).bit_count()
                        if common > best_common:
                            best_common = common
                            best_u, best_v = u, v

            if best_u == -1:
                return []

            # Expand clique greedily
            C = [best_u, best_v]
            U = adj[best_u] & adj[best_v]

            while U:
                # Find vertex in U with max connections within U
                best_candidate = -1
                best_score = -1
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                for u in candidates:
                    score = (adj[u] & U).bit_count()
                    if score > best_score:
                        best_score = score
                        best_candidate = u
                if best_candidate == -1:
                    break
                C.append(best_candidate)
                U &= adj[best_candidate]

            return C

        def remove_clique_edges(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def validate(cliques, adj_mask_orig, n):
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None):
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []

            while True:
                if time.time() > deadline:
                    break
                C = find_large_clique(adj, n, rng)
                if not C:
                    break
                cliques.append(C)
                remove_clique_edges(C, adj)

            return cliques

        def try_merge(cliques, adj_orig, n, deadline):
            """Try to merge cliques that form a larger clique together."""
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        C_j = cliques[j]
                        # Check if union is a clique
                        is_clique = True
                        for u in C_i:
                            for v in C_j:
                                if not (adj_orig[u] & (1 << v)):
                                    is_clique = False
                                    break
                            if not is_clique:
                                break
                        if is_clique:
                            new_cliques = [C_i + C_j] + [cliques[k] for k in range(i + 1, len(cliques)) if k != j]
                            i = 0
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                        i += 1
                cliques = new_cliques
            return cliques

        best_cliques = None
        best_count = float('inf')

        # Deterministic run
        det_deadline = start_time + time_budget * 0.3
        det_cliques = greedy_partition(adj_orig, n, rng=None, deadline=det_deadline)
        if validate(det_cliques, adj_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)

        # Randomized restarts
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(5, int(remaining / 0.5))
            rng_master = random.Random(seed + 1000)

            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 2.0:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                restart_deadline = min(start_time + time_budget - 2.0, time.time() + max(0.2, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_orig, n, rng=rng, deadline=restart_deadline)
                    if validate(cliques, adj_orig, n):
                        # Try merging
                        merge_deadline = min(start_time + time_budget - 2.0, time.time() + 0.3)
                        improved = try_merge(cliques, adj_orig, n, merge_deadline)
                        if validate(improved, adj_orig, n):
                            if len(improved) < best_count:
                                best_count = len(improved)
                                best_cliques = improved
                        else:
                            if len(cliques) < best_count:
                                best_count = len(cliques)
                                best_cliques = cliques
                except:
                    continue

        if best_cliques is None:
            # Fallback: one clique per edge
            best_cliques = []
            for u in range(n):
                for v in iter_bits(adj_orig[u]):
                    if v > u:
                        best_cliques.append([u, v])

        return best_cliques

    results = []
    # Instance 1: n=32
    t1 = min(45, TOTAL_BUDGET - 65)
    results.append(solve_instance(32, 0.6, 601, t1))

    # Instance 2: n=64
    elapsed = time.time() - start_time
    t2 = max(20, TOTAL_BUDGET - elapsed - 3)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results
