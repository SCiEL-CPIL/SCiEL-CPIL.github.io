import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 143  # safety margin from 150s wall-clock limit

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def solve_instance(n, p, seed, time_budget):
        adj_mask_orig = get_graph(n, p, seed)

        if not any(adj_mask_orig):
            return []

        def edge_common(adj, n):
            """List (common, u, v) for every remaining edge."""
            edges = []
            for u in range(n):
                m = adj[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        edges.append(((adj[u] & adj[v]).bit_count(), u, v))
                    m ^= lsb
            return edges

        def expand_clique(seed_u, seed_v, adj, rng=None):
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            while U:
                best_u = -1
                best_score = -1
                best_score2 = -1
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                for u in candidates:
                    score = (adj[u] & U).bit_count()
                    score2 = adj[u].bit_count()
                    if score > best_score or (score == best_score and score2 > best_score2):
                        best_score = score
                        best_score2 = score2
                        best_u = u
                if best_u == -1:
                    break
                C.append(best_u)
                U &= adj[best_u]
            return C

        def expand_clique_multi(edges, adj, rng=None, k=4):
            """Try the k lowest-common seed edges, keep the largest maximal clique.
            Measured finding: checking multiple starting vertices and selecting the
            one yielding the largest clique reduces the number of cliques."""
            if not edges:
                return [0, 1] if n > 1 else []
            edges_sorted = sorted(edges, key=lambda x: x[0])
            kk = min(k, len(edges_sorted))
            best_C = None
            best_len = -1
            for idx in range(kk):
                common, u, v = edges_sorted[idx]
                C = expand_clique(u, v, adj, rng)
                if len(C) > best_len:
                    best_len = len(C)
                    best_C = C
            if best_C is None:
                u, v = edges_sorted[0][1], edges_sorted[0][2]
                best_C = [u, v]
            return best_C

        def remove_clique_edges(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def is_clique_union(C1, C2, adj):
            for u in C1:
                for v in C2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def is_clique(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def merge_cliques(cliques, adj, n, deadline):
            """Merge cliques whose union is a clique (in-place, clean)."""
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                i = 0
                while i < len(cliques):
                    merged_any = False
                    for j in range(i + 1, len(cliques)):
                        if is_clique_union(cliques[i], cliques[j], adj):
                            cliques[i] = cliques[i] + cliques[j]
                            del cliques[j]
                            changed = True
                            merged_any = True
                            break
                    if merged_any:
                        i = 0
                    else:
                        i += 1
            return cliques

        def reassign_vertices(cliques, adj, n, deadline):
            """Move a vertex out of a 2-clique into another clique if it shrinks count."""
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                for a in range(len(cliques)):
                    if time.time() >= deadline:
                        break
                    Ca = cliques[a]
                    if len(Ca) < 2:
                        continue
                    for b in range(len(cliques)):
                        if a == b:
                            continue
                        Cb = cliques[b]
                        for w in list(Ca):
                            rest_a = [x for x in Ca if x != w]
                            new_b = Cb + [w]
                            if len(rest_a) >= 2 and is_clique(rest_a, adj) and is_clique(new_b, adj):
                                if len(Ca) == 2:
                                    cliques = [cliques[k] for k in range(len(cliques)) if k != a]
                                    cliques[b] = new_b
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
            return cliques

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None, temperature=0.0, k=4):
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []
            while True:
                if time.time() > deadline:
                    break
                edges = edge_common(adj, n)
                if not edges:
                    break
                # Select seed: deterministic lowest-common, or soft-min over near-hardest.
                edges.sort(key=lambda x: x[0])
                if rng is None:
                    chosen = edges[:k]
                else:
                    kk = min(len(edges), max(8, len(edges) // 3))
                    cand = edges[:kk]
                    best_key = float('inf')
                    best_idx = 0
                    for i, (common, u, v) in enumerate(cand):
                        key = common + temperature * rng.random()
                        if key < best_key:
                            best_key = key
                            best_idx = i
                    chosen = [cand[best_idx]] + edges[:max(0, k - 1)]
                C = expand_clique_multi(chosen, adj, rng, k=k)
                if len(C) < 2:
                    u, v = edges[0][1], edges[0][2]
                    C = [u, v]
                cliques.append(C)
                remove_clique_edges(C, adj)
            cliques = merge_cliques(cliques, adj, n, deadline)
            cliques = reassign_vertices(cliques, adj, n, deadline)
            return cliques

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
                m = adj_mask_orig[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
                    m ^= lsb
            return True

        best_cliques = None
        best_count = float('inf')

        # Deterministic multi-seed run first
        det_deadline = start_time + time_budget * 0.30
        det_cliques = greedy_partition(adj_mask_orig, n, rng=None, deadline=det_deadline,
                                       temperature=0.0, k=4)
        if validate(det_cliques, adj_mask_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)

        # Randomized soft-min restarts
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(1, int(remaining / 1.2))
            rng_master = random.Random(seed + 1000)
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1:
                    break
                rng = random.Random(rng_master.randint(0, 2 ** 31))
                temp = rng_master.uniform(0.0, 1.5)
                kk = rng_master.choice([3, 4, 5])
                restart_deadline = min(start_time + time_budget - 1,
                                       time.time() + max(0.5, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_mask_orig, n, rng=rng,
                                               deadline=restart_deadline,
                                               temperature=temp, k=kk)
                    if validate(cliques, adj_mask_orig, n):
                        if len(cliques) < best_count:
                            best_count = len(cliques)
                            best_cliques = cliques
                except Exception:
                    continue

        if best_cliques is None:
            best_cliques = []
            for u in range(n):
                m = adj_mask_orig[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v > u:
                        best_cliques.append([u, v])
                    m ^= lsb

        return best_cliques

    results = []
    # Instance 1
    t1 = min(50, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))

    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(10, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results
