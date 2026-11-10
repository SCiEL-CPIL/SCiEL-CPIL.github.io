import time
import numpy as np
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0

    triples = list(combinations(range(n), 3))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_to_idx = {}
    for idx, (i, j) in enumerate(all_pairs):
        pair_to_idx[(i, j)] = idx
    pair_to_idx[(j, i)] = idx

    def count_edges(adj):
        return sum(1 for (i, j) in all_pairs if (adj[i] >> j) & 1)

    def find_violation(adj):
        for (i, j, k) in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k), common
        return None, 0

    def is_valid(adj):
        for (i, j, k) in triples:
            if (adj[i] & adj[j] & adj[k]).bit_count() >= 3:
                return False
        return True

    def can_add(adj, u, v):
        # Case A: v in part B, u in part A
        # Need pairs (a,b) in N(v)\{u} with |N(u)∩N(a)∩N(b)| >= 2
        mask_v = adj[v]
        mask_u = adj[u]
        candidates = mask_v & ~(1 << u)
        # Collect candidates as list
        cand = []
        m = candidates
        while m:
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            cand.append(x)
            m ^= lsb
        for a in cand:
            mask_a = adj[a]
            if not (mask_v >> a) & 1:
                continue
            # iterate over b in cand after a
            for b in cand:
                if b <= a:
                    continue
                if not (mask_v >> b) & 1:
                    continue
                common = mask_u & mask_a & adj[b]
                if common.bit_count() >= 2:
                    return False
        # Case B: u in part B, v in part A
        candidates2 = mask_u & ~(1 << v)
        cand2 = []
        m = candidates2
        while m:
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            cand2.append(x)
            m ^= lsb
        for a in cand2:
            mask_a = adj[a]
            if not (mask_u >> a) & 1:
                continue
            for b in cand2:
                if b <= a:
                    continue
                if not (mask_u >> b) & 1:
                    continue
                common = mask_v & mask_a & adj[b]
                if common.bit_count() >= 2:
                    return False
        return True

    def clean(adj, rng):
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            viol, common = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            nbrs = []
            m = common
            while m:
                lsb = m & (-m)
                x = lsb.bit_length() - 1
                nbrs.append(x)
                m ^= lsb
            # pick random edge to remove
            for _t in range(30):
                a = [i, j, k][rng.integers(0, 3)]
                b = nbrs[rng.integers(0, len(nbrs))]
                if (adj[a] >> b) & 1:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    break
            else:
                a, b = i, nbrs[0]
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)

    def greedy_add(adj, rng):
        missing = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    def local_search_1for1(adj, rng, max_rounds=20):
        improved = True
        rnd = 0
        while improved and rnd < max_rounds:
            if time.time() > deadline - 5:
                break
            improved = False
            edges = [(i, j) for (i, j) in all_pairs if (adj[i] >> j) & 1]
            rng.shuffle(edges)
            for u, v in edges:
                if time.time() > deadline - 5:
                    break
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                missing = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
                rng.shuffle(missing)
                added = 0
                for a, b in missing:
                    if can_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        added += 1
                        if added >= 2:
                            break
                if added == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                elif added >= 1:
                    improved = True
            rnd += 1

    def local_search_2for1(adj, rng, max_rounds=10):
        improved = True
        rnd = 0
        while improved and rnd < max_rounds:
            if time.time() > deadline - 5:
                break
            improved = False
            pairs = [(i, j) for (i, j) in all_pairs if (adj[i] >> j) & 1]
            rng.shuffle(pairs)
            for u1, v1 in pairs:
                if time.time() > deadline - 5:
                    break
                # remove two edges
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                # try to add 3 edges
                missing = [(i, j) for (i, j) in all_pairs if not ((adj[i] >> j) & 1)]
                rng.shuffle(missing)
                added = 0
                for a, b in missing:
                    if can_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        added += 1
                        if added >= 3:
                            break
                if added < 3:
                    # restore
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    # remove the added edges
                    for a, b in missing[:added]:
                        if (adj[a] >> b) & 1 and (a, b) != (u1, v1):
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                else:
                    improved = True
            rnd += 1

    def hill_climb(adj, rng):
        # Try removing a single edge and re-adding greedily
        edges = [(i, j) for (i, j) in all_pairs if (adj[i] >> j) & 1]
        rng.shuffle(edges)
        best = adj[:]
        best_count = count_edges(adj)
        for u, v in edges[:100]:
            if time.time() > deadline - 5:
                break
            trial = adj[:]
            trial[u] &= ~(1 << v)
            trial[v] &= ~(1 << u)
            greedy_add(trial, rng)
            cnt = count_edges(trial)
            if cnt > best_count:
                best = trial[:]
                best_count = cnt
        return best, best_count

    best_adj = [0] * n
    best_count = 0
    rng = np.random.default_rng(42)

    # Phase 1: Random restarts with clean + greedy
    while time.time() < deadline - 30:
        p = 0.4 + 0.3 * rng.random()
        adj = [0] * n
        for (i, j) in all_pairs:
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        clean(adj, rng)
        greedy_add(adj, rng)
        local_search_1for1(adj, rng, max_rounds=15)
        greedy_add(adj, rng)
        local_search_1for1(adj, rng, max_rounds=10)

        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            if best_count >= 55:
                break

    # Phase 2: More intensive local search on best
    if time.time() < deadline - 10:
        adj = best_adj[:]
        local_search_1for1(adj, rng, max_rounds=30)
        greedy_add(adj, rng)
        local_search_2for1(adj, rng, max_rounds=15)
        greedy_add(adj, rng)
        local_search_1for1(adj, rng, max_rounds=20)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Phase 3: Hill climbing
    if time.time() < deadline - 5:
        adj, cnt = hill_climb(best_adj, rng)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Final validation
    if not is_valid(best_adj):
        clean(best_adj, rng)

    edges = []
    for (i, j) in all_pairs:
        if (best_adj[i] >> j) & 1:
            edges.append([i, j])
    return edges
