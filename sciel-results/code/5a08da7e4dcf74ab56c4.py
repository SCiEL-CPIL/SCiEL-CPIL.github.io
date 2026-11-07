import time
import random

_DEADLINE = 130.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _deadline_hit():
    return _elapsed() > _DEADLINE


def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)

        adj_c = [0] * k
        adj_i = [0] * (n - k)

        for (u, v) in edges:
            if u < k and v < k:
                adj_c[u] |= (1 << v)
                adj_c[v] |= (1 << u)
            elif u < k and v >= k:
                idx_v = v - k
                adj_i[idx_v] |= (1 << u)
            elif u >= k and v < k:
                idx_v = u - k
                adj_i[idx_v] |= (1 << v)

        uncovered = [0] * k
        for u in range(k):
            uncovered[u] = adj_c[u]

        def _find_max_clique(cand_mask):
            if cand_mask == 0:
                return 0
            best = 0
            temp = cand_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Greedy: take v, intersect with neighbors
                sub = cand_mask & uncovered[v] & ~lsb
                sub_c = sub
                while sub_c:
                    lsb2 = sub_c & -sub_c
                    v2 = lsb2.bit_length() - 1
                    sub_c ^= lsb2
                    sub = sub & uncovered[v2]
                size = (sub | lsb).bit_count()
                if size > best.bit_count():
                    best = sub | lsb
            return best

        def _cover_cross_for_vertex(idx):
            """Cover all cross edges for independent vertex k+idx."""
            remaining = adj_i[idx]
            while remaining:
                if _deadline_hit():
                    temp = remaining
                    while temp:
                        lsb = temp & -temp
                        u = lsb.bit_length() - 1
                        temp ^= lsb
                        yield [k + idx, u]
                    return

                # Find a clique in `remaining` that is also a clique in uncovered
                # We want a set S ⊆ remaining such that S is a clique in uncovered
                # Try greedy: pick vertex with max degree in (remaining ∩ uncovered-neighborhood)
                best_mask = 0
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    # Neighbors of u in both remaining and uncovered
                    neigh = remaining & uncovered[u] & ~lsb
                    # Greedy expand
                    cur = lsb
                    cand = neigh
                    while cand:
                        lsb2 = cand & -cand
                        v2 = lsb2.bit_length() - 1
                        cand ^= lsb2
                        cur |= lsb2
                        cand &= uncovered[v2]
                    if cur.bit_count() > best_mask.bit_count():
                        best_mask = cur

                if best_mask == 0:
                    # No edges in remaining are uncovered; pick single vertex
                    lsb = remaining & -remaining
                    best_mask = lsb

                # Form clique {k+idx} ∪ S
                clique = [k + idx]
                temp = best_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    clique.append(u)
                yield clique

                # Remove edges within best_mask from uncovered
                verts = []
                temp = best_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    verts.append(u)
                    temp ^= lsb
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        w = verts[j]
                        uncovered[u] &= ~(1 << w)
                        uncovered[w] &= ~(1 << u)

                remaining &= ~best_mask

        cliques = []

        # Process independent vertices, sorted by degree descending
        i_indices = list(range(n - k))
        i_indices.sort(key=lambda idx: adj_i[idx].bit_count(), reverse=True)

        for idx in i_indices:
            for c in _cover_cross_for_vertex(idx):
                cliques.append(c)

        # Process remaining uncovered edges in C
        full_mask = (1 << k) - 1
        while True:
            if _deadline_hit():
                for u in range(k):
                    for w in range(u + 1, k):
                        if (uncovered[u] >> w) & 1:
                            cliques.append([u, w])
                            uncovered[u] &= ~(1 << w)
                            uncovered[w] &= ~(1 << u)
                break

            has_edge = False
            for u in range(k):
                if uncovered[u]:
                    has_edge = True
                    break
            if not has_edge:
                break

            S_mask = _find_max_clique(full_mask)
            if S_mask.bit_count() < 2:
                # Find any edge
                for u in range(k):
                    if uncovered[u]:
                        lsb = uncovered[u] & -uncovered[u]
                        w = lsb.bit_length() - 1
                        S_mask = (1 << u) | (1 << w)
                        break

            clique = []
            temp = S_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                temp ^= lsb
                clique.append(u)
            cliques.append(clique)

            verts = []
            temp = S_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                verts.append(u)
                temp ^= lsb
            for i in range(len(verts)):
                u = verts[i]
                for j in range(i + 1, len(verts)):
                    w = verts[j]
                    uncovered[u] &= ~(1 << w)
                    uncovered[w] &= ~(1 << u)

        # Local improvement: try to merge cliques that form a larger clique
        # For each pair of cliques, if their union is a clique, merge them
        improved = True
        while improved and not _deadline_hit():
            improved = False
            new_cliques = []
            used = [False] * len(cliques)

            for i in range(len(cliques)):
                if used[i]:
                    continue
                cur = set(cliques[i])
                # Try to merge with other cliques
                for j in range(i + 1, len(cliques)):
                    if used[j]:
                        continue
                    candidate = cur | set(cliques[j])
                    # Check if candidate is a clique in the original graph
                    is_c = True
                    cand_list = list(candidate)
                    for a in range(len(cand_list)):
                        ua = cand_list[a]
                        for b in range(a + 1, len(cand_list)):
                            ub = cand_list[b]
                        if ua >= k or ub >= k:
                            # Cross edge: check adj_i
                            if ua >= k:
                                if not (adj_i[ua - k] & (1 << ub)):
                                    is_c = False
                                    break
                            else:
                                if not (adj_i[ub - k] & (1 << ua)):
                                    is_c = False
                                    break
                        else:
                            if not (adj_c[ua] & (1 << ub)):
                                is_c = False
                                break
                        if not is_c:
                            break
                    if is_c:
                        cur = candidate
                        used[j] = True
                        improved = True
                new_cliques.append(list(cur))
                used[i] = True

            if improved:
                cliques = new_cliques

        # Final type conversion
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
