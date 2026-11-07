import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    n = P['n']
    edges = P['edges']

    # Build adjacency and edge set
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))

    # Adjacency matrix for fast lookup
    adj_matrix = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj_matrix[u][v] = True
        adj_matrix[v][u] = True

    def expand_biclique(u, v, rem_adj):
        """Expand edge (u,v) into maximal biclique using remaining adjacency."""
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Candidates for L: connected to all in R via remaining edges
            cand_L = set(range(n))
            for x in R:
                cand_L &= rem_adj[x]
            cand_L -= L
            cand_L -= R
            if cand_L:
                L |= cand_L
                changed = True

            # Candidates for R: connected to all in L via remaining edges
            cand_R = set(range(n))
            for x in L:
                cand_R &= rem_adj[x]
            cand_R -= L
            cand_R -= R
            if cand_R:
                R |= cand_R
                changed = True
        return L, R

    def greedy_partition(seed):
        """Greedy set cover: repeatedly pick the edge that yields the largest biclique."""
        random.seed(seed)
        remaining = set(edge_set)
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        bicliques = []

        while remaining:
            if time.time() > deadline - 8.0:
                break

            # Try a subset of candidate edges for speed; use all if small enough
            candidates = list(remaining)
            deg = [len(rem_adj[i]) for i in range(n)]
            candidates.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))

            # Evaluate top candidates (at least 20, or all if fewer)
            eval_k = min(len(candidates), max(20, len(candidates) // 2))
            # Add randomness to which candidates we evaluate
            if eval_k < len(candidates):
                # Shuffle the tail and pick some from it too
                tail = candidates[eval_k:]
                random.shuffle(tail)
                extra = tail[:max(5, len(candidates) - eval_k)]
                candidates = candidates[:eval_k] + extra
                random.shuffle(candidates)

            best_L = None
            best_R = None
            best_score = -1

            for u, v in candidates:
                L, R = expand_biclique(u, v, rem_adj)
                score = len(L) * len(R)
                if score > best_score:
                    best_score = score
                    best_L = L
                    best_R = R

            if best_L is None:
                u, v = next(iter(remaining))
                best_L = {u}
                best_R = {v}

            bicliques.append((frozenset(best_L), frozenset(best_R)))

            # Remove covered edges
            for u_l in best_L:
                for v_r in best_R:
                    e = _edge_key(u_l, v_r)
                    if e in remaining:
                        remaining.remove(e)
                        rem_adj[u_l].discard(v_r)
                        rem_adj[v_r].discard(u_l)

        return bicliques

    def try_merge(bicliques, adj_matrix):
        """Try to merge two bicliques if no cross-edges exist in original graph."""
        bicliques = [(list(L), list(R)) for L, R in bicliques]
        merged_any = True
        while merged_any:
            merged_any = False
            for i in range(len(bicliques)):
                for j in range(i + 1, len(bicliques)):
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]

                    # Check: no edges between Li and Rj, and Lj and Ri
                    valid = True
                    for u in Li:
                        for v in Rj:
                            if adj_matrix[u][v]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for u in Lj:
                            for v in Ri:
                                if adj_matrix[u][v]:
                                    valid = False
                                    break
                                if not valid:
                                    break

                    if valid:
                        new_L = set(Li) | set(Lj)
                        new_R = set(Ri) | set(Rj)
                        bicliques[i] = (list(new_L), list(new_R))
                        bicliques.pop(j)
                        merged_any = True
                        break
                if merged_any:
                    break
        return bicliques

    def local_search(bicliques, adj_matrix, deadline):
        """Local search: merge compatible bicliques and try vertex reassignment."""
        bicliques = [(list(L), list(R)) for L, R in bicliques]

        def build_edge_map(bcliques):
            emap = {}
            for idx, (L, R) in enumerate(bcliques):
                for u in L:
                    for v in R:
                        e = _edge_key(u, v)
                        emap[e] = idx
            return emap

        for iteration in range(300):
            if time.time() > deadline - 5.0:
                break

            # Try to merge any two bicliques
            merged = False
            for i in range(len(bicliques)):
                for j in range(i + 1, len(bicliques)):
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]
                    valid = True
                    for u in Li:
                        for v in Rj:
                            if adj_matrix[u][v]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for u in Lj:
                            for v in Ri:
                                if adj_matrix[u][v]:
                                    valid = False
                                    break
                                if not valid:
                                    break
                    if valid:
                        new_L = set(Li) | set(Lj)
                        new_R = set(Ri) | set(Rj)
                        bicliques[i] = (list(new_L), list(new_R))
                        bicliques.pop(j)
                        merged = True
                        break
                if merged:
                    break

            if not merged:
                # Try vertex reassignment: move a vertex from one side to help merge
                moved = False
                for i in range(len(bicliques)):
                    Li, Ri = bicliques[i]
                    if len(Li) <= 1:
                        continue
                    for u in Li:
                        # Check if removing u from L still leaves a valid biclique
                        new_L = [x for x in Li if x != u]
                        if not new_L:
                            continue
                        # The edges (u, v) for v in Ri must be coverable
                        # They will form a new single-vertex biclique or be merged
                        # For now, just check if it reduces total count after merge
                        bicliques[i] = (new_L, list(Ri))
                        bicliques.append(([u], list(Ri)))
                        # Try to merge the new single-vertex biclique with others
                        new_idx = len(bicliques) - 1
                        can_merge = False
                        for j in range(len(bicliques)):
                            if j == new_idx or j == i:
                                continue
                            Lj, Rj = bicliques[j]
                            # Check if [u] x Ri can merge with Lj x Rj
                            # Need no edges between {u} and Rj, and Lj and Ri
                            ok = True
                            for v in Rj:
                                if adj_matrix[u][v]:
                                    ok = False
                                    break
                            if ok:
                                for x in Lj:
                                    for v in Ri:
                                        if adj_matrix[x][v]:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                            if ok:
                                new_L2 = set(Lj) | {u}
                                new_R2 = set(Rj) | set(Ri)
                                bicliques[j] = (list(new_L2), list(new_R2))
                                bicliques.pop(new_idx)
                                can_merge = True
                                break
                        if not can_merge:
                            # Revert
                            bicliques.pop()
                            bicliques[i] = (Li, list(Ri))
                            continue
                        moved = True
                        break
                    if moved:
                        break

                if not moved:
                    break

        return bicliques

    best_bicliques = None
    best_count = float('inf')

    # Phase 1: Multiple greedy restarts with varying strategies
    num_restarts = 200
    for s in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        b = greedy_partition(s)
        b = try_merge(b, adj_matrix)
        if len(b) < best_count:
            best_count = len(b)
            best_bicliques = b

    # Phase 2: Local search on best solutions
    if best_bicliques is not None:
        improved = local_search(best_bicliques, adj_matrix, deadline)
        if len(improved) < best_count:
            best_count = len(improved)
            best_bicliques = improved

    # Phase 3: Apply local search to second-best as well
    # (already covered by the loop above if we found multiple good ones)

    if best_bicliques is None:
        return [[[u], [v]] for u, v in edges]

    # Final validation
    covered = set()
    valid = True
    for L, R in best_bicliques:
        if not L or not R:
            valid = False
            break
        if set(L) & set(R):
            valid = False
            break
        for u in L:
            for v in R:
                k = _edge_key(u, v)
                if k not in edge_set:
                    valid = False
                    break
                if k in covered:
                    valid = False
                    break
                covered.add(k)

    if not valid or covered != edge_set:
        return [[[u], [v]] for u, v in edges]

    result = []
    for L, R in best_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])

    return result

# EVOLVE-BLOCK-END
