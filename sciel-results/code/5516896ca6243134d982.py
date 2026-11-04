import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 143.0   # safety margin under 150s hard kill
    D = 11

    # ---- Build supports and vectorized conflict graph (intersection >= 3) ----
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)

    # One-hot matrix M[i, j] = 1 iff j in supports[i]
    M = np.zeros((N_sup, D), dtype=np.int8)
    for i, s in enumerate(supports):
        for j in s:
            M[i, j] = 1
    # Intersection sizes
    inter = M @ M.T  # (N_sup, N_sup), entries in {0,1,2,3,4}
    conflict = (inter >= 3)
    # Store neighbor bitsets for fast degree computation
    neighbor = [np.flatnonzero(conflict[i]) for i in range(N_sup)]
    conflict_rows = [np.zeros(N_sup, dtype=bool) for _ in range(N_sup)]
    for i in range(N_sup):
        conflict_rows[i][neighbor[i]] = True

    def greedy_min_degree(avail, rng):
        """Greedy independent set via minimum-degree heuristic (vectorized degrees)."""
        avail = np.array(sorted(avail), dtype=np.int64)
        selected = []
        in_set = np.zeros(N_sup, dtype=bool)
        # degree[u] = number of neighbors of u that are still available
        deg = np.zeros(N_sup, dtype=np.int64)
        for u in avail:
            deg[u] = conflict_rows[u][avail].sum()
        while avail.size > 0:
            if time.time() > deadline - 12.0:
                break
            m = deg[avail].min()
            cands = avail[deg[avail] == m]
            u = int(cands[rng.integers(len(cands))])
            selected.append(u)
            in_set[u] = True
            # remove u and its neighbors
            to_remove = np.unique(np.concatenate(([u], neighbor[u])))
            mask = np.isin(avail, to_remove)
            avail = avail[~mask]
            # update degrees: for each remaining node, decrement if u was a neighbor
            # neighbors of remaining nodes that equal u
            # compute which remaining nodes are neighbors of u
            rem_is_neigh = conflict_rows[u][avail]
            if rem_is_neigh.any():
                deg[avail[rem_is_neigh]] -= 1
        return selected

    def local_search(selected, rng):
        """1-2 swaps: try removing one selected node and adding two compatible nodes."""
        selected = set(selected)
        improved = True
        stall = 0
        while improved and time.time() < deadline - 12.0:
            improved = False
            stall += 1
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for v in sel_list:
                if time.time() > deadline - 12.0:
                    break
                sel_minus_v = selected - {v}
                # Candidates are neighbors of v (excluding v itself)
                # that are not in sel_minus_v and not blocked by sel_minus_v
                cands = []
                for u in neighbor[v]:
                    if u == v:
                        continue
                    if u in sel_minus_v:
                        continue
                    # Check if u conflicts with any node in sel_minus_v
                    is_blocked = False
                    for w in sel_minus_v:
                        if conflict_rows[u][w]:
                            is_blocked = True
                            break
                    if not is_blocked:
                        cands.append(u)
                
                if len(cands) < 2:
                    continue
                
                found = False
                n = len(cands)
                for i in range(n):
                    u1 = cands[i]
                    for j in range(i + 1, n):
                        u2 = cands[j]
                        if not conflict_rows[u1][u2]:
                            selected.discard(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break # Restart while loop with new set
            if not improved and stall > 5:
                break
        return selected

    best_iset = set()
    best_size = 0
    rng = np.random.default_rng(42)

    n_restarts = 500
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        avail = set(range(N_sup))
        iset = set(greedy_min_degree(avail, rng))
        iset = local_search(iset, rng)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset = set(iset)
            if time.time() < deadline - 20.0:
                iset2 = local_search(iset, rng)
                if len(iset2) > best_size:
                    best_size = len(iset2)
                    best_iset = set(iset2)
        if r % 25 == 24 and best_iset and time.time() < deadline - 15.0:
            imp = local_search(best_iset, rng)
            if len(imp) > best_size:
                best_size = len(imp)
                best_iset = set(imp)

    # Final exact polish pass on the best set (add any non-conflicting node)
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for x in range(N_sup):
                if x in best_iset:
                    continue
                if not np.any(conflict_rows[x][list(best_iset)]):
                    best_iset.add(x)
                    changed = True

    # ---- Assemble vectors ----
    vectors = []
    # Type A: ±2 e_i  (22 vectors), always compatible
    for i in range(D):
        for sgn in (1, -1):
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * sgn
            vectors.append(v)
    # Type D: 16 sign vectors per selected support
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
