import time
import random
import sys

N = 9
MAX_V = 1 << N

sys.setrecursionlimit(100000)

# Precompute neighbors for all vertices
_NEIGHBORS = [[] for _ in range(MAX_V)]
for v in range(MAX_V):
    _NEIGHBORS[v] = [v ^ (1 << b) for b in range(N)]

# Precompute popcount on a numpy array for SA distance checks
import numpy as np
_POP = np.fromiter((bin(i).count('1') for i in range(MAX_V)), dtype=np.uint8, count=MAX_V)


def construct():
    deadline = time.time() + 112.0  # safety margin under the 150s wall-clock kill

    best_path = [0, 1]
    best_len = 2

    # Symmetry-breaking canonical start: 0 -> 1 (flip bit 0)
    # After placing 0,1, a vertex x is a valid extension iff it is adjacent to the
    # current end AND adjacent to no other used vertex (block_count == 1).
    # With used = {0,1}: block_count[x] = (# used neighbors of x). x is valid
    # from cur=1 iff block_count[x]==1 and x is a neighbor of 1.

    def compute_block_count(path):
        """block_count[v] = number of neighbors of v that are in path."""
        bc = [0] * MAX_V
        for v in path:
            for nb in _NEIGHBORS[v]:
                bc[nb] += 1
        return bc

    def valid_candidates(cur, used_set, bc):
        """Candidates adjacent to cur with bc==1 (adjacent only to cur)."""
        out = []
        for nb in _NEIGHBORS[cur]:
            if nb in used_set:
                continue
            if bc[nb] == 1:
                out.append(nb)
        return out

    def warnsdorff_deg(v, used_set, bc):
        """Number of valid extensions from v (neighbors with bc==0)."""
        deg = 0
        for nn in _NEIGHBORS[v]:
            if nn in used_set:
                continue
            if bc[nn] == 0:
                deg += 1
        return deg

    def dfs(cur, used_set, bc, path, depth_limit, rng):
        """Randomized DFS with Warnsdorff ordering; updates best_path."""
        nonlocal best_path, best_len, deadline

        if time.time() > deadline:
            return

        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

        if len(path) >= depth_limit:
            return

        cands = valid_candidates(cur, used_set, bc)
        if not cands:
            return

        # Warnsdorff: prefer low degree (avoid getting trapped)
        scored = [(warnsdorff_deg(v, used_set, bc), rng.random(), v) for v in cands]
        scored.sort(key=lambda t: (t[0], t[1]))

        for _deg, _r, v in scored:
            if time.time() > deadline:
                return
            # place v
            path.append(v)
            used_set.add(v)
            for nn in _NEIGHBORS[v]:
                bc[nn] += 1

            dfs(v, used_set, bc, path, depth_limit, rng)

            # undo
            path.pop()
            used_set.remove(v)
            for nn in _NEIGHBORS[v]:
                bc[nn] -= 1

            if time.time() > deadline:
                return

    def extend_both_ends(path, max_iters=40):
        """Greedily extend at both ends using block_count; returns new path."""
        nonlocal deadline
        used_set = set(path)
        bc = compute_block_count(path)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            # Try front
            first = path[0]
            for nb in _NEIGHBORS[first]:
                if nb in used_set:
                    continue
                if bc[nb] == 1:
                    path.insert(0, nb)
                    used_set.add(nb)
                    for nn in _NEIGHBORS[nb]:
                        bc[nn] += 1
                    improved = True
                    break
            if improved:
                continue
            # Try back
            last = path[-1]
            for nb in _NEIGHBORS[last]:
                if nb in used_set:
                    continue
                if bc[nb] == 1:
                    path.append(nb)
                    used_set.add(nb)
                    for nn in _NEIGHBORS[nb]:
                        bc[nn] += 1
                    improved = True
                    break
        return path

    def sa_refine(path, iterations=40000):
        """Simulated annealing with 2-bit flip moves preserving validity.
        Returns a (possibly) improved path of the same length."""
        nonlocal deadline
        cur = list(path)
        L = len(cur)
        used_set = set(cur)
        # For fast validity: a vertex u is "safe" for position i if it is at
        # distance >= 2 from all path vertices except the two adjacent ones.
        best = list(cur)
        best_len_local = L
        T = 4.0
        cooling = 0.99985

        for it in range(iterations):
            if time.time() > deadline:
                break
            if L < 3:
                break
            pos = rng.randint(1, L - 2)
            v = cur[pos]
            b1 = rng.randint(0, N - 1)
            b2 = rng.randint(0, N - 1)
            while b2 == b1:
                b2 = rng.randint(0, N - 1)
            nv = v ^ (1 << b1) ^ (1 << b2)
            if nv == v or nv in used_set:
                continue
            # adjacent to both neighbors?
            if _POP[nv ^ cur[pos - 1]] != 1:
                continue
            if _POP[nv ^ cur[pos + 1]] != 1:
                continue
            # distance >= 2 from all other vertices
            ok = True
            for i, u in enumerate(cur):
                if i == pos - 1 or i == pos + 1:
                    continue
                if _POP[nv ^ u] < 2:
                    ok = False
                    break
            if not ok:
                continue
            # valid move
            used_set.discard(v)
            used_set.add(nv)
            cur[pos] = nv
            # accept with probability T (cooling)
            if rng.random() > T:
                cur[pos] = v
                used_set.discard(nv)
                used_set.add(v)
            T *= cooling
        return cur

    # ---------------- Search driver ----------------
    rng = random.Random(12345)

    # Phase 1: multiple time-sliced randomized DFS from the symmetry-broken start.
    # Each DFS explores a different random branch; the fast block_count makes deep
    # exploration feasible, and multi-start gives diverse coverage of Q_9.
    search_id = 0
    while time.time() < deadline - 25.0:
        rng = random.Random(1000 + search_id)
        path = [0, 1]
        used_set = {0, 1}
        bc = compute_block_count(path)
        # deeper search on early attempts, shallower later for diversity
        depth_limit = 60 if search_id < 6 else 45
        dfs(1, used_set, bc, path, depth_limit, rng)
        search_id += 1
        if search_id % 3 == 0 and time.time() < deadline - 25.0:
            # extend the best path found so far at both ends
            ext = extend_both_ends(list(best_path), max_iters=40)
            if len(ext) > best_len:
                best_len = len(ext)
                best_path = ext

    # Phase 2: extend best path at both ends
    ext = extend_both_ends(list(best_path), max_iters=60)
    if len(ext) > best_len:
        best_len = len(ext)
        best_path = ext

    # Phase 3: SA refinement to unlock additional extensions, then re-extend
    if time.time() < deadline - 8.0:
        rng = random.Random(777)
        refined = sa_refine(list(best_path), iterations=60000)
        ext2 = extend_both_ends(refined, max_iters=60)
        if len(ext2) > best_len:
            best_len = len(ext2)
            best_path = ext2

    # Phase 4: if time remains, a couple more DFS restarts from best start
    if time.time() < deadline - 6.0:
        for k in range(3):
            if time.time() > deadline - 5.0:
                break
            rng = random.Random(9000 + k)
            path = [0, 1]
            used_set = {0, 1}
            bc = compute_block_count(path)
            dfs(1, used_set, bc, path, 50, rng)
            ext = extend_both_ends(list(best_path), max_iters=30)
            if len(ext) > best_len:
                best_len = len(ext)
                best_path = ext

    return best_path
