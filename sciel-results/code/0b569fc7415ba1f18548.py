import time
import random
import numpy as np


def _has_c4(adj, n):
    """Check if graph has a C4: any pair with 2+ common neighbors."""
    for u in range(n):
        au = adj[u]
        for v in au:
            if v > u:
                common = au & adj[v]
                if len(common) >= 2:
                    return True
    return False


def _min_degree(adj, n):
    return min(len(adj[i]) for i in range(n))


def _bipartite_has_c4(adj, n, A, B):
    """Check C4 only within the bipartite structure (faster)."""
    # For each pair in A, common neighbors in B must be < 2
    for i in range(len(A)):
        ai = adj[A[i]]
        for j in range(i + 1, len(A)):
            if len(ai & adj[A[j]]) >= 2:
                return True
    for i in range(len(B)):
        bi = adj[B[i]]
        for j in range(i + 1, len(B)):
            if len(bi & adj[B[j]]) >= 2:
                return True
    return False


def _try_bipartite(n, target_deg, seed, time_limit):
    """Build a bipartite C4-free graph on n vertices with min degree >= target_deg."""
    rng = random.Random(seed)
    # Split vertices into two partitions
    # Try balanced and slightly unbalanced splits
    best = None

    for a_size in range(max(1, n // 3), n // 2 + 1):
        A = list(range(a_size))
        B = list(range(a_size, n))
        b_size = len(B)

        # Target degree for A-vertices: at least target_deg
        # Target degree for B-vertices: at least target_deg
        # But B-vertices can only connect to A, so max degree = a_size
        # And A-vertices max degree = b_size
        if a_size < target_deg or b_size < target_deg:
            continue

        # Greedy: for each A-vertex, pick target_deg neighbors in B
        # such that no two A-vertices share 2+ B-neighbors
        # This is like a packing / set system problem
        # Use greedy with randomization

        adj = [set() for _ in range(n)]
        # Map A-vertex i (0-indexed in A) to actual vertex i
        # Map B-vertex j (0-indexed in B) to actual vertex a_size + j

        # For each A-vertex, we need to pick target_deg B-neighbors
        # Constraint: any pair of A-vertices share at most 1 B-neighbor
        # This is equivalent to: the incidence matrix has pairwise inner product <= 1

        # Greedy with backtracking-free approach:
        # Process A-vertices one by one, picking neighbors that minimize conflicts

        a_neighbors = []  # list of sets of B-indices for each A-vertex
        for i in range(a_size):
            # Pick target_deg B-vertices for A-vertex i
            # Prioritize B-vertices with low current "load" (number of A-neighbors)
            # But also avoid creating 2+ shared with any previous A-vertex

            b_load = [0] * b_size
            for prev_set in a_neighbors:
                for b in prev_set:
                    b_load[b] += 1

            # Sort B-vertices by load (ascending), with random tie-breaking
            b_order = sorted(range(b_size), key=lambda b: (b_load[b], rng.random()))

            chosen = []
            # Greedily pick B-vertices
            for b in b_order:
                if len(chosen) >= target_deg:
                    break
                # Check: adding b to A-vertex i, does it create 2+ shared with any prev?
                ok = True
                for prev_set in a_neighbors:
                    if b in prev_set:
                        # Count how many shared
                        shared = len(chosen & prev_set)
                        if shared >= 1:
                            # Adding b makes shared = shared+1 >= 2
                            ok = False
                            break
                if ok:
                    chosen.append(b)

            if len(chosen) < target_deg:
                return None  # Can't achieve target for this A-vertex

            a_neighbors.append(set(chosen))
            # Update adj
            actual_i = A[i]
            for b in chosen:
                actual_b = B[b]
                adj[actual_i].add(actual_b)
                adj[actual_b].add(actual_i)

        # Verify: check all A-vertices have degree >= target_deg
        for i in range(a_size):
            if len(adj[A[i]]) < target_deg:
                return None
        # Check all B-vertices have degree >= target_deg
        for j in range(b_size):
            if len(adj[B[j]]) < target_deg:
                return None

        # Verify C4-free
        if _bipartite_has_c4(adj, n, A, B):
            return None

        # Build edge list
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    return None


def _try_greedy_general(n, target_deg, seed, time_limit):
    """General greedy: add edges to maximize min degree while avoiding C4."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Priority: add edges involving lowest-degree vertices first
    # Use a simple approach: multiple passes, each pass tries to raise the min degree

    for _pass in range(50):
        if time.time() > time_limit:
            break
        # Find vertices below target
        low_verts = [i for i in range(n) if len(adj[i]) < target_deg]
        if not low_verts:
            break

        # For each low vertex, try to add edges to non-neighbors
        for u in low_verts:
            if len(adj[u]) >= target_deg:
                continue
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            rng.shuffle(candidates)
            added = 0
            for v in candidates:
                if len(adj[u]) >= target_deg:
                    break
                # Check if adding (u,v) creates a C4
                # C4 created iff exists a,b: u~a, a~b, b~v, a!=b, a!=v, b!=u
                nu = adj[u] - {v}
                nv = adj[v] - {u}
                creates_c4 = False
                for a in nu:
                    if a in nv:
                        continue  # triangle, not C4
                    for b in adj[a]:
                        if b in nv and b != a and b != u and b != v:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                if not creates_c4:
                    adj[u].add(v)
                    adj[v].add(u)
                    added += 1
            if added == 0:
                # No progress, reshuffle and try again in next pass
                pass

    md = min(len(adj[i]) for i in range(n))
    if md >= target_deg:
        if _has_c4(adj, n):
            return None
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    best_n = 39
    best_edges = []

    # Try n from 47 down to 40
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        target_deg = max(1, n - 39)

        # First try bipartite construction (faster, more reliable)
        for seed in range(300):
            if time.time() > deadline:
                break
            edges = _try_bipartite(n, target_deg, seed, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break

        if best_n >= n:
            continue

        # Then try general greedy
        for seed in range(200):
            if time.time() > deadline:
                break
            edges = _try_greedy_general(n, target_deg, seed, deadline)
            if edges is not None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break

        if time.time() > deadline:
            break

    if best_n <= 39:
        return {'n': 39, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
