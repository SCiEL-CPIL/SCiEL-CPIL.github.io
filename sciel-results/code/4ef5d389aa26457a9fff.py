"""Fewest four-cycles in graphs with 31 vertices and 91 edges.

Strategy: build a C4-free polarity graph of PG(2,5) with 90 edges, then add
the single edge that creates the fewest C4s.  The polarity graph of PG(2,5)
is known to be C4-free.  Adding one edge creates C4s only through 3-paths
between the new edge's endpoints.  We try all 8 bilinear-form variants to
ensure we get exactly 90 edges, then do a local-search polish (edge swaps
that keep 91 edges) to reduce C4 count further.

The C4 count is computed exactly via:
  C4 = (1/2) * sum_{i<j} C(common_neighbors(i,j), 2)
"""
import time
import random
import numpy as np


def _pg2_points(q):
    """Generate all points of PG(2, q-1) over GF(q) as normalized tuples."""
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                idx = 0
                while idx < 3 and (x, y, z)[idx] == 0:
                    idx += 1
                factor = (x, y, z)[idx]
                inv_factor = pow(factor, q - 2, q)
                norm = tuple((x, y, z)[i] * inv_factor % q for i in range(3))
                if norm not in seen:
                    seen.add(norm)
                    points.append(norm)
    return points


def _bilinear(p, q_pt, q_mod=5, variant=0):
    """Symmetric bilinear form for polarity graph. variant selects coefficients."""
    x1, y1, z1 = p
    x2, y2, z2 = q_pt
    m = q_mod
    if variant == 0:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2) % m
    elif variant == 1:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2 + z1 * x2 + z2 * x1) % m
    elif variant == 2:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2 + z1 * y2 + z2 * y1) % m
    elif variant == 3:
        return (x1 * x2 + y1 * y2 + z1 * z2 + 2 * x1 * y2 + 2 * y1 * x2) % m
    elif variant == 4:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * z2 + z1 * x2) % m
    elif variant == 5:
        return (x1 * x2 + y1 * y2 + z1 * z2 + y1 * z2 + z1 * y2) % m
    elif variant == 6:
        return (x1 * x2 + y1 * y2 + z1 * z2 + 3 * x1 * y2 + 3 * y1 * x2) % m
    elif variant == 7:
        return (x1 * x2 + y1 * y2 + z1 * z2 + x1 * y2 + y1 * x2 + x1 * z2 + z1 * x2) % m
    return (x1 * x2 + y1 * y2 + z1 * z2) % m


def _count_c4(adj, n):
    """Exact C4 count: (1/2) * sum_{i<j} C(cn(i,j), 2) where cn = (A^2)[i,j]."""
    A = adj.astype(np.int32)
    A2 = A @ A
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            cn = A2[i, j]
            if cn >= 2:
                total += cn * (cn - 1) // 2
    return total // 2


def _c4s_containing_edge(adj, i, j, n):
    """Count C4s that contain the specific edge (i,j)."""
    count = 0
    for a in range(n):
        if a == i or a == j:
            continue
        if adj[j, a]:
            for b in range(n):
                if b == i or b == j or b == a:
                    continue
                if adj[i, b] and adj[a, b]:
                    count += 1
    return count


def _c4_delta_add_edge(adj, i, j, n):
    """Compute the change in C4 count when adding edge (i,j) to the graph."""
    count = 0
    for b in range(n):
        if b == i or b == j:
            continue
        if not adj[i, b]:
            continue
        for a in range(n):
            if a == i or a == j or a == b:
                continue
            if adj[j, a] and adj[b, a]:
                count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n = 31
    target_edges = 91

    points = _pg2_points(5)
    assert len(points) == 31

    # Try all variants and pick the one closest to 90 edges, preferring exactly 90
    best_adj = None
    best_edge_diff = float('inf')
    for variant in range(8):
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if _bilinear(points[i], points[j], 5, variant) == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        ec = int(adj.sum() // 2)
        diff = abs(ec - 90)
        if diff < best_edge_diff or (diff == best_edge_diff and best_adj is not None and ec == 90):
            best_edge_diff = diff
            best_adj = adj
        if ec == 90:
            break

    if best_adj is None:
        random.seed(42)
        best_adj = np.zeros((n, n), dtype=np.int8)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for i, j in all_pairs[:target_edges]:
            best_adj[i, j] = 1
            best_adj[j, i] = 1

    adj = best_adj.copy()
    current_edges = int(adj.sum() // 2)

    # Adjust to exactly target_edges
    if current_edges < target_edges:
        need = target_edges - current_edges
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        scores = []
        for (i, j) in non_edges:
            s = _c4_delta_add_edge(adj, i, j, n)
            scores.append((s, i, j))
        scores.sort()
        for s, i, j in scores[:need]:
            adj[i, j] = 1
            adj[j, i] = 1
    elif current_edges > target_edges:
        excess = current_edges - target_edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        # Remove edges with fewest C4s first (to preserve C4-free structure as much as possible)
        # Actually, we want to remove edges that are in the most C4s to reduce C4 count
        # But if the graph is C4-free, all have 0 C4s, so just remove any
        edge_c4 = {}
        for (i, j) in edges:
            edge_c4[(i, j)] = _c4s_containing_edge(adj, i, j, n)
        sorted_edges = sorted(edge_c4.items(), key=lambda x: -x[1])
        for e, c in sorted_edges[:excess]:
            i, j = e
            adj[i, j] = 0
            adj[j, i] = 0

    current_c4 = _count_c4(adj, n)
    best_adj = adj.copy()
    best_c4 = current_c4

    # Local search: greedy + random swaps
    random.seed(2024)
    no_improve = 0
    max_no_improve = 200000

    while time.time() < deadline and no_improve < max_no_improve:
        if time.time() > deadline:
            break

        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i, j]]

        if not edges or not non_edges:
            break

        improved = False

        # Greedy: try best swaps
        edge_c4 = {}
        for (i, j) in edges:
            edge_c4[(i, j)] = _c4s_containing_edge(adj, i, j, n)

        ne_c4 = {}
        for (k, l) in non_edges:
            ne_c4[(k, l)] = _c4_delta_add_edge(adj, k, l, n)

        sorted_edges = sorted(edge_c4.items(), key=lambda x: -x[1])
        sorted_non_edges = sorted(ne_c4.items(), key=lambda x: x[1])

        tried = set()
        for ei, (e, ec) in enumerate(sorted_edges[:5]):
            if time.time() > deadline:
                break
            for ni, (ne, nc) in enumerate(sorted_non_edges[:5]):
                if time.time() > deadline:
                    break
                key = (e, ne)
                if key in tried:
                    continue
                tried.add(key)

                i, j = e
                k, l = ne

                if {k, l} == {i, j}:
                    continue
                if adj[k, l]:
                    continue

                adj_temp = adj.copy()
                adj_temp[i, j] = 0
                adj_temp[j, i] = 0
                adj_temp[k, l] = 1
                adj_temp[l, k] = 1

                if int(adj_temp.sum() // 2) != target_edges:
                    continue

                c4_temp = _count_c4(adj_temp, n)
                if c4_temp < best_c4:
                    adj = adj_temp
                    best_adj = adj.copy()
                    best_c4 = c4_temp
                    no_improve = 0
                    improved = True
                    break
            if improved:
                break

        if not improved:
            no_improve += 1
            for _ in range(10):
                if time.time() > deadline:
                    break
                i, j = random.choice(edges)
                k, l = random.choice(non_edges)

                if {k, l} == {i, j}:
                    continue
                if adj[k, l]:
                    continue

                adj_temp = adj.copy()
                adj_temp[i, j] = 0
                adj_temp[j, i] = 0
                adj_temp[k, l] = 1
                adj_temp[l, k] = 1

                if int(adj_temp.sum() // 2) != target_edges:
                    continue

                c4_temp = _count_c4(adj_temp, n)
                if c4_temp < best_c4:
                    adj = adj_temp
                    best_adj = adj.copy()
                    best_c4 = c4_temp
                    no_improve = 0
                    improved = True
                    break

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                result.append([i, j])

    assert len(result) == target_edges, f"Expected {target_edges} edges, got {len(result)}"

    return result
