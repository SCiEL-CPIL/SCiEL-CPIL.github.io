import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 140.0
    N = 16
    M = N * (N - 1) // 2  # 120 possible edges

    def time_ok():
        return time.time() < DEADLINE

    # Precompute edge index mapping
    edge_idx = {}
    edges_list = []
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            edge_idx[(u, v)] = idx
            edges_list.append((u, v))
            idx += 1

    # Build conflict pairs: edges that form a triangle when both present
    # For each triple (u,v,w), edges (u,v), (v,w), (u,w) form a triangle
    # So any two of these three edges conflict (can't both be present)
    conflict_pairs = []
    for u in range(N):
        for v in range(u + 1, N):
            for w in range(v + 1, N):
                e1 = edge_idx[(u, v)]
                e2 = edge_idx[(v, w)]
                e3 = edge_idx[(u, w)]
                # Any pair among these three conflicts
                conflict_pairs.append((e1, e2))
                conflict_pairs.append((e1, e3))
                conflict_pairs.append((e2, e3))

    # Precompute conflict neighbors for each edge
    conflict_neighbors = [[] for _ in range(M)]
    for i, j in conflict_pairs:
        conflict_neighbors[i].append(j)
        conflict_neighbors[j].append(i)

    # Fast 3-colorability check
    def is_3_colorable(x):
        adj = [0] * N
        for i in range(M):
            if x[i]:
                u, v = edges_list[i]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        deg = [bin(adj[i]).count('1') for i in range(N)]
        order = list(range(N))
        order.sort(key=lambda i: -deg[i])
        adj_mask = [0] * N
        for i, v in enumerate(order):
            m = 0
            for j, w in enumerate(order):
                if j != i and (adj[v] >> w) & 1:
                    m |= (1 << j)
            adj_mask[i] = m

        def bt(i, colors):
            if i == N:
                return True
            mask = adj_mask[i]
            for c in range(3):
                if (colors[c] & mask) == 0:
                    newc = colors[c] | (1 << i)
                    if c == 0:
                        nc = (newc, colors[1], colors[2])
                    elif c == 1:
                        nc = (colors[0], newc, colors[2])
                    else:
                        nc = (colors[0], colors[1], newc)
                    if bt(i + 1, nc):
                        return True
            return False

        return bt(0, (0, 0, 0))

    best_score = -1
    best_x = None
    rng = random.Random(42)

    for restart in range(150):
        if not time_ok():
            break

        # Initialize with dense random, then fix triangles
        x = [1 if rng.random() < 0.55 else 0 for _ in range(M)]

        # Fix triangle violations iteratively
        for _ in range(100):
            found = False
            for i, j in conflict_pairs:
                if x[i] and x[j]:
                    if rng.random() < 0.5:
                        x[i] = 0
                    else:
                        x[j] = 0
                    found = True
            if not found:
                break

        # Local search
        for it in range(30000):
            if not time_ok():
                break

            # Check triangle-free
            tri_ok = True
            for i, j in conflict_pairs:
                if x[i] and x[j]:
                    tri_ok = False
                    break

            if not tri_ok:
                # Find and fix a violation
                for i, j in conflict_pairs:
                    if x[i] and x[j]:
                        if rng.random() < 0.5:
                            x[i] = 0
                        else:
                            x[j] = 0
                        break
                continue

            # Check 3-colorability
            if is_3_colorable(x):
                # Try to add an edge that doesn't create a triangle
                zeros = [i for i in range(M) if not x[i]]
                if zeros:
                    rng.shuffle(zeros)
                    for i in zeros[:20]:  # Try up to 20 random zeros
                        creates_tri = any(x[j] for j in conflict_neighbors[i])
                        if not creates_tri:
                            x[i] = 1
                            break
                continue

            # Valid solution
            score = sum(x)
            if score > best_score:
                best_score = score
                best_x = list(x)

            # Try to add an edge
            zeros = [i for i in range(M) if not x[i]]
            if zeros:
                rng.shuffle(zeros)
                for i in zeros[:10]:
                    creates_tri = any(x[j] for j in conflict_neighbors[i])
                    if not creates_tri:
                        x[i] = 1
                        break
            else:
                # All edges present, can't improve
                break

    if best_x is not None:
        edges = []
        for i in range(M):
            if best_x[i]:
                u, v = edges_list[i]
                edges.append([u, v])
        return edges

    # Fallback: Clebsch graph
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    edges = []
    n = len(vertices)
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vertices[u_idx] ^ vertices[v_idx]
            if bin(xor_val).count('1') == 4:
                edges.append([u_idx, v_idx])
    return edges
