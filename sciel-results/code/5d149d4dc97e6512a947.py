import time
import random
import numpy as np

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def build_adj():
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        adj[0, 1] = 1
        adj[1, 0] = 1
        return adj

    def build_masks(adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for i in range(n):
                if adj[u, i]:
                    m |= (1 << i)
            masks[u] = m
        return masks

    def compute_triangles(adj, masks):
        """Return list of (u, v, w) for all triangles."""
        tris = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    cm = masks[u] & masks[v]
                    while cm:
                        lsb = cm & -cm
                        w = lsb.bit_length() - 1
                        if w > v:
                            tris.append((u, v, w))
                        cm &= cm - 1
        return tris

    def score_from_triangles(tris, deg):
        mx = 0
        for (u, v, w) in tris:
            s = deg[u] + deg[v] + deg[w]
            if s > mx:
                mx = s
        return mx

    def get_score(adj, masks, deg):
        tris = compute_triangles(adj, masks)
        return score_from_triangles(tris, deg)

    # Initial graph
    adj = build_adj()
    masks = build_masks(adj)
    deg = adj.sum(axis=1).astype(int).tolist()

    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Current state
    current_adj = adj.copy()
    current_masks = masks.copy()
    current_deg = deg.copy()
    current_tris = compute_triangles(current_adj, current_masks)
    current_score = score_from_triangles(current_tris, current_deg)

    best_adj = current_adj.copy()
    best_score = current_score
    best_masks = current_masks.copy()
    best_deg = current_deg.copy()

    # Track worst triangle info for guided moves
    def get_worst_triangles(tris, deg, k=5):
        """Return top-k triangles by degree sum, with their edge pairs."""
        scored = []
        for (u, v, w) in tris:
            s = deg[u] + deg[v] + deg[w]
            scored.append((s, (u, v), (u, w), (v, w), (u, v, w)))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    # Simulated annealing with triangle-targeted moves
    T_start = 4.0
    T_end = 0.05
    total_iters = 300000

    for it in range(total_iters):
        if time.time() > deadline:
            break

        frac = min(1.0, it / total_iters)
        T = T_start * (T_end / T_start) ** frac

        # Get worst triangles (cached, only recompute when graph changes)
        worst_tris = get_worst_triangles(current_tris, current_deg, k=4)

        # Choose move type: 70% target worst triangle, 30% random
        if worst_tris and random.random() < 0.70:
            # Pick a random edge from one of the worst triangles
            wt = random.choice(worst_tris)
            edge_pairs = [wt[1], wt[2], wt[3]]
            u, v = random.choice(edge_pairs)
        else:
            # Pick random existing edge
            existing = [(i, j) for (i, j) in all_pairs if current_adj[i, j]]
            if not existing:
                break
            u, v = random.choice(existing)

        # Choose non-edge to add: prefer edges that won't create high-degree triangles
        # Build candidate non-edges with scores
        non_edges = []
        for a, b in all_pairs:
            if not current_adj[a, b]:
                # Estimate: if we add (a,b), which triangles are created?
                # Triangles created involve a common neighbor of a and b
                common = current_masks[a] & current_masks[b]
                # Max degree among common neighbors
                max_common_deg = 0
                while common:
                    lsb = common & -common
                    c = lsb.bit_length() - 1
                    if current_deg[c] > max_common_deg:
                        max_common_deg = current_deg[c]
                    common &= common - 1
                # Score: lower is better (avoid creating high-degree triangles)
                est_max = max(current_deg[a] + 1, current_deg[b] + 1, max_common_deg)
                penalty = current_deg[a] + current_deg[b] + max_common_deg
                non_edges.append((a, b, penalty))

        if not non_edges:
            break

        # Weighted selection: lower penalty = higher weight
        # Use softmax-like weighting
        if T > 0.1:
            # Temperature-scaled random
            weights = []
            for a, b, p in non_edges:
                w = np.exp(-p / max(T, 0.1))
                weights.append(w)
            total_w = sum(weights)
            if total_w < 1e-15:
                x, y = random.choice(non_edges)[0], random.choice(non_edges)[1]
            else:
                r = random.random() * total_w
                cum = 0.0
                x, y = non_edges[0][0], non_edges[0][1]
                for (a, b, p), w in zip(non_edges, weights):
                    cum += w
                    if r <= cum:
                        x, y = a, b
                        break
        else:
            # Greedy: pick lowest penalty
            best_idx = 0
            best_p = non_edges[0][2]
            for idx, (a, b, p) in enumerate(non_edges):
                if p < best_p:
                    best_p = p
                    best_idx = idx
            x, y = non_edges[best_idx][0], non_edges[best_idx][1]

        # Ensure we're not removing and adding the same edge
        if (u == x and v == y) or (u == y and v == x):
            continue

        # Apply move
        current_adj[u, v] = 0
        current_adj[v, u] = 0
        current_adj[x, y] = 1
        current_adj[y, x] = 1

        current_masks[u] &= ~(1 << v)
        current_masks[v] &= ~(1 << u)
        current_masks[x] |= (1 << y)
        current_masks[y] |= (1 << x)

        current_deg[u] -= 1
        current_deg[v] -= 1
        current_deg[x] += 1
        current_deg[y] += 1

        # Recompute triangles and score
        new_tris = compute_triangles(current_adj, current_masks)
        new_score = score_from_triangles(new_tris, current_deg)

        # Accept/reject
        if new_score < current_score:
            current_score = new_score
            current_tris = new_tris
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
                best_masks = current_masks.copy()
                best_deg = current_deg.copy()
        else:
            delta = new_score - current_score
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                current_score = new_score
                current_tris = new_tris
            else:
                # Revert
                current_adj[u, v] = 1
                current_adj[v, u] = 1
                current_adj[x, y] = 0
                current_adj[y, x] = 0

                current_masks[u] |= (1 << v)
                current_masks[v] |= (1 << u)
                current_masks[x] &= ~(1 << y)
                current_masks[y] &= ~(1 << x)

                current_deg[u] += 1
                current_deg[v] += 1
                current_deg[x] -= 1
                current_deg[y] -= 1

    # Final deterministic polish: try all single edge swaps
    polish_adj = best_adj.copy()
    polish_masks = build_masks(polish_adj)
    polish_deg = polish_adj.sum(axis=1).astype(int).tolist()
    polish_tris = compute_triangles(polish_adj, polish_masks)
    polish_score = score_from_triangles(polish_tris, polish_deg)

    improved = True
    polish_deadline = deadline - 3
    while improved and time.time() < polish_deadline:
        improved = False
        for u, v in all_pairs:
            if time.time() > polish_deadline:
                break
            if not polish_adj[u, v]:
                continue
            for x, y in all_pairs:
                if time.time() > polish_deadline:
                    break
                if x == u or x == v or y == u or y == v:
                    continue
                if polish_adj[x, y]:
                    continue

                # Try swap
                polish_adj[u, v] = 0
                polish_adj[v, u] = 0
                polish_adj[x, y] = 1
                polish_adj[y, x] = 1

                polish_masks[u] &= ~(1 << v)
                polish_masks[v] &= ~(1 << u)
                polish_masks[x] |= (1 << y)
                polish_masks[y] |= (1 << x)

                polish_deg[u] -= 1
                polish_deg[v] -= 1
                polish_deg[x] += 1
                polish_deg[y] += 1

                new_tris = compute_triangles(polish_adj, polish_masks)
                new_polish_score = score_from_triangles(new_tris, polish_deg)

                if new_polish_score < polish_score:
                    polish_score = new_polish_score
                    polish_tris = new_tris
                    improved = True
                else:
                    # Revert
                    polish_adj[u, v] = 1
                    polish_adj[v, u] = 1
                    polish_adj[x, y] = 0
                    polish_adj[y, x] = 0

                    polish_masks[u] |= (1 << v)
                    polish_masks[v] |= (1 << u)
                    polish_masks[x] &= ~(1 << y)
                    polish_masks[y] &= ~(1 << x)

                    polish_deg[u] += 1
                    polish_deg[v] += 1
                    polish_deg[x] -= 1
                    polish_deg[y] -= 1

    if polish_score < best_score:
        best_adj = polish_adj
        best_score = polish_score

    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])

    return {'n': 16, 'edges': best_edges}
