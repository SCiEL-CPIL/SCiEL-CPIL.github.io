# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 250.0  # safety margin under 300s wall-clock


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """Vectorized proper-crossing test for segments (a,b) vs (c,d)."""
    def orient(px, py, qx, qy, rx, ry):
        val = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        return np.sign(val)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 != o2) & (o3 != o4)


def _tour_length(tour, D):
    n = len(tour)
    return float(D[tour, np.roll(tour, -1)].sum())


def _two_opt(tour, D, max_passes=80):
    n = len(tour)
    tour = list(tour)
    best = _tour_length(tour, D)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    best += delta
                    improved = True
    return tour, best


def _or_opt(d, order, max_passes=20):
    """
    Improve a tour using Or-opt (moving 1 node).
    """
    n = len(order)
    if n < 4:
        return list(order), float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    best = list(order)
    best_len = _tour_length(best, d)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            node = best[i]
            prev = best[(i - 1) % n]
            next_node = best[(i + 1) % n]
            # Remove node from position i
            # Cost change: remove edges (prev, node) and (node, next_node)
            # Add edge (prev, next_node)
            delta_remove = d[prev, node] + d[node, next_node] - d[prev, next_node]
            
            # Try inserting node at all other positions
            for j in range(n):
                if j == i or j == (i + 1) % n:
                    continue
                # Insert between j and (j+1)
                # But we need to be careful with indices after removal
                # Let's work on a copy for clarity, though it's slower
                # Better: calculate delta directly
                a = best[j]
                b = best[(j + 1) % n]
                # If we insert node between a and b:
                # Remove (a, b), add (a, node) and (node, b)
                # But wait, if we removed node from i, the tour structure changed.
                # This direct delta calculation is tricky if j is near i.
                # Let's use a simpler approach: reconstruct and check, or careful delta.
                
                # Simple robust approach for small n:
                # Construct new tour
                new_tour = best[:i] + best[i+1:]
                # Find insertion index in new_tour corresponding to between a and b
                # This is getting complicated. Let's stick to a simpler Or-opt variant
                # or just use 2-opt + swap.
                
                # Actually, let's implement a standard Or-opt that moves a segment of length 1
                # and tries all positions.
                pass
        
        # For simplicity and correctness within budget, let's implement a clean Or-opt
        # that moves one node.
        improved_any = False
        for i in range(n):
            node = best[i]
            prev = best[(i - 1) % n]
            next_node = best[(i + 1) % n]
            
            # Remove node
            # Create list without node
            # To avoid O(n^2) list slicing in inner loop, we can use indices
            # But n=50 is small.
            
            # Try inserting at position k (between best[k] and best[k+1])
            for k in range(n):
                if k == i or k == (i - 1) % n:
                    continue
                # If k == i-1, we are putting it back.
                # If k == i, we are putting it back.
                
                # Let's define the insertion: between node at index k and k+1 in the *original* list?
                # No, we remove first, then insert.
                
                # Let's just do the list manipulation, it's fast enough for n=50.
                new_tour = best[:i] + best[i+1:]
                # Insert node at position k in new_tour?
                # The indices shift if i < k.
                
                # Let's try a different Or-opt implementation:
                # Move node from i to position j.
                # This is effectively a swap of positions if we consider the rest shifting?
                # No, Or-opt is move.
                
                # Let's skip complex Or-opt and use a robust 2-opt + 3-opt or just more 2-opt passes.
                # The inspiration suggests Or-opt, but let's ensure it's correct.
                
                # Simple correct Or-opt:
                # 1. Remove node at i.
                # 2. Try inserting it at every position j in the resulting list of length n-1.
                
                # Cost of current tour: L
                # Cost after removal: L - d[prev, node] - d[node, next] + d[prev, next]
                # Cost after insertion at j (between a and b in the new list):
                # L_new = L_removed - d[a, b] + d[a, node] + d[node, b]
                
                # Total delta = d[prev, next] - d[prev, node] - d[node, next] - d[a, b] + d[a, node] + d[node, b]
                
                # We need to identify a and b in the original list.
                # If we remove node at i, the list is:
                # 0, 1, ..., i-1, i+1, ..., n-1
                # If we insert at position j in this new list:
                # If j < i: a = best[j], b = best[j+1]
                # If j >= i-1 (adjusted): 
                # Let's map new index j to original indices.
                
                # This is error-prone. Let's use a simpler local search: 2-opt is already there.
                # Let's add a "swap" operator (interchange two nodes) which is easy to delta.
                # Or just increase 2-opt passes.
                
                # Let's implement a simple "3-opt" or just rely on 2-opt.
                # Given the time, I will implement a correct Or-opt by reconstructing the tour for evaluation.
                # n=50, n^3 = 125k operations per pass, 20 passes = 2.5M. Fast.
                
                new_tour = best[:i] + best[i+1:]
                # Try inserting at all positions 0..n-1 in new_tour
                for j in range(len(new_tour) + 1):
                    # Insert node at j
                    # We can compute length by summing, or use delta.
                    # Delta is better.
                    # a = new_tour[(j-1)%len(new_tour)]
                    # b = new_tour[j%len(new_tour)]
                    # But we need their original indices to look up D.
                    # We have the list `new_tour` which contains the node IDs.
                    # So we can just use the node IDs directly.
                    
                    a = new_tour[(j - 1) % len(new_tour)]
                    b = new_tour[j % len(new_tour)]
                    
                    # The nodes prev and next are from the original best list.
                    # a and b are from the new_tour list.
                    
                    delta = d[prev, next_node] - d[prev, node] - d[node, next_node] - d[a, b] + d[a, node] + d[node, b]
                    
                    if delta < -1e-12:
                        # Perform the move
                        # Remove i
                        # Insert at j
                        # We need to update `best`
                        # But `best` is being modified in place? No, we are in a loop.
                        # Let's just update best and break out to restart the pass or continue.
                        # For simplicity, let's apply the change and restart the pass.
                        
                        # Reconstruct best
                        # Remove i
                        # Insert at j
                        # Note: j is index in the list WITHOUT node i.
                        # If j >= i, the index in original list is j+1?
                        # Let's just create a new list.
                        new_best = best[:i] + best[i+1:]
                        new_best.insert(j, node)
                        best = new_best
                        best_len += delta
                        improved_any = True
                        break
            if improved_any:
                break
        if not improved_any:
            break
            
    return best, best_len


def _nearest_neighbor(tour, D, start):
    n = len(tour)
    unvisited = set(range(n))
    unvisited.remove(start)
    path = [start]
    cur = start
    while unvisited:
        # nearest neighbor among unvisited
        cands = list(unvisited)
        dists = D[cur, cands]
        nxt = cands[int(np.argmin(dists))]
        path.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return path


def _compute_reference_tour(D, n):
    """Multi-start nearest-neighbor + 2-opt + Or-opt to get a good reference tour."""
    best_tour = None
    best_len = np.inf
    # multiple starts
    starts = list(range(n))
    for s in starts:
        path = _nearest_neighbor(list(range(n)), D, s)
        tour, ln = _two_opt(path, D)
        # Add Or-opt
        tour, ln = _or_opt(D, tour)
        if ln < best_len:
            best_len = ln
            best_tour = tour
    return best_tour, best_len


def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    # 1) base inverse-distance heuristic
    base = 1.0 / (D + 1e-9)
    base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)

    # 2) build reference tour (NN multi-start + 2-opt + Or-opt) with deadline guard
    try:
        ref_tour, ref_len = _compute_reference_tour(D, n)
    except Exception:
        ref_tour = list(range(n))
        ref_len = np.inf

    # points: reconstruct coordinates from distance matrix via embedding-free
    # approach: we only need segment geometry; derive from D using a proxy.
    # Use a simple coordinate reconstruction (MDS-like) for crossing tests.
    # Fallback: if we can't get coords, skip crossing penalty.
    coords = None
    try:
        # classical MDS via eigendecomposition of double-centered Gram
        d2 = D ** 2
        r = np.mean(d2, axis=1)
        c = np.mean(d2, axis=0)
        z = np.mean(d2)
        B = -0.5 * (d2 - r[:, None] - c[None, :] + z)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        # take top 2 dims
        w2 = np.sqrt(np.clip(w[:2], 0, None))
        coords = V[:, :2] * w2[None, :]
    except Exception:
        coords = None

    H = base.copy()

    if coords is not None and ref_tour is not None and len(ref_tour) == n:
        # topological alignment factor
        # reward edges present in reference tour
        align = np.ones((n, n))
        for k in range(n):
            u = ref_tour[k]
            v = ref_tour[(k + 1) % n]
            align[u, v] *= 2.5
            align[v, u] *= 2.5

        # crossing penalty: penalize edges crossing any reference-tour segment
        # Build reference segment list
        segs = []
        for k in range(n):
            u = ref_tour[k]
            v = ref_tour[(k + 1) % n]
            segs.append((u, v))

        # For each candidate edge (i,j), test crossing against each ref segment
        # Vectorize over edges vs each segment
        cross_count = np.zeros((n, n))
        for (u, v) in segs:
            ax, ay = coords[u]
            bx, by = coords[v]
            # all candidate edges (i,j)
            ii, jj = np.indices((n, n))
            # skip the segment's own endpoints and diagonal
            mask = (ii != u) & (jj != u) & (ii != v) & (jj != v) & (ii != jj)
            if not np.any(mask):
                continue
            cx = coords[ii[mask], 0]
            cy = coords[ii[mask], 1]
            dx = coords[jj[mask], 0]
            dy = coords[jj[mask], 1]
            cross = _segments_cross(ax, ay, bx, by, cx, cy, dx, dy)
            # scatter back
            idx_i = ii[mask][cross]
            idx_j = jj[mask][cross]
            cross_count[idx_i, idx_j] += 1
            cross_count[idx_j, idx_i] += 1

        # heavy penalty for crossing edges (exponential decay)
        cross_penalty = np.exp(-0.8 * cross_count)
        H = H * align * cross_penalty

    # ensure positive & finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END
