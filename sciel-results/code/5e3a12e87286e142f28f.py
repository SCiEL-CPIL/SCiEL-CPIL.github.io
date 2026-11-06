import time
import random
import sys

def _build_k32():
    """Build the 32-vertex 4-coloring from GF(16) cubic residues."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

_K32 = _build_k32()

def _validate_fast(colors, n):
    """Fast validation: no color class contains a triangle."""
    adj = [0] * (4 * n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c * n + i] |= (1 << j)
            adj[c * n + j] |= (1 << i)
            idx += 1

    for c in range(4):
        base = c * n
        for i in range(n):
            nbrs = adj[base + i]
            temp = nbrs
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                temp ^= lsb
                if adj[base + j] & nbrs:
                    return False
    return True

def _add_vertex_bitmask(colors, n, deadline, max_attempts=500):
    """
    Try to add one vertex using bitmask-based conflict tracking.
    Returns (new_colors, n+1) on success, (None, n) on failure.
    """
    # Precompute adjacency bitmasks for each color
    adj_bm = [0] * (4 * n)  # adj_bm[c*n + v] = bitmask of neighbors of v in color c
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_bm[c * n + i] |= (1 << j)
            adj_bm[c * n + j] |= (1 << i)
            idx += 1

    # Precompute: for each vertex v, total degree
    deg = [0] * n
    for c in range(4):
        for v in range(n):
            bm = adj_bm[c * n + v]
            deg[v] += bm.bit_count()

    best_assignment = None

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        # Choose vertex ordering: mix of random and most-constrained-first
        # Use a weighted approach: high degree vertices more likely to come first
        if attempt % 3 == 0:
            # Most constrained first
            order = sorted(range(n), key=lambda v: -deg[v])
        elif attempt % 3 == 1:
            # Random
            order = list(range(n))
            random.shuffle(order)
        else:
            # Semi-random: sort by degree with some noise
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda v: -deg[v] + random.uniform(-2, 2))

        assignment = [-1] * n
        # conflict_bm[c] = bitmask of vertices that cannot be assigned color c
        conflict_bm = [0] * 4
        # count[c] = number of vertices assigned to color c
        count = [0] * 4

        success = True
        for v in order:
            # Find valid colors
            valid = []
            for c in range(4):
                if not (conflict_bm[c] >> v) & 1:
                    valid.append(c)

            if not valid:
                success = False
                break

            # Choose color: prefer least-used valid color with some randomness
            if len(valid) == 1:
                c = valid[0]
            else:
                # Weight inversely by count, add small random perturbation
                weights = []
                for cc in valid:
                    w = 1.0 / (count[cc] + 1.0) + random.uniform(0, 0.1)
                    weights.append(w)
                total = sum(weights)
                r = random.random() * total
                cum = 0.0
                c = valid[0]
                for idx2, cc in enumerate(valid):
                    cum += weights[idx2]
                    if r <= cum:
                        c = cc
                        break

            assignment[v] = c
            count[c] += 1

            # Update conflict: all neighbors of v in color c now conflict with color c
            nbrs = adj_bm[c * n + v]
            conflict_bm[c] |= nbrs

        if success:
            best_assignment = assignment
            break

    if best_assignment is None:
        return None, n

    new_colors = list(colors)
    for v in range(n):
        new_colors.append(best_assignment[v])
    return new_colors, n + 1

def _add_vertex_backtrack_bm(colors, n, deadline):
    """
    Backtracking search to add a vertex using bitmask conflict tracking.
    """
    adj_bm = [0] * (4 * n)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_bm[c * n + i] |= (1 << j)
            adj_bm[c * n + j] |= (1 << i)
            idx += 1

    deg = [0] * n
    for c in range(4):
        for v in range(n):
            deg[v] += adj_bm[c * n + v].bit_count()

    order = sorted(range(n), key=lambda v: -deg[v])
    # Map: position in order -> vertex
    pos_to_v = order
    # Map: vertex -> position in order
    v_to_pos = [0] * n
    for pos, v in enumerate(order):
        v_to_pos[v] = pos

    assignment = [-1] * n
    conflict_bm = [0] * 4

    nodes_visited = 0
    max_nodes = 200000

    def backtrack(pos):
        nonlocal nodes_visited
        if time.time() > deadline:
            return False
        if nodes_visited > max_nodes:
            return False
        nodes_visited += 1

        if pos == n:
            return True

        v = pos_to_v[pos]
        v_mask = 1 << v

        # Try colors, preferring those with fewer conflicts
        color_order = sorted(range(4), key=lambda c: (conflict_bm[c] >> v) & 1)

        for c in color_order:
            if (conflict_bm[c] >> v) & 1:
                continue

            assignment[v] = c
            nbrs = adj_bm[c * n + v]
            old_conflict = conflict_bm[c]
            conflict_bm[c] = old_conflict | nbrs

            if backtrack(pos + 1):
                return True

            conflict_bm[c] = old_conflict
            assignment[v] = -1

        return False

    if backtrack(0):
        new_colors = list(colors)
        for v in range(n):
            new_colors.append(assignment[v])
        return new_colors, n + 1

    return None, n

def _local_search_fix(colors, n, deadline, max_iter=200):
    """
    Local search: try to fix a near-miss assignment by flipping individual colors.
    Used when greedy/backtrack fails but we have a partial assignment.
    """
    # This is a fallback; for now just return None
    # A proper implementation would take a partial assignment and try to repair it
    return None, n

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 40s margin before 150s kill

    best_n = 32
    best_colors = list(_K32)

    current_n = best_n
    current_colors = best_colors

    # Phase 1: Greedy extension with bitmask (fast)
    phase1_deadline = deadline - 40  # Leave 40s for later phases
    for _ in range(30):
        if time.time() > phase1_deadline:
            break
        new_colors, new_n = _add_vertex_bitmask(current_colors, current_n, phase1_deadline)
        if new_colors is None:
            break
        current_n = new_n
        current_colors = new_colors
        if current_n > best_n:
            best_n = current_n
            best_colors = current_colors

    # Phase 2: Backtrack for one more vertex if greedy got stuck
    if time.time() < deadline - 15:
        new_colors, new_n = _add_vertex_backtrack_bm(current_colors, current_n, deadline - 10)
        if new_colors is not None and new_n > current_n:
            current_n = new_n
            current_colors = new_colors
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors

    # Final validation
    if time.time() < deadline:
        if not _validate_fast(best_colors, best_n):
            best_n = 32
            best_colors = list(_K32)

    return {'n': best_n, 'colors': best_colors}
