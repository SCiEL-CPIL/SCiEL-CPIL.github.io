"""Combined: efficient edge-list SA with soft-max surrogate for acceptance + targeted worst-triangle search."""
import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 140.0  # safety margin before 150s kill

def _score(adj, deg):
    """Compute max degree sum over all triangles (hard max)."""
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
    return best

def _soft_score(adj, deg, beta=2.0):
    """Soft-max surrogate: log-sum-exp of triangle degree sums.
    Provides a smooth landscape that penalizes ALL high-sum triangles,
    not just the maximum. Lower is better.
    """
    max_s = 0
    tri_sums = []
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    tri_sums.append(s)
                    if s > max_s:
                        max_s = s
    if not tri_sums:
        return 0.0
    # log-sum-exp for numerical stability
    log_sum = 0.0
    for s in tri_sums:
        log_sum += math.exp(beta * (s - max_s))
    return max_s + math.log(log_sum) / beta

def _apply_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[u][v] = adj[v][u] = False
    deg[u] -= 1
    deg[v] -= 1
    adj[x][y] = adj[y][x] = True
    deg[x] += 1
    deg[y] += 1

def _undo_swap(adj, deg, e_remove, e_add):
    u, v = e_remove
    x, y = e_add
    adj[x][y] = adj[y][x] = False
    deg[x] -= 1
    deg[y] -= 1
    adj[u][v] = adj[v][u] = True
    deg[u] += 1
    deg[v] += 1

def _find_worst_triangle(adj, deg):
    """Find the triangle with maximum degree sum."""
    worst_tri = None
    worst_sum = 0
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    s = deg[i] + deg[j] + deg[k]
                    if s > worst_sum:
                        worst_sum = s
                        worst_tri = (i, j, k)
    return worst_tri, worst_sum

def construct():
    t0 = time.time()
    global_best_score = float('inf')
    global_best_adj = None
    global_best_deg = None

    num_restarts = 12

    for seed in range(num_restarts):
        if time.time() - t0 > DEADLINE:
            break

        rng = random.Random(seed * 7919 + 31)

        # Initialize: K_{8,8} + one internal edge
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        # Random internal edge for diversity
        part = rng.choice([list(range(8)), list(range(8, 16))])
        u, v = rng.sample(part, 2)
        adj[u][v] = adj[v][u] = True

        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1

        cur_score = _score(adj, deg)
        cur_soft = _soft_score(adj, deg)
        best_adj = [row[:] for row in adj]
        best_deg = deg[:]
        best_score = cur_score

        # Build efficient edge/non-edge lists with index maps
        edges = []
        nonedges = []
        edge_index = {}
        nonedge_index = {}

        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    edges.append((i, j))
                    edge_index[(i, j)] = len(edges) - 1
                else:
                    nonedges.append((i, j))
                    nonedge_index[(i, j)] = len(nonedges) - 1

        def remove_edge_from_list(e):
            idx = edge_index[e]
            last = len(edges) - 1
            last_edge = edges[last]
            edges[idx] = last_edge
            edge_index[last_edge] = idx
            edges.pop()
            del edge_index[e]
            nonedges.append(e)
            nonedge_index[e] = len(nonedges) - 1

        def add_edge_to_list(e):
            idx = nonedge_index[e]
            last = len(nonedges) - 1
            last_nonedge = nonedges[last]
            nonedges[idx] = last_nonedge
            nonedge_index[last_nonedge] = idx
            nonedges.pop()
            del nonedge_index[e]
            edges.append(e)
            edge_index[e] = len(edges) - 1

        # Simulated annealing with adaptive temperature
        max_iters = 40000
        start_temp = 4.0
        end_temp = 0.05
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        iters_done = 0

        # Track for adaptive reheat
        no_improve_count = 0
        reheat_threshold = 3000
        reheat_temp = 2.0

        while iters_done < max_iters:
            if time.time() - t0 > DEADLINE:
                break

            # Targeted search: find worst triangle
            worst_tri, worst_sum = _find_worst_triangle(adj, deg)

            # Decide which edge to remove
            if worst_tri and rng.random() < 0.6:
                i, j, k = worst_tri
                candidates = [(i, j), (i, k), (j, k)]
                e_remove = candidates[rng.randrange(3)]
            else:
                e_remove = edges[rng.randrange(len(edges))]

            e_add = nonedges[rng.randrange(len(nonedges))]

            u, v = e_remove
            x, y = e_add

            if (u == x and v == y) or (u == y and v == x):
                iters_done += 1
                temp *= cooling_rate
                continue

            # Perform swap
            _apply_swap(adj, deg, e_remove, e_add)
            new_score = _score(adj, deg)
            new_soft = _soft_score(adj, deg)

            # Acceptance: use soft-max for smooth guidance, hard max for tracking
            if new_score < cur_score:
                accept = True
                no_improve_count = 0
            elif new_score == cur_score:
                # Accept with moderate probability to explore
                accept = rng.random() < 0.4
                no_improve_count += 1
            else:
                # Use soft-max delta for smoother acceptance
                delta_soft = new_soft - cur_soft
                delta_hard = new_score - cur_score
                # Blend: use soft delta if it's informative, else hard
                if delta_soft < 0.01:
                    eff_delta = delta_soft
                else:
                    eff_delta = delta_hard
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-eff_delta / temp) > rng.random()
                no_improve_count += 1

            if accept:
                cur_score = new_score
                cur_soft = new_soft
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    no_improve_count = 0
                remove_edge_from_list(e_remove)
                add_edge_to_list(e_add)
            else:
                _undo_swap(adj, deg, e_remove, e_add)

            iters_done += 1
            temp *= cooling_rate

            # Adaptive reheat: if stuck, boost temperature
            if no_improve_count > reheat_threshold:
                temp = max(temp, reheat_temp)
                no_improve_count = 0
                reheat_threshold = int(reheat_threshold * 1.3)
                reheat_temp = max(0.5, reheat_temp * 0.7)

        # Update global best
        if best_score < global_best_score:
            global_best_score = best_score
            global_best_adj = best_adj
            global_best_deg = best_deg

    # Final polish: greedy local search on best solution
    if global_best_adj is not None and time.time() - t0 < DEADLINE - 5:
        adj = [row[:] for row in global_best_adj]
        deg = global_best_deg[:]
        cur_score = _score(adj, deg)
        improved = True
        while improved and time.time() - t0 < DEADLINE - 5:
            improved = False
            # Try all single edge swaps that reduce the score
            for i in range(len(edges)):
                if time.time() - t0 > DEADLINE - 5:
                    break
                e_remove = edges[i]
                for j in range(len(nonedges)):
                    e_add = nonedges[j]
                    u, v = e_remove
                    x, y = e_add
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    # Quick check: will this create a triangle with higher sum?
                    _apply_swap(adj, deg, e_remove, e_add)
                    new_score = _score(adj, deg)
                    if new_score < cur_score:
                        cur_score = new_score
                        remove_edge_from_list(e_remove)
                        add_edge_to_list(e_add)
                        improved = True
                        # Restart scan from beginning
                        break
                    else:
                        _undo_swap(adj, deg, e_remove, e_add)
                if improved:
                    break
            if cur_score < global_best_score:
                global_best_score = cur_score
                global_best_adj = [row[:] for row in adj]
                global_best_deg = deg[:]

    if global_best_adj is None:
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        global_best_adj = adj

    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if global_best_adj[i][j]]}
