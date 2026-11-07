import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2

    # Precompute edge indices for quick access
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)

    # Precompute 4-cycles (squares in the prism)
    cycles_4 = []
    for i in range(m):
        v1 = i
        v2 = (i + 1) % m
        v3 = m + i
        v4 = m + (i + 1) % m

        e1 = edge_idx.get((v1, v2))
        e2 = edge_idx.get((v2, v4))
        e3 = edge_idx.get((v4, v3))
        e4 = edge_idx.get((v3, v1))

        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycles_4.append([e1, e2, e3, e4])

    # Adjacency list for connectivity checks
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected (signing is NOT a coboundary).
        Uses BFS to check if the signing is a coboundary.
        """
        f = [-1] * n
        f[0] = 0
        queue = deque([0])

        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip

                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True

        return False

    def build_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(np.float64)
        S[v_arr, u_arr] = signs.astype(np.float64)
        return S

    def get_objective(signs):
        S = build_matrix(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def fast_radius_update(S, idx, old_sign, new_sign):
        """
        Compute new spectral radius after flipping edge idx.
        For small n (<= 18), full eigvalsh is fast enough.
        """
        S_new = S.copy()
        u, v = edges[idx]
        delta = new_sign - old_sign
        S_new[u, v] += delta
        S_new[v, u] += delta
        eigvals = np.linalg.eigvalsh(S_new)
        return float(np.max(np.abs(eigvals)))

    rng = np.random.default_rng(3)

    best_signs = None
    best_obj = float('inf')

    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()

    def local_search(initial_signs, time_limit=None, max_no_improve=100):
        """
        Local search with single-flip, 4-cycle-flip, and double-flip moves.
        """
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline

        current_signs = initial_signs.copy()

        if not is_lift_connected(current_signs):
            return

        current_S = build_matrix(current_signs)
        current_obj = float(np.max(np.abs(np.linalg.eigvalsh(current_S))))

        update_best(current_signs, current_obj)

        no_improve_count = 0
        start_time = time.time()

        while time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False

            # 1. Try single flips
            best_flip_idx = -1
            best_flip_obj = current_obj
            best_flip_S = None

            for i in range(num_edges):
                if time.time() > time_limit:
                    break
                old = current_signs[i]
                current_signs[i] = -old

                if is_lift_connected(current_signs):
                    S_new = current_S.copy()
                    u, v = edges[i]
                    delta = -old - old  # new_sign - old_sign = (-old) - old = -2*old
                    S_new[u, v] += delta
                    S_new[v, u] += delta
                    obj = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                    if obj < best_flip_obj - 1e-12:
                        best_flip_obj = obj
                        best_flip_idx = i
                        best_flip_S = S_new
                current_signs[i] = old

            if best_flip_idx != -1:
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_obj = best_flip_obj
                current_S = best_flip_S
                update_best(current_signs, current_obj)
                improved = True
                no_improve_count = 0
            else:
                # 2. Try 4-cycle flips
                best_cycle_idx = -1
                best_cycle_obj = current_obj
                best_cycle_S = None

                for c_idx, cyc in enumerate(cycles_4):
                    if time.time() > time_limit:
                        break
                    S_new = current_S.copy()
                    for e in cyc:
                        old = current_signs[e]
                        new = -old
                        current_signs[e] = new
                        u, v = edges[e]
                        delta = new - old
                        S_new[u, v] += delta
                        S_new[v, u] += delta

                    if is_lift_connected(current_signs):
                        obj = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                        if obj < best_cycle_obj - 1e-12:
                            best_cycle_obj = obj
                            best_cycle_idx = c_idx
                            best_cycle_S = S_new.copy()

                    # Restore
                    for e in cyc:
                        current_signs[e] = -current_signs[e]

                if best_cycle_idx != -1:
                    for e in cycles_4[best_cycle_idx]:
                        current_signs[e] = -current_signs[e]
                    current_obj = best_cycle_obj
                    current_S = best_cycle_S
                    update_best(current_signs, current_obj)
                    improved = True
                    no_improve_count = 0
                else:
                    # 3. Try pair flips
                    found_pair = False
                    for _ in range(80):
                        if time.time() > time_limit:
                            break
                        e1 = rng.integers(num_edges)
                        e2 = rng.integers(num_edges)
                        if e1 == e2:
                            continue

                        old1 = current_signs[e1]
                        old2 = current_signs[e2]
                        current_signs[e1] = -old1
                        current_signs[e2] = -old2

                        if is_lift_connected(current_signs):
                            S_new = current_S.copy()
                            u1, v1 = edges[e1]
                            u2, v2 = edges[e2]
                            delta1 = -old1 - old1
                            delta2 = -old2 - old2
                            S_new[u1, v1] += delta1
                            S_new[v1, u1] += delta1
                            S_new[u2, v2] += delta2
                            S_new[v2, u2] += delta2

                            obj = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                            if obj < current_obj - 1e-12:
                                current_obj = obj
                                current_S = S_new
                                update_best(current_signs, current_obj)
                                found_pair = True
                                improved = True
                                no_improve_count = 0
                                break
                        else:
                            current_signs[e1] = old1
                            current_signs[e2] = old2

                    if not found_pair:
                        no_improve_count += 1
                        if no_improve_count >= max_no_improve:
                            break

    def polish(signs):
        """High-precision polish: iterate single + pair flips until convergence"""
        nonlocal best_signs, best_obj
        current_signs = signs.copy()

        if not is_lift_connected(current_signs):
            return

        current_S = build_matrix(current_signs)
        current_obj = float(np.max(np.abs(np.linalg.eigvalsh(current_S))))

        update_best(current_signs, current_obj)

        # Multi-pass exact local optimization
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = False

            # Single flip sweep
            for i in range(num_edges):
                if time.time() > deadline - 2.0:
                    break
                old = current_signs[i]
                current_signs[i] = -old

                if is_lift_connected(current_signs):
                    S_new = current_S.copy()
                    u, v = edges[i]
                    delta = -old - old
                    S_new[u, v] += delta
                    S_new[v, u] += delta
                    obj = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                    if obj < current_obj - 1e-14:
                        current_obj = obj
                        current_S = S_new
                        improved = True
                    else:
                        current_signs[i] = old
                else:
                    current_signs[i] = old

            if not improved:
                # Try pair flips
                for _ in range(150):
                    if time.time() > deadline - 2.0:
                        break
                    e1 = rng.integers(num_edges)
                    e2 = rng.integers(num_edges)
                    if e1 == e2:
                        continue

                    old1 = current_signs[e1]
                    old2 = current_signs[e2]
                    current_signs[e1] = -old1
                    current_signs[e2] = -old2

                    if is_lift_connected(current_signs):
                        S_new = current_S.copy()
                        u1, v1 = edges[e1]
                        u2, v2 = edges[e2]
                        delta1 = -old1 - old1
                        delta2 = -old2 - old2
                        S_new[u1, v1] += delta1
                        S_new[v1, u1] += delta1
                        S_new[u2, v2] += delta2
                        S_new[v2, u2] += delta2

                        obj = float(np.max(np.abs(np.linalg.eigvalsh(S_new))))
                        if obj < current_obj - 1e-14:
                            current_obj = obj
                            current_S = S_new
                            improved = True
                            break
                    else:
                        current_signs[e1] = old1
                        current_signs[e2] = old2

            if not improved:
                break

        update_best(current_signs, current_obj)

    # Strategy 1: Exhaustive search for small m (m=4: 12 edges, m=5: 15 edges)
    if num_edges <= 15:
        max_combo = 1 << num_edges
        for k in range(max_combo):
            if time.time() > deadline - 5.0:
                break
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            if not is_lift_connected(signs):
                continue
            S = build_matrix(signs)
            eigvals = np.linalg.eigvalsh(S)
            radius = float(np.max(np.abs(eigvals)))
            if radius < best_obj:
                best_obj = radius
                best_signs = signs.copy()

        if best_signs is not None:
            # Polish the exhaustive result
            polish(best_signs)
            return best_signs

    # Strategy 2: Multiple restarts with local search
    # 1. All 1s
    signs = np.ones(num_edges, dtype=np.int8)
    if is_lift_connected(signs):
        local_search(signs, deadline - 40.0, max_no_improve=50)

    # 2. Alternating signs on top cycle
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            signs[i] = 1 if (u % 2 == 0) else -1
    if is_lift_connected(signs):
        local_search(signs, deadline - 35.0, max_no_improve=50)

    # 3. Alternating signs on bottom cycle
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u >= m and v >= m:
            signs[i] = 1 if ((u - m) % 2 == 0) else -1
    if is_lift_connected(signs):
        local_search(signs, deadline - 30.0, max_no_improve=50)

    # 4. Alternating signs on both cycles
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            signs[i] = 1 if (u % 2 == 0) else -1
        elif u >= m and v >= m:
            signs[i] = 1 if ((u - m) % 2 == 0) else -1
    if is_lift_connected(signs):
        local_search(signs, deadline - 25.0, max_no_improve=50)

    # 5. Random restarts
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, deadline - 5.0, max_no_improve=30)

    # Final polish on best solution
    if best_signs is not None:
        polish(best_signs)

    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)

    return best_signs
