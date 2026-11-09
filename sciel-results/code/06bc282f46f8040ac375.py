import time
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1

    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    M = len(valid_nodes)

    if M == 0:
        return [1]

    adj = np.zeros((M, M), dtype=np.int8)
    for i in range(M):
        for j in range(i + 1, M):
            if (valid_nodes[i] + valid_nodes[j]) in squares:
                adj[i, j] = 1
                adj[j, i] = 1

    c = np.ones(M)
    A_ub = []
    b_ub = []
    for i in range(M):
        for j in range(i + 1, M):
            if adj[i, j] == 1:
                row = np.zeros(M)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(1)

    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.zeros((1, M))
        b_ub = np.array([1])

    bounds = [(0, 1)] * M

    best_solution = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0 and res.x is not None:
            x_lp = res.x
            order = np.argsort(-x_lp)
            selected = []
            selected_set = set()
            for idx in order:
                if x_lp[idx] < 0.3:
                    continue
                conflict = False
                for sel in selected_set:
                    if adj[idx, sel] == 1:
                        conflict = True
                        break
                if not conflict:
                    selected.append(idx)
                    selected_set.add(idx)
            if selected:
                best_solution = selected
    except Exception:
        pass

    if best_solution is None:
        remaining = set(range(M))
        best_solution = []
        while remaining:
            best_node = None
            best_deg = float('inf')
            for node in remaining:
                deg = sum(adj[node, other] for other in remaining if other != node)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
            best_solution.append(best_node)
            remaining.remove(best_node)
            to_remove = set()
            for other in remaining:
                if adj[best_node, other] == 1:
                    to_remove.add(other)
            remaining -= to_remove

    # Local search with proper conflict checking
    def local_search(initial):
        selected = list(initial)
        selected_set = set(selected)

        while (time.time() - start_time) < deadline - 5.0:
            improved = False

            # Phase 1: Try to add any non-conflicting node
            for node in range(M):
                if node in selected_set:
                    continue
                can_add = True
                for sel in selected_set:
                    if adj[node, sel] == 1:
                        can_add = False
                        break
                if can_add:
                    selected.append(node)
                    selected_set.add(node)
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: Try swaps (remove one, add one)
            for remove_idx in list(selected):
                for add_node in range(M):
                    if add_node in selected_set:
                        continue
                    can_add = True
                    for sel in selected_set:
                        if sel == remove_idx:
                            continue
                        if adj[add_node, sel] == 1:
                            can_add = False
                            break
                    if can_add:
                        selected.remove(remove_idx)
                        selected_set.discard(remove_idx)
                        selected.append(add_node)
                        selected_set.add(add_node)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return selected

    best_solution = local_search(best_solution)
    result = [valid_nodes[i] for i in best_solution]

    # Final verification
    for a in result:
        for b in result:
            if a <= b:
                s = a + b
                r = math.isqrt(s)
                if r * r == s:
                    return [1]

    return result
