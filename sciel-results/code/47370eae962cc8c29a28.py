import time
import random
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256

    # Perfect squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1

    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)

    node_set = set(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    n = len(valid_nodes)

    # Build conflict adjacency
    conflicts = [set() for _ in range(n)]
    for i_idx in range(n):
        v = valid_nodes[i_idx]
        for j_idx in range(i_idx + 1, n):
            w = valid_nodes[j_idx]
            if v + w in squares:
                conflicts[i_idx].add(j_idx)
                conflicts[j_idx].add(i_idx)

    # Build conflict masks for fast checking
    conflict_masks = [0] * n
    for i in range(n):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask

    def is_valid_set(selected):
        """Verify selected indices form an independent set."""
        sel_set = set(selected)
        for idx in sel_set:
            for nb in conflicts[idx]:
                if nb in sel_set:
                    return False
        return True

    def greedy_mis(order):
        """Greedy: process nodes in given order, add if no conflict with already selected."""
        selected = set()
        for idx in order:
            if all(nb not in selected for nb in conflicts[idx]):
                selected.add(idx)
        return selected

    def local_search(selected, max_iters=5000):
        """Local search: try adding single nodes, then swap-based improvements."""
        selected = set(selected)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and (time.time() - start_time) < deadline - 1:
            improved = False
            iter_count += 1

            # Try adding a single unselected node
            for idx in range(n):
                if idx not in selected:
                    if all(nb not in selected for nb in conflicts[idx]):
                        selected.add(idx)
                        improved = True
                        break
            if improved:
                continue

            # Try swap: remove one selected, add one unselected if it improves or allows more
            best_swap = None
            best_swap_gain = 0
            for rem in list(selected):
                if (time.time() - start_time) > deadline - 1:
                    break
                for add in range(n):
                    if add in selected or add == rem:
                        continue
                    # Check if add conflicts with selected minus rem
                    temp_sel = selected - {rem}
                    if all(nb not in temp_sel for nb in conflicts[add]):
                        # This swap is valid (size stays same, but may enable future additions)
                        # We'll try it and see if we can then add more
                        new_sel = temp_sel | {add}
                        # Try to greedily add more
                        for idx2 in range(n):
                            if idx2 not in new_sel:
                                if all(nb not in new_sel for nb in conflicts[idx2]):
                                    new_sel.add(idx2)
                        gain = len(new_sel) - len(selected)
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = new_sel
                            if gain >= 2:
                                break
                if best_swap_gain >= 2:
                    break

            if best_swap is not None and best_swap_gain > 0:
                selected = best_swap
                improved = True
        return selected

    def ilp_mis():
        """Solve maximum independent set via ILP (LP relaxation + rounding + repair)."""
        # First get a good lower bound via greedy
        degrees = [len(conflicts[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        lb = greedy_mis(order)
        lb_size = len(lb)

        # Use LP relaxation with scipy
        c = -np.ones(n)  # minimize -sum(x_i)

        # Build constraint matrix
        edge_list = []
        for i in range(n):
            for j in conflicts[i]:
                if i < j:
                    edge_list.append((i, j))

        if not edge_list:
            return set(range(n))

        m = len(edge_list)
        A_ub = np.zeros((m, n))
        b_ub = np.ones(m)
        for k, (i, j) in enumerate(edge_list):
            A_ub[k, i] = 1
            A_ub[k, j] = 1

        bounds = [(0, 1)] * n

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                x = res.x
                selected = set()
                # Sort by LP value descending for better rounding
                order_by_lp = sorted(range(n), key=lambda i: -x[i])
                for i in order_by_lp:
                    if x[i] > 0.3:
                        if all(nb not in selected for nb in conflicts[i]):
                            selected.add(i)

                if len(selected) < lb_size:
                    return lb
                return selected
            else:
                return lb
        except Exception:
            return lb

    def branch_and_bound(selected, upper_bound, time_limit=5.0):
        """Branch and bound for MIS. Returns best independent set found."""
        best_set = set(selected)
        best_size = len(selected)
        start_bb = time.time()

        # Compute upper bound: current size + number of unselected nodes that could potentially be added
        unselected = [i for i in range(n) if i not in selected]

        # Sort unselected by degree (ascending) for better pruning
        unselected.sort(key=lambda x: len(conflicts[x]))

        def bb_recursion(current_set, remaining, depth=0):
            nonlocal best_set, best_size
            if (time.time() - start_bb) > time_limit:
                return

            # Upper bound: current size + len(remaining)
            if len(current_set) + len(remaining) <= best_size:
                return

            if not remaining:
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                return

            # Pick a node to branch on
            # Heuristic: pick node with highest degree in remaining
            pivot = max(remaining, key=lambda x: len(conflicts[x] & set(remaining)))
            pivot_idx = remaining.index(pivot)

            # Branch 1: include pivot
            new_remaining = [r for r in remaining if r != pivot and r not in conflicts[pivot]]
            bb_recursion(current_set | {pivot}, new_remaining, depth + 1)

            # Branch 2: exclude pivot
            remaining_excl = [r for r in remaining if r != pivot]
            bb_recursion(current_set, remaining_excl, depth + 1)

        bb_recursion(set(selected), unselected)
        return best_set

    best = set()
    best_size = 0

    random.seed(42)

    # Order 1: natural order
    order = list(range(n))
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)

    # Order 2: reverse
    order = list(range(n - 1, -1, -1))
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)

    # Order 3: sorted by degree (fewest conflicts first)
    degrees = [len(conflicts[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)

    # Order 4: sorted by degree (most conflicts first)
    order = sorted(range(n), key=lambda x: -degrees[x])
    sel = greedy_mis(order)
    sel = local_search(sel)
    if len(sel) > best_size:
        best = sel
        best_size = len(sel)

    # ILP-based initialization
    if (time.time() - start_time) < deadline - 10:
        try:
            ilp_sel = ilp_mis()
            ilp_sel = local_search(ilp_sel, max_iters=10000)
            if len(ilp_sel) > best_size:
                best = ilp_sel
                best_size = len(ilp_sel)
        except Exception:
            pass

    # Branch and bound on current best
    if (time.time() - start_time) < deadline - 5:
        try:
            bb_sel = branch_and_bound(best, best_size + n, time_limit=3.0)
            if len(bb_sel) > best_size:
                best = bb_sel
                best_size = len(bb_sel)
        except Exception:
            pass

    # Random orderings
    for trial in range(50):
        if (time.time() - start_time) > deadline - 2:
            break
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_mis(order)
        sel = local_search(sel)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)

    # More aggressive local search on best
    if (time.time() - start_time) < deadline - 3:
        best = local_search(best, max_iters=20000)
        best_size = len(best)

    # Additional random restarts with local search if time permits
    for trial in range(30):
        if (time.time() - start_time) > deadline - 2:
            break
        order = list(range(n))
        random.shuffle(order)
        sel = greedy_mis(order)
        sel = local_search(sel, max_iters=5000)
        if len(sel) > best_size:
            best = sel
            best_size = len(sel)

    # Final branch and bound if time permits
    if (time.time() - start_time) < deadline - 3:
        try:
            bb_sel = branch_and_bound(best, best_size + n, time_limit=2.0)
            if len(bb_sel) > best_size:
                best = bb_sel
                best_size = len(bb_sel)
        except Exception:
            pass

    # Final verification
    if not is_valid_set(best):
        order = list(range(n))
        best = greedy_mis(order)

    # One more local search pass on the final result
    if (time.time() - start_time) < deadline - 1:
        best = local_search(best, max_iters=5000)

    # Convert back to values
    result = sorted(valid_nodes[i] for i in best)

    if not result:
        result = [1]

    return result
