"""Optimized solver: greedy + local search for max reciprocal sum with lcm constraint."""
import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    candidates = list(range(2, 257))
    n = len(candidates)
    weights = [Fraction(1, x) for x in candidates]

    # Precompute conflict: conflict[i][j] = True if lcm(candidates[i], candidates[j]) <= 256
    # Use a set of frozensets or a 2D list
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(candidates[i], candidates[j]) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_compatible(idx_set):
        """Check if all pairs in idx_set are compatible (no conflicts)."""
        lst = list(idx_set)
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                if conflict[lst[a]][lst[b]]:
                    return False
        return True

    # Greedy: sort by weight desc, add if compatible
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    selected = set()
    for i in order:
        if all(not conflict[i][j] for j in selected):
            selected.add(i)

    current_weight = sum(weights[i] for i in selected)
    best_weight = current_weight
    best_solution = set(selected)

    # Local search: try swaps, additions, removals
    def try_improve(selected, current_weight):
        nonlocal best_weight, best_solution
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Try adding a single element
            for i in range(n):
                if i in selected:
                    continue
                if all(not conflict[i][j] for j in selected):
                    new_sel = selected | {i}
                    new_w = current_weight + weights[i]
                    if new_w > current_weight:
                        selected = new_sel
                        current_weight = new_w
                        improved = True
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best_solution = set(selected)
                        break

            # Try swapping: remove one, add one or more
            for rem in list(selected):
                if time.time() > deadline:
                    break
                sel_without = selected - {rem}
                w_without = current_weight - weights[rem]
                # Try adding one
                for add in range(n):
                    if add in sel_without:
                        continue
                    if all(not conflict[add][j] for j in sel_without):
                        new_w = w_without + weights[add]
                        if new_w > current_weight:
                            selected = sel_without | {add}
                            current_weight = new_w
                            improved = True
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_solution = set(selected)
                            break
                if improved:
                    break

        return selected, current_weight

    selected, current_weight = try_improve(selected, current_weight)

    # Additional local search: try removing low-weight elements and adding multiple high-weight
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            # Try removing the lowest weight element and see if we can add better ones
            if not selected:
                break
            min_idx = min(selected, key=lambda i: weights[i])
            sel_without = selected - {min_idx}
            w_without = current_weight - weights[min_idx]
            # Greedily add back best compatible elements
            new_sel = set(sel_without)
            new_w = w_without
            for i in sorted(range(n), key=lambda i: weights[i], reverse=True):
                if i in new_sel:
                    continue
                if all(not conflict[i][j] for j in new_sel):
                    new_sel.add(i)
                    new_w += weights[i]
            if new_w > current_weight:
                selected = new_sel
                current_weight = new_w
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_solution = set(selected)

    result = [candidates[i] for i in best_solution]
    return result if result else [256]
