import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # For a candidate (idx, r), the set of residues it FORBIDS on each other
    # modulus k is { r + t*gcd(idx,k) mod m_k : t = 0..g-1 }.
    # A good residue forbids few classes on the moduli we still need to place.
    def forbidden_count(idx, r, selected):
        # selected is a list of (idx2, r2); count conflicts (should be 0 if valid)
        return sum(1 for (idx2, r2) in selected
                   if (r - r2) % gcd_tab[idx][idx2] == 0)

    def check_residue(idx, r, selected):
        """selected is a list of (idx2, r2)"""
        for (idx2, r2) in selected:
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    # Greedy with given order, choosing the least-constraining valid residue.
    def try_greedy(order):
        selected = []
        remaining = set(order)
        for idx in order:
            m = moduli[idx]
            remaining.discard(idx)
            best_r = -1
            best_score = None
            for r in range(m):
                if not check_residue(idx, r, selected):
                    continue
                # Score: how many residue-classes this forbids on the moduli
                # we still have to place. Lower is better.
                cost = 0
                for j in remaining:
                    g = gcd_tab[idx][j]
                    cost += g  # number of forbidden classes on modulus j
                if best_score is None or cost < best_score:
                    best_score = cost
                    best_r = r
            if best_r >= 0:
                selected.append((idx, best_r))
        return selected

    # Backtracking with MRV heuristic
    def backtrack(selected, remaining, depth=0):
        if time.time() > deadline - 5:
            return False, selected
        if not remaining:
            return True, selected

        best_idx = -1
        best_count = float('inf')
        best_residues = None
        for idx in remaining:
            m = moduli[idx]
            valid = [r for r in range(m) if check_residue(idx, r, selected)]
            cnt = len(valid)
            if cnt < best_count:
                best_count = cnt
                best_idx = idx
                best_residues = valid
            if cnt == 0:
                break

        if best_count == 0 or best_idx == -1:
            return False, selected

        remaining2 = [idx for idx in remaining if idx != best_idx]
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            success, result = backtrack(new_selected, remaining2, depth + 1)
            if success:
                return True, result
        return False, selected

    # Local search: try to add more moduli (no removal)
    def add_loop(best):
        selected_idx = set(idx for (idx, _) in best)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    if check_residue(idx, r, best):
                        best.append((idx, r))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break
        return best

    # Local search: remove one, add two
    def local_search(best):
        selected_idx = set(idx for (idx, _) in best)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for pos in range(len(best)):
                if time.time() > deadline - 5:
                    break
                old_idx, old_r = best[pos]
                for i in range(n):
                    if i in selected_idx or i == old_idx:
                        continue
                    for j in range(n):
                        if j in selected_idx or j == old_idx or j == i:
                            continue
                        base = best[:pos] + best[pos + 1:]
                        found = False
                        for r_i in range(moduli[i]):
                            if not check_residue(i, r_i, base):
                                continue
                            for r_j in range(moduli[j]):
                                if (r_i - r_j) % gcd_tab[i][j] == 0:
                                    continue
                                if not check_residue(j, r_j, base):
                                    continue
                                best[pos] = (i, r_i)
                                best.append((j, r_j))
                                selected_idx.discard(old_idx)
                                selected_idx.add(i)
                                selected_idx.add(j)
                                improved = True
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # Local search: swap one existing residue to free a new modulus
    def swap_add(best):
        selected_idx = set(idx for (idx, _) in best)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                for pos in range(len(best)):
                    old_idx, old_r = best[pos]
                    old_m = moduli[old_idx]
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
                        ok_old = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        if found_new:
                            best[pos] = (old_idx, new_r)
                            best.append((new_idx, r_new))
                            selected_idx.add(new_idx)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    best = []
    best_len = 0

    # Deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        sorted(range(n), key=lambda i: (-len([p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23] if moduli[i] % p == 0]), moduli[i])),
        sorted(range(n), key=lambda i: (len([p for p in [2, 3, 5, 7, 11, 13, 17, 19, 23] if moduli[i] % p == 0]), moduli[i])),
        sorted(range(n), key=lambda i: -sum(gcd_tab[i][j] for j in range(n) if j != i)),
    ]

    for order in orderings:
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random restarts
    random.seed(42)
    n_restarts = 3000
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Improve the best solution
    if time.time() < deadline - 15:
        best = add_loop(best)
        best_len = len(best)

    if time.time() < deadline - 5:
        best = local_search(best)
        best_len = len(best)

    if time.time() < deadline - 5:
        best = swap_add(best)
        best_len = len(best)

    return [[r, moduli[idx]] for (idx, r) in best]
