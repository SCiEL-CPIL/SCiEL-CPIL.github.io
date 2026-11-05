import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 88.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute prime factor counts for ordering heuristic
    def count_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    prime_factor_counts = [count_prime_factors(m) for m in moduli]

    # For each pair (i, j) and each residue r_j of modulus j,
    # precompute the set of residues r_i of modulus i that are INCOMPATIBLE with r_j
    # Two residues r_i (mod m_i) and r_j (mod m_j) are incompatible iff (r_i - r_j) % gcd(m_i, m_j) == 0
    # So for fixed r_j, incompatible r_i are those where r_i ≡ r_j (mod g), where g = gcd(m_i, m_j)
    
    # Precompute forbidden residues: forb[i][j][r_j] = set of r_i values that conflict
    # This is expensive, so we'll compute on the fly but cache per pair

    def get_forbidden_residues(idx_i, idx_j, r_j):
        """Return set of residues for modulus idx_i that are incompatible with residue r_j of modulus idx_j."""
        g = gcd_tab[idx_i][idx_j]
        m_i = moduli[idx_i]
        # r_i is incompatible if r_i ≡ r_j (mod g)
        # So r_i = r_j mod g, r_i + g, r_i + 2g, ... < m_i
        # The number of such residues is m_i // g
        forbidden = set()
        base = r_j % g
        for r_i in range(base, m_i, g):
            forbidden.add(r_i)
        return forbidden

    # CSP backtracking with forward checking
    def csp_solve(order, time_limit):
        """
        Solve CSP with given variable ordering.
        Returns (selected, max_found) where selected is list of (idx, residue, modulus)
        and max_found is the maximum number of variables assigned in any complete/partial solution.
        """
        selected = []
        max_found = 0
        assigned = [False] * n
        assigned_residue = [-1] * n

        # Domains: for each unassigned variable, track which residues are still possible
        # We'll use forward checking: when we assign a variable, we prune domains of others

        def backtrack(depth):
            nonlocal max_found
            if time.time() > time_limit:
                return False

            if len(selected) > max_found:
                max_found = len(selected)

            if depth == len(order):
                return True

            idx = order[depth]
            m = moduli[idx]

            # Try residues in order of least constraining value
            # Residues that forbid fewer options for remaining variables are preferred
            # Simple heuristic: try residue 1 first, then others
            residues_to_try = list(range(m))
            # Sort by how many they'd forbid in remaining unassigned vars
            # (expensive, so just use a simple order)
            residues_to_try.sort(key=lambda r: r if r != 1 else -1000)

            for r in residues_to_try:
                # Check compatibility with all already assigned
                compatible = True
                for (idx2, r2, m2) in selected:
                    g = gcd_tab[idx][idx2]
                    if (r - r2) % g == 0:
                        compatible = False
                        break

                if not compatible:
                    continue

                # Forward check: verify that all unassigned variables still have at least one valid residue
                feasible = True
                for depth2 in range(depth + 1, len(order)):
                    idx2 = order[depth2]
                    m2 = moduli[idx2]
                    # Check if any residue of idx2 is compatible with all assigned + this new assignment
                    has_valid = False
                    for r2 in range(m2):
                        ok = True
                        for (idx3, r3, m3) in selected:
                            g = gcd_tab[idx2][idx3]
                            if (r2 - r3) % g == 0:
                                ok = False
                                break
                        if ok:
                            g_new = gcd_tab[idx2][idx]
                            if (r2 - r) % g_new == 0:
                                ok = False
                        if ok:
                            has_valid = True
                            break
                    if not has_valid:
                        feasible = False
                        break

                if not feasible:
                    continue

                # Assign
                selected.append((idx, r, m))
                if backtrack(depth + 1):
                    return True
                selected.pop()

            return False

        backtrack(0)
        return selected, max_found

    # Greedy with forward checking (faster, used for initial solutions)
    def greedy_with_fc(order):
        selected = []
        for idx in order:
            m = moduli[idx]
            found = False
            # Try residues, check forward feasibility
            residues_to_try = [1] + [r for r in range(m) if r != 1]
            for r in residues_to_try:
                # Check compatibility with selected
                compatible = True
                for (idx2, r2, m2) in selected:
                    g = gcd_tab[idx][idx2]
                    if (r - r2) % g == 0:
                        compatible = False
                        break
                if not compatible:
                    continue

                # Forward check: can remaining unassigned vars still be placed?
                # Only check a subset for speed
                feasible = True
                remaining = [i for i in order if i not in [s[0] for s in selected] and i != idx]
                # Check just the next few
                for idx2 in remaining[:3]:
                    m2 = moduli[idx2]
                    has_valid = False
                    for r2 in range(m2):
                        ok = True
                        for (idx3, r3, m3) in selected:
                            g = gcd_tab[idx2][idx3]
                            if (r2 - r3) % g == 0:
                                ok = False
                                break
                        if ok:
                            g_new = gcd_tab[idx2][idx]
                            if (r2 - r) % g_new == 0:
                                ok = False
                        if ok:
                            has_valid = True
                            break
                    if not has_valid:
                        feasible = False
                        break

                if feasible:
                    selected.append((idx, r, m))
                    found = True
                    break
        return selected

    best = []
    best_len = 0

    # Orderings based on different heuristics
    orderings = [
        # Most prime factors first (hardest to place)
        sorted(range(n), key=lambda i: -prime_factor_counts[i]),
        # Largest modulus first
        sorted(range(n), key=lambda i: -moduli[i]),
        # Smallest modulus first
        sorted(range(n), key=lambda i: moduli[i]),
        # Most composite first (most divisors)
        sorted(range(n), key=lambda i: -sum(1 for d in range(2, moduli[i]+1) if moduli[i] % d == 0)),
    ]

    # Try greedy with forward checking first (fast)
    for order in orderings:
        res = greedy_with_fc(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Random restarts with greedy FC
    random.seed(12345)
    for _ in range(3000):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        res = greedy_with_fc(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    # Now try CSP backtracking on promising orderings (slower but can find better)
    if time.time() < deadline - 15:
        # Use the best greedy solution's ordering as a starting point
        for order in orderings[:2]:
            if time.time() > deadline - 15:
                break
            time_limit = min(deadline - 5, time.time() + 10)
            res, max_found = csp_solve(order, time_limit)
            if len(res) > best_len:
                best = res
                best_len = len(res)

    # Local search: try to add more moduli to best solution
    if time.time() < deadline - 10:
        selected_idx = set(idx for (idx, _, _) in best)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                for r in range(m):
                    compatible = True
                    for (idx2, r2, m2) in best:
                        g = gcd_tab[idx][idx2]
                        if (r - r2) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        best.append((idx, r, m))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break

    # Local search: try swapping residues to enable adding new moduli
    if time.time() < deadline - 5:
        selected_idx = set(idx for (idx, _, _) in best)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                for pos in range(len(best)):
                    old_idx, old_r, old_m = best[pos]
                    for new_r in range(old_m):
                        if new_r == old_r:
                            continue
                        # Check if new_r is compatible with all other selected (excluding old_idx)
                        ok_old = True
                        for qpos in range(len(best)):
                            if qpos == pos:
                                continue
                            other_idx, other_r, other_m = best[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        # Now check if new_idx can be added with some residue
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best)):
                                if qpos == pos:
                                    if (r_new - new_r) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r, other_m = best[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        if found_new:
                            best[pos] = (old_idx, new_r, old_m)
                            best.append((new_idx, r_new, m_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

    return [[r, m] for (_, r, m) in best]
