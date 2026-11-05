"""
Maximum disjoint residue classes with distinct moduli 2..128.

Strategy:
- For each modulus m in 2..128, choose a residue r in [0, m).
- Two choices (r1, m1) and (r2, m2) are DISJOINT iff (r1 - r2) % gcd(m1, m2) != 0.
- We want to maximize the number of moduli we can assign.

Approach:
1. Precompute gcd table and conflict masks using bitsets.
2. Greedy randomized restarts with MRV ordering for initial solutions.
3. Branch-and-bound with upper bound from a relaxed greedy coloring.
4. Local search: try to add missing moduli, try 1-for-1 and 1-for-2 swaps.
5. Deadline guard to return best found before 90s.
"""
import time
import math
import random

MAXIMUM = 128


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    MAX_M = MAXIMUM
    candidates = list(range(2, MAX_M + 1))
    n = len(candidates)

    # Precompute gcd table
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])

    # Precompute for each pair (i, j) and each residue r of i,
    # which residues of j are FORBIDDEN (i.e., (r - rj) % gcd == 0)
    # conflict_mask[i][j] is a list of length candidates[i], each entry is a set of forbidden residues for j
    # Actually, let's use: for each i, j, the forbidden residues of j given residue r of i
    # forbidden[i][j][r] = set of rj such that (r - rj) % gcd(i,j) == 0
    # This is: rj ≡ r (mod g) where g = gcd(candidates[i], candidates[j])
    # So rj in {r % g, r % g + g, r % g + 2g, ...} < candidates[j]

    # For speed, precompute as lists
    forbidden = [[None] * n for _ in range(n)]
    for i in range(n):
        mi = candidates[i]
        for j in range(n):
            if i == j:
                continue
            mj = candidates[j]
            g = gcd_tab[i][j]
            # forbidden[i][j][r] = list of rj in [0, mj) such that (r - rj) % g == 0
            # i.e., rj ≡ r (mod g)
            mask = []
            for r in range(mi):
                base = r % g
                lst = list(range(base, mj, g))
                mask.append(lst)
            forbidden[i][j] = mask

    def is_compatible(selected, new_idx, new_r):
        """Check if (new_idx, new_r) is disjoint from all in selected."""
        for (idx, r) in selected:
            g = gcd_tab[new_idx][idx]
            if (new_r - r) % g == 0:
                return False
        return True

    def get_valid_residues(selected, idx):
        """Get all valid residues for idx given current selection."""
        m = candidates[idx]
        valid = set(range(m))
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            base = sel_r % g
            for r in range(base, m, g):
                valid.discard(r)
        return sorted(valid)

    def get_valid_residues_fast(selected, idx, selected_idx_list):
        """Faster version using precomputed forbidden masks."""
        m = candidates[idx]
        valid = [True] * m
        for k in range(len(selected_idx_list)):
            sel_idx = selected_idx_list[k]
            sel_r = selected[k]
            if sel_idx == idx:
                continue
            # forbidden[sel_idx][idx][sel_r] gives forbidden residues for idx
            for rj in forbidden[sel_idx][idx][sel_r]:
                valid[rj] = False
        return [r for r in range(m) if valid[r]]

    def greedy_randomized(order, selected_set):
        """Greedy assignment with random residue order."""
        selected = []
        for idx in order:
            m = candidates[idx]
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(selected, idx, r):
                    selected.append((idx, r))
                    selected_set.add(idx)
                    break
        return selected

    def local_search_improve(solution, max_iters=500):
        """Improve solution by trying to add missing moduli or swap."""
        improved = True
        iters = 0
        solution = list(solution)
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            selected_idx_set = set(idx for (idx, _) in solution)

            # Try to add any missing modulus
            added = False
            for idx in range(n):
                if idx in selected_idx_set:
                    continue
                valid_residues = get_valid_residues(solution, idx)
                if valid_residues:
                    solution.append((idx, valid_residues[0]))
                    selected_idx_set.add(idx)
                    improved = True
                    added = True
                    break

            if improved:
                continue

            # Try swap: remove one, try to add two
            for i in range(len(solution)):
                if time.time() > deadline - 5:
                    break
                removed = solution[i]
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)

                added = 0
                new_solution = list(reduced)
                for idx in range(n):
                    if idx in reduced_idx_set:
                        continue
                    valid_residues = get_valid_residues(new_solution, idx)
                    if valid_residues:
                        new_solution.append((idx, valid_residues[0]))
                        reduced_idx_set.add(idx)
                        added += 1
                        if added >= 2:
                            break

                if len(new_solution) > len(solution):
                    solution = new_solution
                    improved = True
                    break

        return solution

    def local_search_improve_v2(solution, max_iters=300):
        """More aggressive local search with 2-for-1 swaps too."""
        improved = True
        iters = 0
        solution = list(solution)
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            selected_idx_set = set(idx for (idx, _) in solution)

            # Try to add any missing modulus
            for idx in range(n):
                if idx in selected_idx_set:
                    continue
                valid_residues = get_valid_residues(solution, idx)
                if valid_residues:
                    solution.append((idx, valid_residues[0]))
                    selected_idx_set.add(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try 1-for-2 swap
            for i in range(len(solution)):
                if time.time() > deadline - 5:
                    break
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)

                added = 0
                new_solution = list(reduced)
                for idx in range(n):
                    if idx in reduced_idx_set:
                        continue
                    valid_residues = get_valid_residues(new_solution, idx)
                    if valid_residues:
                        new_solution.append((idx, valid_residues[0]))
                        reduced_idx_set.add(idx)
                        added += 1
                        if added >= 2:
                            break

                if len(new_solution) > len(solution):
                    solution = new_solution
                    improved = True
                    break

        return solution

    def branch_and_bound(selected, available_indices, best_len, time_limit):
        """
        Branch and bound to try to extend the solution.
        selected: list of (idx, residue)
        available_indices: list of indices not yet assigned
        Returns: best solution found
        """
        if time.time() > time_limit:
            return selected
        if not available_indices:
            return selected

        # MRV: pick index with fewest valid residues
        best_idx = None
        best_residues = None
        best_count = n + 1
        for idx in available_indices:
            residues = get_valid_residues(selected, idx)
            if not residues:
                continue
            if len(residues) < best_count:
                best_count = len(residues)
                best_idx = idx
                best_residues = residues

        if best_idx is None:
            return selected

        best_result = list(selected)
        for r in best_residues:
            if time.time() > time_limit:
                break
            new_selected = selected + [(best_idx, r)]
            remaining = [idx for idx in available_indices if idx != best_idx]
            result = branch_and_bound(new_selected, remaining, best_len, time_limit)
            if len(result) > len(best_result):
                best_result = result

        return best_result

    # Phase 1: Greedy + random restarts to get good initial solutions
    random.seed(42)
    best_solution = []
    best_len = 0

    # First, try a deterministic greedy with MRV ordering
    def greedy_mrv():
        selected = []
        selected_set = set()
        remaining = list(range(n))
        while remaining:
            # Pick index with fewest valid residues
            best_idx = None
            best_residues = None
            best_count = n + 1
            for idx in remaining:
                residues = get_valid_residues(selected, idx)
                if not residues:
                    continue
                if len(residues) < best_count:
                    best_count = len(residues)
                    best_idx = idx
                    best_residues = residues
            if best_idx is None:
                break
            # Pick the first valid residue
            selected.append((best_idx, best_residues[0]))
            selected_set.add(best_idx)
            remaining.remove(best_idx)
        return selected

    sol = greedy_mrv()
    if len(sol) > best_len:
        best_solution = sol
        best_len = len(sol)

    # Random restarts
    for restart in range(200):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        selected_set = set()
        sol = greedy_randomized(order, selected_set)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)

    # Phase 2: Local search improvement
    if best_solution:
        improved_sol = local_search_improve(best_solution)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)

    # Phase 3: More local search
    if best_solution:
        improved_sol = local_search_improve_v2(best_solution)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(improved_sol)

    # Phase 4: Branch-and-bound to try to improve further
    if time.time() < deadline - 10 and best_solution:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        available = [idx for idx in range(n) if idx not in selected_idx_set]

        bt_result = branch_and_bound(
            list(best_solution), available, best_len, deadline - 5
        )
        if len(bt_result) > best_len:
            best_solution = bt_result
            best_len = len(best_solution)

    # Phase 5: Final local search
    if best_solution:
        improved_sol = local_search_improve(best_solution, max_iters=200)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(improved_sol)

    # Final validation: ensure all pairs are truly disjoint
    def validate(solution):
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                idx1, r1 = solution[i]
                idx2, r2 = solution[j]
                g = gcd_tab[idx1][idx2]
                if (r1 - r2) % g == 0:
                    return False
        return True

    if not validate(best_solution):
        # Repair: remove conflicting pairs
        repaired = []
        for item in best_solution:
            if all(is_compatible(repaired, item[0], item[1]) for _ in [1]):
                repaired.append(item)
        best_solution = repaired

    # Convert to required format [[r, m], ...]
    result = [[r, candidates[idx]] for (idx, r) in best_solution]
    return result
