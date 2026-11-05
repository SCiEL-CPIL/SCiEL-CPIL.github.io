import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    MAX_M = 128
    # Moduli 2..128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs of moduli
    gcd_tab = [[1] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    def is_disjoint(r1, m1_idx, r2, m2_idx):
        """Check if residue classes (r1, m1) and (r2, m2) are disjoint.
        Disjoint iff (r1 - r2) % gcd(m1, m2) != 0."""
        g = gcd_tab[m1_idx][m2_idx]
        return (r1 - r2) % g != 0

    def get_conflict_residues(selected, idx):
        """For a given modulus index idx, find which residues conflict with selected.
        Returns a boolean array of length moduli[idx] where True means conflict."""
        m = moduli[idx]
        conflict = [False] * m
        for (sel_idx, sel_r) in selected:
            g = gcd_tab[idx][sel_idx]
            # Residues r in [0, m) that conflict: (r - sel_r) % g == 0
            # i.e., r % g == sel_r % g
            base = sel_r % g
            for r in range(base, m, g):
                conflict[r] = True
        return conflict

    def count_valid(selected, idx):
        """Count valid residues for idx given selected."""
        conflict = get_conflict_residues(selected, idx)
        return sum(1 for c in conflict if not c)

    def find_valid_residues(selected, idx):
        """Find all valid residues for idx given selected."""
        conflict = get_conflict_residues(selected, idx)
        return [r for r in range(moduli[idx]) if not conflict[r]]

    # Phase 1: Greedy with multiple orderings
    def greedy(order):
        selected = []
        for idx in order:
            valid = find_valid_residues(selected, idx)
            if valid:
                # Pick a random valid residue to diversify
                r = random.choice(valid)
                selected.append((idx, r))
        return selected

    # Phase 2: Backtracking with MRV heuristic
    def backtrack(selected, available, time_limit):
        """Try to add more elements to selected from available using backtracking."""
        if time.time() > time_limit:
            return selected

        if not available:
            return selected

        # MRV: pick index with fewest valid residues
        best_idx = None
        best_count = float('inf')
        best_residues = None
        for idx in available:
            cnt = count_valid(selected, idx)
            if cnt == 0:
                continue
            if cnt < best_count:
                best_count = cnt
                best_idx = idx
                best_residues = find_valid_residues(selected, idx)
                if cnt == 1:
                    break

        if best_idx is None:
            return selected

        remaining = [idx for idx in available if idx != best_idx]

        # Try residues in order of least constraining (or just all)
        for r in best_residues:
            new_selected = selected + [(best_idx, r)]
            result = backtrack(new_selected, remaining, time_limit)
            if len(result) > len(selected):
                return result

        return selected

    # Phase 3: Local search - try to add missing or swap
    def local_search(solution, max_iters=500):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            selected_idx_set = set(idx for (idx, _) in solution)
            missing = [idx for idx in range(n) if idx not in selected_idx_set]

            # Try to add any missing modulus
            for idx in missing:
                valid = find_valid_residues(solution, idx)
                if valid:
                    solution.append((idx, valid[0]))
                    improved = True
                    break

            if improved:
                continue

            # Try swap: remove one, try to add two
            for i in range(len(solution)):
                removed = solution[i]
                reduced = [x for k, x in enumerate(solution) if k != i]
                reduced_idx_set = set(idx for (idx, _) in reduced)
                missing_after = [idx for idx in range(n) if idx not in reduced_idx_set]

                added = 0
                new_solution = list(reduced)
                for idx in missing_after:
                    valid = find_valid_residues(new_solution, idx)
                    if valid:
                        new_solution.append((idx, valid[0]))
                        added += 1
                        if added >= 2:
                            break

                if len(new_solution) > len(solution):
                    solution = new_solution
                    improved = True
                    break

        return solution

    # Main search
    random.seed(42)
    best_solution = []
    best_len = 0

    # Phase 1: Greedy with random restarts
    for restart in range(300):
        if time.time() > deadline - 30:
            break
        order = list(range(n))
        random.shuffle(order)
        sol = greedy(order)
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)

    # Phase 2: Local search on best
    if best_solution:
        improved_sol = local_search(best_solution)
        if len(improved_sol) > best_len:
            best_solution = improved_sol
            best_len = len(best_solution)

    # Phase 3: Backtracking to add more
    if time.time() < deadline - 12 and best_solution:
        selected_idx_set = set(idx for (idx, _) in best_solution)
        available = [idx for idx in range(n) if idx not in selected_idx_set]
        bt_result = backtrack(list(best_solution), available, deadline - 5)
        if len(bt_result) > best_len:
            best_solution = bt_result
            best_len = len(best_solution)

    # Final validation
    def validate(solution):
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                idx1, r1 = solution[i]
                idx2, r2 = solution[j]
                if not is_disjoint(r1, idx1, r2, idx2):
                    return False
        return True

    if not validate(best_solution):
        # Repair
        repaired = []
        for item in best_solution:
            idx, r = item
            if all(is_disjoint(r, idx, r2, idx2) for (idx2, r2) in repaired):
                repaired.append(item)
        best_solution = repaired

    # Convert to required format [[r, m], ...]
    result = [[r, moduli[idx]] for (idx, r) in best_solution]
    return result
