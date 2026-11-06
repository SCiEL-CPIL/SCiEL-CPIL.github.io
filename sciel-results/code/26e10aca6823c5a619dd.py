"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START_TIME = None


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect (coprime moduli)
    return (r2 - r1) % g != 0


def _can_add(r_new, m_new, existing):
    """Check if new class is disjoint from all existing."""
    for (r, m) in existing:
        if not _disjoint(r_new, m_new, r, m):
            return False
    return True


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    deadline = _START_TIME + 80.0  # safety margin before 90s kill

    # Candidate moduli from 2 to 128
    moduli = list(range(2, 129))
    n = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # constraint_score: how many other moduli share a factor (higher = more constrained)
    constraint = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd_table[i][j] > 1:
                constraint[i] += 1
                constraint[j] += 1

    # Build several deterministic orderings
    order_most_constrained = sorted(range(n), key=lambda i: (-constraint[i], -moduli[i]))
    order_least_constrained = sorted(range(n), key=lambda i: (constraint[i], moduli[i]))
    order_desc = list(range(n - 1, -1, -1))  # moduli 128..2
    order_asc = list(range(n))               # moduli 2..128

    def greedy_pass(order, residue_mode, rng):
        """
        Greedy: iterate moduli in `order`, pick a valid residue.
        residue_mode: 0 = first valid (deterministic), 1 = random valid,
                      2 = pick residue minimizing future blocking.
        Returns list of [r, m].
        """
        current = []
        used = set()
        for idx in order:
            m = moduli[idx]
            if m in used:
                continue
            if not current:
                # First class: any residue works; pick 0 for determinism or random
                r = rng.randrange(m) if residue_mode == 1 else 0
                current.append([r, m])
                used.add(m)
                continue

            if residue_mode == 0:
                # First valid residue
                for r in range(m):
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used.add(m)
                        break
            elif residue_mode == 1:
                # Random valid residue
                residues = list(range(m))
                rng.shuffle(residues)
                for r in residues:
                    if _can_add(r, m, current):
                        current.append([r, m])
                        used.add(m)
                        break
            else:
                # residue_mode == 2: pick residue that blocks fewest future moduli
                best_r = None
                best_score = None
                for r in range(m):
                    if not _can_add(r, m, current):
                        continue
                    score = 0
                    for j in order:
                        m2 = moduli[j]
                        if m2 in used or m2 == m:
                            continue
                        g = gcd_table[idx][j]
                        if g > 1:
                            blocked = m2 // g
                            score += blocked
                    if best_score is None or score < best_score:
                        best_score = score
                        best_r = r
                if best_r is not None:
                    current.append([best_r, m])
                    used.add(m)
        return current

    def local_search(sol):
        """Try to improve sol by adding missing moduli and swapping residues."""
        if not sol:
            return sol
        sol = [x[:] for x in sol]
        used = set(m for _, m in sol)
        improved = True
        while improved:
            improved = False
            # Try adding a missing modulus
            for m in moduli:
                if m in used:
                    continue
                for r in range(m):
                    if _can_add(r, m, sol):
                        sol.append([r, m])
                        used.add(m)
                        improved = True
                        break
                if improved:
                    break
            # Try swapping a residue to free up a modulus
            if not improved:
                for i, (r1, m1) in enumerate(sol):
                    for r_new in range(m1):
                        if r_new == r1:
                            continue
                        old = sol[i]
                        sol[i] = [r_new, m1]
                        ok = all(_disjoint(r_new, m1, r2, m2)
                                 for j, (r2, m2) in enumerate(sol) if j != i)
                        if ok:
                            for m_add in moduli:
                                if m_add in used:
                                    continue
                                for r_add in range(m_add):
                                    if _can_add(r_add, m_add, sol):
                                        sol.append([r_add, m_add])
                                        used.add(m_add)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        else:
                            sol[i] = old
                    if improved:
                        break
                    else:
                        sol[i] = old
        return sol

    def _count_valid_residues(m_idx, assigned):
        """Count how many residues are valid for modulus m_idx given current assignments."""
        m = moduli[m_idx]
        count = 0
        for r in range(m):
            valid = True
            for (r2, m2_idx) in assigned:
                if not _disjoint(r, m, r2, moduli[m2_idx]):
                    valid = False
                    break
            if valid:
                count += 1
        return count

    def csp_solve(assigned, unassigned, depth, max_depth):
        """
        CSP backtracking with MRV heuristic.
        assigned: list of (residue, modulus_idx)
        unassigned: set of modulus indices not yet assigned
        Returns list of (residue, modulus_idx) or None if no solution.
        """
        if not unassigned:
            return assigned[:]
        
        if depth >= max_depth:
            return None
        
        # MRV: pick unassigned modulus with fewest valid residues
        min_count = float('inf')
        best_idx = None
        candidates = []
        for idx in unassigned:
            count = _count_valid_residues(idx, assigned)
            if count == 0:
                return None  # Dead end
            if count < min_count:
                min_count = count
                best_idx = idx
                candidates = [idx]
            elif count == min_count:
                candidates.append(idx)
        
        # Try residues for the chosen modulus
        m = moduli[best_idx]
        valid_residues = []
        for r in range(m):
            if all(_disjoint(r, m, r2, moduli[m2_idx]) for (r2, m2_idx) in assigned):
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        # Shuffle for diversity
        random.shuffle(valid_residues)
        
        for r in valid_residues:
            new_assigned = assigned + [(r, best_idx)]
            new_unassigned = unassigned - {best_idx}
            result = csp_solve(new_assigned, new_unassigned, depth + 1, max_depth)
            if result is not None:
                return result
        
        return None

    best_solution = []
    best_size = 0

    def update(sol):
        nonlocal best_solution, best_size
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = [x[:] for x in sol]

    # Phase 1: CSP-based approach with MRV
    # Start with an empty assignment and try to build up
    if time.time() < deadline - 30:
        # Use the most constrained ordering to guide CSP
        all_indices = set(range(n))
        # Try CSP with increasing max_depth
        for max_depth in [10, 20, 30, 40, 50, 60, 64]:
            if time.time() > deadline - 30:
                break
            result = csp_solve([], all_indices, 0, max_depth)
            if result is not None:
                sol = [[r, moduli[idx]] for (r, idx) in result]
                update(sol)
                # Try to extend with local search
                if time.time() < deadline - 20:
                    sol2 = local_search(sol)
                    update(sol2)
                if best_size >= 64:
                    break

    # Phase 2: deterministic greedy passes
    deterministic_orders = [
        (order_most_constrained, 0),
        (order_most_constrained, 2),
        (order_least_constrained, 0),
        (order_least_constrained, 2),
        (order_desc, 0),
        (order_desc, 2),
        (order_asc, 0),
        (order_asc, 2),
    ]
    for order, mode in deterministic_orders:
        if time.time() > deadline:
            break
        sol = greedy_pass(order, mode, random)
        update(sol)
        if time.time() < deadline - 15:
            sol2 = local_search(sol)
            update(sol2)

    # Phase 3: randomized restarts
    rng = random
    for restart in range(200):
        if time.time() > deadline:
            break
        if restart % 3 == 0:
            order = order_most_constrained[:]
            rng.shuffle(order)
        elif restart % 3 == 1:
            order = order_least_constrained[:]
            rng.shuffle(order)
        else:
            order = list(range(n))
            rng.shuffle(order)
        mode = rng.choice([0, 1, 2])
        sol = greedy_pass(order, mode, rng)
        update(sol)
        if len(sol) >= best_size and time.time() < deadline - 10:
            sol2 = local_search(sol)
            update(sol2)

    # Final local search on the best solution
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
        update(best_solution)

    if not best_solution:
        return [[1, 2]]

    return best_solution
# EVOLVE-BLOCK-END
