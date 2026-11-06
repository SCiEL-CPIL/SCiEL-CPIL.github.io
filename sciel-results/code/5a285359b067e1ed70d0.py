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
                # residue_mode == 2: pick residue that minimizes conflict potential
                # with existing classes (count how many existing classes have
                # gcd > 1, weighted by how many residues are blocked)
                best_r = None
                best_score = None
                for r in range(m):
                    if not _can_add(r, m, current):
                        continue
                    # Score: for each existing class, if gcd(m, m_existing) > 1,
                    # the residue r blocks a specific residue class mod g.
                    # We want to pick r that is "least aligned" with existing
                    # classes' residue classes. Count the number of existing
                    # classes where (r - r_existing) % g == 0 would be a conflict
                    # if we picked the wrong residue — but since we already
                    # verified disjointness, instead score by how many future
                    # moduli this residue would restrict.
                    # Better heuristic: for each existing (r_e, m_e) with
                    # g = gcd(m, m_e) > 1, the residue r determines which
                    # residue class mod g we occupy. Count how many existing
                    # classes we are "close to" in the mod-g sense.
                    score = 0
                    for (r_e, m_e) in current:
                        g = _gcd(m, m_e)
                        if g > 1:
                            # We occupy residue class r % g mod g.
                            # The existing occupies r_e % g mod g.
                            # They are disjoint (guaranteed), so r % g != r_e % g.
                            # For future moduli m_f with gcd(m_f, m) = h > 1,
                            # our choice of r % h matters. Count how many
                            # distinct residue classes mod h we are "pinned" to.
                            # Simpler: just count the number of existing classes
                            # sharing a factor, as a proxy for constraint tightness.
                            score += 1
                    if best_score is None or score < best_score:
                        best_score = score
                        best_r = r
                if best_r is None:
                    # Fallback: try each residue
                    for r in range(m):
                        if _can_add(r, m, current):
                            best_r = r
                            break
                if best_r is not None:
                    current.append([best_r, m])
                    used.add(m)
        return current

    def local_search(sol):
        """Try to improve sol by adding missing moduli and optimizing residues."""
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
            # Try optimizing residues to enable adding more moduli
            if not improved:
                # For each class, try all residues and keep the one that
                # allows the most additional moduli to be added
                best_improvement = 0
                best_change = None
                for i, (r1, m1) in enumerate(sol):
                    for r_new in range(m1):
                        if r_new == r1:
                            continue
                        # Check if this residue is valid with all others
                        ok = all(_disjoint(r_new, m1, r2, m2)
                                 for j, (r2, m2) in enumerate(sol) if j != i)
                        if not ok:
                            continue
                        # Try this residue and see how many moduli we can add
                        sol[i][0] = r_new
                        count_added = 0
                        added = []
                        for m_add in moduli:
                            if m_add in used:
                                continue
                            for r_add in range(m_add):
                                if _can_add(r_add, m_add, sol):
                                    sol.append([r_add, m_add])
                                    used.add(m_add)
                                    added.append([r_add, m_add])
                                    count_added += 1
                                    break
                        # Undo additions
                        for a in added:
                            sol.pop()
                            used.discard(a[1])
                        if count_added > best_improvement:
                            best_improvement = count_added
                            best_change = (i, r_new)
                        # Restore original residue
                        sol[i][0] = r1
                if best_change is not None:
                    i, r_new = best_change
                    sol[i][0] = r_new
                    # Now add all possible moduli
                    for m_add in moduli:
                        if m_add in used:
                            continue
                        for r_add in range(m_add):
                            if _can_add(r_add, m_add, sol):
                                sol.append([r_add, m_add])
                                used.add(m_add)
                                improved = True
                                break
        return sol

    best_solution = []
    best_size = 0

    def update(sol):
        nonlocal best_solution, best_size
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = [x[:] for x in sol]

    # Phase 1: deterministic greedy passes with different orderings and residue modes
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

    # Phase 2: randomized restarts
    rng = random
    for restart in range(300):
        if time.time() > deadline:
            break
        # Random order, sometimes biased toward constrained
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
        # Local search on the best found so far occasionally
        if len(sol) >= best_size and time.time() < deadline - 10:
            sol2 = local_search(sol)
            update(sol2)

    # Final local search on the best solution
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
        update(best_solution)
        # Additional polish: try to add any remaining moduli with optimized residues
        if best_solution and time.time() < deadline - 2:
            used = set(m for _, m in best_solution)
            for m in moduli:
                if m in used:
                    continue
                for r in range(m):
                    if _can_add(r, m, best_solution):
                        best_solution.append([r, m])
                        used.add(m)
                        break
            update(best_solution)

    if not best_solution:
        return [[1, 2]]

    return best_solution
# EVOLVE-BLOCK-END
