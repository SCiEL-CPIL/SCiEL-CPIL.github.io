import time
import random
import math
from math import gcd

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Uses a constraint-propagation backtracking approach: for each new modulus, compute the set of
    forbidden residues (union of forbidden APs from previous selections) and pick a valid residue
    that maximizes future flexibility.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MAX_M = 128
    MODULI = list(range(2, MAX_M + 1))

    # Precompute gcd table
    gcd_table = {}
    for i, m1 in enumerate(MODULI):
        for m2 in MODULI[i:]:
            g = gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            if m1 != m2:
                gcd_table[(m2, m1)] = g

    def is_compatible(r, m, chosen):
        """Check if (r, m) is compatible with all pairs in chosen list [(r2, m2), ...]."""
        for r2, m2 in chosen:
            g = gcd_table[(m, m2)]
            if g == 1:
                return False
            if (r - r2) % g == 0:
                return False
        return True

    def compute_forbidden(m, chosen):
        """
        For modulus m, compute the set of forbidden residues based on chosen pairs.
        Returns a boolean array of length m where True means forbidden.
        """
        forbidden = [False] * m
        for r2, m2 in chosen:
            g = gcd_table[(m, m2)]
            if g == 1:
                # All residues are forbidden (classes always intersect)
                return None  # Signal: impossible
            # Forbidden residues: r such that r % g == r2 % g
            rem = r2 % g
            for r in range(rem, m, g):
                forbidden[r] = True
        return forbidden

    def count_allowed(m, chosen):
        """Count how many residues are allowed for modulus m given chosen pairs."""
        forbidden = compute_forbidden(m, chosen)
        if forbidden is None:
            return 0
        return sum(1 for f in forbidden if not f)

    def find_best_residue(m, chosen):
        """Find the best residue for modulus m, preferring residues that leave most options for others."""
        forbidden = compute_forbidden(m, chosen)
        if forbidden is None:
            return -1
        allowed = [r for r in range(m) if not forbidden[r]]
        if not allowed:
            return -1
        # Pick the residue that minimizes the maximum gcd with other moduli
        # This leaves more flexibility
        best_r = allowed[0]
        best_score = -1
        for r in allowed:
            # Score: minimize the sum of gcds with existing moduli (smaller gcds are easier to avoid)
            score = sum(gcd_table[(m, m2)] for _, m2 in chosen)
            # Actually, we want to pick r such that for each existing (r2, m2),
            # (r - r2) % gcd(m, m2) != 0, which is already guaranteed by being allowed.
            # For future flexibility, prefer residues that are "spread out"
            # Use a simple heuristic: pick the residue that is furthest from existing residues mod small gcds
            min_dist = float('inf')
            for r2, m2 in chosen:
                g = gcd_table[(m, m2)]
                dist = min((r - r2) % g, g - (r - r2) % g)
                min_dist = min(min_dist, dist)
            if min_dist > best_score:
                best_score = min_dist
                best_r = r
        return best_r

    def greedy_build(moduli_order, chosen):
        """Greedy: add moduli in given order, pick a valid residue if possible."""
        result = list(chosen)
        for m in moduli_order:
            if any(m2 == m for _, m2 in result):
                continue
            r = find_best_residue(m, result)
            if r >= 0:
                result.append((r, m))
        return result

    def backtracking_search(moduli_pool, chosen, depth_limit=30, timeout=5.0):
        """
        Recursive backtracking with constraint propagation.
        Try to add more moduli from moduli_pool to chosen.
        """
        start_bt = time.time()
        best = list(chosen)
        best_size = len(best)

        if not moduli_pool or depth_limit <= 0:
            return best

        # Sort moduli by how constrained they are (fewer allowed residues = more urgent)
        def constraint_score(m):
            return count_allowed(m, chosen)

        sorted_pool = sorted(moduli_pool, key=constraint_score)

        # Try each modulus in order of constraint
        for m in sorted_pool:
            if time.time() - start_bt > timeout:
                break
            if any(m2 == m for _, m2 in best):
                continue

            forbidden = compute_forbidden(m, chosen)
            if forbidden is None:
                continue

            allowed = [r for r in range(m) if not forbidden[r]]
            if not allowed:
                continue

            # Try each allowed residue
            for r in allowed:
                new_chosen = chosen + [(r, m)]
                new_pool = [pm for pm in moduli_pool if pm != m]

                result = backtracking_search(new_pool, new_chosen, depth_limit - 1, timeout - (time.time() - start_bt))
                if len(result) > best_size:
                    best = result
                    best_size = len(result)

                if time.time() - start_bt > timeout:
                    break

        return best

    def local_search(selected, rng, max_iters=10000):
        """Try to improve by adding, removing, or swapping moduli."""
        current = [list(x) for x in selected]
        current_size = len(current)
        best = [list(x) for x in current]
        best_size = current_size
        moduli_in = set(m for _, m in current)
        moduli_out = [m for m in MODULI if m not in moduli_in]

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 2:
                break

            op = rng.random()
            if op < 0.4 and moduli_out:
                # Try to add a random modulus from moduli_out
                m = rng.choice(moduli_out)
                forbidden = compute_forbidden(m, current)
                if forbidden is None:
                    continue
                allowed = [r for r in range(m) if not forbidden[r]]
                if allowed:
                    r = rng.choice(allowed)
                    new_sel = current + [[r, m]]
                    if len(new_sel) > best_size:
                        best = [list(x) for x in new_sel]
                        best_size = len(new_sel)
                    current = new_sel
                    moduli_in.add(m)
                    moduli_out.remove(m)
            elif op < 0.7:
                # Try to change a residue of an existing modulus
                if not current:
                    continue
                idx = rng.randrange(len(current))
                r_old, m = current[idx]
                forbidden = compute_forbidden(m, [(r2, m2) for j, (r2, m2) in enumerate(current) if j != idx])
                if forbidden is not None:
                    allowed = [r for r in range(m) if not forbidden[r] and r != r_old]
                    if allowed:
                        r_new = rng.choice(allowed)
                        current[idx][0] = r_new
            else:
                # Try to remove a modulus and re-add a different one
                if not current:
                    continue
                idx = rng.randrange(len(current))
                removed = current.pop(idx)
                moduli_in.discard(removed[1])
                moduli_out.append(removed[1])
                # Try to add a random modulus
                if moduli_out:
                    m = rng.choice(moduli_out)
                    forbidden = compute_forbidden(m, current)
                    if forbidden is not None:
                        allowed = [r for r in range(m) if not forbidden[r]]
                        if allowed:
                            r = rng.choice(allowed)
                            current.append([r, m])
                            moduli_in.add(m)
                            moduli_out.remove(m)
                            if len(current) > best_size:
                                best = [list(x) for x in current]
                                best_size = len(current)

        return best

    best_result = []
    best_size = 0

    rng = random.Random(42)

    # Phase 1: Greedy with multiple random orderings
    for restart in range(100):
        if time.time() - start_time > DEADLINE - 10:
            break

        order = list(MODULI)
        rng.shuffle(order)

        sel = greedy_build(order, [])
        if len(sel) > best_size:
            best_size = len(sel)
            best_result = [list(x) for x in sel]

    # Phase 2: Backtracking search to improve
    if time.time() - start_time < DEADLINE - 15:
        used_moduli = set(m for _, m in best_result)
        unused = [m for m in MODULI if m not in used_moduli]
        if unused:
            improved = backtracking_search(unused, best_result, depth_limit=10, timeout=10.0)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = [list(x) for x in improved]

    # Phase 3: Local search
    if time.time() - start_time < DEADLINE - 5:
        improved = local_search(best_result, rng, max_iters=20000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = [list(x) for x in improved]

    # Final polish: try to add any remaining moduli
    if time.time() - start_time < DEADLINE - 3:
        moduli_in = set(m for _, m in best_result)
        moduli_out = [m for m in MODULI if m not in moduli_in]
        for m in moduli_out:
            if time.time() - start_time > DEADLINE - 3:
                break
            forbidden = compute_forbidden(m, best_result)
            if forbidden is not None:
                allowed = [r for r in range(m) if not forbidden[r]]
                if allowed:
                    best_result.append([allowed[0], m])

    # Ensure canonical form and distinct moduli
    if not best_result:
        return [[1, 2]]

    # Verify and clean up
    result = []
    seen_moduli = set()
    for r, m in best_result:
        if m in seen_moduli:
            continue
        if 0 <= r < m and 2 <= m <= 128:
            result.append([r, m])
            seen_moduli.add(m)

    return result
