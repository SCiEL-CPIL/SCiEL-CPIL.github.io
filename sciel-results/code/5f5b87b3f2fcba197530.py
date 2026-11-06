import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin

    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))

    # Precompute gcd table
    n_mod = len(moduli)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute index for modulus value
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    best_result = [[1, 2]]  # fallback
    best_count = 1

    def is_valid_triangle(i, j, k):
        """Check if adding k to {i, j} creates a bad triangle (gcd 2 for all pairs)."""
        g_ij = gcd_table[i][j]
        g_ik = gcd_table[i][k]
        g_jk = gcd_table[j][k]
        # A triangle with all gcds equal to 2 is unsatisfiable
        # because we need 3 distinct residues mod 2.
        if g_ij == 2 and g_ik == 2 and g_jk == 2:
            return False
        return True

    def try_add_residue_deterministic(m_idx, selected):
        """Find a residue for modulus moduli[m_idx] disjoint from all selected, deterministically."""
        m = moduli[m_idx]
        constraints = []
        for (r_i, m_i) in selected:
            m_i_idx = mod_to_idx[m_i]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))

        # Sort constraints by g (smallest g is most restrictive)
        constraints.sort(key=lambda x: x[0])

        for r in range(m):
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def try_add_residue_random(m_idx, selected):
        """Find a residue for modulus moduli[m_idx] disjoint from all selected, with random shuffle."""
        m = moduli[m_idx]
        constraints = []
        for (r_i, m_i) in selected:
            m_i_idx = mod_to_idx[m_i]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))

        # Sort constraints by g (smallest g is most restrictive)
        constraints.sort(key=lambda x: x[0])

        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy_pass(order, use_random_residue=False):
        """One greedy pass with given ordering."""
        selected = []  # list of [residue, modulus]
        used_moduli = set()

        for idx in order:
            if time.time() > deadline:
                break
            m = moduli[idx]
            if m in used_moduli:
                continue

            # Check compatibility with selected moduli
            compatible = True
            for (r_i, m_i) in selected:
                m_i_idx = mod_to_idx[m_i]
                if gcd_table[idx][m_i_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue

            # Check for bad triangles with any pair in selected
            if len(selected) >= 2:
                bad_triangle = False
                for a in range(len(selected)):
                    for b in range(a + 1, len(selected)):
                        m_a = selected[a][1]
                        m_b = selected[b][1]
                        idx_a = mod_to_idx[m_a]
                        idx_b = mod_to_idx[m_b]
                        if not is_valid_triangle(idx_a, idx_b, idx):
                            bad_triangle = True
                            break
                    if bad_triangle:
                        break
                if bad_triangle:
                    continue

            # Try to find a valid residue
            if use_random_residue:
                r = try_add_residue_random(idx, selected)
            else:
                r = try_add_residue_deterministic(idx, selected)
            if r is not None:
                selected.append([r, m])
                used_moduli.add(m)

        return selected

    def local_search_add(current):
        """Try to add more moduli to current selection."""
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            used = set(m for _, m in current)
            for idx in range(n_mod):
                if time.time() > deadline - 2.0:
                    break
                m = moduli[idx]
                if m in used:
                    continue
                compatible = True
                for (r_i, m_i) in current:
                    m_i_idx = mod_to_idx[m_i]
                    if gcd_table[idx][m_i_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue

                # Check bad triangles
                bad_triangle = False
                if len(current) >= 2:
                    for a in range(len(current)):
                        for b in range(a + 1, len(current)):
                            m_a = current[a][1]
                            m_b = current[b][1]
                            idx_a = mod_to_idx[m_a]
                            idx_b = mod_to_idx[m_b]
                            if not is_valid_triangle(idx_a, idx_b, idx):
                                bad_triangle = True
                                break
                        if bad_triangle:
                            break
                if bad_triangle:
                    continue

                r = try_add_residue_deterministic(idx, current)
                if r is not None:
                    current.append([r, m])
                    used.add(m)
                    improved = True
        return current

    # Strategy 1: Try different orderings with deterministic residues
    # Ordering by modulus ascending (smaller moduli first)
    order_asc = list(range(n_mod))
    result = greedy_pass(order_asc, use_random_residue=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Ordering by modulus descending
    order_desc = list(range(n_mod - 1, -1, -1))
    result = greedy_pass(order_desc, use_random_residue=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Ordering by number of divisors (fewer divisors first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count

    order_divisors = sorted(range(n_mod), key=lambda i: num_divisors(moduli[i]))
    result = greedy_pass(order_divisors, use_random_residue=False)
    if len(result) > best_count:
        best_count = len(result)
        best_result = result

    # Strategy 2: Randomized passes with different biases
    base_primes = [None, 2, 3, 5, 7]
    num_restarts_per_prime = 50

    for bp in base_primes:
        for restart in range(num_restarts_per_prime):
            if time.time() > deadline - 5.0:
                break
            order = list(range(n_mod))

            if bp:
                multiples = [i for i in order if moduli[i] % bp == 0]
                others = [i for i in order if moduli[i] % bp != 0]
                random.shuffle(multiples)
                random.shuffle(others)
                order = multiples + others
            else:
                random.shuffle(order)

            result = greedy_pass(order, use_random_residue=True)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result

    # Strategy 3: Local search on best result
    if time.time() < deadline - 5.0:
        current = local_search_add([[r, m] for r, m in best_result])
        if len(current) > best_count:
            best_count = len(current)
            best_result = current

    # Strategy 4: More diverse random restarts with different residue strategies
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for restart in range(30):
            if time.time() > deadline - 2.0:
                break
            order = list(range(n_mod))
            random.shuffle(order)
            use_random = random.random() < 0.5
            result = greedy_pass(order, use_random_residue=use_random)
            if len(result) > best_count:
                best_count = len(result)
                best_result = result

    # Final local search
    if time.time() < deadline - 1.0:
        current = local_search_add([[r, m] for r, m in best_result])
        if len(current) > best_count:
            best_count = len(current)
            best_result = current

    return best_result
