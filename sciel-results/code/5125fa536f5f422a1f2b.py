import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table for all pairs of moduli
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        gcd_table[i][i] = moduli[i]
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute number of distinct prime factors (omega) for ordering heuristics
    def omega(m):
        count = 0
        d = 2
        while d * d <= m:
            if m % d == 0:
                count += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            count += 1
        return count

    omega_cache = {m: omega(m) for m in moduli}

    # Precompute a few deterministic orderings (indices into moduli)
    orderings = []
    orderings.append(sorted(range(n_mod), key=lambda i: (-omega_cache[moduli[i]], moduli[i])))  # most constrained first
    orderings.append(sorted(range(n_mod), key=lambda i: (omega_cache[moduli[i]], moduli[i])))   # least constrained first
    orderings.append(sorted(range(n_mod), key=lambda i: moduli[i]))                             # ascending
    orderings.append(sorted(range(n_mod), key=lambda i: -moduli[i]))                            # descending

    def find_residue(m_idx, selected_pairs, selected_indices):
        """Find a residue r in [0, m) disjoint from all selected classes."""
        m = moduli[m_idx]
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            g = gcd_table[m_idx][selected_indices[k]]
            if g > 1:
                constraints.append((g, r_i))
        if not constraints:
            return 0
        # Try residues in a smart order: 0, then the rest
        for r in range(m):
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy_pass(order_indices):
        """One greedy pass given an ordering of modulus indices."""
        selected_pairs = []
        selected_indices = []
        used = set()
        for m_idx in order_indices:
            m = moduli[m_idx]
            if m in used:
                continue
            # Must have gcd > 1 with every selected modulus (else always intersects)
            compatible = True
            for s_idx in selected_indices:
                if gcd_table[m_idx][s_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            r = find_residue(m_idx, selected_pairs, selected_indices)
            if r is not None:
                selected_pairs.append((r, m))
                selected_indices.append(m_idx)
                used.add(m)
        return selected_pairs, selected_indices

    best_pairs = []
    best_count = 0

    # Deterministic orderings
    for order in orderings:
        if time.time() > deadline:
            break
        pairs, _ = greedy_pass(order)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # Randomized restarts
    random.seed(42)
    all_indices = list(range(n_mod))
    while time.time() < deadline - 12.0:
        shuffled = all_indices[:]
        random.shuffle(shuffled)
        pairs, _ = greedy_pass(shuffled)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # Local search: repeatedly try to add any unused compatible modulus
    if time.time() < deadline - 4.0 and best_pairs:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            cur_pairs = best_pairs[:]
            cur_indices = []
            used = set()
            for (r, m) in cur_pairs:
                idx = mod_to_idx[m]
                cur_indices.append(idx)
                used.add(m)
            for m_idx in range(n_mod):
                if time.time() > deadline - 2.0:
                    break
                m = moduli[m_idx]
                if m in used:
                    continue
                compatible = True
                for s_idx in cur_indices:
                    if gcd_table[m_idx][s_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = find_residue(m_idx, cur_pairs, cur_indices)
                if r is not None:
                    cur_pairs.append((r, m))
                    cur_indices.append(m_idx)
                    used.add(m)
                    improved = True
            if len(cur_pairs) > len(best_pairs):
                best_pairs = cur_pairs
                best_count = len(cur_pairs)

    if not best_pairs:
        return [[1, 2]]
    return [[r, m] for (r, m) in best_pairs]
