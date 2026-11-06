"""Combined: correct gcd>1 clique constraint + greedy/local search/randomized restarts
to maximize disjoint residue classes with distinct moduli 2..128."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START = None
MAX_MOD = 128


def _prime_factors(m):
    """Distinct prime factors of m."""
    facs = set()
    d = 2
    while d * d <= m:
        if m % d == 0:
            facs.add(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        facs.add(m)
    return facs


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 85.0  # Safety margin before 90s kill

    moduli = list(range(2, MAX_MOD + 1))
    n_mod = len(moduli)
    mod_index = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute omega for ordering heuristics
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

    # Precompute orderings
    orderings = []
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], m)))  # Most constrained first
    orderings.append(sorted(moduli, key=lambda m: (omega_cache[m], m)))   # Least constrained first
    orderings.append(sorted(moduli))                                       # Ascending
    orderings.append(sorted(moduli, reverse=True))                         # Descending
    orderings.append(sorted(moduli, key=lambda m: (-omega_cache[m], -m)))  # Most constrained, largest first

    def _find_residue(m_idx, selected_pairs, selected_mod_indices):
        """Find a residue r in [0, moduli[m_idx]) disjoint from all selected classes."""
        m = moduli[m_idx]
        constraints = []
        for k, (r_i, m_i) in enumerate(selected_pairs):
            m_i_idx = selected_mod_indices[k]
            g = gcd_table[m_idx][m_i_idx]
            if g > 1:
                constraints.append((g, r_i))

        if not constraints:
            return 0

        for r in range(m):
            valid = True
            for (g, r_i) in constraints:
                if (r - r_i) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy_pass(order):
        """One greedy pass given an ordering of moduli. Returns list of (r, m) pairs."""
        selected_pairs = []
        selected_mod_indices = []
        used = set()

        for m in order:
            if m in used:
                continue
            m_idx = mod_index[m]

            # CRITICAL: gcd must be > 1 with ALL selected moduli
            compatible = True
            for m_sel_idx in selected_mod_indices:
                if gcd_table[m_idx][m_sel_idx] == 1:
                    compatible = False
                    break
            if not compatible:
                continue

            r = _find_residue(m_idx, selected_pairs, selected_mod_indices)
            if r is not None:
                selected_pairs.append((r, m))
                selected_mod_indices.append(m_idx)
                used.add(m)

        return selected_pairs

    best_pairs = []
    best_count = 0

    # Run deterministic orderings
    for order in orderings:
        if time.time() > deadline:
            break
        pairs = greedy_pass(order)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # Randomized restarts
    random.seed(42)
    num_restarts = 3000
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random ordering
        shuffled = moduli[:]
        random.shuffle(shuffled)
        pairs = greedy_pass(shuffled)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_pairs = pairs[:]

    # Local search: try to add more moduli to the best solution
    if time.time() < deadline - 5.0 and best_pairs:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_pairs = best_pairs[:]
            used = set(m for _, m in current_pairs)
            current_mod_indices = [mod_index[m] for _, m in current_pairs]

            # Try adding each unused modulus
            for m in moduli:
                if time.time() > deadline - 2.0:
                    break
                if m in used:
                    continue

                m_idx = mod_index[m]

                # Check gcd compatibility with all selected
                compatible = True
                for m_sel_idx in current_mod_indices:
                    if gcd_table[m_idx][m_sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue

                r = _find_residue(m_idx, current_pairs, current_mod_indices)
                if r is not None:
                    current_pairs.append((r, m))
                    current_mod_indices.append(m_idx)
                    used.add(m)
                    improved = True

            if len(current_pairs) > len(best_pairs):
                best_pairs = current_pairs
                best_count = len(best_pairs)

    # Additional randomized local search passes
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            # Start from a random subset of the best solution and try to extend
            if not best_pairs:
                break
            subset_size = max(1, len(best_pairs) // 2)
            subset = random.sample(best_pairs, min(subset_size, len(best_pairs)))
            current_pairs = subset[:]
            used = set(m for _, m in current_pairs)
            current_mod_indices = [mod_index[m] for _, m in current_pairs]

            # Try adding all unused moduli greedily
            for m in moduli:
                if time.time() > deadline - 10.0:
                    break
                if m in used:
                    continue
                m_idx = mod_index[m]
                compatible = True
                for m_sel_idx in current_mod_indices:
                    if gcd_table[m_idx][m_sel_idx] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                r = _find_residue(m_idx, current_pairs, current_mod_indices)
                if r is not None:
                    current_pairs.append((r, m))
                    current_mod_indices.append(m_idx)
                    used.add(m)

            if len(current_pairs) > best_count:
                best_count = len(current_pairs)
                best_pairs = current_pairs[:]

    # Safety fallback
    if not best_pairs:
        best_pairs = [(1, 2)]

    # Enforce canonical residues and distinct moduli
    seen = set()
    out = []
    for r, m in best_pairs:
        if m in seen:
            continue
        if not (0 <= r < m):
            continue
        seen.add(m)
        out.append([r, m])

    if not out:
        out = [[1, 2]]

    return out
# EVOLVE-BLOCK-END
