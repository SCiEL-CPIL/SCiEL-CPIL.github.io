"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Moduli in [20, 200]
    moduli = list(range(20, 201))

    # Precompute gcd for all pairs of moduli
    n_mod = len(moduli)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    mod_idx = {m: i for i, m in enumerate(moduli)}
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def can_add_residue(classes, m, r):
        """Check if residue r for modulus m is disjoint from all classes in classes."""
        for (m2, r2) in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def count_future_valid_residues(classes, m, r, used_moduli):
        """Count how many valid residues remain for future moduli if we pick (m, r).
        A future modulus m2 (gcd>1 with m, not used) has valid residues r2 where
        (r2 - r) % gcd(m, m2) != 0. We count total valid residues across all
        future moduli as a flexibility score (higher = better)."""
        total = 0
        for m2 in moduli:
            if m2 in used_moduli or m2 == m:
                continue
            g = gcd(m, m2)
            if g == 1:
                continue
            # Number of residues r2 in [0, m2) with (r2 - r) % g != 0
            # Total residues: m2
            # Residues with (r2 - r) % g == 0: floor((m2-1 - r_mod)/g) + 1 if r_mod < m2 else 0
            # Simpler: count = m2 - (number with r2 ≡ r mod g)
            # Number of r2 in [0, m2) with r2 ≡ r (mod g):
            #   Let r0 = r % g. Residues r2 ≡ r0 (mod g) in [0, m2):
            #   count = m2 // g + (1 if m2 % g > r0 else 0)  -- wait, need care
            #   Actually: r2 = r0 + k*g for k >= 0, r2 < m2
            #   k ranges from 0 to floor((m2 - 1 - r0)/g) if r0 < m2 else -1
            if r0 := r % g:
                pass
            r0 = r % g
            if r0 < m2:
                conflict_count = (m2 - 1 - r0) // g + 1
            else:
                conflict_count = 0
            total += m2 - conflict_count
        return total

    def pick_best_residue_flexible(classes, m, used_moduli):
        """Pick the residue for modulus m that maximizes flexibility for future moduli.
        Among valid residues, prefer the one with highest future-valid-residue count."""
        valid_residues = []
        for r in range(m):
            if can_add_residue(classes, m, r):
                valid_residues.append(r)
        if not valid_residues:
            return None
        if len(valid_residues) == 1:
            return valid_residues[0]
        # Score each valid residue by flexibility
        best_r = valid_residues[0]
        best_score = -1
        for r in valid_residues:
            flex = count_future_valid_residues(classes, m, r, used_moduli)
            if flex > best_score:
                best_score = flex
                best_r = r
        return best_r

    def score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total

    def solve_greedy_flexible(order):
        """Greedy construction with flexibility-based residue selection."""
        classes = []
        used = set()
        for m in order:
            if m in used:
                continue
            # Check pairwise non-coprime with all existing
            compatible = True
            for (m2, r2) in classes:
                if gcd(m, m2) == 1:
                    compatible = False
                    break
            if not compatible:
                continue

            r = pick_best_residue_flexible(classes, m, used)
            if r is not None:
                classes.append([m, r])
                used.add(m)
        return classes

    def local_search(classes, max_time=20.0):
        """Improve solution: add moduli, remove-and-readd, re-optimize residues."""
        classes = [list(c) for c in classes]
        current_score = score(classes)
        used_moduli = set(m for m, r in classes)
        start = time.time()

        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False

            # Phase 1: Try to add each unused modulus (small first)
            for m in sorted([m for m in moduli if m not in used_moduli]):
                # Check non-coprime
                compatible = True
                for (m2, r2) in classes:
                    if gcd(m, m2) == 1:
                        compatible = False
                        break
                if not compatible:
                    continue

                r = pick_best_residue_flexible(classes, m, used_moduli)
                if r is not None:
                    new_classes = classes + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > current_score:
                        classes = new_classes
                        current_score = new_score
                        used_moduli.add(m)
                        improved = True
                        break  # Restart scan after addition

            # Phase 2: Try removing the largest modulus and re-adding better ones
            if not improved and classes:
                idx_to_remove = max(range(len(classes)), key=lambda i: classes[i][0])
                m_remove = classes[idx_to_remove][0]
                temp_classes = [c for i, c in enumerate(classes) if i != idx_to_remove]
                temp_used = set(m for m, r in temp_classes)
                temp_score = score(temp_classes)

                added_any = False
                for m_add in sorted([m for m in moduli if m not in temp_used]):
                    compatible = True
                    for (m2, r2) in temp_classes:
                        if gcd(m_add, m2) == 1:
                            compatible = False
                            break
                    if not compatible:
                        continue

                    r = pick_best_residue_flexible(temp_classes, m_add, temp_used)
                    if r is not None:
                        new_classes = temp_classes + [[m_add, r]]
                        new_score = score(new_classes)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            used_moduli = set(m for m, r in classes)
                            improved = True
                            added_any = True
                            break

            # Phase 3: Re-optimize residues of existing classes for flexibility
            if not improved:
                # Try re-picking residues for each class to maximize flexibility
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    others = [c for j, c in enumerate(classes) if j != i]
                    r_new = pick_best_residue_flexible(others, m_i, used_moduli - {m_i})
                    if r_new is not None and r_new != r_i:
                        # Check if this actually helps (it might not change score, but may enable additions)
                        # For now, accept if valid; score unchanged but flexibility may improve
                        classes[i] = [m_i, r_new]
                        improved = True

        return classes, current_score

    # --- Main search ---
    best_classes = []
    best_score = Fraction(0, 1)

    # Initial greedy with ascending moduli
    classes = solve_greedy_flexible(sorted(moduli))
    classes, sc = local_search(classes, max_time=15.0)
    if sc > best_score:
        best_score = sc
        best_classes = [list(c) for c in classes]

    # Multiple random restarts
    random.seed(42)
    for restart in range(30):
        if time.time() > deadline:
            break

        # Random order biased towards small moduli
        order = moduli.copy()
        random.shuffle(order)
        # Bias: sort by random weight + m/100 to favor small moduli
        order.sort(key=lambda m: random.random() * 50 + m / 100.0)

        classes = solve_greedy_flexible(order)
        classes, sc = local_search(classes, max_time=10.0)
        if sc > best_score:
            best_score = sc
            best_classes = [list(c) for c in classes]

    return best_classes
# EVOLVE-BLOCK-END
