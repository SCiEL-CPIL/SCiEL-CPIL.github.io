import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # Precompute gcd table for moduli 20..200
    MOD_MIN, MOD_MAX = 20, 200
    moduli = list(range(MOD_MIN, MOD_MAX + 1))
    n_mod = len(moduli)
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(n_mod):
            gcd_table[i][j] = math.gcd(moduli[i], moduli[j])

    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is compatible with all selected pairs."""
        for m_sel, r_sel in selected:
            g = math.gcd(m_new, m_sel)
            if r_new % g == r_sel % g:
                return False
        return True

    def find_valid_residues(m, selected):
        """Return list of valid residues for modulus m given selected pairs."""
        valid = []
        for r in range(m):
            ok = True
            for m_sel, r_sel in selected:
                g = math.gcd(m, m_sel)
                if r % g == r_sel % g:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid

    def score(selected):
        return sum(1.0 / m for m, r in selected)

    # --- Phase 1: Greedy initialization (multiple orders) ---
    best_selected = []
    best_score = 0.0

    # Try greedy in increasing order
    selected = []
    for m in range(MOD_MIN, MOD_MAX + 1):
        r = None
        for rr in range(m):
            if is_compatible(m, rr, selected):
                r = rr
                break
        if r is not None:
            selected.append((m, r))
    s = score(selected)
    if s > best_score:
        best_score = s
        best_selected = list(selected)

    # Try greedy in decreasing order
    selected = []
    for m in range(MOD_MAX, MOD_MIN - 1, -1):
        r = None
        for rr in range(m):
            if is_compatible(m, rr, selected):
                r = rr
                break
        if r is not None:
            selected.append((m, r))
    s = score(selected)
    if s > best_score:
        best_score = s
        best_selected = list(selected)

    # Try greedy with random tie-breaking (a few iterations)
    for _ in range(3):
        if time_left() < 5:
            break
        selected = []
        for m in range(MOD_MIN, MOD_MAX + 1):
            valid_rs = []
            for rr in range(m):
                if is_compatible(m, rr, selected):
                    valid_rs.append(rr)
            if valid_rs:
                r = random.choice(valid_rs)
                selected.append((m, r))
        s = score(selected)
        if s > best_score:
            best_score = s
            best_selected = list(selected)

    # --- Phase 2: Simulated Annealing ---
    current_selected = list(best_selected)
    current_score = best_score
    best_selected_sa = list(best_selected)
    best_score_sa = best_score

    # SA parameters
    T0 = 0.05
    Tend = 0.001
    alpha = 0.995  # cooling rate

    T = T0
    step = 0
    max_steps = 500000  # upper bound, but time-limited

    while time_left() > 3 and T > Tend:
        step += 1

        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Add a random (m, r)
            m = random.randint(MOD_MIN, MOD_MAX)
            # Check if m already in selected
            if any(sm == m for sm, sr in current_selected):
                continue
            r = random.randint(0, m - 1)
            if not is_compatible(m, r, current_selected):
                continue
            new_selected = current_selected + [(m, r)]
            new_score = score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_selected = new_selected
                current_score = new_score
        elif move_type < 0.5:
            # Remove a random element
            if len(current_selected) <= 1:
                continue
            idx = random.randint(0, len(current_selected) - 1)
            removed = current_selected[idx]
            new_selected = current_selected[:idx] + current_selected[idx+1:]
            new_score = score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_selected = new_selected
                current_score = new_score
        elif move_type < 0.75:
            # Swap: remove one, add another
            if len(current_selected) < 2:
                continue
            idx = random.randint(0, len(current_selected) - 1)
            removed_m = current_selected[idx][0]
            # Try to add a different modulus
            m = random.randint(MOD_MIN, MOD_MAX)
            if m == removed_m:
                continue
            if any(sm == m for sm, sr in current_selected):
                continue
            r = random.randint(0, m - 1)
            # Check compatibility with all except the one we're removing
            temp_selected = current_selected[:idx] + current_selected[idx+1:]
            if not is_compatible(m, r, temp_selected):
                continue
            new_selected = temp_selected + [(m, r)]
            new_score = score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_selected = new_selected
                current_score = new_score
        else:
            # Reassign residue for a random existing modulus
            if len(current_selected) < 1:
                continue
            idx = random.randint(0, len(current_selected) - 1)
            m = current_selected[idx][0]
            old_r = current_selected[idx][1]
            # Try a different residue
            r = random.randint(0, m - 1)
            if r == old_r:
                continue
            # Check compatibility with all others
            temp_selected = current_selected[:idx] + current_selected[idx+1:]
            if not is_compatible(m, r, temp_selected):
                continue
            new_selected = temp_selected + [(m, r)]
            new_score = score(new_selected)
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_selected = new_selected
                current_score = new_score

        # Update best
        if current_score > best_score_sa:
            best_score_sa = current_score
            best_selected_sa = list(current_selected)

        # Cool down
        T *= alpha

        # Periodic restart: if stuck, try to add high-value moduli
        if step % 5000 == 0 and time_left() > 10:
            # Try to add any missing high-value moduli
            for m in range(MOD_MIN, MOD_MAX + 1):
                if any(sm == m for sm, sr in current_selected):
                    continue
                valid_rs = find_valid_residues(m, current_selected)
                if valid_rs:
                    r = random.choice(valid_rs)
                    current_selected.append((m, r))
                    current_score = score(current_selected)
                    if current_score > best_score_sa:
                        best_score_sa = current_score
                        best_selected_sa = list(current_selected)
                    break

    # Use the best found
    if best_score_sa > best_score:
        best_selected = best_selected_sa
        best_score = best_score_sa

    # --- Phase 3: Final local improvement ---
    if time_left() > 5:
        improved = True
        while improved and time_left() > 3:
            improved = False
            for m in range(MOD_MIN, MOD_MAX + 1):
                if any(sm == m for sm, sr in best_selected):
                    continue
                valid_rs = find_valid_residues(m, best_selected)
                if valid_rs:
                    best_selected.append((m, random.choice(valid_rs)))
                    best_score = score(best_selected)
                    improved = True

    # Convert to required format
    result = [[m, r] for m, r in best_selected]

    # Verify validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            m1, r1 = result[i]
            m2, r2 = result[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                # Invalid - this shouldn't happen, but fallback
                return [[m, r] for m, r in best_selected_sa]

    return result
