# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    all_moduli = list(range(20, 201))

    # Precompute gcd table
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for m2 in all_moduli[i:]:
            g = math.gcd(m1, m2)
            gcd_table[(m1, m2)] = g
            if m1 != m2:
                gcd_table[(m2, m1)] = g

    def get_gcd(m1, m2):
        if m1 == m2:
            return m1
        key = (m1, m2) if m1 < m2 else (m2, m1)
        return gcd_table.get(key, math.gcd(m1, m2))

    def is_compatible(m_new, r_new, selected):
        for m_old, r_old in selected:
            g = get_gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def get_valid_residues(m, selected):
        valid = []
        for r in range(m):
            if is_compatible(m, r, selected):
                valid.append(r)
        return valid

    def score_float(classes):
        return sum(1.0 / m for m, r in classes)

    def score_frac(classes):
        total = Fraction(0, 1)
        for m, r in classes:
            total += Fraction(1, m)
        return total

    # ---------- Conflict-aware greedy construction ----------
    # For a candidate modulus m, compute how many residues it would block
    # across the current selection. Prefer moduli with minimal "conflict cost".
    def conflict_cost(m, selected):
        """Estimate how restrictive m is: for each selected class, the gcd
        determines how many residue classes of m are blocked."""
        cost = 0
        for m_old, r_old in selected:
            g = get_gcd(m, m_old)
            # m residues, gcd g -> exactly g residue classes of m are blocked
            # (those r where r ≡ r_old mod g). Fraction blocked = g/m.
            cost += g / m
        return cost

    def greedy_conflict_aware(order_seed=None):
        """Build solution: repeatedly pick the modulus with smallest conflict cost
        among remaining, choosing a residue that minimizes future blocking."""
        remaining = all_moduli[:]
        if order_seed is not None:
            remaining = list(order_seed)
        selected = []
        while remaining:
            if time.time() > deadline - 5:
                break
            best_m = None
            best_cost = float('inf')
            best_r = None
            for m in remaining:
                # Quick feasibility check
                # Count valid residues
                n_valid = 0
                for r in range(m):
                    if is_compatible(m, r, selected):
                        n_valid += 1
                        if n_valid > 2:  # early exit, just need feasibility
                            break
                if n_valid == 0:
                    continue
                c = conflict_cost(m, selected)
                # We want small m (high 1/m) and small conflict cost.
                # Combined objective: maximize 1/m - lambda * cost
                obj = 1.0 / m - 0.5 * c
                if obj > best_cost or (best_m is None):
                    # track by obj; use best_cost as placeholder
                    if best_m is None or obj > best_cost:
                        best_cost = obj
                        best_m = m
            if best_m is None:
                break
            # Pick residue: choose one that minimizes conflict cost for future
            valid = get_valid_residues(best_m, selected)
            if not valid:
                remaining.remove(best_m)
                continue
            # For each candidate residue, estimate how much it blocks future moduli
            best_r = valid[0]
            best_r_score = float('inf')
            for r in valid[:min(len(valid), 8)]:  # sample
                # Estimate: sum over remaining moduli of gcd(m, r) penalty
                # cheaper proxy: prefer r that is coprime-ish to many
                r_score = 0.0
                for m2 in remaining:
                    if m2 == best_m:
                        continue
                    g = get_gcd(best_m, m2)
                    # residue r blocks class r mod g; penalty ~ 1 if r mod g == 0
                    # rough: count how many m2 have (r % g == 0)
                    if g > 1 and r % g == 0:
                        r_score += 1.0 / m2
                if r_score < best_r_score:
                    best_r_score = r_score
                    best_r = r
            selected.append([best_m, best_r])
            remaining.remove(best_m)
        return selected

    # ---------- Residue-assignment SA on a fixed modulus set ----------
    def sa_residues(modulus_set, max_iter=20000):
        """SA that fixes the set of moduli and only changes residues.
        Returns best valid assignment or None if infeasible."""
        # Initialize: greedy assign residues
        selected = []
        for m in modulus_set:
            valid = get_valid_residues(m, selected)
            if not valid:
                return None
            r = random.choice(valid)
            selected.append([m, r])
        cur_score = score_frac(selected)
        best = [c[:] for c in selected]
        best_score = cur_score

        T = 0.02
        cooling = 0.9995
        it = 0
        while it < max_iter:
            if time.time() > deadline - 3:
                break
            it += 1
            T *= cooling
            # Pick a random class, change its residue
            idx = random.randint(0, len(selected) - 1)
            m = selected[idx][0]
            others = [selected[j] for j in range(len(selected)) if j != idx]
            valid = get_valid_residues(m, others)
            if len(valid) <= 1:
                continue
            new_r = random.choice(valid)
            if new_r == selected[idx][1]:
                continue
            old_r = selected[idx][1]
            selected[idx][1] = new_r
            # score unchanged (same modulus)
            if random.random() < 0.9:
                # accept most residue changes (they don't hurt score, may help future)
                pass
            else:
                selected[idx][1] = old_r
            # Track best (score same, but keep for final)
            if selected == best or True:
                pass
        # Final: verify and return
        # Re-verify validity
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m1, r1 = selected[i]
                m2, r2 = selected[j]
                g = get_gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return None
        return selected

    # ---------- Full SA over set membership (from inspiration 1) ----------
    def sa_set(initial, max_iter=80000):
        current = [c[:] for c in initial]
        current_score = score_frac(current)
        best = [c[:] for c in current]
        best_score = current_score
        selected_set = set(m for m, r in current)

        T = 0.03
        cooling = 0.9998
        it = 0
        while it < max_iter:
            if time.time() > deadline - 3:
                break
            it += 1
            T *= cooling
            move = random.random()
            new_state = None
            new_score = None

            if move < 0.4 and current:
                # Change residue
                idx = random.randint(0, len(current) - 1)
                m = current[idx][0]
                others = [current[j] for j in range(len(current)) if j != idx]
                valid = get_valid_residues(m, others)
                if len(valid) > 1:
                    new_r = random.choice(valid)
                    new_state = [c[:] for c in current]
                    new_state[idx][1] = new_r
                    new_score = current_score  # same

            elif move < 0.6 and current:
                # Remove
                idx = random.randint(0, len(current) - 1)
                m_old = current[idx][0]
                new_state = [c[:] for c in current]
                del new_state[idx]
                new_score = current_score - Fraction(1, m_old)

            elif move < 0.85:
                # Add
                unselected = [m for m in all_moduli if m not in selected_set]
                if unselected:
                    m_new = random.choice(unselected)
                    valid = get_valid_residues(m_new, current)
                    if valid:
                        r_new = random.choice(valid)
                        new_state = [c[:] for c in current]
                        new_state.append([m_new, r_new])
                        new_score = current_score + Fraction(1, m_new)

            else:
                # Swap
                if current:
                    unselected = [m for m in all_moduli if m not in selected_set]
                    if unselected:
                        idx = random.randint(0, len(current) - 1)
                        m_old = current[idx][0]
                        m_new = random.choice(unselected)
                        others = [current[j] for j in range(len(current)) if j != idx]
                        valid = get_valid_residues(m_new, others)
                        if valid:
                            r_new = random.choice(valid)
                            new_state = [c[:] for c in current]
                            new_state[idx] = [m_new, r_new]
                            new_score = current_score - Fraction(1, m_old) + Fraction(1, m_new)

            if new_state is not None and new_score is not None:
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current = new_state
                    current_score = new_score
                    selected_set = set(m for m, r in current)
                    if current_score > best_score:
                        best_score = current_score
                        best = [c[:] for c in current]
        return best, float(best_score)

    # ---------- Main search ----------
    best_classes = None
    best_score = Fraction(0, 1)

    def consider(classes):
        nonlocal best_classes, best_score
        if not classes:
            return
        # validate
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = get_gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return
        s = score_frac(classes)
        if s > best_score:
            best_score = s
            best_classes = [c[:] for c in classes]

    random.seed(12345)

    # Phase 1: conflict-aware greedy (deterministic + randomized)
    consider(greedy_conflict_aware())
    for _ in range(15):
        if time.time() > deadline - 40:
            break
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        consider(greedy_conflict_aware(shuffled))

    # Phase 2: SA over set membership from the best
    if best_classes and time.time() < deadline - 10:
        sa_res, sa_sc = sa_set(best_classes, max_iter=120000)
        consider(sa_res)

    # Phase 3: more SA restarts
    while time.time() < deadline - 5:
        if best_classes:
            sa_res, sa_sc = sa_set(best_classes, max_iter=40000)
            consider(sa_res)
        else:
            break

    # Phase 4: local improvement - try adding missing moduli
    if best_classes and time.time() < deadline - 2:
        used = set(m for m, r in best_classes)
        cur = [c[:] for c in best_classes]
        for m_new in all_moduli:
            if m_new in used:
                continue
            valid = get_valid_residues(m_new, cur)
            if valid:
                cur.append([m_new, valid[0]])
                used.add(m_new)
        consider(cur)

    if not best_classes:
        return [[20, 0]]
    return best_classes
# EVOLVE-BLOCK-END
