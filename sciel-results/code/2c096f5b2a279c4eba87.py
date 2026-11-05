import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def is_compatible(m, r, selected):
        for m_old, r_old in selected:
            g = math.gcd(m, m_old)
            if r % g == r_old % g:
                return False
        return True

    def get_score(selected):
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return float(total)

    def find_valid_residue(m, selected):
        # Try residues in order of smallest value first for dense packing
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None

    # Phase 1: Dense greedy construction
    # Process moduli in increasing order, but pick the smallest valid residue
    selected = []
    for m in range(20, 201):
        if time.time() > deadline - 60:  # Leave time for local search
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))

    best_score = get_score(selected)
    best_selected = list(selected)

    # Phase 2: Local search with hill climbing
    # Try adding moduli not in selected, removing and re-adding, swapping residues
    all_moduli = list(range(20, 201))
    selected_set = set(m for m, r in selected)

    # Precompute gcd table for speed
    gcd_table = {}
    for i in range(20, 201):
        for j in range(20, 201):
            if i != j:
                gcd_table[(i, j)] = math.gcd(i, j)

    def quick_score_diff(selected, m_add, r_add, m_remove=None, r_remove=None):
        """Estimate score change without full recomputation."""
        delta = Fraction(1, m_add)
        if m_remove is not None:
            delta -= Fraction(1, m_remove)
        return float(delta)

    def is_valid_assignment(m, r, selected, exclude_m=None):
        for m_old, r_old in selected:
            if m_old == exclude_m:
                continue
            g = math.gcd(m, m_old)
            if r % g == r_old % g:
                return False
        return True

    # Local search: try to add unselected moduli
    unselected = [m for m in all_moduli if m not in selected_set]
    random.shuffle(unselected)

    local_search_time_limit = deadline - 5
    step = 0
    while time.time() < local_search_time_limit:
        step += 1
        improved = False

        # Try adding an unselected modulus
        if unselected:
            # Try a few random unselected moduli
            candidates = random.sample(unselected, min(5, len(unselected)))
            for m_new in candidates:
                if m_new in selected_set:
                    continue
                r_new = find_valid_residue(m_new, selected)
                if r_new is not None:
                    new_selected = selected + [(m_new, r_new)]
                    new_score = get_score(new_selected)
                    if new_score > best_score:
                        selected = new_selected
                        best_score = new_score
                        best_selected = list(selected)
                        selected_set.add(m_new)
                        unselected.remove(m_new)
                        improved = True
                        break

        # Try swapping a low-value modulus for a higher-value one
        if not improved and time.time() < local_search_time_limit:
            # Sort selected by 1/m descending (we want to keep high 1/m)
            # Try removing a low-priority modulus and adding a high-priority one
            selected_sorted = sorted(selected, key=lambda x: 1.0/x[0], reverse=True)
            # Consider removing one of the last few (lowest 1/m)
            for idx in range(len(selected_sorted) - 1, max(len(selected_sorted) - 4, -1), -1):
                m_remove, r_remove = selected_sorted[idx]
                temp_selected = [x for x in selected if x[0] != m_remove]
                selected_set_temp = set(m for m, r in temp_selected)
                # Try to add a modulus not in temp_selected
                candidates_add = [m for m in all_moduli if m not in selected_set_temp and m != m_remove]
                candidates_add.sort(key=lambda m: 1.0/m, reverse=True)
                for m_add in candidates_add[:10]:
                    r_add = find_valid_residue(m_add, temp_selected)
                    if r_add is not None:
                        new_selected = temp_selected + [(m_add, r_add)]
                        new_score = get_score(new_selected)
                        if new_score > best_score:
                            selected = new_selected
                            best_score = new_score
                            best_selected = list(selected)
                            selected_set = set(m for m, r in selected)
                            unselected = [m for m in all_moduli if m not in selected_set]
                            improved = True
                            break
                if improved:
                    break

        # Try re-choosing residues for existing moduli to free up space
        if not improved and time.time() < local_search_time_limit:
            # Pick a random selected modulus and try a different residue
            if selected:
                idx = random.randint(0, len(selected) - 1)
                m_cur, r_cur = selected[idx]
                # Try other residues for this modulus
                other_residues = [r for r in range(m_cur) if r != r_cur]
                random.shuffle(other_residues)
                for r_new in other_residues[:5]:
                    temp_selected = [(m, r) for (m, r) in selected if m != m_cur]
                    if is_valid_assignment(m_cur, r_new, temp_selected):
                        new_selected = temp_selected + [(m_cur, r_new)]
                        # Check if this allows adding a new modulus
                        selected_set_temp = set(m for m, r in new_selected)
                        candidates_add = [m for m in all_moduli if m not in selected_set_temp]
                        candidates_add.sort(key=lambda m: 1.0/m, reverse=True)
                        for m_add in candidates_add[:5]:
                            r_add = find_valid_residue(m_add, new_selected)
                            if r_add is not None:
                                final_selected = new_selected + [(m_add, r_add)]
                                new_score = get_score(final_selected)
                                if new_score > best_score:
                                    selected = final_selected
                                    best_score = new_score
                                    best_selected = list(selected)
                                    selected_set = set(m for m, r in selected)
                                    unselected = [m for m in all_moduli if m not in selected_set]
                                    improved = True
                                    break
                        if improved:
                            break

        # If no improvement after many steps, try a random restart of local search
        if not improved and step % 100 == 0:
            # Perturb: randomly change a few residues
            if selected:
                for _ in range(min(3, len(selected))):
                    idx = random.randint(0, len(selected) - 1)
                    m_cur, r_cur = selected[idx]
                    temp_selected = [(m, r) for (m, r) in selected if m != m_cur]
                    r_new = find_valid_residue(m_cur, temp_selected)
                    if r_new is not None:
                        selected[idx] = (m_cur, r_new)

    # Final check
    if get_score(best_selected) >= best_score:
        return [[m, r] for m, r in best_selected]
    else:
        return [[m, r] for m, r in selected]
