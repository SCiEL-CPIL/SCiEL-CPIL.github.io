import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin

    def is_valid(selected, m, r):
        """Check if (m, r) is disjoint from all (m_j, r_j) in selected."""
        for m_j, r_j in selected:
            g = math.gcd(m, m_j)
            if r % g == r_j % g:
                return False
        return True

    def find_valid_residue(selected, m):
        """Find a valid residue for modulus m given selected pairs. Return None if none exists."""
        for r in range(m):
            if is_valid(selected, m, r):
                return r
        return None

    def greedy_pass(order=None, random_seed=None):
        """Generates a solution using a greedy approach with given order."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()

        if order is None:
            order = list(range(20, 201))

        selected = []
        for m in order:
            if time.time() > deadline:
                break
            # Try residues in random order for diversity
            rs = list(range(m))
            if random_seed is not None:
                rng.shuffle(rs)
            else:
                # Deterministic: try 0,1,2,... but also try random
                pass
            for r in rs:
                if is_valid(selected, m, r):
                    selected.append((m, r))
                    break
        score = sum(1.0 / m for m, r in selected)
        return selected, score

    def local_search(selected, max_time=None):
        """Improves a given solution using local search."""
        if max_time is None:
            max_time = deadline - 5
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        best_score = current_score
        best_solution = list(selected)

        improved = True
        while improved and time.time() < max_time:
            improved = False

            # Move type 1: Add a new modulus
            m_set = {m for m, r in selected}
            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                r = find_valid_residue(selected, m_add)
                if r is not None:
                    new_selected = list(selected) + [(m_add, r)]
                    new_score = current_score + 1.0 / m_add
                    if new_score > best_score + 1e-12:
                        selected = new_selected
                        current_score = new_score
                        best_score = new_score
                        improved = True

            # Move type 2: Remove one modulus and try to add multiple
            for idx in range(len(selected)):
                if time.time() > max_time:
                    break
                m_to_remove, r_to_remove = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                new_selected = list(others)
                new_score = current_score - 1.0 / m_to_remove

                # Try to add any modulus not in new_selected
                m_set_new = {m for m, r in new_selected}
                for m_add in range(20, 201):
                    if m_add in m_set_new:
                        continue
                    r = find_valid_residue(new_selected, m_add)
                    if r is not None:
                        new_selected.append((m_add, r))
                        new_score += 1.0 / m_add
                        m_set_new.add(m_add)

                if new_score > best_score + 1e-12:
                    selected = new_selected
                    current_score = new_score
                    best_score = new_score
                    improved = True
                    break

        return best_solution, best_score

    def swap_search(selected, max_time=None):
        """Try swapping residues to enable adding more moduli."""
        if max_time is None:
            max_time = deadline - 5
        selected = [tuple(x) for x in selected]
        current_score = sum(1.0 / m for m, r in selected)
        best_score = current_score
        best_solution = list(selected)

        while time.time() < max_time:
            improved = False
            m_set = {m for m, r in selected}

            # For each missing modulus, see if changing one existing residue allows it
            for m_add in range(20, 201):
                if m_add in m_set:
                    continue
                if time.time() > max_time:
                    break

                # Find which existing moduli conflict with m_add
                conflicting = []
                for idx, (m_j, r_j) in enumerate(selected):
                    g = math.gcd(m_add, m_j)
                    # Check if there exists a residue r for m_add that doesn't conflict
                    # Actually, we need to check if ALL residues of m_add conflict with r_j
                    # i.e., r_j % g covers all residues mod g... no, that's not right.
                    # (m_add, r) conflicts with (m_j, r_j) iff r % g == r_j % g
                    # So for a given m_j, the conflicting residues are those r where r % g == r_j % g
                    # There are m_add/g such residues.
                    # If m_add/g == m_add (i.e., g=1), no conflict ever.
                    # Otherwise, there's always some r that doesn't conflict.
                    # So the only issue is when multiple existing moduli together block all residues.
                    pass

                # Try changing one residue at a time
                for idx in range(len(selected)):
                    m_j, r_j = selected[idx]
                    g = math.gcd(m_add, m_j)
                    if g == 1:
                        continue  # No conflict possible
                    # Try all new residues for m_j
                    for new_r_j in range(m_j):
                        if new_r_j == r_j:
                            continue
                        # Check if new_r_j is valid with all other selected
                        others = selected[:idx] + selected[idx+1:]
                        valid = True
                        for m_k, r_k in others:
                            g2 = math.gcd(m_j, m_k)
                            if new_r_j % g2 == r_k % g2:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Now check if m_add can be added with some residue
                        test_selected = others + [(m_j, new_r_j)]
                        r_add = find_valid_residue(test_selected, m_add)
                        if r_add is not None:
                            new_selected = test_selected + [(m_add, r_add)]
                            new_score = current_score + 1.0 / m_add
                            if new_score > best_score + 1e-12:
                                selected = new_selected
                                current_score = new_score
                                best_score = new_score
                                improved = True
                                m_set.add(m_add)
                                break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return best_solution, best_score

    best_score = 0.0
    best_solution = []

    # 1. Run deterministic greedy (smallest moduli first)
    sol, score = greedy_pass(order=list(range(20, 201)), random_seed=None)
    if score > best_score:
        best_score = score
        best_solution = sol

    # 2. Run multiple randomized greedy passes with different orderings
    num_passes = 300
    for i in range(num_passes):
        if time.time() > deadline - 40:  # Reserve 40s for local search
            break
        seed = i
        # Different orderings: random shuffle, reverse, etc.
        if i % 3 == 0:
            order = list(range(20, 201))
            random.Random(seed).shuffle(order)
        elif i % 3 == 1:
            order = list(range(200, 19, -1))
            random.Random(seed).shuffle(order)
        else:
            order = list(range(20, 201))
            # Weight toward smaller moduli
            random.Random(seed).shuffle(order)

        sol, score = greedy_pass(order=order, random_seed=seed)
        if score > best_score:
            best_score = score
            best_solution = sol

    # 3. Apply local search to the best solution found
    if time.time() < deadline - 10:
        improved_sol, improved_score = local_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    # 4. Apply swap search if time permits
    if time.time() < deadline - 5:
        improved_sol, improved_score = swap_search(best_solution)
        if improved_score > best_score:
            best_score = improved_score
            best_solution = improved_sol

    # 5. Final polish: try to add any missing small moduli
    if time.time() < deadline - 2:
        m_set = {m for m, r in best_solution}
        for m_add in range(20, 201):
            if m_add in m_set:
                continue
            r = find_valid_residue(best_solution, m_add)
            if r is not None:
                best_solution.append((m_add, r))
                best_score += 1.0 / m_add
                m_set.add(m_add)

    return [[m, r] for m, r in best_solution]
