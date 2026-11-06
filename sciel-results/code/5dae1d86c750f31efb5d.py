"""Combined program: prime-partition structural decomposition + bitmask exact solver + greedy/local search."""
# EVOLVE-BLOCK-START
import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    moduli = list(range(2, 129))
    n_moduli = len(moduli)

    # Precompute gcd table
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factorization for each modulus
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    prime_sets = [prime_factors(m) for m in moduli]

    # Partition moduli into connected components based on shared prime factors
    # Two moduli are in the same component if they share at least one prime factor
    # (transitively). Moduli in different components are always disjoint regardless of residues.
    parent = list(range(n_moduli))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            if prime_sets[i] & prime_sets[j]:
                union(i, j)

    components = defaultdict(list)
    for i in range(n_moduli):
        components[find(i)].append(i)

    # For each component, we need to choose a subset of moduli and residues
    # such that all chosen pairs are pairwise disjoint.
    # Moduli in different components are automatically disjoint.

    # For small components, we can try exact search.
    # For large components, we use greedy + local search.

    def can_add(selected, r_new, m_idx_new, comp_indices):
        """Check if (r_new, moduli[m_idx_new]) is disjoint from all selected in the same component."""
        for (r_sel, m_idx_sel) in selected:
            g = gcd_table[m_idx_sel][m_idx_new]
            if g > 1 and (r_new - r_sel) % g == 0:
                return False
        return True

    def greedy_for_component(comp_indices, order):
        """Greedy: process moduli in given order, pick first valid residue for each."""
        selected = []
        for m_idx in order:
            m = moduli[m_idx]
            for r in range(m):
                if can_add(selected, r, m_idx, comp_indices):
                    selected.append((r, m_idx))
                    break
        return selected

    def try_add_to_component(selected, comp_indices):
        """Try to add any missing modulus in this component with a valid residue."""
        selected_set = set(si[1] for si in selected)
        for m_idx in comp_indices:
            if m_idx in selected_set:
                continue
            m = moduli[m_idx]
            for r in range(m):
                if can_add(selected, r, m_idx, comp_indices):
                    return selected + [(r, m_idx)]
        return None

    def local_search_component(selected, comp_indices, max_iter=2000):
        """Local search within a component to improve the solution."""
        best = selected[:]
        best_len = len(best)

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            improved = False

            # Try to add a new modulus
            new_sol = try_add_to_component(best, comp_indices)
            if new_sol is not None:
                best = new_sol
                best_len += 1
                improved = True
                continue

            # Try swapping a residue to enable future additions
            for i, (r_old, m_idx) in enumerate(best):
                m = moduli[m_idx]
                # Try a subset of residues to save time
                step = max(1, m // 15)
                for r_new in range(0, m, step):
                    if r_new == r_old:
                        continue
                    ok = True
                    for j, (r_j, m_idx_j) in enumerate(best):
                        if i == j:
                            continue
                        g = gcd_table[m_idx][m_idx_j]
                        if g > 1 and (r_new - r_j) % g == 0:
                            ok = False
                            break
                    if ok:
                        test_best = best[:]
                        test_best[i] = (r_new, m_idx)
                        new_sol = try_add_to_component(test_best, comp_indices)
                        if new_sol is not None:
                            best = new_sol
                            best_len += 1
                            improved = True
                            break
                if improved:
                    break

            if not improved:
                break

        return best

    # Solve each component independently
    all_selected = []
    for root, comp_indices in components.items():
        if len(comp_indices) == 1:
            # Single modulus: always include it
            m_idx = comp_indices[0]
            all_selected.append((0, m_idx))
            continue

        # For this component, run multiple strategies
        best_comp = []
        best_len = 0

        # Strategy 1: Greedy with ascending modulus order
        order_asc = sorted(comp_indices, key=lambda i: moduli[i])
        sol = greedy_for_component(comp_indices, order_asc)
        sol = local_search_component(sol, comp_indices, max_iter=1000)
        if len(sol) > best_len:
            best_len = len(sol)
            best_comp = sol

        if time.time() > deadline:
            all_selected.extend(best_comp)
            continue

        # Strategy 2: Greedy with descending modulus order
        order_desc = sorted(comp_indices, key=lambda i: -moduli[i])
        sol = greedy_for_component(comp_indices, order_desc)
        sol = local_search_component(sol, comp_indices, max_iter=1000)
        if len(sol) > best_len:
            best_len = len(sol)
            best_comp = sol

        if time.time() > deadline:
            all_selected.extend(best_comp)
            continue

        # Strategy 3: Random restarts
        random.seed(42 + root)
        for attempt in range(30):
            if time.time() > deadline:
                break
            random_order = comp_indices[:]
            random.shuffle(random_order)
            sol = greedy_for_component(comp_indices, random_order)
            sol = local_search_component(sol, comp_indices, max_iter=500)
            if len(sol) > best_len:
                best_len = len(sol)
                best_comp = sol

        # Final local search on best
        best_comp = local_search_component(best_comp, comp_indices, max_iter=2000)
        all_selected.extend(best_comp)

    # Global local search: try to add any modulus not yet selected
    selected_set = set(si[1] for si in all_selected)
    for m_idx in range(n_moduli):
        if m_idx in selected_set:
            continue
        m = moduli[m_idx]
        for r in range(m):
            ok = True
            for (r_sel, m_idx_sel) in all_selected:
                g = gcd_table[m_idx_sel][m_idx]
                if g > 1 and (r - r_sel) % g == 0:
                    ok = False
                    break
            if ok:
                all_selected.append((r, m_idx))
                break

    # Convert to required format
    result = [[r, moduli[m_idx]] for r, m_idx in all_selected]

    if not result:
        result = [[1, 2]]

    return result
# EVOLVE-BLOCK-END
