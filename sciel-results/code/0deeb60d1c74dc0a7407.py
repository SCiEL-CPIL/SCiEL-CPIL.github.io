import time
import math
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 80.0
    MAX_M = 128

    # Precompute gcd table
    moduli = list(range(2, MAX_M + 1))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factors for each modulus
    def prime_factors(m):
        facs = set()
        d = 2
        mm = m
        while d * d <= mm:
            if mm % d == 0:
                facs.add(d)
                while mm % d == 0:
                    mm //= d
            d += 1
        if mm > 1:
            facs.add(mm)
        return facs

    # For each prime p, find all moduli divisible by p
    prime_to_mods = defaultdict(list)
    for m in moduli:
        for p in prime_factors(m):
            prime_to_mods[p].append(m)

    # For a prime p, the moduli divisible by p: we need distinct residues mod p
    # This is a bipartite matching: moduli on left, residues mod p on right
    # Edge exists if residue r < modulus m (always true since r < p <= m for m divisible by p)
    # Actually r must be in [0, m-1] and r mod p = r (since r < p)
    # So we need to assign distinct residues in [0, p-1] to moduli, where residue r is valid if r < m

    def max_matching_prime_family(p, mods_in_family):
        """Find max number of moduli from family that can have distinct residues mod p."""
        # mods_in_family: list of moduli all divisible by p
        # We need to assign distinct residues r in [0, p-1] such that r < m for each assigned modulus
        # This is a bipartite matching: left = moduli, right = residues 0..p-1
        # Edge: modulus m connected to residue r iff r < m

        # Use DFS-based bipartite matching (Hopcroft-Karp would be faster but this is small)
        # Sort moduli by size (smaller first - more constrained)
        mods_sorted = sorted(mods_in_family, key=lambda m: m)

        match_r = [-1] * p  # match_r[r] = index of modulus matched to residue r
        match_m = [-1] * len(mods_sorted)  # match_m[i] = residue matched to modulus i

        def dfs(i, visited):
            m = mods_sorted[i]
            for r in range(min(p, m)):  # r must be < m and < p
                if visited[r]:
                    continue
                visited[r] = True
                if match_r[r] == -1 or dfs(match_r[r], visited):
                    match_r[r] = i
                    match_m[i] = r
                    return True
            return False

        count = 0
        for i in range(len(mods_sorted)):
            visited = [False] * p
            if dfs(i, visited):
                count += 1

        # Extract the assignment
        assignment = []
        for i in range(len(mods_sorted)):
            if match_m[i] != -1:
                assignment.append((mods_sorted[i], match_m[i]))
        return assignment

    # Strategy: For each prime p, compute the max matching in its family
    # Then try to combine families, checking cross-family compatibility

    # First, get the best assignment for each prime family
    family_assignments = {}
    for p, mods in prime_to_mods.items():
        if len(mods) < 2:
            continue
        assignment = max_matching_prime_family(p, mods)
        if len(assignment) > 0:
            family_assignments[p] = assignment

    # Sort families by size (largest first)
    families_sorted = sorted(family_assignments.items(), key=lambda x: -len(x[1]))

    # Greedy: pick families, check cross-compatibility
    best = []
    best_count = 0

    # Try different orderings of families
    family_orderings = [
        families_sorted,
        sorted(families_sorted, key=lambda x: x[0]),  # by prime
        sorted(families_sorted, key=lambda x: x[0], reverse=True),
    ]

    for ordering in family_orderings:
        if time.time() > deadline:
            break
        selected = []
        selected_mods = set()
        for p, assignment in ordering:
            if time.time() > deadline:
                break
            # Try to add as many as possible from this family
            # Sort by residue (or by modulus)
            for m, r in sorted(assignment, key=lambda x: x[1]):
                if m in selected_mods:
                    continue
                # Check compatibility with all selected
                ok = True
                for (m2, r2) in selected:
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    selected_mods.add(m)

        if len(selected) > best_count:
            best_count = len(selected)
            best = selected[:]

    # Local search: try to add more moduli by trying different residues
    def try_add_modulus(selected, m):
        """Try to find a residue for modulus m that's compatible with all in selected."""
        for r in range(m):
            ok = True
            for (r2, m2) in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                return r
        return None

    # Polish: try to add moduli not yet selected
    selected_mods = set(m for _, m in best)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for m in moduli:
            if m in selected_mods:
                continue
            r = try_add_modulus(best, m)
            if r is not None:
                best.append((r, m))
                selected_mods.add(m)
                improved = True

    # Try swap-based local search
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        improved = False
        for idx in range(len(best)):
            r, m = best[idx]
            # Try different residues for this modulus
            for r_new in range(m):
                if r_new == r:
                    continue
                # Check if r_new is valid against all others
                valid = True
                for j in range(len(best)):
                    if j == idx:
                        continue
                    r2, m2 = best[j]
                    g = math.gcd(m, m2)
                    if r_new % g == r2 % g:
                        valid = False
                        break
                if not valid:
                    continue
                # Temporarily swap
                best[idx] = (r_new, m)
                # Try to add a missing modulus
                selected_mods_now = set(m2 for _, m2 in best)
                added = False
                for m2 in moduli:
                    if m2 in selected_mods_now:
                        continue
                    r2 = try_add_modulus(best, m2)
                    if r2 is not None:
                        best.append((r2, m2))
                        added = True
                        break
                if not added:
                    # Revert
                    best[idx] = (r, m)
                else:
                    improved = True
                    break
            if improved:
                break

    if not best:
        best = [(1, 2)]

    return [[r, m] for r, m in best]
