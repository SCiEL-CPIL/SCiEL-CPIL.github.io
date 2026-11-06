import time
import math
from math import gcd
from itertools import product

MAX_MOD = 128

def _factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s limit

    # Moduli we can choose from: 2..128
    all_moduli = list(range(2, MAX_MOD + 1))

    # Precompute gcd for each pair of moduli
    n = len(all_moduli)
    idx_to_mod = {i: all_moduli[i] for i in range(n)}
    mod_to_idx = {m: i for i, m in enumerate(all_moduli)}

    # gcd[i][j] = gcd(all_moduli[i], all_moduli[j])
    gcd_matrix = [[1] * n for _ in range(n)]
    for i in range(n):
        gcd_matrix[i][i] = all_moduli[i]
        for j in range(i + 1, n):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # For a set of chosen (residue, modulus) pairs, the constraint on a new modulus m
    # with residue r is: for each chosen (r_i, m_i), we need r != r_i mod gcd(m, m_i).
    # The set of forbidden residues mod g = gcd(m, m_i) is {r_i mod g}.
    # So for modulus m, the allowed residues are those r in [0, m) such that for all
    # chosen i, r mod gcd(m, m_i) != r_i mod gcd(m, m_i).

    # We do a backtracking search. Order: try to include moduli in some order.
    # Strategy: sort moduli by "constraint tightness" - moduli with many small gcds
    # with others are more constrained. Let's order by descending number of divisors
    # or by descending modulus (larger moduli have more residues to choose from, so
    # they're more flexible; smaller moduli are more constrained).
    # Actually, let's order by ascending modulus: smaller moduli have fewer residues,
    # so they're more constrained and should be placed first.

    # But wait - we want to maximize the count. Let's try a different approach:
    # Use a greedy + local search, or a proper branch and bound.

    # Let me implement a backtracking search with good pruning:
    # - Order moduli by ascending value (more constrained first)
    # - For each modulus, determine allowed residues given current assignment
    # - Use upper bound: current_count + remaining_moduli

    # Sort moduli in ascending order
    sorted_moduli = sorted(all_moduli)  # 2, 3, 4, ..., 128
    sorted_indices = [mod_to_idx[m] for m in sorted_moduli]

    # We'll work with indices into all_moduli
    # State: assignment[mod_idx] = residue or None
    assignment = [None] * n
    chosen = []  # list of (residue, mod_idx) for chosen moduli

    best_count = 0
    best_assignment = None

    def get_allowed_residues(mod_idx, current_chosen):
        m = all_moduli[mod_idx]
        # Start with all residues 0..m-1
        allowed = set(range(m))
        for (r_i, m_i_idx) in current_chosen:
            g = gcd_matrix[mod_idx][m_i_idx]
            if g == 1:
                continue  # No constraint
            # Forbidden: r_i mod g
            forbidden_val = r_i % g
            # Remove all residues r where r % g == forbidden_val
            # This is: r = forbidden_val + k*g for k = 0, ..., (m-1-forbidden_val)//g
            for r in range(forbidden_val, m, g):
                if r in allowed:
                    allowed.discard(r)
        return allowed

    def backtrack(pos, current_chosen, current_count):
        nonlocal best_count, best_assignment

        if time.time() > deadline:
            return

        # Upper bound: current_count + (number of moduli from pos to end)
        remaining = n - pos
        if current_count + remaining <= best_count:
            return

        if pos == n:
            if current_count > best_count:
                best_count = current_count
                best_assignment = list(assignment)
            return

        mod_idx = sorted_indices[pos]
        m = all_moduli[mod_idx]

        # Option 1: Skip this modulus
        backtrack(pos + 1, current_chosen, current_count)

        # Option 2: Include this modulus with some residue
        allowed = get_allowed_residues(mod_idx, current_chosen)
        if allowed:
            # Try each allowed residue
            for r in sorted(allowed):
                assignment[mod_idx] = r
                current_chosen.append((r, mod_idx))
                backtrack(pos + 1, current_chosen, current_count + 1)
                current_chosen.pop()
                assignment[mod_idx] = None

    backtrack(0, [], 0)

    if best_assignment is None:
        # Fallback to the original solution
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    # Extract result
    result = []
    for mod_idx in range(n):
        if best_assignment[mod_idx] is not None:
            r = best_assignment[mod_idx]
            m = all_moduli[mod_idx]
            result.append([r, m])

    if not result:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return result
