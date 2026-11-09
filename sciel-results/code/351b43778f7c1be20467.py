"""Planted 3-SAT solver: spectral init + multi-restart WalkSAT with clause-weighted scoring."""
import time
import math
import random

import numpy as np


def _spectral_init(n, clauses, rng):
    """
    Build a clause-variable incidence structure and use SVD to get a sign
    pattern that may correlate with the planted solution.
    Returns a list of booleans of length n.
    """
    m = len(clauses)
    if m == 0:
        return [rng.random() < 0.5 for _ in range(n)]

    # Build incidence: for each clause, record signed literals
    # We create a matrix A of shape (m, n) where A[i, j] = sign of literal j in clause i
    # (0 if variable j not in clause i)
    A = np.zeros((m, n), dtype=np.float64)
    for i, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            A[i, var] = 1.0 if lit > 0 else -1.0

    # SVD: A = U S V^T
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return [rng.random() < 0.5 for _ in range(n)]

    # The planted solution should align with the top singular vectors.
    # Use the first few right singular vectors (rows of Vt) to form a score per variable.
    # Take the sum of top k singular vectors weighted by singular values.
    k = min(5, Vt.shape[0])
    # Vt is (k, n) for top k
    top_vt = Vt[:k, :]
    top_s = s[:k]
    # Weighted sum
    scores = top_vt.T @ top_s  # shape (n,)
    # Sign of score gives the assignment
    assignment = [scores[i] > 0 for i in range(n)]
    return assignment


def _walksat(n, clauses, assignment, rng, max_steps=50000, alpha=0.55):
    """
    WalkSAT with clause-weighted variable selection.
    Returns (assignment, is_satisfied).
    """
    m = len(clauses)
    # Precompute clause literal indices and signs for speed
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vs = [abs(l) - 1 for l in clause]
        ss = [1 if l > 0 else -1 for l in clause]
        clause_vars.append(vs)
        clause_signs.append(ss)

    # Current unsat count per clause (0 or 1)
    unsat_count = np.zeros(m, dtype=np.int32)
    # For each variable, track how many clauses it appears in and with what sign
    # var_clause_list[var] = list of (clause_idx, sign)
    var_clause_list = [[] for _ in range(n)]
    for ci, (vs, ss) in enumerate(zip(clause_vars, clause_signs)):
        for v, s in zip(vs, ss):
            var_clause_list[v].append((ci, s))

    def clause_sat(ci):
        vs = clause_vars[ci]
        ss = clause_signs[ci]
        for v, s in zip(vs, ss):
            if (assignment[v] and s == 1) or (not assignment[v] and s == -1):
                return True
        return False

    # Initialize unsat counts
    for ci in range(m):
        if not clause_sat(ci):
            unsat_count[ci] = 1

    total_unsat = int(unsat_count.sum())
    if total_unsat == 0:
        return assignment, True

    # Precompute for each clause the list of (var, sign)
    # WalkSAT loop
    for step in range(max_steps):
        if total_unsat == 0:
            return assignment, True

        # Pick a random unsatisfied clause
        ci = rng.randrange(m)
        while unsat_count[ci] == 0:
            ci = rng.randrange(m)

        vs = clause_vars[ci]
        ss = clause_signs[ci]

        # Compute for each literal in this clause: how many other unsat clauses it would satisfy
        # and how many it would break
        best_var = -1
        best_score = -1e18
        best_is_flip = False

        for v, s in zip(vs, ss):
            # If we flip var v:
            # Currently in clause ci (unsat), flipping v will satisfy ci if s matches new value
            # new value = not assignment[v]
            # literal is satisfied if (assignment[v] and s==1) or (not assignment[v] and s==-1)
            # After flip: (not assignment[v] and s==1) or (assignment[v] and s==-1)
            # Since ci is unsat, currently the literal is NOT satisfied.
            # After flip, it WILL be satisfied (since we flip the variable that makes it true)
            # So flipping v satisfies clause ci: +1 sat
            # Now count effect on other clauses where v appears:
            gain = 0  # clauses that become satisfied
            loss = 0  # clauses that become unsatisfied
            for (oci, osign) in var_clause_list[v]:
                if oci == ci:
                    continue
                # Current state of literal (v, osign) in clause oci:
                cur_sat = (assignment[v] and osign == 1) or (not assignment[v] and osign == -1)
                # After flip:
                new_sat = (not assignment[v] and osign == 1) or (assignment[v] and osign == -1)
                if cur_sat and not new_sat:
                    loss += 1
                elif not cur_sat and new_sat:
                    gain += 1
            score = gain - loss
            if score > best_score:
                best_score = score
                best_var = v
                best_is_flip = True

        if best_var < 0:
            # Shouldn't happen
            break

        # With probability alpha, pick the best variable; else random from clause
        if rng.random() < alpha:
            flip_var = best_var
        else:
            flip_var = vs[rng.randrange(3)]

        # Flip and update
        old_val = assignment[flip_var]
        assignment[flip_var] = not old_val
        for (oci, osign) in var_clause_list[flip_var]:
            cur_sat_before = (old_val and osign == 1) or (not old_val and osign == -1)
            new_sat_after = ((not old_val) and osign == 1) or (old_val and osign == -1)
            if cur_sat_before and not new_sat_after:
                unsat_count[oci] += 1
                total_unsat += 1
            elif not cur_sat_before and new_sat_after:
                unsat_count[oci] -= 1
                total_unsat -= 1

    return assignment, total_unsat == 0


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0  # safety margin before 300s limit

    # Use multiple seeds for robustness
    best_assignment = None
    best_unsat = float('inf')

    # First try spectral initialization
    rng = random.Random(42)
    spec_assign = _spectral_init(n, clauses, rng)

    # Try WalkSAT from spectral init
    assign, sat = _walksat(n, clauses, spec_assign, rng, max_steps=30000, alpha=0.6)
    if sat:
        return assign
    # Count unsat
    m = len(clauses)
    unsat_c = sum(1 for c in clauses if not any(
        assign[abs(l)-1] == (l > 0) for l in c
    ))
    best_assignment = list(assign)
    best_unsat = unsat_c

    # Multi-restart with different random inits
    restart = 0
    while time.time() < deadline:
        restart += 1
        seed = 1000 + restart * 7
        rng_r = random.Random(seed)
        init = [rng_r.random() < 0.5 for _ in range(n)]
        assign, sat = _walksat(n, clauses, init, rng_r, max_steps=20000, alpha=0.55)
        if sat:
            return assign
        unsat_c = sum(1 for c in clauses if not any(
            assign[abs(l)-1] == (l > 0) for l in c
        ))
        if unsat_c < best_unsat:
            best_unsat = unsat_c
            best_assignment = list(assign)

        # If we're getting close to deadline, do a final polish
        if time.time() > deadline - 30.0:
            break

    # Final polish: try a few more restarts with the best assignment as starting point
    while time.time() < deadline - 5.0:
        rng_p = random.Random(int(time.time() * 1000) % 100000)
        assign = list(best_assignment)
        assign, sat = _walksat(n, clauses, assign, rng_p, max_steps=5000, alpha=0.7)
        if sat:
            return assign
        unsat_c = sum(1 for c in clauses if not any(
            assign[abs(l)-1] == (l > 0) for l in c
        ))
        if unsat_c < best_unsat:
            best_unsat = unsat_c
            best_assignment = list(assign)

    return best_assignment
