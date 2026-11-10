"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_clause_data(n, clauses):
    """Precompute clause structure as numpy arrays for vectorized operations."""
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_sgn = np.zeros((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_sgn[i, j] = 1 if lit > 0 else -1
    var_to_clauses = {}
    for v in range(n):
        var_to_clauses[v] = []
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))
    return clause_vars, clause_sgn, var_to_clauses


def _compute_sat_state(clause_vars, clause_sgn, assignment):
    """Compute per-clause satisfaction state: sat_count (how many literals satisfied)."""
    signs = clause_sgn
    truth = assignment[clause_vars]
    sat_lit = (signs == 1) & truth | (signs == -1) & (~truth)
    sat_count = sat_lit.sum(axis=1)
    return sat_count


def _phase_seed(n, clauses, rng):
    """Initialize assignment using phase information from clause polarities."""
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment


def _count_unsat(assignment, clauses):
    """Count number of unsatisfied clauses."""
    count = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                sat = True
                break
        if not sat:
            count += 1
    return count


def _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline, noise_start, noise_end):
    """
    Vectorized WalkSAT with an adaptive noise schedule.
    noise ramps linearly from noise_start to noise_end across max_steps.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)

    for step in range(max_steps):
        if time.time() >= deadline:
            break

        unsat_mask = sat_count == 0
        n_unsat = int(unsat_mask.sum())
        if n_unsat == 0:
            break

        # Adaptive noise: low early (greedy), high late (escape)
        progress = step / max(1, max_steps)
        noise = noise_start + (noise_end - noise_start) * progress

        unsat_indices = np.where(unsat_mask)[0]
        ci = int(rng.choice(unsat_indices.tolist()))

        lits_vars = clause_vars[ci]
        lits_sgn = clause_sgn[ci]

        if rng.random() < noise:
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            for (ci2, li2) in var_to_clauses[v]:
                ci2 = int(ci2)
                li2 = int(li2)
                s2 = int(clause_sgn[ci2, li2])
                if s2 == 1:
                    old_lit_sat = old_val
                    new_lit_sat = not old_val
                else:
                    old_lit_sat = not old_val
                    new_lit_sat = old_val
                if new_lit_sat and not old_lit_sat:
                    sat_count[ci2] += 1
                elif old_lit_sat and not new_lit_sat:
                    sat_count[ci2] -= 1
        else:
            best_gain = -999
            best_v = -1
            for li in range(3):
                v = int(lits_vars[li])
                s = int(lits_sgn[li])
                old_val = assignment[v]
                new_val = not old_val
                gain = 0
                for (ci2, li2) in var_to_clauses[v]:
                    ci2 = int(ci2)
                    s2 = int(clause_sgn[ci2, li2])
                    if s2 == 1:
                        cur_lit_sat = old_val
                        new_lit_sat = new_val
                    else:
                        cur_lit_sat = not old_val
                        new_lit_sat = not new_val
                    if ci2 == ci:
                        new_sc = 0
                        for jj in range(3):
                            vv = int(clause_vars[ci, jj])
                            ss = int(clause_sgn[ci, jj])
                            val = new_val if vv == v else assignment[vv]
                            if ss == 1:
                                if val:
                                    new_sc += 1
                            else:
                                if not val:
                                    new_sc += 1
                        if new_sc > 0:
                            gain += 1
                    else:
                        if new_lit_sat and not cur_lit_sat:
                            if sat_count[ci2] == 0:
                                gain += 1
                        elif cur_lit_sat and not new_lit_sat:
                            if sat_count[ci2] == 1:
                                gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v >= 0:
                old_val = assignment[best_v]
                assignment[best_v] = not assignment[best_v]
                new_val = not old_val
                for (ci2, li2) in var_to_clauses[best_v]:
                    ci2 = int(ci2)
                    s2 = int(clause_sgn[ci2, li2])
                    if s2 == 1:
                        old_lit_sat = old_val
                        new_lit_sat = new_val
                    else:
                        old_lit_sat = not old_val
                        new_lit_sat = not new_val
                    if new_lit_sat and not old_lit_sat:
                        sat_count[ci2] += 1
                    elif old_lit_sat and not new_lit_sat:
                        sat_count[ci2] -= 1

    return assignment.tolist()


def _minconflicts(assignment, clauses, n, rng, max_steps, deadline):
    """MinConflicts: pick most violated clause, flip var that minimizes violations."""
    for _ in range(max_steps):
        if time.time() >= deadline:
            return assignment
        unsat_clauses = []
        for c in clauses:
            sat = False
            for lit in c:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(c)
        if not unsat_clauses:
            return assignment
        clause = rng.choice(unsat_clauses)
        best_var = -1
        best_cost = float('inf')
        for lit in clause:
            var = abs(lit) - 1
            old_val = assignment[var]
            assignment[var] = not assignment[var]
            cost = 0
            for c in clauses:
                sat = False
                for l2 in c:
                    if assignment[abs(l2) - 1] == (l2 > 0):
                        sat = True
                        break
                if not sat:
                    cost += 1
            assignment[var] = old_val
            if cost < best_cost:
                best_cost = cost
                best_var = var
            elif cost == best_cost and rng.random() < 0.5:
                best_var = var
        if best_var >= 0:
            assignment[best_var] = not assignment[best_var]
    return assignment


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin before 330s kill

    m = len(clauses)

    clause_vars, clause_sgn, var_to_clauses = _build_clause_data(n, clauses)

    best_assignment = None
    best_unsat_count = float('inf')

    # Strategy 1: Phase-seeded vectorized WalkSAT with adaptive noise, multiple restarts.
    # Mix fresh phase seeds, fresh random, and mutation-from-best restarts.
    max_restarts = 120
    for attempt in range(max_restarts):
        if time.time() >= deadline - 5:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)

        if attempt == 0:
            assignment = _phase_seed(n, clauses, rng)
        elif attempt % 5 == 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif best_assignment is not None:
            # Mutation-from-best: preserve partial structure
            assignment = best_assignment[:]
            n_flip = max(1, n // 4)
            for _ in range(n_flip):
                idx = rng.randrange(n)
                assignment[idx] = not assignment[idx]
        else:
            assignment = _phase_seed(n, clauses, rng)

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # Time-aware step budget: scale remaining time per restart
        remaining = max(0.0, deadline - 5 - time.time())
        est_restarts_left = max(1, max_restarts - attempt)
        per_restart = remaining / est_restarts_left
        # Cap per-restart steps to a reasonable range
        steps = int(min(6000, max(800, per_restart * 1500)))

        # Adaptive noise: start greedy, end more random to escape local optima
        noise_start = 0.1
        noise_end = 0.4
        assignment = _vectorized_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses,
            n, m, rng, steps, deadline, noise_start, noise_end
        )

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

    # Strategy 2: MinConflicts from best assignment found so far
    if best_assignment is not None and time.time() < deadline - 10:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        assignment = _minconflicts(assignment, clauses, n, rng, 5000, deadline)
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

    # Strategy 3: Pure random vectorized WalkSAT from scratch (last resort)
    for attempt in range(30):
        if time.time() >= deadline - 2:
            break
        seed = 100000 + attempt * 31337
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        assignment = _vectorized_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses,
            n, m, rng, 3000, deadline, 0.15, 0.5
        )

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment

    return [False] * n
# EVOLVE-BLOCK-END
