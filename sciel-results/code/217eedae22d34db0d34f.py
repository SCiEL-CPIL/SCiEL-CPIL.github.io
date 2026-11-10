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
    # For each variable, precompute which (clause, position) pairs contain it
    var_to_clauses = {}
    for v in range(n):
        var_to_clauses[v] = []
    for i in range(m):
        for j in range(3):
            v = int(clause_vars[i, j])
            var_to_clauses[v].append((i, j))
    # Incidence matrix: lit_sat[i, j] is 1 if literal (i,j) is satisfied
    # Build a variable->literal-position map for fast impact scoring
    return clause_vars, clause_sgn, var_to_clauses


def _compute_sat_state(clause_vars, clause_sgn, assignment):
    """Compute per-clause satisfaction state: sat_count (how many literals satisfied)."""
    var_idx = clause_vars
    signs = clause_sgn
    truth = assignment[var_idx]
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


def _vectorized_impact_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline):
    """
    Vectorized WalkSAT with impact-based literal selection.

    Maintains sat_count per clause. On each step:
      1. Pick a random unsatisfied clause.
      2. For each of the 3 literals in it, compute net gain = (# newly-satisfied
         clauses) - (# newly-unsatisfied clauses) from flipping that variable.
      3. With probability `noise`, pick a random literal; else pick max-gain
         (random tie-break).
      4. Apply the flip and update sat_count incrementally.

    The gain computation is vectorized: for a candidate variable v, we gather
    all clauses containing v, compute old/new literal satisfaction, and use
    sat_count to determine which clauses flip status.
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    m_arr = clause_vars.shape[0]

    for step in range(max_steps):
        if time.time() >= deadline:
            break

        unsat_mask = sat_count == 0
        n_unsat = int(unsat_mask.sum())
        if n_unsat == 0:
            break

        # Pick a random unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        ci = int(rng.choice(unsat_indices.tolist()))

        lits_vars = clause_vars[ci]  # (3,)
        lits_sgn = clause_sgn[ci]    # (3,)

        # Adaptive noise: start greedy (low noise), increase over the walk
        progress = step / max(1, max_steps)
        noise = 0.15 + 0.35 * progress

        if rng.random() < noise:
            # Random flip of one literal in the clause
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
        else:
            # Greedy: compute gain for each of the 3 literals
            gains = np.zeros(3, dtype=np.int32)
            for li in range(3):
                v = int(lits_vars[li])
                old_val = assignment[v]
                new_val = not old_val

                # Gather all clause-positions containing v
                vcl = var_to_clauses[v]
                if not vcl:
                    gains[li] = 0
                    continue

                # Vectorized gain computation
                # For each (ci2, li2) in vcl:
                #   old_lit_sat = (sgn==1 and old_val) or (sgn==-1 and not old_val)
                #   new_lit_sat = (sgn==1 and new_val) or (sgn==-1 and not new_val)
                #   If clause ci2 is currently unsat (sat_count==0) and new_lit_sat: +1
                #   If clause ci2 is currently sat (sat_count==1) and old_lit_sat and not new_lit_sat: -1
                #   (clauses with sat_count>=2 are unaffected in status)

                # Build arrays
                cidx = np.array([p[0] for p in vcl], dtype=np.int32)
                lidx = np.array([p[1] for p in vcl], dtype=np.int32)
                s2 = clause_sgn[cidx, lidx]  # (k,)

                # Old and new literal satisfaction
                old_lit_sat = (s2 == 1) & old_val | (s2 == -1) & (~old_val)
                new_lit_sat = (s2 == 1) & new_val | (s2 == -1) & (~new_val)

                sc = sat_count[cidx]  # (k,) current sat counts

                # Clauses that become satisfied: currently unsat, new_lit_sat
                newly_sat = (sc == 0) & new_lit_sat
                # Clauses that become unsat: currently sat-by-one, old_lit_sat, not new_lit_sat
                newly_unsat = (sc == 1) & old_lit_sat & (~new_lit_sat)

                gains[li] = int(newly_sat.sum()) - int(newly_unsat.sum())

            # Pick max gain with random tie-break
            max_gain = gains.max()
            best_candidates = np.where(gains == max_gain)[0]
            li = int(rng.choice(best_candidates.tolist()))
            v = int(lits_vars[li])

        # Apply the flip
        old_val = assignment[v]
        assignment[v] = not assignment[v]
        new_val = not old_val

        # Update sat_count for all clauses containing v
        for (ci2, li2) in var_to_clauses[v]:
            ci2 = int(ci2)
            li2 = int(li2)
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


def _vectorized_minconflicts(assignment, clauses, n, clause_vars, clause_sgn, rng, max_steps, deadline):
    """
    Vectorized MinConflicts:
    - Maintains sat_count per clause.
    - Each step: pick a random unsat clause, then for each variable in it,
      compute the total unsat count after flipping (vectorized over all clauses).
    - Flip the variable that minimizes total unsat.
    """
    assignment = np.array(assignment, dtype=bool)
    m = clause_vars.shape[0]
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)

    for step in range(max_steps):
        if time.time() >= deadline:
            break

        n_unsat = int((sat_count == 0).sum())
        if n_unsat == 0:
            break

        # Pick a random unsatisfied clause
        unsat_indices = np.where(sat_count == 0)[0]
        ci = int(rng.choice(unsat_indices.tolist()))

        # For each of the 3 variables in this clause, compute new total unsat
        best_var = -1
        best_cost = float('inf')

        for li in range(3):
            v = int(clause_vars[ci, li])
            old_val = assignment[v]
            new_val = not old_val

            # Compute new sat_count if v is flipped
            # For each clause containing v, determine if its sat count changes
            vcl = var_to_clauses[v]
            if not vcl:
                cost = n_unsat
            else:
                cidx = np.array([p[0] for p in vcl], dtype=np.int32)
                lidx = np.array([p[1] for p in vcl], dtype=np.int32)
                s2 = clause_sgn[cidx, lidx]

                old_lit_sat = (s2 == 1) & old_val | (s2 == -1) & (~old_val)
                new_lit_sat = (s2 == 1) & new_val | (s2 == -1) & (~new_val)

                sc = sat_count[cidx].copy()
                # Update: clauses gaining a satisfied literal
                gain = new_lit_sat & (~old_lit_sat)
                loss = old_lit_sat & (~new_lit_sat)
                sc_new = sc + gain.astype(np.int32) - loss.astype(np.int32)

                # Total unsat after flip
                # Clauses in cidx: use sc_new; others: use sat_count
                # Unsat count = (# unsat in cidx) + (# unsat not in cidx)
                # # unsat in cidx = (sc_new == 0).sum()
                # # unsat not in cidx = total_unsat - (# unsat in cidx originally)
                # # unsat in cidx originally = (sc == 0).sum()
                unsat_in_cidx_new = int((sc_new == 0).sum())
                unsat_in_cidx_old = int((sc == 0).sum())
                cost = n_unsat - unsat_in_cidx_old + unsat_in_cidx_new

            if cost < best_cost:
                best_cost = cost
                best_var = v
            elif cost == best_cost and rng.random() < 0.5:
                best_var = v

        if best_var >= 0:
            old_val = assignment[best_var]
            assignment[best_var] = not assignment[best_var]
            new_val = not old_val
            # Update sat_count
            for (ci2, li2) in var_to_clauses[best_var]:
                ci2 = int(ci2)
                li2 = int(li2)
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


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin before 330s kill

    m = len(clauses)

    # Precompute clause structure
    clause_vars, clause_sgn, var_to_clauses = _build_clause_data(n, clauses)

    best_assignment = None
    best_unsat_count = float('inf')

    def _try_update(assignment):
        nonlocal best_assignment, best_unsat_count
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        return unsat_count == 0

    # Strategy 1: Phase-seeded vectorized impact WalkSAT with multiple restarts
    for attempt in range(120):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)

        if attempt == 0:
            assignment = _phase_seed(n, clauses, rng)
        elif attempt == 1:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            # Mix: start from best, mutate some
            if best_assignment is not None:
                assignment = best_assignment[:]
                for _ in range(n // 3):
                    idx = rng.randrange(n)
                    assignment[idx] = not assignment[idx]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # Vectorized impact WalkSAT
        steps = 1500 + attempt * 200
        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, steps, deadline
        )

        if _try_update(assignment):
            return assignment

    # Strategy 2: Vectorized MinConflicts from best assignment found so far
    if best_assignment is not None and time.time() < deadline - 5:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        assignment = _vectorized_minconflicts(
            assignment, clauses, n, clause_vars, clause_sgn, rng, 8000, deadline
        )
        if _try_update(assignment):
            return assignment

    # Strategy 3: More random-seed impact WalkSAT restarts
    for attempt in range(40):
        if time.time() >= deadline:
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

        assignment = _vectorized_impact_walksat(
            assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, 2500, deadline
        )

        if _try_update(assignment):
            return assignment

    # Final: return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
