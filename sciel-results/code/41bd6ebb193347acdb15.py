"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random


def solve(instance):
    """Simulated annealing with incremental clause tracking for 3-SAT."""
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # safety margin before 330s kill

    # Precompute literal indices (0-based) and sign for each clause
    # clause_lits[i] = list of (var_idx, sign) where sign=+1 means positive literal
    clause_lits = []
    for c in clauses:
        lits = [(abs(l) - 1, 1 if l > 0 else -1) for l in c]
        clause_lits.append(lits)

    # For each variable, track which clauses contain it (as positive or negative literal)
    # pos_clauses[v] = list of clause indices where v appears positively
    # neg_clauses[v] = list of clause indices where v appears negatively
    pos_clauses = [[] for _ in range(n)]
    neg_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for var, sign in lits:
            if sign == 1:
                pos_clauses[var].append(ci)
            else:
                neg_clauses[var].append(ci)

    # Initialize assignment
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Compute initial clause satisfaction status
    # clause_sat[i] = number of satisfied literals in clause i
    clause_sat = [0] * m
    for ci, lits in enumerate(clause_lits):
        sat_count = 0
        for var, sign in lits:
            if assignment[var] == (sign == 1):
                sat_count += 1
        clause_sat[ci] = sat_count

    unsat_count = sum(1 for s in clause_sat if s == 0)

    # Track best solution found
    best_unsat = unsat_count
    best_assignment = assignment[:]

    # Simulated annealing parameters
    T_start = 2.0
    T_end = 0.01
    total_steps = 500000  # will be limited by deadline

    # Adaptive: use all available time
    step = 0
    while time.time() < deadline:
        # Current temperature (linear cooling)
        progress = min(step / total_steps, 1.0)
        T = T_start * math.exp(-10.0 * progress)
        if T < T_end:
            T = T_end

        # If all clauses satisfied, return immediately
        if unsat_count == 0:
            return assignment

        # Pick a random variable to flip
        v = rng.randint(0, n - 1)

        # Compute delta: change in unsat_count if we flip v
        # For each clause containing v, flipping v changes its sat count by:
        #   if v was satisfying the clause (literal true), it becomes unsatisfying: delta = +1
        #   if v was not satisfying, it becomes satisfying: delta = -1
        delta = 0
        # Clauses where v appears positively:
        #   if assignment[v] is True, the positive literal is satisfied -> flipping makes it unsat: +1
        #   if assignment[v] is False, the positive literal is not satisfied -> flipping makes it sat: -1
        if assignment[v]:
            for ci in pos_clauses[v]:
                # v's positive literal was satisfying; check if clause becomes unsat
                # only if sat count was 1 (only this literal was satisfying)
                if clause_sat[ci] == 1:
                    delta += 1
        else:
            for ci in pos_clauses[v]:
                # v's positive literal was not satisfying; flipping makes it sat
                # only helps if clause was unsat (sat count 0)
                if clause_sat[ci] == 0:
                    delta -= 1

        # Clauses where v appears negatively:
        #   if assignment[v] is False, the negative literal is satisfied -> flipping makes it unsat: +1
        #   if assignment[v] is True, the negative literal is not satisfied -> flipping makes it sat: -1
        if not assignment[v]:
            for ci in neg_clauses[v]:
                if clause_sat[ci] == 1:
                    delta += 1
        else:
            for ci in neg_clauses[v]:
                if clause_sat[ci] == 0:
                    delta -= 1

        # Accept or reject
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            # Apply flip
            assignment[v] = not assignment[v]
            # Update clause_sat for all clauses containing v
            if assignment[v]:
                # v is now True: positive literals become sat, negative become unsat
                for ci in pos_clauses[v]:
                    clause_sat[ci] += 1
                for ci in neg_clauses[v]:
                    clause_sat[ci] -= 1
            else:
                # v is now False: negative literals become sat, positive become unsat
                for ci in neg_clauses[v]:
                    clause_sat[ci] += 1
                for ci in pos_clauses[v]:
                    clause_sat[ci] -= 1

            # Update unsat_count
            unsat_count += delta

            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

        step += 1

    # Return best assignment found
    if best_unsat == 0:
        return best_assignment
    # If we didn't find a satisfying assignment, return the best we have
    # (still return something valid as a list of booleans)
    return best_assignment
# EVOLVE-BLOCK-END
