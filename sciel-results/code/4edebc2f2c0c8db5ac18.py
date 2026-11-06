"""Planted 3-SAT solver (50 vars): phase-biased WalkSAT + unit propagation + restarts."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280.0  # safety margin under 300s

    # Precompute: clause_lits[i] = list of (var_0based, sign) where sign=1 means positive
    clause_lits = []
    for c in clauses:
        clause_lits.append([(abs(l) - 1, 1 if l > 0 else -1) for l in c])

    # Precompute var_occ[v] = list of (clause_idx, sign_in_that_clause)
    var_occ = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for (v, s) in lits:
            var_occ[v].append((ci, s))

    def evaluate(assignment):
        """Return (num_unsat, clause_unsat_bool_list)."""
        clause_unsat = [False] * m
        num_unsat = 0
        for ci, lits in enumerate(clause_lits):
            sat = False
            for (v, s) in lits:
                if assignment[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                clause_unsat[ci] = True
                num_unsat += 1
        return num_unsat, clause_unsat

    def unit_propagate(assignment, clause_unsat, num_unsat):
        """Iteratively apply unit propagation. Returns updated (assignment, num_unsat, clause_unsat)."""
        changed = True
        while changed and num_unsat > 0:
            changed = False
            for ci in range(m):
                if not clause_unsat[ci]:
                    continue
                # Find unsatisfied literals in this clause
                unsat_lits = []
                for (v, s) in clause_lits[ci]:
                    if assignment[v] != (s == 1):
                        unsat_lits.append((v, s))
                if len(unsat_lits) == 1:
                    v, s = unsat_lits[0]
                    if assignment[v] != (s == 1):
                        assignment[v] = (s == 1)
                        changed = True
            if changed:
                num_unsat, clause_unsat = evaluate(assignment)
        return assignment, num_unsat, clause_unsat

    def get_phase_bias():
        """Compute phase bias: prefer sign that appears more often."""
        pos = [0] * n
        neg = [0] * n
        for lits in clause_lits:
            for (v, s) in lits:
                if s == 1:
                    pos[v] += 1
                else:
                    neg[v] += 1
        bias = []
        for v in range(n):
            if pos[v] > neg[v]:
                bias.append(True)
            elif neg[v] > pos[v]:
                bias.append(False)
            else:
                bias.append(None)
        return bias

    phase_bias = get_phase_bias()

    def init_assignment(rng):
        """Initialize with phase bias, random for ties."""
        a = []
        for v in range(n):
            if phase_bias[v] is not None:
                a.append(phase_bias[v])
            else:
                a.append(rng.random() < 0.5)
        return a

    def walk_sat(assignment, max_flips, rng):
        """WalkSAT with weighted clause selection and phase-saving flip choice."""
        num_unsat, clause_unsat = evaluate(assignment)
        if num_unsat == 0:
            return assignment, True

        # Phase memory
        phase = [assignment[v] for v in range(n)]

        for step in range(max_flips):
            if time.time() > deadline:
                break
            if num_unsat == 0:
                break

            # Collect unsatisfied clauses with weights
            unsat_idx = [ci for ci in range(m) if clause_unsat[ci]]
            if not unsat_idx:
                break

            # Weight by number of unsat literals (harder clauses weighted more)
            weights = []
            for ci in unsat_idx:
                w = 1
                for (v, s) in clause_lits[ci]:
                    if assignment[v] != (s == 1):
                        w += 1
                weights.append(w)
            total_w = sum(weights)
            r = rng.random() * total_w
            acc = 0.0
            chosen_ci = unsat_idx[0]
            for i, ci in enumerate(unsat_idx):
                acc += weights[i]
                if r <= acc:
                    chosen_ci = ci
                    break

            # Choose best literal to flip in chosen clause
            best_v = None
            best_score = float('inf')
            for (v, s) in clause_lits[chosen_ci]:
                # Compute delta: change in number of unsat clauses if we flip v
                delta = 0
                for (ci2, s2) in var_occ[v]:
                    if clause_unsat[ci2]:
                        # Clause is unsat. If flipping v satisfies it, delta -= 1
                        if assignment[v] != (s == 1):
                            # v is currently unsat in this clause context
                            # Check if all other literals are unsat
                            others_all_unsat = True
                            for (v3, s3) in clause_lits[ci2]:
                                if v3 != v and assignment[v3] == (s3 == 1):
                                    others_all_unsat = False
                                    break
                            if others_all_unsat:
                                delta -= 1
                    else:
                        # Clause is satisfied. If v is the unique satisfying literal, flipping breaks it
                        unique = True
                        for (v3, s3) in clause_lits[ci2]:
                            if v3 != v and assignment[v3] == (s3 == 1):
                                unique = False
                                break
                        if unique:
                            delta += 1

                # Phase penalty: prefer flipping toward saved phase
                target = not assignment[v]
                pen = 0.0 if target == phase[v] else 0.4

                score = delta + pen
                if score < best_score:
                    best_score = score
                    best_v = v

            if best_v is None:
                break

            assignment[best_v] = not assignment[best_v]
            phase[best_v] = assignment[best_v]

            num_unsat, clause_unsat = evaluate(assignment)

            # Periodic unit propagation
            if step % 20 == 0 and num_unsat > 0:
                assignment, num_unsat, clause_unsat = unit_propagate(assignment, clause_unsat, num_unsat)

        return assignment, (num_unsat == 0)

    # Main loop with restarts
    best_assign = None
    best_unsat = float('inf')

    base_budget = 3000
    max_budget = 150000
    restart = 0

    while time.time() < deadline - 5:
        restart += 1
        rng = random.Random(1000 + restart)

        assignment = init_assignment(rng)
        # Apply initial unit propagation
        num_unsat, clause_unsat = evaluate(assignment)
        assignment, num_unsat, clause_unsat = unit_propagate(assignment, clause_unsat, num_unsat)

        if num_unsat == 0:
            return assignment

        budget = min(base_budget + restart * 2000, max_budget)
        assignment, solved = walk_sat(assignment, budget, rng)

        num_unsat, _ = evaluate(assignment)
        if num_unsat < best_unsat:
            best_unsat = num_unsat
            best_assign = assignment[:]

        if num_unsat == 0:
            return assignment

    # Return best
    if best_assign is not None:
        return best_assign
    return [False] * n
# EVOLVE-BLOCK-END
