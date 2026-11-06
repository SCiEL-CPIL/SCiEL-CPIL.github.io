"""WalkSAT with preprocessing (unit propagation + pure literal elimination) and clause-weighted variable selection."""
import time
import random
import math

def solve(instance):
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)

    # Work with 0-based literals internally
    # Convert clauses to 0-based signed literals
    clauses0 = [[l - 1 if l > 0 else l + 1 for l in c] for c in clauses]

    # Assignment: None = unassigned, True/False = assigned
    assignment = [None] * n

    # Helper: value of a literal given assignment
    # lit is 0-based signed: positive means var True, negative means var False
    def lit_val(lit):
        var = abs(lit) - 1
        if assignment[var] is None:
            return None
        return assignment[var] == (lit > 0)

    # Helper: check if clause is satisfied
    def clause_sat(clause):
        for lit in clause:
            v = lit_val(lit)
            if v is True:
                return True
        return False

    # Helper: check if clause is falsified (all literals False)
    def clause_falsified(clause):
        for lit in clause:
            v = lit_val(lit)
            if v is None or v is True:
                return False
        return True

    # Preprocessing: Unit propagation and pure literal elimination
    # Iteratively apply until fixpoint
    def preprocess():
        """Apply unit propagation and pure literal elimination. Returns True if no contradiction."""
        changed = True
        while changed:
            changed = False

            # Unit propagation
            unit_clauses = []
            for c in clauses0:
                if clause_sat(c):
                    continue
                # Count unassigned and false literals
                unassigned = 0
                false_lit = None
                for lit in c:
                    v = lit_val(lit)
                    if v is None:
                        unassigned += 1
                    elif v is False:
                        false_lit = lit
                    else:
                        # v is True, clause satisfied (shouldn't happen if we checked)
                        pass
                if unassigned == 0:
                    # Clause is falsified - contradiction
                    return False
                if unassigned == 1:
                    unit_clauses.append(c)

            # Assign unit clauses
            for c in unit_clauses:
                for lit in c:
                    v = lit_val(lit)
                    if v is None:
                        var = abs(lit) - 1
                        assignment[var] = (lit > 0)
                        changed = True
                        break

            # Pure literal elimination
            # Find variables that appear with only one sign
            pos_vars = set()
            neg_vars = set()
            for c in clauses0:
                if clause_sat(c):
                    continue
                for lit in c:
                    var = abs(lit) - 1
                    if assignment[var] is not None:
                        continue
                    if lit > 0:
                        pos_vars.add(var)
                    else:
                        neg_vars.add(var)

            pure_vars = (pos_vars - neg_vars) | (neg_vars - pos_vars)
            for var in pure_vars:
                if var in pos_vars:
                    assignment[var] = True
                else:
                    assignment[var] = False
                changed = True

        return True

    if not preprocess():
        # Contradiction during preprocessing - shouldn't happen for satisfiable instances
        # Fall back to random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        return assignment

    # Check if all satisfied
    def all_satisfied():
        for c in clauses0:
            if not clause_sat(c):
                return False
        return True

    if all_satisfied():
        return [a if a is not None else rng.random() < 0.5 for a in assignment]

    # Fill remaining unassigned variables randomly
    for i in range(n):
        if assignment[i] is None:
            assignment[i] = rng.random() < 0.5

    # WalkSAT with clause-weighted variable selection
    # Weight each clause, pick a random unsatisfied clause, then pick a variable
    # from that clause weighted by the number of satisfied literals it would flip to false

    # Clause weights
    clause_weights = [1.0] * len(clauses0)

    def count_unsat():
        count = 0
        for c in clauses0:
            if not clause_sat(c):
                count += 1
        return count

    # Main search loop with restarts
    max_restarts = 50
    steps_per_restart = 5000
    noise = 0.2  # probability of random flip

    best_assignment = assignment[:]
    best_unsat = count_unsat()

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # On restart after first, re-randomize
        if restart > 0:
            for i in range(n):
                if assignment[i] is None:
                    assignment[i] = rng.random() < 0.5
                else:
                    # Keep preprocessed assignments but allow some randomization
                    if rng.random() < 0.1:
                        assignment[i] = not assignment[i]
            # Re-apply preprocessing lightly
            # Just do unit propagation once
            for c in clauses0:
                if clause_sat(c):
                    continue
                unassigned = 0
                false_lit = None
                for lit in c:
                    v = lit_val(lit)
                    if v is None:
                        unassigned += 1
                    elif v is False:
                        false_lit = lit
                if unassigned == 1:
                    for lit in c:
                        v = lit_val(lit)
                        if v is None:
                            var = abs(lit) - 1
                            assignment[var] = (lit > 0)
                            break

        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for i, c in enumerate(clauses0):
                if not clause_sat(c):
                    unsat_clauses.append(i)

            if not unsat_clauses:
                break

            # Pick a clause weighted by clause_weights
            total_w = sum(clause_weights[i] for i in unsat_clauses)
            r = rng.random() * total_w
            cumulative = 0
            chosen_clause_idx = unsat_clauses[0]
            for idx in unsat_clauses:
                cumulative += clause_weights[idx]
                if r <= cumulative:
                    chosen_clause_idx = idx
                    break

            clause = clauses0[chosen_clause_idx]

            # For each literal in the clause, compute the damage:
            # number of currently satisfied clauses that would become unsatisfied
            # if we flip this variable
            best_var = None
            best_damage = float('inf')
            best_random = rng.random()

            for lit in clause:
                var = abs(lit) - 1
                # Compute damage: how many satisfied clauses contain the negation of this literal
                damage = 0
                for j, c in enumerate(clauses0):
                    if j == chosen_clause_idx:
                        continue
                    if clause_sat(c):
                        # Check if flipping var would falsify this clause
                        # The clause is currently satisfied; it would become unsatisfied
                        # if the only satisfied literal is the one that flips
                        sat_lits = 0
                        would_flip = False
                        for l in c:
                            lv = lit_val(l)
                            if lv is True:
                                sat_lits += 1
                                if abs(l) == var and (l > 0) == (lit < 0):
                                    would_flip = True
                            elif lv is False:
                                pass
                            else:
                                pass
                        if sat_lits == 1 and would_flip:
                            damage += clause_weights[j]

                # With probability noise, pick randomly
                if rng.random() < noise:
                    if best_var is None or rng.random() < 0.5:
                        best_var = var
                        best_damage = damage
                else:
                    if damage < best_damage or (damage == best_damage and rng.random() < 0.5):
                        best_var = var
                        best_damage = damage

            if best_var is None:
                best_var = abs(clause[0]) - 1

            # Flip the variable
            assignment[best_var] = not assignment[best_var]

            # Update best
            cur_unsat = count_unsat()
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            if cur_unsat == 0:
                break

        if best_unsat == 0:
            break

    # Return the best assignment found
    if best_unsat == 0:
        return best_assignment
    else:
        # Return best_assignment or current assignment
        return best_assignment
