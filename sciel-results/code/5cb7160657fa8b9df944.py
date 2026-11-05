"""Optimized 3-SAT solver with unit propagation, geometric restarts, and min-confl WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 285.0  # safety margin below 330s kill
    num_clauses = len(clauses)

    # Build variable-to-clauses index for fast incremental updates
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    def is_clause_satisfied(ci, assignment):
        for lit in clauses[ci]:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def compute_unsat_count(assignment):
        """Count unsatisfied clauses."""
        count = 0
        for ci in range(num_clauses):
            sat = False
            for lit in clauses[ci]:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def unit_propagate(assignment, clauses_copy):
        """
        Perform unit propagation. Returns (reduced_clauses, fixed_vars, success).
        fixed_vars maps var_idx -> bool for variables forced by unit propagation.
        Returns success=False if a contradiction is found.
        """
        fixed = {}
        clauses_to_process = list(clauses_copy)
        i = 0
        while i < len(clauses_to_process):
            clause = clauses_to_process[i]
            # Remove already-fixed literals
            remaining = []
            unit_lit = None
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in fixed:
                    if fixed[var_idx] == (lit > 0):
                        # Clause is satisfied, skip
                        break
                    else:
                        # Literal is false, remove it
                        continue
                if len(remaining) == 0 and unit_lit is None:
                    unit_lit = lit
                    remaining.append(lit)
                else:
                    remaining.append(lit)
            else:
                # No break, so clause wasn't satisfied
                if len(remaining) == 0:
                    # Empty clause: contradiction
                    return None, fixed, False
                elif len(remaining) == 1:
                    # Unit clause: force the variable
                    lit = remaining[0]
                    var_idx = abs(lit) - 1
                    if var_idx in fixed:
                        if fixed[var_idx] != (lit > 0):
                            return None, fixed, False
                    else:
                        fixed[var_idx] = (lit > 0)
                        # Continue processing, don't increment i
                        continue
                else:
                    clauses_to_process[i] = remaining
                    i += 1
                    continue
                i += 1
                continue
            # Clause was satisfied, skip
            i += 1

        return clauses_to_process, fixed, True

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def walksat_with_minconfl(assignment, rng, p_walk, max_steps, deadline):
        """
        WalkSAT with min-confl heuristic.
        Returns (unsat_count, assignment) at the end.
        """
        # Initialize unsat set
        unsat_set = set()
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat_set.add(ci)

        cur_unsat = len(unsat_set)
        if cur_unsat == 0:
            return 0, assignment

        steps = 0
        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            # Min-confl: for each literal in the clause, count conflicts
            # A conflict is a satisfied clause that would be broken by flipping
            best_var = None
            best_conflict = float('inf')

            for lit in clause:
                var_idx = abs(lit) - 1
                conflict = 0
                for other_ci in var_to_clauses[var_idx]:
                    if other_ci == ci:
                        continue
                    if other_ci in unsat_set:
                        # Already unsat, flipping won't hurt
                        continue
                    # Check if this clause is satisfied and would be broken
                    olit = None
                    for l in clauses[other_ci]:
                        if abs(l) - 1 == var_idx:
                            olit = l
                            break
                    if olit is None:
                        continue
                    if assignment[var_idx] == (olit > 0):
                        # This literal is satisfied; check if any other literal saves it
                        other_sat = False
                        for ll in clauses[other_ci]:
                            if ll != olit and assignment[abs(ll) - 1] == (ll > 0):
                                other_sat = True
                                break
                        if not other_sat:
                            conflict += 1

                if conflict < best_conflict:
                    best_conflict = conflict
                    best_var = var_idx

            if best_var is None:
                break

            # WalkSAT: with probability p_walk, pick random literal
            if rng.random() < p_walk:
                lit = rng.choice(clause)
                best_var = abs(lit) - 1

            # Flip the variable
            assignment[best_var] = not assignment[best_var]

            # Update unsat set incrementally
            for other_ci in var_to_clauses[best_var]:
                was_unsat = other_ci in unsat_set
                now_sat = False
                for lit in clauses[other_ci]:
                    if assignment[abs(lit) - 1] == (lit > 0):
                        now_sat = True
                        break
                if now_sat:
                    if was_unsat:
                        unsat_set.discard(other_ci)
                else:
                    if not was_unsat:
                        unsat_set.add(other_ci)

            cur_unsat = len(unsat_set)
            if cur_unsat == 0:
                return 0, assignment

        return cur_unsat, assignment

    def walksat_random(assignment, rng, max_steps, deadline):
        """
        Basic WalkSAT with random variable selection.
        Returns (unsat_count, assignment) at the end.
        """
        unsat_set = set()
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat_set.add(ci)

        cur_unsat = len(unsat_set)
        if cur_unsat == 0:
            return 0, assignment

        steps = 0
        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1

            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]
            lit = rng.choice(clause)
            var_idx = abs(lit) - 1

            assignment[var_idx] = not assignment[var_idx]

            for other_ci in var_to_clauses[var_idx]:
                was_unsat = other_ci in unsat_set
                now_sat = False
                for l in clauses[other_ci]:
                    if assignment[abs(l) - 1] == (l > 0):
                        now_sat = True
                        break
                if now_sat:
                    if was_unsat:
                        unsat_set.discard(other_ci)
                else:
                    if not was_unsat:
                        unsat_set.add(other_ci)

            cur_unsat = len(unsat_set)
            if cur_unsat == 0:
                return 0, assignment

        return cur_unsat, assignment

    # Try unit propagation first
    clauses_copy = [list(c) for c in clauses]
    reduced_clauses, fixed_vars, up_success = unit_propagate(None, clauses_copy)

    best_assignment = None
    best_unsat = num_clauses + 1

    if up_success and reduced_clauses is not None:
        # Try solving the reduced problem
        # First, check if all clauses are satisfied with fixed vars
        all_sat = True
        for clause in reduced_clauses:
            sat = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in fixed_vars:
                    if fixed_vars[var_idx] == (lit > 0):
                        sat = True
                        break
            if not sat:
                all_sat = False
                break

        if all_sat:
            # All clauses satisfied with just unit propagation
            assignment = [False] * n
            for var_idx, val in fixed_vars.items():
                assignment[var_idx] = val
            return assignment

        # Solve the reduced problem with local search
        remaining_vars = [v for v in range(n) if v not in fixed_vars]
        num_remaining = len(remaining_vars)

        if num_remaining == 0:
            assignment = [False] * n
            for var_idx, val in fixed_vars.items():
                assignment[var_idx] = val
            return assignment

        # Build mapping from original var to reduced var
        var_map = {}
        for i, v in enumerate(remaining_vars):
            var_map[v] = i

        # Transform clauses to reduced form
        reduced_clauses_mapped = []
        for clause in reduced_clauses:
            new_clause = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in fixed_vars:
                    if fixed_vars[var_idx] == (lit > 0):
                        # Clause satisfied, skip
                        break
                    else:
                        continue
                else:
                    new_lit = lit if lit > 0 else -lit
                    new_lit = var_map[var_idx] + 1
                    if lit < 0:
                        new_lit = -new_lit
                    new_clause.append(new_lit)
            else:
                if new_clause:
                    reduced_clauses_mapped.append(new_clause)

        # Now solve the reduced problem
        n_red = num_remaining
        clauses_red = reduced_clauses_mapped
        num_clauses_red = len(clauses_red)

        # Build var_to_clauses for reduced problem
        var_to_clauses_red = defaultdict(list)
        for ci, clause in enumerate(clauses_red):
            for lit in clause:
                var_to_clauses_red[abs(lit) - 1].append(ci)

        def is_clause_satisfied_red(ci, assignment):
            for lit in clauses_red[ci]:
                if assignment[abs(lit) - 1] == (lit > 0):
                    return True
            return False

        def compute_unsat_count_red(assignment):
            count = 0
            for ci in range(num_clauses_red):
                sat = False
                for lit in clauses_red[ci]:
                    if assignment[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count

        def make_random_assignment_red(seed):
            rng = random.Random(seed)
            return [rng.random() < 0.5 for _ in range(n_red)]

        def walksat_red(assignment, rng, p_walk, max_steps, deadline):
            unsat_set = set()
            for ci in range(num_clauses_red):
                if not is_clause_satisfied_red(ci, assignment):
                    unsat_set.add(ci)

            cur_unsat = len(unsat_set)
            if cur_unsat == 0:
                return 0, assignment

            steps = 0
            while cur_unsat > 0 and steps < max_steps:
                if time.time() > deadline:
                    break
                steps += 1

                ci = rng.choice(list(unsat_set))
                clause = clauses_red[ci]

                best_var = None
                best_conflict = float('inf')

                for lit in clause:
                    var_idx = abs(lit) - 1
                    conflict = 0
                    for other_ci in var_to_clauses_red[var_idx]:
                        if other_ci == ci:
                            continue
                        if other_ci in unsat_set:
                            continue
                        olit = None
                        for l in clauses_red[other_ci]:
                            if abs(l) - 1 == var_idx:
                                olit = l
                                break
                        if olit is None:
                            continue
                        if assignment[var_idx] == (olit > 0):
                            other_sat = False
                            for ll in clauses_red[other_ci]:
                                if ll != olit and assignment[abs(ll) - 1] == (ll > 0):
                                    other_sat = True
                                    break
                            if not other_sat:
                                conflict += 1

                    if conflict < best_conflict:
                        best_conflict = conflict
                        best_var = var_idx

                if best_var is None:
                    break

                if rng.random() < p_walk:
                    lit = rng.choice(clause)
                    best_var = abs(lit) - 1

                assignment[best_var] = not assignment[best_var]

                for other_ci in var_to_clauses_red[best_var]:
                    was_unsat = other_ci in unsat_set
                    now_sat = False
                    for lit in clauses_red[other_ci]:
                        if assignment[abs(lit) - 1] == (lit > 0):
                            now_sat = True
                            break
                    if now_sat:
                        if was_unsat:
                            unsat_set.discard(other_ci)
                    else:
                        if not was_unsat:
                            unsat_set.add(other_ci)

                cur_unsat = len(unsat_set)
                if cur_unsat == 0:
                    return 0, assignment

            return cur_unsat, assignment

        best_assignment_red = None
        best_unsat_red = num_clauses_red + 1

        for attempt in range(5000):
            if time.time() > deadline:
                break

            seed = attempt * 7919 + 13
            assignment_red = make_random_assignment_red(seed)
            cur_unsat = compute_unsat_count_red(assignment_red)

            if cur_unsat < best_unsat_red:
                best_unsat_red = cur_unsat
                best_assignment_red = assignment_red[:]
            if cur_unsat == 0:
                # Convert back to original assignment
                assignment = [False] * n
                for var_idx, val in fixed_vars.items():
                    assignment[var_idx] = val
                for i, v in enumerate(remaining_vars):
                    assignment[v] = assignment_red[i]
                return assignment

            rng = random.Random(attempt * 31337 + 42)
            p_walk = 0.1
            max_steps = 5000

            new_unsat, assignment_red = walksat_red(assignment_red, rng, p_walk, max_steps, deadline)
            cur_unsat = new_unsat

            if cur_unsat < best_unsat_red:
                best_unsat_red = cur_unsat
                best_assignment_red = assignment_red[:]
            if cur_unsat == 0:
                assignment = [False] * n
                for var_idx, val in fixed_vars.items():
                    assignment[var_idx] = val
                for i, v in enumerate(remaining_vars):
                    assignment[v] = assignment_red[i]
                return assignment

        # If we didn't solve the reduced problem, fall through to full problem
        if best_assignment_red is not None:
            # Convert partial solution back
            assignment = [False] * n
            for var_idx, val in fixed_vars.items():
                assignment[var_idx] = val
            for i, v in enumerate(remaining_vars):
                assignment[v] = best_assignment_red[i]
            best_assignment = assignment
            best_unsat = best_unsat_red

    # Solve the full problem with local search
    for attempt in range(10000):
        if time.time() > deadline:
            break

        seed = attempt * 7919 + 13
        assignment = make_random_assignment(seed)
        cur_unsat = compute_unsat_count(assignment)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        rng = random.Random(attempt * 31337 + 42)
        p_walk = 0.1
        max_steps = 5000

        if attempt % 2 == 0:
            new_unsat, assignment = walksat_with_minconfl(assignment, rng, p_walk, max_steps, deadline)
        else:
            new_unsat, assignment = walksat_random(assignment, rng, max_steps, deadline)

        cur_unsat = new_unsat

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

    return best_assignment if best_assignment else make_random_assignment(0)
# EVOLVE-BLOCK-END
