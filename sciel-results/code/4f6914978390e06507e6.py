# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 295.0

    m = len(clauses)

    # Convert to 0-based signed literals
    clauses_0based = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0based.append(clause_0)

    # Build watched literal data structures for DPLL
    # For each variable, store list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_0based):
        for var, sign in clause:
            var_clauses[var].append((ci, sign))

    def _compute_unsat(assignment):
        """Compute unsatisfied clause count for a full assignment."""
        unsat = 0
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                unsat += 1
        return unsat

    def _is_satisfied(assignment):
        for clause in clauses_0based:
            sat = False
            for var, sign in clause:
                if assignment[var] == (sign > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    # DPLL with watched literals and VSIDS
    def _dpll_vsids(rng):
        """
        DPLL with watched literals, VSIDS variable selection, and restarts.
        Uses a trail-based approach for efficient backtracking.
        """
        # VSIDS data
        activity = [1.0] / n
        bump = 1.0
        var_order = list(range(n))

        # Assignment trail
        # We'll use a simpler approach: recursive DPLL with trail
        # For 150 variables, we can use a different strategy

        # Let's use a portfolio: try DPLL with different random seeds
        # and also WalkSAT, returning the first solution found

        # Strategy: Use a hybrid approach
        # 1. Try quick WalkSAT (fast, good for easy instances)
        # 2. If that fails, try DPLL with VSIDS
        # 3. If that fails, try more WalkSAT with different parameters

        # Quick WalkSAT
        result = _walksat_quick(rng, max_restarts=20, steps_per_restart=5000)
        if result is not None and _is_satisfied(result):
            return result

        # DPLL with VSIDS
        result = _dpll_search(rng)
        if result is not None and _is_satisfied(result):
            return result

        # More WalkSAT with different parameters
        result = _walksat_extended(rng, max_restarts=50, steps_per_restart=20000)
        if result is not None and _is_satisfied(result):
            return result

        return result

    def _walksat_quick(rng, max_restarts, steps_per_restart):
        """Quick WalkSAT for easy instances."""
        best_assignment = None
        best_unsat = m + 1

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            assignment = [rng.random() < 0.5 for _ in range(n)]
            clause_unsat = np.zeros(m, dtype=bool)
            for ci, clause in enumerate(clauses_0based):
                sat = False
                for var, sign in clause:
                    if assignment[var] == (sign > 0):
                        sat = True
                        break
                clause_unsat[ci] = not sat

            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_assignment

                if unsat_count == 0:
                    return assignment

                unsat_indices = np.where(clause_unsat)[0]
                ci = int(rng.choice(unsat_indices))
                clause = clauses_0based[ci]

                # Simple WalkSAT: flip random literal in clause with probability 0.5
                # otherwise flip the literal that breaks fewest other clauses
                if rng.random() < 0.5:
                    var, sign = clause[rng.randrange(len(clause))]
                else:
                    best_var = -1
                    best_damage = float('inf')
                    for var, sign in clause:
                        damage = 0
                        for other_ci, other_sign in var_clauses[var]:
                            if other_ci == ci:
                                continue
                            if not clause_unsat[other_ci]:
                                # Would flipping break this clause?
                                only_sat = True
                                for v2, s2 in clauses_0based[other_ci]:
                                    if v2 == var:
                                        continue
                                    if assignment[v2] == (s2 > 0):
                                        only_sat = False
                                        break
                                if only_sat:
                                    damage += 1
                        if damage < best_damage:
                            best_damage = damage
                            best_var = var
                    var = best_var

                if var == -1:
                    var, sign = clause[rng.randrange(len(clause))]

                old_val = assignment[var]
                assignment[var] = not old_val

                for other_ci, other_sign in var_clauses[var]:
                    other_clause = clauses_0based[other_ci]
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat

                if unsat_count == 0:
                    return assignment

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

        return best_assignment

    def _walksat_extended(rng, max_restarts, steps_per_restart):
        """Extended WalkSAT with adaptive noise."""
        best_assignment = None
        best_unsat = m + 1

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            assignment = [rng.random() < 0.5 for _ in range(n)]
            clause_unsat = np.zeros(m, dtype=bool)
            for ci, clause in enumerate(clauses_0based):
                sat = False
                for var, sign in clause:
                    if assignment[var] == (sign > 0):
                        sat = True
                        break
                clause_unsat[ci] = not sat

            unsat_count = int(np.sum(clause_unsat))
            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            noise = 0.5
            for step in range(steps_per_restart):
                if time.time() > deadline:
                    return best_assignment

                if unsat_count == 0:
                    return assignment

                unsat_indices = np.where(clause_unsat)[0]
                ci = int(rng.choice(unsat_indices))
                clause = clauses_0based[ci]

                best_var = -1
                best_score = -float('inf')

                for var, sign in clause:
                    score = 0
                    for other_ci, other_sign in var_clauses[var]:
                        if other_ci == ci:
                            continue
                        if clause_unsat[other_ci]:
                            if assignment[var] != (other_sign > 0):
                                score += 1
                        else:
                            only_sat = True
                            for v2, s2 in clauses_0based[other_ci]:
                                if v2 == var:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                score -= 1

                    score += rng.random() * noise
                    if score > best_score:
                        best_score = score
                        best_var = var

                if best_var == -1:
                    var, sign = clause[rng.randrange(len(clause))]
                    best_var = var

                old_val = assignment[best_var]
                assignment[best_var] = not old_val

                for other_ci, other_sign in var_clauses[best_var]:
                    other_clause = clauses_0based[other_ci]
                    sat = False
                    for v, s in other_clause:
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    was_unsat = clause_unsat[other_ci]
                    is_unsat = not sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    clause_unsat[other_ci] = is_unsat

                if unsat_count == 0:
                    return assignment

                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]

                if step % 1000 == 0 and step > 0:
                    noise = max(0.1, noise * 0.9)

        return best_assignment

    def _dpll_search(rng):
        """
        DPLL with VSIDS variable activity and watched literals.
        For 150 variables, use a trail-based approach.
        """
        # Use a simplified DPLL that works well for planted SAT
        # Key insight: for planted SAT, the solution is "close" to random
        # in the sense that most variables are determined by a few key clauses

        # Approach: Use a combination of:
        # 1. Identify "pivotal" variables (appear in many clauses)
        # 2. Try assignments for pivotal variables first
        # 3. Use unit propagation to simplify

        # Build clause-to-variable index for quick lookup
        # For each clause, which variables it contains
        # We already have var_clauses

        # VSIDS-based DPLL with trail
        # For efficiency, we'll use a recursive approach with backtracking

        # Precompute for each clause the list of variables
        # Already have clauses_0based

        # Strategy: Randomized DPLL with restarts
        # For each restart, use a different random variable ordering
        # Use unit propagation to simplify

        best_result = None
        best_unsat = m + 1

        max_restarts = 100

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random variable ordering with VSIDS-like bias
            # Variables that appear in more clauses get higher priority
            var_freq = [0] * n
            for clause in clauses_0based:
                for var, sign in clause:
                    var_freq[var] += 1

            # Create weighted random ordering
            weights = [var_freq[v] + 1 for v in range(n)]
            total_w = sum(weights)
            var_order = []
            remaining = list(range(n))
            remaining_weights = weights[:]
            for _ in range(n):
                r = rng.random() * sum(remaining_weights)
                cum = 0
                idx = 0
                for i, w in enumerate(remaining_weights):
                    cum += w
                    if r <= cum:
                        idx = i
                        break
                var_order.append(remaining[idx])
                remaining_weights.pop(idx)
                remaining.pop(idx)

            # DPLL search
            assignment = [None] * n  # None = unassigned
            # Use a stack-based approach to avoid deep recursion

            # Simplified: try to assign variables in order, with backtracking
            # Use unit propagation to prune

            def _propagate(assignment):
                """
                Apply unit propagation.
                Returns (success, new_assignment) where success indicates no contradiction.
                """
                changed = True
                while changed:
                    changed = False
                    for ci, clause in enumerate(clauses_0based):
                        # Check if clause is unit (exactly one unassigned literal)
                        unassigned_lits = []
                        satisfied = False
                        for var, sign in clause:
                            if assignment[var] is None:
                                unassigned_lits.append((var, sign))
                            elif assignment[var] == (sign > 0):
                                satisfied = True
                                break

                        if satisfied:
                            continue

                        if len(unassigned_lits) == 1:
                            var, sign = unassigned_lits[0]
                            assignment[var] = (sign > 0)
                            changed = True
                        elif len(unassigned_lits) == 0:
                            # Clause is falsified
                            return False, assignment

                return True, assignment

            # Recursive DPLL with depth limit
            import sys
            sys.setrecursionlimit(10000)

            def _dpll_recursive(assignment, depth):
                nonlocal best_result, best_unsat

                if time.time() > deadline:
                    return False

                if depth > 100:  # Limit depth to avoid infinite recursion
                    # Evaluate current assignment (fill remaining randomly)
                    full_assignment = assignment[:]
                    for v in range(n):
                        if full_assignment[v] is None:
                            full_assignment[v] = rng.random() < 0.5
                    unsat = _compute_unsat(full_assignment)
                    if unsat < best_unsat:
                        best_unsat = unsat
                        best_result = full_assignment[:]
                    return False

                # Apply unit propagation
                ok, assignment = _propagate(assignment)
                if not ok:
                    return False

                # Check if all assigned
                all_assigned = True
                for v in range(n):
                    if assignment[v] is None:
                        all_assigned = False
                        break

                if all_assigned:
                    if _is_satisfied(assignment):
                        best_result = assignment[:]
                        best_unsat = 0
                        return True
                    return False

                # Select next variable using VSIDS-like heuristic
                # Prefer variables in unsatisfied or unit clauses
                var_to_assign = None
                best_priority = -1

                for v in var_order:
                    if assignment[v] is not None:
                        continue

                    # Compute priority based on clause involvement
                    priority = 0
                    for ci, sign in var_clauses[v]:
                        clause = clauses_0based[ci]
                        # Check if clause is "tight" (few unassigned literals)
                        unassigned_count = 0
                        for var2, sign2 in clause:
                            if assignment[var2] is None:
                                unassigned_count += 1
                        if unassigned_count <= 2:
                            priority += 10
                        elif unassigned_count == 3:
                            priority += 1

                    if priority > best_priority:
                        best_priority = priority
                        var_to_assign = v

                if var_to_assign is None:
                    # Pick first unassigned in order
                    for v in var_order:
                        if assignment[v] is None:
                            var_to_assign = v
                            break

                if var_to_assign is None:
                    return False

                # Try both values, randomize order
                first_val = rng.random() < 0.5
                for val in [first_val, not first_val]:
                    assignment[var_to_assign] = val

                    # Apply unit propagation
                    ok, new_assignment = _propagate(assignment)
                    if ok:
                        if _dpll_recursive(new_assignment, depth + 1):
                            return True

                    # Backtrack
                    assignment[var_to_assign] = None

                return False

            # Start DPLL
            if _dpll_recursive(assignment[:], 0):
                if best_unsat == 0:
                    return best_result

        return best_result

    # Main solver: portfolio of strategies
    rng = random.Random(42)

    # Time budget allocation
    # 1. Quick WalkSAT: ~5s
    # 2. DPLL with VSIDS: ~10s
    # 3. Extended WalkSAT: remaining time

    # Strategy 1: Quick WalkSAT
    result = _walksat_quick(rng, max_restarts=20, steps_per_restart=5000)
    if result is not None and _is_satisfied(result):
        return result

    # Strategy 2: DPLL with VSIDS
    if time.time() < deadline - 20:
        result2 = _dpll_search(random.Random(123))
        if result2 is not None and _is_satisfied(result2):
            return result2

    # Strategy 3: Extended WalkSAT
    if time.time() < deadline - 5:
        result3 = _walksat_extended(random.Random(777), max_restarts=50, steps_per_restart=20000)
        if result3 is not None and _is_satisfied(result3):
            return result3

    # Return best result found
    best_result = result
    best_unsat = m + 1

    for r in [result, result2, result3]:
        if r is None:
            continue
        unsat = _compute_unsat(r)
        if unsat < best_unsat:
            best_unsat = unsat
            best_result = r

    return best_result
# EVOLVE-BLOCK-END
