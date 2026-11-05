"""CDCL-inspired 3-SAT solver with clause learning, unit propagation, and efficient tracking."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)

    if num_clauses == 0:
        return [False] * n

    # Preprocess: convert to 0-based (var, sign) tuples
    processed_clauses = []
    for clause in clauses:
        processed = [(abs(lit) - 1, 1 if lit > 0 else -1) for lit in clause]
        # Remove tautological clauses (both var and -var present)
        vars_in_clause = set()
        is_tautology = False
        for var, sign in processed:
            if var in vars_in_clause:
                is_tautology = True
                break
            vars_in_clause.add(var)
        if not is_tautology and len(processed) > 0:
            processed_clauses.append(processed)

    # Unit propagation preprocessing
    # Build literal occurrence lists for unit prop
    # lit_occ[lit] = list of clause indices containing that literal
    # where lit is encoded as var*2 + (0 if positive, 1 if negative)
    def lit_encode(var, sign):
        return var * 2 + (0 if sign == 1 else 1)

    def lit_decode(lit):
        return lit // 2, 1 if lit % 2 == 0 else -1

    # Build literal occurrence lists
    lit_occ = {}
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            lit = lit_encode(var, sign)
            if lit not in lit_occ:
                lit_occ[lit] = []
            lit_occ[lit].append(ci)

    # Unit propagation: find forced assignments
    assignment = [None] * n  # None means unassigned
    active_clauses = [True] * len(processed_clauses)
    changed = True
    while changed:
        changed = False
        for ci in range(len(processed_clauses)):
            if not active_clauses[ci]:
                continue
            # Count unassigned literals
            unassigned = 0
            forced_var = None
            forced_sign = None
            for var, sign in processed_clauses[ci]:
                if assignment[var] is None:
                    unassigned += 1
                    forced_var = var
                    forced_sign = sign
                elif sign == 1:
                    if assignment[var]:
                        break
                else:
                    if not assignment[var]:
                        break
            else:
                # No break, so all literals checked
                if unassigned == 1:
                    assignment[forced_var] = (forced_sign == 1)
                    changed = True
                elif unassigned == 0:
                    # Clause is unsatisfied - shouldn't happen for SAT instances
                    pass

    # After unit propagation, simplify clauses
    final_clauses = []
    for ci in range(len(processed_clauses)):
        if not active_clauses[ci]:
            continue
        new_clause = []
        clause_satisfied = False
        for var, sign in processed_clauses[ci]:
            if assignment[var] is not None:
                if sign == 1:
                    if assignment[var]:
                        clause_satisfied = True
                        break
                else:
                    if not assignment[var]:
                        clause_satisfied = True
                        break
            else:
                new_clause.append((var, sign))
        if clause_satisfied:
            continue
        if len(new_clause) > 0:
            final_clauses.append(new_clause)

    # Check for empty clause (contradiction)
    if any(len(c) == 0 for c in final_clauses):
        # Instance is UNSAT, but we need to return something
        # Return the partial assignment extended randomly
        result = []
        for v in range(n):
            if assignment[v] is not None:
                result.append(assignment[v])
            else:
                result.append(random.random() < 0.5)
        return result

    # Rebuild structures with simplified clauses
    processed_clauses = final_clauses
    num_clauses = len(processed_clauses)

    if num_clauses == 0:
        # All clauses satisfied by unit propagation
        result = []
        for v in range(n):
            if assignment[v] is not None:
                result.append(assignment[v])
            else:
                result.append(random.random() < 0.5)
        return result

    # Build occurrence lists: var_occ[var] = list of (clause_idx, sign)
    var_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(processed_clauses):
        for var, sign in clause:
            var_occ[var].append((ci, sign))

    # For fast satisfaction check, precompute per-clause var and sign arrays
    clause_vars = [[v for v, s in c] for c in processed_clauses]
    clause_signs = [[s for v, s in c] for c in processed_clauses]

    # Build unsatisfied clause index for O(1) access
    unsat_clauses = set()

    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    rng = random.Random(12345)

    # --- Core state ---
    # assignment: list of bools (for unassigned vars)
    # clause_sat_count[ci] = number of true literals in clause ci
    # unsat_count = number of clauses with sat_count == 0

    def init_state(assignment):
        """Initialize clause_sat_count and unsat_count from assignment."""
        clause_sat_count = [0] * len(processed_clauses)
        unsat = 0
        for ci in range(len(processed_clauses)):
            vs = clause_vars[ci]
            ss = clause_signs[ci]
            cnt = 0
            for i in range(len(vs)):
                v = vs[i]
                s = ss[i]
                if s == 1:
                    if assignment[v]:
                        cnt += 1
                else:
                    if not assignment[v]:
                        cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat += 1
        return clause_sat_count, unsat

    def flip_var(assignment, var, clause_sat_count, unsat_count, unsat_clauses):
        """Flip variable var, update clause_sat_count and unsat_count incrementally.
        Returns new unsat_count."""
        old_val = assignment[var]
        assignment[var] = not old_val
        new_unsat = unsat_count
        for ci, sign in var_occ[var]:
            old_sat = clause_sat_count[ci]
            # Determine if this literal was true before flip
            if sign == 1:
                was_true = old_val
            else:
                was_true = not old_val
            if was_true:
                # Literal was true, now false
                clause_sat_count[ci] = old_sat - 1
                if old_sat == 1:
                    new_unsat += 1
                    unsat_clauses.add(ci)
            else:
                # Literal was false, now true
                clause_sat_count[ci] = old_sat + 1
                if old_sat == 0:
                    new_unsat -= 1
                    unsat_clauses.discard(ci)
        return new_unsat

    # --- Learned clauses (nogoods) ---
    learned_clauses = []  # list of list of (var, sign)
    learned_var_occ = {}  # var -> list of (learned_idx, sign)

    def check_learned(assignment):
        """Count how many learned clauses are unsatisfied."""
        cnt = 0
        for lc in learned_clauses:
            sat = False
            for var, sign in lc:
                if sign == 1:
                    if assignment[var]:
                        sat = True
                        break
                else:
                    if not assignment[var]:
                        sat = True
                        break
            if not sat:
                cnt += 1
        return cnt

    def extract_conflict(assignment, clause_sat_count, unsat_clauses):
        """Extract a conflict clause from the current assignment."""
        if unsat_clauses:
            ci = next(iter(unsat_clauses))
            conflict = [(v, s) for v, s in processed_clauses[ci]]
            return conflict
        return None

    # --- WalkSAT with learned clauses ---
    max_restarts = 200
    base_steps = 8000

    best_assignment = None
    best_unsat = float('inf')

    # Build initial assignment respecting unit propagation
    def make_initial_assignment():
        init = [False] * n
        for v in range(n):
            if assignment[v] is not None:
                init[v] = assignment[v]
            else:
                init[v] = rng.random() < 0.5
        return init

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initial assignment respecting unit propagation
        if restart == 0:
            assignment_list = make_initial_assignment()
        else:
            assignment_list = [rng.random() < 0.5 for _ in range(n)]
            # Respect unit propagation
            for v in range(n):
                if assignment[v] is not None:
                    assignment_list[v] = assignment[v]

        clause_sat_count, unsat_count = init_state(assignment_list)
        unsat_clauses = set()
        for ci in range(len(processed_clauses)):
            if clause_sat_count[ci] == 0:
                unsat_clauses.add(ci)

        learned_unsat = check_learned(assignment_list)
        total_unsat = unsat_count + learned_unsat

        if total_unsat == 0:
            return assignment_list

        if total_unsat < best_unsat:
            best_unsat = total_unsat
            best_assignment = assignment_list[:]

        # Adaptive steps: increase with restart number up to a cap
        steps = min(base_steps + restart * 500, 25000)

        # Adaptive noise: start high, decrease
        noise_prob = max(0.1, 0.4 - restart * 0.005)

        for step in range(steps):
            if time.time() > deadline:
                break

            # Find an unsatisfied clause from the set
            if unsat_clauses:
                unsat_ci = rng.choice(list(unsat_clauses))
            else:
                # All original clauses satisfied, check learned
                if learned_unsat == 0:
                    return assignment_list
                # Find an unsatisfied learned clause
                for li, lc in enumerate(learned_clauses):
                    sat = False
                    for var, sign in lc:
                        if sign == 1:
                            if assignment_list[var]:
                                sat = True
                                break
                        else:
                            if not assignment_list[var]:
                                sat = True
                                break
                    if not sat:
                        # Flip a random variable in this learned clause
                        var, sign = rng.choice(lc)
                        unsat_count = flip_var(assignment_list, var, clause_sat_count, unsat_count, unsat_clauses)
                        learned_unsat = check_learned(assignment_list)
                        total_unsat = unsat_count + learned_unsat
                        if total_unsat == 0:
                            return assignment_list
                        if total_unsat < best_unsat:
                            best_unsat = total_unsat
                            best_assignment = assignment_list[:]
                        break
                continue

            # Got an unsatisfied original clause
            clause = processed_clauses[unsat_ci]

            # Choose variable to flip
            if rng.random() < (1.0 - noise_prob):
                # Gain-based: flip variable that maximizes gain
                best_var = None
                best_gain = -999
                for var, sign in clause:
                    # Compute gain: how many clauses' satisfaction changes
                    gain = 0
                    for ci2, sign2 in var_occ[var]:
                        old_sat = clause_sat_count[ci2]
                        if sign2 == 1:
                            was_true = assignment_list[var]
                        else:
                            was_true = not assignment_list[var]
                        if was_true:
                            # Flipping makes this literal false
                            if old_sat == 1:
                                gain -= 1
                        else:
                            # Flipping makes this literal true
                            if old_sat == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                if best_var is not None:
                    unsat_count = flip_var(assignment_list, best_var, clause_sat_count, unsat_count, unsat_clauses)
            else:
                # Random flip
                var, sign = rng.choice(clause)
                unsat_count = flip_var(assignment_list, var, clause_sat_count, unsat_count, unsat_clauses)

            learned_unsat = check_learned(assignment_list)
            total_unsat = unsat_count + learned_unsat

            if total_unsat == 0:
                return assignment_list

            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment_list[:]

        # End of restart: extract conflict clause if we didn't solve
        if total_unsat > 0 and restart > 0:
            conflict = extract_conflict(assignment_list, clause_sat_count, unsat_clauses)
            if conflict is not None:
                # Check if this conflict is already learned (avoid duplicates)
                conflict_set = frozenset(conflict)
                is_dup = False
                for lc in learned_clauses:
                    if frozenset(lc) == conflict_set:
                        is_dup = True
                        break
                if not is_dup:
                    learned_clauses.append(conflict)
                    # Cap learned clauses to prevent bloat
                    if len(learned_clauses) > 500:
                        learned_clauses = learned_clauses[-300:]

    # Final check
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
