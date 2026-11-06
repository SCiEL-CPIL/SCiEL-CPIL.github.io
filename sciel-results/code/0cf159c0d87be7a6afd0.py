"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0

    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)

    # Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append(i)

    # Build literal-to-clause index for faster lookup
    # lit_clauses[var_idx][0] = clauses where var is positive
    # lit_clauses[var_idx][1] = clauses where var is negative
    lit_clauses = [[[], []] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            lit_clauses[var_idx][0 if is_pos else 1].append(i)

    def init_sat_counts(assignment):
        sat_counts = [0] * m
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_counts[i] = cnt
        return sat_counts

    def count_unsat_from_counts(sat_counts):
        return sum(1 for c in sat_counts if c == 0)

    def walk_sat(assignment, rng, max_flips):
        """WalkSAT with adaptive noise and incremental clause tracking."""
        sat_counts = init_sat_counts(assignment)
        unsat_count = count_unsat_from_counts(sat_counts)
        if unsat_count == 0:
            return True

        unsat_clauses = [i for i, c in enumerate(sat_counts) if c == 0]
        if not unsat_clauses:
            return True

        flip_count = 0
        while flip_count < max_flips:
            if not unsat_clauses:
                return True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]

            # Adaptive noise: decrease over time
            noise = 0.5 * max(0.0, 1.0 - flip_count / max_flips)

            best_var = -1
            best_delta = float("inf")

            # For each literal in the chosen unsat clause, compute delta in unsat count
            for var_idx, is_pos in cv:
                delta = 0

                # Clauses where var_idx appears positively (flipping from False->True satisfies, True->False unsatisfies)
                # Clauses where var_idx appears negatively (flipping from True->False satisfies, False->True unsatisfies)
                
                # If assignment[var_idx] is currently True:
                #   Flipping to False:
                #     - Clauses with var_idx positive: lose satisfaction (delta += count of cnt==1)
                #     - Clauses with var_idx negative: gain satisfaction (delta -= count of cnt==0)
                # If assignment[var_idx] is currently False:
                #   Flipping to True:
                #     - Clauses with var_idx positive: gain satisfaction (delta -= count of cnt==0)
                #     - Clauses with var_idx negative: lose satisfaction (delta += count of cnt==1)
                
                if assignment[var_idx]:
                    # Currently True, flipping to False
                    for ci2 in lit_clauses[var_idx][0]:  # positive occurrences
                        cnt = sat_counts[ci2]
                        if cnt == 1:
                            delta += 1
                    for ci2 in lit_clauses[var_idx][1]:  # negative occurrences
                        cnt = sat_counts[ci2]
                        if cnt == 0:
                            delta -= 1
                else:
                    # Currently False, flipping to True
                    for ci2 in lit_clauses[var_idx][0]:  # positive occurrences
                        cnt = sat_counts[ci2]
                        if cnt == 0:
                            delta -= 1
                    for ci2 in lit_clauses[var_idx][1]:  # negative occurrences
                        cnt = sat_counts[ci2]
                        if cnt == 1:
                            delta += 1

                # Apply noise
                delta += rng.gauss(0.0, noise)

                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx

            if best_var < 0:
                # Fallback: random flip within the clause
                best_var = cv[rng.randrange(len(cv))][0]

            # Flip variable and update sat_counts incrementally
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update sat_counts for all clauses containing best_var
            if old_val:
                # Was True, now False
                for ci2 in lit_clauses[best_var][0]:  # positive: was satisfying, now not
                    sat_counts[ci2] -= 1
                for ci2 in lit_clauses[best_var][1]:  # negative: was not satisfying, now is
                    sat_counts[ci2] += 1
            else:
                # Was False, now True
                for ci2 in lit_clauses[best_var][0]:  # positive: was not satisfying, now is
                    sat_counts[ci2] += 1
                for ci2 in lit_clauses[best_var][1]:  # negative: was satisfying, now not
                    sat_counts[ci2] -= 1

            flip_count += 1

            # Rebuild unsat list periodically
            if flip_count % 8 == 0:
                new_unsat = [i for i, c in enumerate(sat_counts) if c == 0]
                unsat_clauses = new_unsat
                if not unsat_clauses:
                    return True

        # Final check
        return all(c > 0 for c in sat_counts)

    def dpll_solve(clauses_idx, assignment, phase, time_budget):
        """
        DPLL with unit propagation and pure literal elimination.
        clauses_idx: list of clause indices to consider
        assignment: list of int (-1 unassigned, 0 False, 1 True)
        phase: list of bool (preferred polarity for each variable)
        time_budget: seconds to spend
        Returns: (success, full_assignment) or (False, None)
        """
        start_time = time.time()
        
        # Work with a copy
        assign = assignment[:]
        # Active clauses: track which clauses are still active
        # We'll use a worklist approach
        
        # Build initial active clause set
        active_clauses = set(clauses_idx)
        
        # Watched literals: for each variable, track which clauses watch it
        # Simplified: just rebuild each time (for 150 vars this is fine)
        
        # Unit propagation queue
        unit_queue = deque()
        
        # Initialize: find unit clauses
        for ci in active_clauses:
            cv = clause_vars[ci]
            unassigned = []
            for var_idx, is_pos in cv:
                if assign[var_idx] == -1:
                    unassigned.append((var_idx, is_pos))
            if len(unassigned) == 1:
                var_idx, is_pos = unassigned[0]
                unit_queue.append((var_idx, is_pos))
            elif len(unassigned) == 0:
                # Check if satisfied
                sat = False
                for var_idx, is_pos in cv:
                    if assign[var_idx] == (1 if is_pos else 0):
                        sat = True
                        break
                if not sat:
                    return False, None
        
        while unit_queue:
            if time.time() - start_time > time_budget:
                return False, None
            
            var_idx, is_pos = unit_queue.popleft()
            if assign[var_idx] != -1:
                continue
            
            # Assign
            assign[var_idx] = 1 if is_pos else 0
            phase[var_idx] = is_pos  # Update phase
            
            # Propagate: check all clauses containing var_idx
            for ci in var_clauses[var_idx]:
                if ci not in active_clauses:
                    continue
                cv = clause_vars[ci]
                
                # Check if clause is now satisfied
                sat = False
                unassigned = []
                for v, p in cv:
                    if assign[v] != -1:
                        if assign[v] == (1 if p else 0):
                            sat = True
                            break
                    else:
                        unassigned.append((v, p))
                
                if sat:
                    active_clauses.discard(ci)
                elif len(unassigned) == 0:
                    # Conflict
                    return False, None
                elif len(unassigned) == 1:
                    unit_queue.append(unassigned[0])
        
        # Pure literal elimination
        changed = True
        while changed and time.time() - start_time <= time_budget:
            changed = False
            for var_idx in range(n):
                if assign[var_idx] != -1:
                    continue
                # Check if var appears only with one polarity in active clauses
                has_pos = False
                has_neg = False
                for ci in lit_clauses[var_idx][0]:
                    if ci in active_clauses:
                        has_pos = True
                        break
                if has_pos:
                    for ci in lit_clauses[var_idx][1]:
                        if ci in active_clauses:
                            has_neg = True
                            break
                if has_pos and not has_neg:
                    assign[var_idx] = 1
                    phase[var_idx] = True
                    changed = True
                    # Propagate
                    for ci in lit_clauses[var_idx][0]:
                        if ci in active_clauses:
                            active_clauses.discard(ci)
                    for ci in lit_clauses[var_idx][1]:
                        if ci not in active_clauses:
                            continue
                        cv = clause_vars[ci]
                        unassigned = []
                        sat = False
                        for v, p in cv:
                            if assign[v] != -1:
                                if assign[v] == (1 if p else 0):
                                    sat = True
                                    break
                            else:
                                unassigned.append((v, p))
                        if sat:
                            active_clauses.discard(ci)
                        elif len(unassigned) == 0:
                            return False, None
                        elif len(unassigned) == 1:
                            unit_queue.append(unassigned[0])
                elif has_neg and not has_pos:
                    assign[var_idx] = 0
                    phase[var_idx] = False
                    changed = True
                    # Propagate
                    for ci in lit_clauses[var_idx][1]:
                        if ci in active_clauses:
                            active_clauses.discard(ci)
                    for ci in lit_clauses[var_idx][0]:
                        if ci not in active_clauses:
                            continue
                        cv = clause_vars[ci]
                        unassigned = []
                        sat = False
                        for v, p in cv:
                            if assign[v] != -1:
                                if assign[v] == (1 if p else 0):
                                    sat = True
                                    break
                            else:
                                unassigned.append((v, p))
                        if sat:
                            active_clauses.discard(ci)
                        elif len(unassigned) == 0:
                            return False, None
                        elif len(unassigned) == 1:
                            unit_queue.append(unassigned[0])
            
            # Process any new unit clauses
            while unit_queue:
                if time.time() - start_time > time_budget:
                    return False, None
                var_idx, is_pos = unit_queue.popleft()
                if assign[var_idx] != -1:
                    continue
                assign[var_idx] = 1 if is_pos else 0
                phase[var_idx] = is_pos
                for ci in var_clauses[var_idx]:
                    if ci not in active_clauses:
                        continue
                    cv = clause_vars[ci]
                    sat = False
                    unassigned = []
                    for v, p in cv:
                        if assign[v] != -1:
                            if assign[v] == (1 if p else 0):
                                sat = True
                                break
                        else:
                            unassigned.append((v, p))
                    if sat:
                        active_clauses.discard(ci)
                    elif len(unassigned) == 0:
                        return False, None
                    elif len(unassigned) == 1:
                        unit_queue.append(unassigned[0])
        
        # Check if all variables are assigned
        all_assigned = all(assign[i] != -1 for i in range(n))
        if all_assigned:
            # Verify
            for ci in range(m):
                cv = clause_vars[ci]
                sat = False
                for var_idx, is_pos in cv:
                    if assign[var_idx] == (1 if is_pos else 0):
                        sat = True
                        break
                if not sat:
                    return False, None
            return True, assign
        
        # If not all assigned, do a simple branch
        # Find an unassigned variable
        branch_var = -1
        for i in range(n):
            if assign[i] == -1:
                branch_var = i
                break
        
        if branch_var == -1:
            return True, assign
        
        # Try preferred polarity first
        preferred = phase[branch_var]
        
        # Try True
        assign_copy = assign[:]
        phase_copy = phase[:]
        active_copy = set(active_clauses)
        
        # Set branch_var to True
        assign_copy[branch_var] = 1
        phase_copy[branch_var] = True
        
        # Propagate
        unit_q = deque()
        for ci in active_copy:
            cv = clause_vars[ci]
            unassigned = []
            for v, p in cv:
                if assign_copy[v] == -1:
                    unassigned.append((v, p))
            if len(unassigned) == 1:
                unit_q.append(unassigned[0])
            elif len(unassigned) == 0:
                sat = False
                for v, p in cv:
                    if assign_copy[v] == (1 if p else 0):
                        sat = True
                        break
                if not sat:
                    # Conflict, try False
                    break
        
        conflict = False
        while unit_q and not conflict:
            var_idx, is_pos = unit_q.popleft()
            if assign_copy[var_idx] != -1:
                continue
            assign_copy[var_idx] = 1 if is_pos else 0
            phase_copy[var_idx] = is_pos
            for ci in var_clauses[var_idx]:
                if ci not in active_copy:
                    continue
                cv = clause_vars[ci]
                sat = False
                unassigned = []
                for v, p in cv:
                    if assign_copy[v] != -1:
                        if assign_copy[v] == (1 if p else 0):
                            sat = True
                            break
                    else:
                        unassigned.append((v, p))
                if sat:
                    active_copy.discard(ci)
                elif len(unassigned) == 0:
                    conflict = True
                    break
                elif len(unassigned) == 1:
                    unit_q.append(unassigned[0])
        
        if not conflict:
            # Check if all assigned
            if all(assign_copy[i] != -1 for i in range(n)):
                # Verify
                for ci in range(m):
                    cv = clause_vars[ci]
                    sat = False
                    for var_idx, is_pos in cv:
                        if assign_copy[var_idx] == (1 if is_pos else 0):
                            sat = True
                            break
                    if not sat:
                        conflict = True
                        break
        
        if not conflict:
            return True, assign_copy
        
        # Try False
        assign_copy = assign[:]
        phase_copy = phase[:]
        active_copy = set(active_clauses)
        
        assign_copy[branch_var] = 0
        phase_copy[branch_var] = False
        
        unit_q = deque()
        for ci in active_copy:
            cv = clause_vars[ci]
            unassigned = []
            for v, p in cv:
                if assign_copy[v] == -1:
                    unassigned.append((v, p))
            if len(unassigned) == 1:
                unit_q.append(unassigned[0])
            elif len(unassigned) == 0:
                sat = False
                for v, p in cv:
                    if assign_copy[v] == (1 if p else 0):
                        sat = True
                        break
                if not sat:
                    return False, None
        
        conflict = False
        while unit_q and not conflict:
            var_idx, is_pos = unit_q.popleft()
            if assign_copy[var_idx] != -1:
                continue
            assign_copy[var_idx] = 1 if is_pos else 0
            phase_copy[var_idx] = is_pos
            for ci in var_clauses[var_idx]:
                if ci not in active_copy:
                    continue
                cv = clause_vars[ci]
                sat = False
                unassigned = []
                for v, p in cv:
                    if assign_copy[v] != -1:
                        if assign_copy[v] == (1 if p else 0):
                            sat = True
                            break
                    else:
                        unassigned.append((v, p))
                if sat:
                    active_copy.discard(ci)
                elif len(unassigned) == 0:
                    conflict = True
                    break
                elif len(unassigned) == 1:
                    unit_q.append(unassigned[0])
        
        if not conflict:
            if all(assign_copy[i] != -1 for i in range(n)):
                for ci in range(m):
                    cv = clause_vars[ci]
                    sat = False
                    for var_idx, is_pos in cv:
                        if assign_copy[var_idx] == (1 if is_pos else 0):
                            sat = True
                            break
                    if not sat:
                        conflict = True
                        break
        
        if not conflict:
            return True, assign_copy
        
        return False, None

    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float("inf")
    best_phase = None

    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Max flips per restart, increasing slightly
        max_flips = 5000 + restart * 500

        if walk_sat(assignment, rng, max_flips):
            return assignment

        # Compute unsat count for bookkeeping
        sat_counts = init_sat_counts(assignment)
        unsat = count_unsat_from_counts(sat_counts)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
            best_phase = assignment[:]

        # If we have a good phase, try DPLL
        if best_phase is not None and restart % 3 == 2:
            # Convert bool assignment to int assignment
            int_assign = [-1] * n
            for i in range(n):
                if best_phase[i]:
                    int_assign[i] = 1
                else:
                    int_assign[i] = 0
            
            phase = best_phase[:]
            
            # Give DPLL a time budget
            remaining = deadline - time.time()
            dpll_budget = min(10.0, remaining * 0.3)
            
            if dpll_budget > 1.0:
                success, result = dpll_solve(list(range(m)), int_assign, phase, dpll_budget)
                if success and result is not None:
                    return [bool(x) for x in result]

        restart += 1

    if best_assignment is not None:
        return best_assignment

    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
