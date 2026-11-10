"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    # Preprocess: convert to 0-based variable indices and signs
    clause_vars = []
    clause_sigs = []
    for c in clauses:
        cv = [abs(l) - 1 for l in c]
        cs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(cv)
        clause_sigs.append(cs)

    # Build var -> list of (clause_idx, pos_in_clause) for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append((i, j))

    rng = random.Random(42)

    def preprocess(assignment):
        """Apply unit propagation and pure literal elimination. Returns (assignment, changed, contradiction)."""
        changed = False

        # Pure literal elimination
        for v in range(n):
            has_pos = False
            has_neg = False
            for (ci, cj) in var_to_clauses[v]:
                if clause_sigs[ci][cj] == 1:
                    has_pos = True
                else:
                    has_neg = True
            if has_pos and not has_neg:
                if not assignment[v]:
                    assignment[v] = True
                    changed = True
            elif has_neg and not has_pos:
                if assignment[v]:
                    assignment[v] = False
                    changed = True

        # Unit propagation: for each clause, if 2 literals are satisfied, force the 3rd
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            sat_count = 0
            unsat_var = -1
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    sat_count += 1
                else:
                    unsat_var = cv[j]
            if sat_count == 2:
                target_val = cs[unsat_var] > 0
                if assignment[unsat_var] != target_val:
                    assignment[unsat_var] = target_val
                    changed = True

        return assignment, changed, False

    def make_random_assignment():
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, changed, _ = preprocess(assignment)
        return assignment

    def walk_sat_run(assignment, max_flips=10000, p=0.55):
        """WalkSAT with optimized list-based unsat clause tracking."""

        # Initialize clause satisfaction status
        clause_sat_count = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        # Initialize unsat_list and unsat_pos for O(1) selection and updates
        unsat_list = []
        unsat_pos = [-1] * m  # -1 if not in unsat_list, else index in unsat_list

        for i in range(m):
            if clause_sat_count[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)

        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None

            if len(unsat_list) == 0:
                return assignment[:]

            # Pick a random unsatisfied clause in O(1)
            idx = rng.randint(0, len(unsat_list) - 1)
            ci = unsat_list[idx]

            # Choose literal to flip
            if rng.random() < p:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Weighted: prefer literals that satisfy more other unsat clauses
                best_li = 0
                best_weight = -1
                for li in range(3):
                    var = clause_vars[ci][li]
                    # Quick count: how many clauses containing var are unsat
                    weight = 0
                    for (oi, oj) in var_to_clauses[var]:
                        if unsat_pos[oi] != -1:
                            weight += 1
                    if weight > best_weight:
                        best_weight = weight
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count and unsat_list for all clauses containing this var
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))

                if was_sat and not now_sat:
                    # Clause became unsat
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        # Add to unsat_list
                        unsat_pos[oi] = len(unsat_list)
                        unsat_list.append(oi)
                elif not was_sat and now_sat:
                    # Clause became sat
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] > 0:
                        # Remove from unsat_list if it was there
                        if unsat_pos[oi] != -1:
                            pos = unsat_pos[oi]
                            last_idx = len(unsat_list) - 1
                            last_clause = unsat_list[last_idx]

                            # Swap with last
                            unsat_list[pos] = last_clause
                            unsat_pos[last_clause] = pos

                            # Pop last
                            unsat_list.pop()
                            unsat_pos[oi] = -1

        # Return None if not solved
        if len(unsat_list) == 0:
            return assignment[:]
        return None

    best_assignment = None
    best_unsat = m + 1

    # Adaptive restart strategy with shorter walks and more restarts
    # For 50-var 3-SAT, optimal p ~ 0.5-0.6, walk length ~ 1000-5000
    max_restarts = 2000

    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        assignment = make_random_assignment()

        # Vary parameters to explore different regions
        # Use a mix of short and long walks
        if restart < 500:
            # Phase 1: short walks, quick exploration
            p = 0.55
            flips = 1000 + (restart % 10) * 200
        elif restart < 1500:
            # Phase 2: medium walks
            p = 0.58
            flips = 3000 + (restart % 10) * 500
        else:
            # Phase 3: longer walks for harder instances
            p = 0.60
            flips = 5000 + (restart % 10) * 1000

        result = walk_sat_run(assignment, max_flips=flips, p=p)
        if result is not None:
            return result

        # Track best (minimum unsat count) for fallback
        unsat_count = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            sat = False
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    sat = True
                    break
            if not sat:
                unsat_count += 1

        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
