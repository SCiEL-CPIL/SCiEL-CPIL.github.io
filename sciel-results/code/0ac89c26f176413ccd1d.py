import random
import time

# EVOLVE-BLOCK-START

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    # Precompute per clause: list of (var_index0, is_positive)
    clause_info = []
    for c in clauses:
        info = [(abs(l) - 1, l > 0) for l in c]
        clause_info.append(info)
    m = len(clause_info)

    # Build variable -> list of (clause_idx, is_positive_in_clause)
    var_clauses = [[] for _ in range(n)]
    for ci, info in enumerate(clause_info):
        for (v, pos) in info:
            var_clauses[v].append((ci, pos))

    # Deadline guard
    t0 = time.time()
    deadline = t0 + 300.0 - 20.0  # 20s safety margin

    rng = random.Random(12345)

    best_mask = None
    best_unsat = None

    # Restart schedule
    restart_step_budgets = [2000, 4000, 8000, 16000, 32000, 64000]
    total_restarts = 0

    while True:
        if time.time() >= deadline:
            break
        remaining_budget = restart_step_budgets[min(total_restarts, len(restart_step_budgets) - 1)]
        if remaining_budget > 100000:
            remaining_budget = 100000
        total_restarts += 1

        # Fresh random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        mask = 0
        for i, b in enumerate(assignment):
            if b:
                mask |= (1 << i)

        # Initialize clause_sat: bool per clause, and var_conflicts
        clause_sat = [False] * m
        var_conflicts = [0] * n
        unsat_count = 0
        for ci, info in enumerate(clause_info):
            sat = False
            for (v, pos) in info:
                if (mask >> v) & 1 == pos:
                    sat = True
                    break
            clause_sat[ci] = sat
            if not sat:
                unsat_count += 1
                for (v, pos) in info:
                    var_conflicts[v] += 1

        if unsat_count == 0:
            return [(mask >> i) & 1 == 1 for i in range(n)]

        if best_mask is None or unsat_count < best_unsat:
            best_mask = mask
            best_unsat = unsat_count

        steps = 0
        max_steps = remaining_budget
        while steps < max_steps:
            if time.time() >= deadline:
                break
            if unsat_count == 0:
                return [(mask >> i) & 1 == 1 for i in range(n)]

            # Pick a random unsatisfied clause
            # To avoid scanning all m each time, keep a list of unsat clause indices
            # For simplicity, scan from a random start
            start = rng.randrange(m)
            clause_idx = None
            for k in range(m):
                idx = (start + k) % m
                if not clause_sat[idx]:
                    clause_idx = idx
                    break

            if clause_idx is None:
                break

            info = clause_info[clause_idx]
            # Choose variable in this clause to flip: prefer lower conflict count + random tie-break
            best_var = None
            best_key = None
            for (v, pos) in info:
                key = var_conflicts[v] + rng.random() * 0.5
                if best_key is None or key < best_key:
                    best_key = key
                    best_var = v
                    best_pos = pos

            v = best_var
            pos = best_pos
            old_val = (mask >> v) & 1 == 1

            # If flipping v doesn't change its value, pick a different variable or just continue
            if old_val == pos:
                # Flipping would set it to the same value; pick another literal in clause
                # Try each literal, pick the one that actually changes
                changed = False
                for (v2, pos2) in info:
                    old2 = (mask >> v2) & 1 == 1
                    if old2 != pos2:
                        v = v2
                        pos = pos2
                        changed = True
                        break
                if not changed:
                    # All literals already satisfied their target? That means clause is sat, contradiction
                    break

            # Flip variable v
            mask ^= (1 << v)
            new_val = (mask >> v) & 1 == 1

            # Update clause_sat and var_conflicts for all clauses containing v
            for (ci, pos_in_clause) in var_clauses[v]:
                old_sat = clause_sat[ci]
                # Determine new sat status
                info_c = clause_info[ci]
                new_sat = False
                for (vv, pp) in info_c:
                    if vv == v:
                        if new_val == pp:
                            new_sat = True
                            break
                    else:
                        if (mask >> vv) & 1 == pp:
                            new_sat = True
                            break
                if old_sat and not new_sat:
                    # Clause became unsat
                    unsat_count += 1
                    for (vv, pp) in info_c:
                        var_conflicts[vv] += 1
                elif not old_sat and new_sat:
                    # Clause became sat
                    unsat_count -= 1
                    for (vv, pp) in info_c:
                        var_conflicts[vv] -= 1
                clause_sat[ci] = new_sat

            if unsat_count == 0:
                return [(mask >> i) & 1 == 1 for i in range(n)]

            if best_mask is None or unsat_count < best_unsat:
                best_mask = mask
                best_unsat = unsat_count

            steps += 1

    # Return best assignment found
    if best_mask is None:
        # Fallback
        return [(mask >> i) & 1 == 1 for i in range(n)]
    return [(best_mask >> i) & 1 == 1 for i in range(n)]

# EVOLVE-BLOCK-END
