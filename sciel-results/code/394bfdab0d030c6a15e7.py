"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)

    # Deadline: 300s wall-clock limit, use 295s for safety margin
    deadline = time.time() + 295.0

    # Precompute clause literal indices (0-based) and sign
    # clause_info[i] = list of (var_idx, is_positive)
    clause_info = []
    for c in clauses:
        clause_info.append([(abs(l) - 1, l > 0) for l in c])

    # For each variable, list of (clause_idx, literal_in_clause) where it appears
    var_occurrences = [[] for _ in range(n)]
    for ci, cl in enumerate(clause_info):
        for li, (v, _) in enumerate(cl):
            var_occurrences[v].append((ci, li))

    def check_satisfied(assignment):
        for cl in clause_info:
            sat = False
            for v, pos in cl:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def solve_one_restart(rng, assignment, max_steps):
        """Run WalkSAT+DLS from given assignment. Returns (success, assignment)."""
        # Build unsatisfied clause set
        unsat_clauses = set()
        for ci, cl in enumerate(clause_info):
            sat = False
            for v, pos in cl:
                if assignment[v] == pos:
                    sat = True
                    break
            if not sat:
                unsat_clauses.add(ci)

        if not unsat_clauses:
            return True, assignment

        for step in range(max_steps):
            if not unsat_clauses:
                return True, assignment

            # DLS: pick the longest unsatisfied clause (ties broken randomly)
            # To avoid O(n) max each step, use a bucket approach or just sample
            # For simplicity with 150 vars, we can find max length unsat clause
            # But that's O(num_clauses) per step. Let's use a different approach:
            # Pick random unsat clause, then within it pick variable to flip
            # using a weighted scheme favoring longer clauses.
            
            # Actually, let's use a proper DLS: maintain max unsat clause length
            # We'll just pick a random unsat clause for speed, but with DLS weighting
            # via picking from a few candidates.
            
            # Simpler effective approach: pick random unsat clause, then flip
            # the variable that appears in the most unsatisfied clauses (most constrained)
            clause_idx = rng.choice(list(unsat_clauses))
            cl = clause_info[clause_idx]
            
            # Count how many unsat clauses each variable in this clause appears in
            # This is the "most constrained" heuristic
            best_var = -1
            best_count = -1
            for v, pos in cl:
                count = 0
                for ci2, _ in var_occurrences[v]:
                    if ci2 in unsat_clauses:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_var = v

            if best_var < 0:
                # Fallback: random
                v, pos = rng.choice(cl)
                best_var = v

            # Flip the variable
            assignment[best_var] = not assignment[best_var]

            # Update unsat_clauses: check clauses containing best_var
            # First, clauses that were unsat and might become sat
            for ci2, li in var_occurrences[best_var]:
                if ci2 in unsat_clauses:
                    # Check if now satisfied
                    cl2 = clause_info[ci2]
                    sat = False
                    for v2, pos2 in cl2:
                        if assignment[v2] == pos2:
                            sat = True
                            break
                    if sat:
                        unsat_clauses.discard(ci2)
                    # else still unsat
                else:
                    # Was sat, check if now unsat
                    cl2 = clause_info[ci2]
                    sat = False
                    for v2, pos2 in cl2:
                        if assignment[v2] == pos2:
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.add(ci2)

            # Deadline check
            if time.time() > deadline:
                return False, assignment

        return False, assignment

    # Multi-restart strategy
    rng = random.Random(0)
    
    # Start with a random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # First try: longer budget
    max_steps_first = 100000
    success, assignment = solve_one_restart(rng, assignment, max_steps_first)
    if success:
        return assignment

    # If not solved, try multiple restarts with shorter budgets
    # Increase restart count as time permits
    restart_budget = 50000
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        assignment = [rng.random() < 0.5 for _ in range(n)]
        success, assignment = solve_one_restart(rng, assignment, restart_budget)
        if success:
            return assignment
        # Gradually reduce budget for later restarts to fit more
        restart_budget = max(10000, int(restart_budget * 0.95))

    # Final check before deadline
    if time.time() < deadline - 1.0:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        success, assignment = solve_one_restart(rng, assignment, 20000)
        if success:
            return assignment

    # Return best effort (last assignment)
    return assignment
# EVOLVE-BLOCK-END
