import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285

    # Pre-compute variable to clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            var_to_clauses[abs(l) - 1].append(i)

    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')

    def clause_satisfied(clause, assignment):
        for l in clause:
            v = assignment[abs(l) - 1]
            if (l > 0 and v) or (l < 0 and not v):
                return True
        return False

    def count_unsatisfied(assignment):
        cnt = 0
        for c in clauses:
            if not clause_satisfied(c, assignment):
                cnt += 1
        return cnt

    def compute_clause_counts(assignment):
        # For each clause, how many literals are true?
        counts = []
        for c in clauses:
            cnt = 0
            for l in c:
                v = assignment[abs(l) - 1]
                if (l > 0 and v) or (l < 0 and not v):
                    cnt += 1
            counts.append(cnt)
        return counts

    # WalkSAT with adaptive parameters
    def walksat(assignment, max_steps, p_noise, seed):
        local_rng = random.Random(seed)
        unsat_set = set()
        for i, c in enumerate(clauses):
            if not clause_satisfied(c, assignment):
                unsat_set.add(i)

        cur_unsat = len(unsat_set)
        if cur_unsat == 0:
            return assignment, 0

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break
            if not unsat_set:
                return assignment, 0

            c_idx = local_rng.choice(list(unsat_set))
            clause = clauses[c_idx]

            best_var = None
            best_score = float('inf')

            for l in clause:
                var_idx = abs(l) - 1
                broken = 0
                for other_c_idx in var_to_clauses[var_idx]:
                    if other_c_idx == c_idx:
                        continue
                    if other_c_idx in unsat_set:
                        continue
                    other_clause = clauses[other_c_idx]
                    other_satisfied_without_var = False
                    for ol in other_clause:
                        o_var = abs(ol) - 1
                        if o_var == var_idx:
                            continue
                        o_val = assignment[o_var]
                        if (ol > 0 and o_val) or (ol < 0 and not o_val):
                            other_satisfied_without_var = True
                            break
                    if not other_satisfied_without_var:
                        broken += 1

                if broken < best_score:
                    best_score = broken
                    best_var = var_idx

            if local_rng.random() < p_noise:
                l = local_rng.choice(clause)
                var_idx = abs(l) - 1
            else:
                var_idx = best_var

            if var_idx is None:
                continue

            assignment[var_idx] = not assignment[var_idx]

            for c_idx2 in var_to_clauses[var_idx]:
                c = clauses[c_idx2]
                sat = clause_satisfied(c, assignment)
                if sat:
                    unsat_set.discard(c_idx2)
                else:
                    unsat_set.add(c_idx2)

        return assignment, len(unsat_set)

    # Multi-restart WalkSAT with varying noise and step counts
    noise_schedule = [0.3, 0.5, 0.7, 0.4, 0.6]
    steps_schedule = [8000, 12000, 6000, 15000, 10000]

    for restart in range(5000):
        if time.time() > deadline - 5:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_count = count_unsatisfied(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        ni = restart % len(noise_schedule)
        si = restart % len(steps_schedule)
        result, final_unsat = walksat(assignment, steps_schedule[si], noise_schedule[ni], restart * 1000 + 7)
        if final_unsat == 0:
            return result
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = result

    if best_assignment:
        return best_assignment
    return [False] * n
