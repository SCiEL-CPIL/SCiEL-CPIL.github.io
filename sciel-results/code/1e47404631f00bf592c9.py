import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.monotonic() + 285.0

    # Preprocess clauses
    # clause_vars[ci] = list of variable indices (0-based)
    # clause_polarity[ci] = list of polarities (True if positive, False if negative)
    # var_clauses[v] = list of (clause_index, polarity)
    
    clause_vars = []
    clause_polarity = []
    for c in clauses:
        vs = []
        ps = []
        for l in c:
            a = abs(l) - 1
            vs.append(a)
            ps.append(1 if l > 0 else 0)
        clause_vars.append(vs)
        clause_polarity.append(ps)

    var_clauses = [[] for _ in range(n)]
    for ci, (vs, ps) in enumerate(zip(clause_vars, clause_polarity)):
        for v, p in zip(vs, ps):
            var_clauses[v].append((ci, p))

    m = len(clauses)

    def init_search():
        # Random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # unsat_lits[ci] = number of unsatisfied literals in clause ci
        unsat_lits = [0] * m
        # conflict_count[v] = number of unsatisfied clauses containing v
        conflict_count = [0] * n
        num_unsat = 0

        for ci in range(m):
            unsat_count = 0
            for v, p in zip(clause_vars[ci], clause_polarity[ci]):
                if assignment[v] != bool(p):
                    unsat_count += 1
            unsat_lits[ci] = unsat_count
            if unsat_count == 0:
                num_unsat += 1
                # All variables in this clause are in conflict
                for v in clause_vars[ci]:
                    conflict_count[v] += 1

        return assignment, unsat_lits, conflict_count, num_unsat

    best_assignment = None
    best_unsat = float('inf')

    def record(assignment, num_unsat):
        nonlocal best_assignment, best_unsat
        if num_unsat < best_unsat:
            best_unsat = num_unsat
            best_assignment = assignment[:]

    max_restarts = 1000
    steps_per_restart = 5000
    stall_limit = 500

    for restart in range(max_restarts):
        if time.monotonic() > deadline:
            break
        
        assignment, unsat_lits, conflict_count, num_unsat = init_search()
        record(assignment, num_unsat)
        
        if num_unsat == 0:
            return assignment

        steps = 0
        stall_counter = 0
        min_unsat_so_far = num_unsat

        while steps < steps_per_restart:
            steps += 1
            if steps % 1024 == 0 and time.monotonic() > deadline:
                record(assignment, num_unsat)
                break

            if num_unsat == 0:
                record(assignment, 0)
                return assignment

            # Find variable with min conflict count
            min_conflicts = min(conflict_count)
            # Collect all vars with min conflicts
            candidates = [v for v in range(n) if conflict_count[v] == min_conflicts]
            v = rng.choice(candidates)
            
            # Flip v
            old_val = assignment[v]
            new_val = not old_val
            assignment[v] = new_val

            # Update unsat_lits and conflict_count
            for (ci, p) in var_clauses[v]:
                old_unsat = unsat_lits[ci]
                
                # Determine if literal v in clause ci was satisfied
                # Satisfied if assignment[v] == bool(p)
                # Before flip: old_val
                lit_was_sat = (old_val == bool(p))
                
                if lit_was_sat:
                    # Literal was satisfied, now it is not.
                    new_unsat = old_unsat + 1
                else:
                    # Literal was not satisfied, now it is.
                    new_unsat = old_unsat - 1
                
                unsat_lits[ci] = new_unsat
                
                # Update conflict counts
                if old_unsat == 0 and new_unsat > 0:
                    # Clause was unsat, now sat.
                    # Remove conflict for all vars in this clause
                    num_unsat -= 1
                    for u in clause_vars[ci]:
                        conflict_count[u] -= 1
                elif old_unsat > 0 and new_unsat == 0:
                    # Clause was sat, now unsat.
                    # Add conflict for all vars in this clause
                    num_unsat += 1
                    for u in clause_vars[ci]:
                        conflict_count[u] += 1
                # Else: no change in clause sat/unsat status

            record(assignment, num_unsat)

            # Stall detection
            if num_unsat < min_unsat_so_far:
                min_unsat_so_far = num_unsat
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter > stall_limit:
                    break

        if time.monotonic() > deadline:
            break

    if best_assignment is None:
        # Fallback: return last assignment if any, or random
        return [False] * n
    return best_assignment
