"""Optimized WalkSAT with dynamic noise and efficient clause updates for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time

DEADLINE_FRACTION = 0.95

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # Precomputation
    var_clauses = [[] for _ in range(n)]
    clause_vars = []
    
    for ci, clause in enumerate(clauses):
        cvars = []
        for lit in clause:
            v = abs(lit) - 1
            pos = lit > 0
            cvars.append((v, pos))
            var_clauses[v].append(ci)
        clause_vars.append(cvars)

    def is_clause_satisfied(ci, assignment):
        cvars = clause_vars[ci]
        for v, pos in cvars:
            if assignment[v] == pos:
                return True
        return False

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    max_restarts = 5000 # Increased restarts due to speed
    max_steps_per_restart = 5000
    base_noise = 0.2
    max_noise = 0.8
    stagnation_limit = 50

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_set = set()
        unsat_list = []
        for ci in range(m):
            if not is_clause_satisfied(ci, assignment):
                unsat_set.add(ci)
                unsat_list.append(ci)
                
        cur_unsat = len(unsat_set)
        
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]
            
        if cur_unsat == 0:
            return assignment

        noise = base_noise
        stagnation = 0
        prev_unsat = cur_unsat

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_set:
                return assignment

            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]

            if cur_unsat < prev_unsat:
                stagnation = 0
                noise = max(base_noise, noise - 0.02)
            else:
                stagnation += 1
                if stagnation > stagnation_limit:
                    noise = min(max_noise, noise + 0.05)
                    stagnation = 0

            prev_unsat = cur_unsat

            if rng.random() < noise:
                v = rng.randint(0, n - 1)
            else:
                # Pick random unsatisfied clause
                # Handle lazy deletion in unsat_list
                c_idx = None
                while True:
                    if not unsat_list:
                        # Should be empty if unsat_set is empty, but check just in case
                        if not unsat_set:
                            break
                        else:
                            # Rebuild list from set if list is empty but set is not (shouldn't happen)
                            unsat_list = list(unsat_set)
                            continue
                    idx = rng.randint(0, len(unsat_list) - 1)
                    ci = unsat_list[idx]
                    if ci in unsat_set:
                        c_idx = ci
                        # Remove from list to avoid re-picking? No, we can keep it, but it's better to remove to keep list small.
                        # Actually, we don't remove it from the list here, we just use it.
                        # We will update the list after the flip.
                        break
                    else:
                        # Stale entry
                        unsat_list[idx] = unsat_list[-1]
                        unsat_list.pop()
                        continue

                if c_idx is None:
                    break # No unsat clauses?

                cvars = clause_vars[c_idx]
                best_v = cvars[0][0]
                best_score = -10000
                
                for v, pos in cvars:
                    repaired = 1 # The chosen clause c_idx is repaired
                    damaged = 0
                    
                    for ci in var_clauses[v]:
                        if ci == c_idx:
                            continue
                        
                        # Check current status
                        cur_sat = is_clause_satisfied(ci, assignment)
                        
                        # Check if v is the only satisfier if cur_sat
                        # Or if v will be the satisfier if not cur_sat
                        
                        # Find v's literal in ci
                        v_pos_in_ci = None
                        for vv, pp in clause_vars[ci]:
                            if vv == v:
                                v_pos_in_ci = pp
                                break
                                
                        if v_pos_in_ci is None:
                            continue
                            
                        new_v_val = not assignment[v]
                        v_satisfies_after = (new_v_val == v_pos_in_ci)
                        
                        if cur_sat:
                            # Check if it relies on v
                            relies_on_v = True
                            for vv, pp in clause_vars[ci]:
                                if vv != v:
                                    if assignment[vv] == pp:
                                        relies_on_v = False
                                        break
                            if relies_on_v:
                                # It will become unsat
                                damaged += 1
                        else:
                            # It is unsat. Will it become sat?
                            if v_satisfies_after:
                                repaired += 1
                            
                    score = repaired - damaged
                    if score > best_score:
                        best_score = score
                        best_v = v
                        
                v = best_v

            # Flip v
            assignment[v] = not assignment[v]
            
            # Update unsat_set and unsat_list
            for ci in var_clauses[v]:
                was_unsat = ci in unsat_set
                is_unsat_now = not is_clause_satisfied(ci, assignment)
                
                if was_unsat and not is_unsat_now:
                    unsat_set.discard(ci)
                    # Leave in list, will be cleaned up
                elif not was_unsat and is_unsat_now:
                    unsat_set.add(ci)
                    unsat_list.append(ci)
                    
            cur_unsat = len(unsat_set)

    if best_unsat_count == 0:
        return best_assignment
        
    return best_assignment if best_assignment else [False]*n
# EVOLVE-BLOCK-END
