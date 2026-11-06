"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Preprocess
    clause_vars = []
    for c in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in c])
    m = len(clause_vars)
    
    # Variable to clauses index
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append(ci)
            
    # Precompute literal signs for fast damage check? 
    # We store (clause_id, is_pos_in_clause) in var_to_clauses for efficiency?
    # Let's stick to clause_id and lookup sign from clause_vars if needed, 
    # or store sign. Storing sign is faster.
    var_to_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_to_clauses[var_idx].append((ci, is_pos))

    start_time = time.time()
    deadline = start_time + 290.0
    
    def is_satisfied(ci, assignment):
        for var_idx, is_pos in clause_vars[ci]:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def count_unsat(assignment):
        return sum(1 for i in range(m) if not is_satisfied(i, assignment))

    def random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def walksat(assignment, max_steps, noise):
        unsat_set = set()
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment, True
            
        for _ in range(max_steps):
            if time.time() > deadline:
                return assignment, False
                
            if not unsat_set:
                return assignment, True
                
            # Pick random unsat clause
            # Using list(unsat_set) is O(U). For small U it's fine.
            # To avoid O(U) list creation every step, we can maintain a list and swap-remove.
            # But for simplicity and speed, let's just pick from list if U is small.
            # Actually, maintaining a list of unsat clauses is faster than set for random access.
            # Let's use a list for unsat clauses.
            
            # Re-initializing list from set is expensive. 
            # Let's keep unsat_list in sync.
            # This implementation below uses a list.
            pass 
            
        return assignment, False

    # Optimized WalkSAT with list-based unsat tracking
    def fast_walksat(assignment, max_steps, noise):
        unsat_list = []
        is_unsat = [False] * m
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_list.append(i)
                is_unsat[i] = True
                
        if not unsat_list:
            return assignment, True
            
        for _ in range(max_steps):
            if time.time() > deadline:
                return assignment, False
                
            if not unsat_list:
                return assignment, True
                
            # Pick random unsat clause
            ci = rng.choice(unsat_list)
            clause = clause_vars[ci]
            
            # Choose variable to flip
            if rng.random() < noise:
                var_idx, is_pos = rng.choice(clause)
                new_val = is_pos
            else:
                best_var = -1
                best_damage = 10**9
                best_val = False
                for var_idx, is_pos in clause:
                    new_val = is_pos
                    # Calculate damage: how many currently satisfied clauses become unsat?
                    # A clause becomes unsat if all its literals are false.
                    # Flipping var_idx to new_val makes the literal (var_idx, new_val) true.
                    # This satisfies clause ci.
                    # It might unsatisfy other clauses where the literal is (var_idx, not new_val).
                    damage = 0
                    opposite = not new_val
                    for (oc, opos) in var_to_clauses[var_idx]:
                        if oc == ci:
                            continue
                        if opos == opposite:
                            # This clause has the literal that will become false
                            # Check if it is currently satisfied (i.e., not in unsat set)
                            # If it is satisfied, and this is its only true literal, it becomes unsat.
                            # However, we don't know if it's the *only* true literal without checking.
                            # Approximation: count it as damage if it's currently satisfied.
                            # This is the standard "damage" heuristic in WalkSAT.
                            if not is_unsat[oc]:
                                damage += 1
                                if damage >= best_damage:
                                    break
                    if damage < best_damage:
                        best_damage = damage
                        best_var = var_idx
                        best_val = new_val
                var_idx = best_var
                new_val = best_val
            
            old_val = assignment[var_idx]
            if old_val == new_val:
                continue
                
            assignment[var_idx] = new_val
            
            # Update unsat status for clauses containing var_idx
            for (ci, is_pos_in_c) in var_to_clauses[var_idx]:
                if is_unsat[ci]:
                    if is_satisfied(ci, assignment):
                        is_unsat[ci] = False
                        # Remove from unsat_list efficiently?
                        # Swap and pop
                        try:
                            idx = unsat_list.index(ci)
                            unsat_list[idx] = unsat_list[-1]
                            unsat_list.pop()
                        except ValueError:
                            pass
                else:
                    if not is_satisfied(ci, assignment):
                        is_unsat[ci] = True
                        unsat_list.append(ci)
                        
        return assignment, len(unsat_list) == 0

    best_assignment = random_assignment()
    best_unsat = count_unsat(best_assignment)
    
    steps_per_restart = 10000
    noise = 0.25
    
    while time.time() < deadline:
        if rng.random() < 0.1:
            assignment = best_assignment[:]
        else:
            assignment = random_assignment()
            
        assignment, solved = fast_walksat(assignment, steps_per_restart, noise)
        
        if solved:
            return assignment
            
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
    return best_assignment
# EVOLVE-BLOCK-END
