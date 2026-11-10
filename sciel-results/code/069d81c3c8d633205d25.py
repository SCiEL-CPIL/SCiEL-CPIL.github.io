"""Simulated annealing based 3-SAT solver for planted instances."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 330s wall-clock limit, leave 10s margin
    deadline = time.time() + 320.0
    
    # Precompute clause literal indices and signs for fast access
    # clause_lits[i] = list of (var_index, sign) where sign is +1 or -1
    # var_index is 0-based
    clause_lits = []
    for c in clauses:
        lits = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            lits.append((v, s))
        clause_lits.append(lits)
    
    # For each variable, precompute list of (clause_index, sign) where it appears
    var_clauses = [[] for _ in range(n)]
    for ci, lits in enumerate(clause_lits):
        for v, s in lits:
            var_clauses[v].append((ci, s))
    
    def initial_solve():
        """Run simulated annealing with a fresh random assignment."""
        rng = random.Random()
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute current unsat count per clause
        clause_unsat = [0] * m
        for ci, lits in enumerate(clause_lits):
            cnt = 0
            for v, s in lits:
                if assignment[v] == (s == 1):
                    cnt += 1
            clause_unsat[ci] = cnt if cnt == 0 else 0  # 1 if unsat
        
        total_unsat = sum(clause_unsat)
        best_assignment = assignment[:]
        best_unsat = total_unsat
        
        # Simulated annealing parameters
        t0 = 1.0
        t_end = 0.01
        # We'll do a fixed number of steps but check time
        max_steps = 500000
        cooling = (t_end / t0) ** (1.0 / max_steps)
        
        temperature = t0
        rng2 = random.Random()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if total_unsat == 0:
                return assignment
            
            # Pick a random variable to flip
            v = rng2.randrange(n)
            
            # Compute delta in unsat count if we flip v
            delta = 0
            for ci, s in var_clauses[v]:
                # Current state of literal (v, s) in clause ci
                current_val = assignment[v] == (s == 1)
                if current_val:
                    # Currently satisfied by this literal
                    # If we flip, this literal becomes false
                    # Clause stays satisfied if another literal is true
                    # We need to check if any other literal in clause ci is true
                    # But we can use clause_unsat: if clause was satisfied (clause_unsat[ci]==0),
                    # and this literal was the only true one, flipping makes it unsat
                    # We need to count how many true literals in clause ci
                    # This is expensive. Let's use a different approach.
                    pass
                else:
                    # Currently false
                    # If we flip, this literal becomes true, clause becomes satisfied
                    pass
            
            # More efficient delta computation:
            # For each clause containing v, check if flipping changes satisfaction
            delta = 0
            for ci, s in var_clauses[v]:
                was_satisfied = (clause_unsat[ci] == 0)
                # After flip, is this literal true?
                new_val = (not assignment[v]) == (s == 1)
                # How many true literals in clause ci currently?
                # We need this. Let's maintain it.
                # Actually, let's just recompute for this clause
                true_count = 0
                for v2, s2 in clause_lits[ci]:
                    if assignment[v2] == (s2 == 1):
                        true_count += 1
                # Current: clause satisfied if true_count > 0
                # After flip of v: 
                #   if v's literal was true (assignment[v] == (s==1)), it becomes false
                #   if v's literal was false, it becomes true
                if assignment[v] == (s == 1):
                    # v's literal was true, becomes false
                    new_true_count = true_count - 1
                else:
                    # v's literal was false, becomes true
                    new_true_count = true_count + 1
                
                new_satisfied = (new_true_count > 0)
                if was_satisfied and not new_satisfied:
                    delta += 1
                elif not was_satisfied and new_satisfied:
                    delta -= 1
            
            # Accept move
            if delta <= 0:
                # Always accept improving or neutral moves
                assignment[v] = not assignment[v]
                total_unsat += delta
                # Update clause_unsat for affected clauses
                for ci, s in var_clauses[v]:
                    # Recompute clause_unsat[ci]
                    true_count = 0
                    for v2, s2 in clause_lits[ci]:
                        if assignment[v2] == (s2 == 1):
                            true_count += 1
                    clause_unsat[ci] = 0 if true_count > 0 else 1
            else:
                # Deteriorating move: accept with probability exp(-delta / temperature)
                if temperature > 1e-10:
                    prob = math.exp(-delta / temperature)
                    if rng2.random() < prob:
                        assignment[v] = not assignment[v]
                        total_unsat += delta
                        for ci, s in var_clauses[v]:
                            true_count = 0
                            for v2, s2 in clause_lits[ci]:
                                if assignment[v2] == (s2 == 1):
                                    true_count += 1
                            clause_unsat[ci] = 0 if true_count > 0 else 1
            
            if total_unsat < best_unsat:
                best_unsat = total_unsat
                best_assignment = assignment[:]
            
            temperature *= cooling
        
        return best_assignment
    
    # Try multiple restarts
    best_result = None
    best_score = m  # worst possible unsat count
    
    while time.time() < deadline:
        result = initial_solve()
        # Check if result is satisfying
        unsat = 0
        for ci, lits in enumerate(clause_lits):
            sat = False
            for v, s in lits:
                if result[v] == (s == 1):
                    sat = True
                    break
            if not sat:
                unsat += 1
        
        if unsat == 0:
            return result
        
        if unsat < best_score:
            best_score = unsat
            best_result = result
        
        # If we've found a good solution, we can stop early
        if best_score <= 2:
            # Do a quick local search to try to fix the remaining clauses
            for _ in range(10000):
                if time.time() > deadline:
                    break
                unsat_clauses = []
                for ci, lits in enumerate(clause_lits):
                    sat = False
                    for v, s in lits:
                        if best_result[v] == (s == 1):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(ci)
                if not unsat_clauses:
                    return best_result
                # Try flipping a variable in an unsat clause
                ci = random.Random().choice(unsat_clauses)
                v, s = random.Random().choice(clause_lits[ci])
                best_result[v] = not best_result[v]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    if best_result is None:
        # Return a random assignment as fallback
        return [random.Random().random() < 0.5 for _ in range(n)]
    
    return best_result
# EVOLVE-BLOCK-END
