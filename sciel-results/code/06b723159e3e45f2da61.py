import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    m = len(clauses)
    
    # Preprocessing
    # Convert clauses to 0-based literal indices
    # Literal l (1-based signed) -> index:
    # if l > 0: var = l-1, sign = +1
    # if l < 0: var = -l-1, sign = -1
    # We store literal as (var, sign) or just index in a flat array?
    # Let's use a flat array of literals where 0..n-1 are +var, n..2n-1 are -var?
    # Or just keep (var, sign) tuples.
    
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8) # 1 for +, -1 for -
    
    var_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            clause_vars[ci, li] = v
            clause_signs[ci, li] = s
            var_clauses[v].append(ci)

    # State
    # assignment: bool array of size n
    # lit_sat: bool array of size (m, 3)
    # clause_sat: bool array of size m
    # num_unsat: int
    
    def init_state(assignment):
        assignment = np.array(assignment, dtype=bool)
        lit_sat = np.zeros((m, 3), dtype=bool)
        for ci in range(m):
            for li in range(3):
                v = clause_vars[ci, li]
                s = clause_signs[ci, li]
                if s == 1:
                    lit_sat[ci, li] = assignment[v]
                else:
                    lit_sat[ci, li] = not assignment[v]
        clause_sat = np.any(lit_sat, axis=1)
        num_unsat = int(np.sum(~clause_sat))
        return assignment, lit_sat, clause_sat, num_unsat

    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 2000
    steps_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, lit_sat, clause_sat, num_unsat = init_state(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment.copy()
            
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
                
            if num_unsat == 0:
                return assignment
                
            # Pick random unsat clause
            unsat_indices = np.where(~clause_sat)[0]
            if len(unsat_indices) == 0:
                return assignment
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])
            
            best_gain = float('-inf')
            best_var = -1
            
            # Evaluate flipping each literal in clause ci
            for li in range(3):
                v = int(clause_vars[ci, li])
                # Gain from satisfying ci is +1
                gain = 1
                
                # Check impact on other clauses containing v
                for c_idx in var_clauses[v]:
                    if c_idx == ci:
                        continue
                    
                    # Check if flipping v helps or hurts c_idx
                    # We need to know the satisfaction status of the OTHER literals in c_idx
                    # and the current satisfaction status of c_idx.
                    
                    # Current state of c_idx:
                    # Is it sat?
                    if clause_sat[c_idx]:
                        # If it's sat, flipping v hurts it ONLY IF v is the ONLY satisfying literal.
                        # Check if other literals are false.
                        # Other literals are at indices 0,1,2 except the one corresponding to v.
                        # Note: v might appear twice? In 3-SAT usually distinct vars. 
                        # If v appears twice, logic changes. Assume standard 3-SAT (distinct vars per clause).
                        
                        other_sat = False
                        for oli in range(3):
                            if oli == li:
                                continue
                            ov = int(clause_vars[c_idx, oli])
                            if ov == v:
                                # Duplicate variable in clause?
                                # If so, this literal is also v.
                                # If v is flipped, this literal also flips.
                                # This case is complex. Standard WalkSAT assumes distinct vars.
                                # Let's assume distinct.
                                pass
                            os = clause_signs[c_idx, oli]
                            if os == 1:
                                if assignment[ov]:
                                    other_sat = True
                                    break
                            else:
                                if not assignment[ov]:
                                    other_sat = True
                                    break
                                    
                        if not other_sat:
                            # v is the only satisfying literal
                            gain -= 1
                    else:
                        # c_idx is UNSAT.
                        # Flipping v helps it ONLY IF the other literals are false (so flipping v makes it true).
                        # Actually, if c_idx is unsat, ALL literals are false.
                        # So the other literals ARE false.
                        # So flipping v to satisfy c_idx will make c_idx sat.
                        # Does flipping v satisfy c_idx?
                        # The literal in c_idx corresponding to v is currently false.
                        # If we flip v, it becomes true.
                        # So yes, it will be satisfied.
                        gain += 1
                        
                # Add noise
                noise = rng.random() * 0.5
                score = gain - noise
                
                if score > best_gain:
                    best_gain = score
                    best_var = v
                    
            if best_var == -1:
                v = rng.randrange(n)
            else:
                v = best_var
                
            # Flip v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update lit_sat and clause_sat for clauses containing v
            for c_idx in var_clauses[v]:
                # Find which literal position in c_idx corresponds to v
                # There might be multiple? Assume 1.
                # Update lit_sat
                for li in range(3):
                    if int(clause_vars[c_idx, li]) == v:
                        s = clause_signs[c_idx, li]
                        if s == 1:
                            lit_sat[c_idx, li] = assignment[v]
                        else:
                            lit_sat[c_idx, li] = not assignment[v]
                
                old_sat = clause_sat[c_idx]
                new_sat = bool(np.any(lit_sat[c_idx]))
                clause_sat[c_idx] = new_sat
                
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                    
            if num_unsat == 0:
                return assignment
                
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment.copy()
                
    if best_assignment is not None:
        return best_assignment
        
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
