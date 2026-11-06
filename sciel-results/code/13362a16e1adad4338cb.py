import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0
    
    m = len(clauses)
    
    # Preprocessing: Convert to 0-based
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=np.int8)
    
    var_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            clause_vars[ci, li] = v
            clause_signs[ci, li] = s
            var_clauses[v].append(ci)
    
    # Build literal index map: for each variable, which (clause, literal_pos) pairs contain it
    var_lit_positions = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            v = int(clause_vars[ci, li])
            var_lit_positions[v].append((ci, li))
    
    # State
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
    
    max_restarts = 5000
    steps_per_restart = 10000
    
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
            best_noise = 0.0
            
            # Adaptive noise: decrease as we get closer to solution
            base_noise = 0.3 * (1.0 + 0.5 * np.cos(step / 1000.0))
            
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
                    if clause_sat[c_idx]:
                        # If it's sat, flipping v hurts it ONLY IF v is the ONLY satisfying literal.
                        other_sat = False
                        for oli in range(3):
                            if oli == li:
                                continue
                            ov = int(clause_vars[c_idx, oli])
                            if ov == v:
                                continue
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
                            gain -= 1
                    else:
                        # c_idx is UNSAT.
                        gain += 1
                        
                # Add noise
                noise = rng.random() * base_noise
                score = gain - noise
                
                if score > best_gain:
                    best_gain = score
                    best_var = v
                    best_noise = noise
                    
            if best_var == -1:
                v = rng.randrange(n)
            else:
                v = best_var
                
            # Flip v
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update lit_sat and clause_sat for clauses containing v
            for c_idx, li_pos in var_lit_positions[v]:
                s = clause_signs[c_idx, li_pos]
                if s == 1:
                    lit_sat[c_idx, li_pos] = assignment[v]
                else:
                    lit_sat[c_idx, li_pos] = not assignment[v]
                
                old_sat = clause_sat[c_idx]
                new_sat = bool(lit_sat[c_idx, 0] or lit_sat[c_idx, 1] or lit_sat[c_idx, 2])
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
                
    # Final polish: try to reduce unsat count further
    if best_assignment is not None:
        assignment, lit_sat, clause_sat, num_unsat = init_state(best_assignment)
        polish_steps = 5000
        for step in range(polish_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return assignment
            
            # Try each variable once
            for v in range(n):
                if time.time() >= deadline:
                    break
                    
                # Calculate gain from flipping v
                gain = 0
                for c_idx, li_pos in var_lit_positions[v]:
                    if clause_sat[c_idx]:
                        # Check if v is the only satisfying literal
                        other_sat = False
                        for oli in range(3):
                            if oli == li_pos:
                                continue
                            ov = int(clause_vars[c_idx, oli])
                            if ov == v:
                                continue
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
                            gain -= 1
                    else:
                        gain += 1
                        
                # Only flip if it improves
                if gain > 0:
                    assignment[v] = not assignment[v]
                    for c_idx, li_pos in var_lit_positions[v]:
                        s = clause_signs[c_idx, li_pos]
                        if s == 1:
                            lit_sat[c_idx, li_pos] = assignment[v]
                        else:
                            lit_sat[c_idx, li_pos] = not assignment[v]
                        
                        old_sat = clause_sat[c_idx]
                        new_sat = bool(lit_sat[c_idx, 0] or lit_sat[c_idx, 1] or lit_sat[c_idx, 2])
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
                        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment.copy()
            
    if best_assignment is not None:
        return best_assignment
        
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
