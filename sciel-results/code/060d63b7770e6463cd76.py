import numpy as np
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Preprocessing: build efficient data structures using numpy
    clause_vars = np.zeros((C, 3), dtype=np.int32)
    clause_signs = np.zeros((C, 3), dtype=np.int8)
    
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_vars[i, j] = v
            clause_signs[i, j] = sign
            var_to_clauses[v].append((i, j))
    
    # Precompute for each variable the list of (clause_idx, pos) it appears in
    # Already have var_to_clauses
    
    # Fast vectorized evaluation
    def evaluate(assignment):
        """Returns (num_unsat, unsat_indices, clause_sat_array, lit_sat_array)"""
        # assignment is numpy bool array of length n
        # Get values for all literals
        vals = assignment[clause_vars]  # shape (C, 3)
        # Literal satisfied if (val == (sign > 0))
        pos_mask = clause_signs > 0  # shape (C, 3)
        lit_sat = (vals.astype(np.int8) == pos_mask.astype(np.int8))  # shape (C, 3)
        clause_sat = lit_sat.any(axis=1)  # shape (C,)
        unsat_indices = np.where(~clause_sat)[0]
        num_unsat = len(unsat_indices)
        return num_unsat, unsat_indices, clause_sat, lit_sat
    
    # Compute delta for flipping variable v
    def compute_delta(v, clause_sat, lit_sat):
        """Returns delta (change in num_unsat if we flip v)"""
        delta = 0
        for (ci, pos) in var_to_clauses[v]:
            # Check the other two literals in clause ci
            # Other positions are those != pos
            other_sat = []
            for p in range(3):
                if p != pos:
                    other_sat.append(lit_sat[ci, p])
            
            if clause_sat[ci]:
                # Clause is currently satisfied
                # After flip, v's literal becomes unsat
                # Clause remains sat iff at least one of other literals is sat
                if not any(other_sat):
                    # Both other literals are unsat, so clause becomes unsat
                    delta -= 1
            else:
                # Clause is currently unsat
                # After flip, v's literal becomes sat
                # Clause becomes sat (regardless of others)
                delta += 1
        return delta
    
    # WalkSAT with proper incremental updates
    def walk_sat(assignment, num_unsat, unsat_indices, clause_sat, lit_sat, rng, max_steps, deadline):
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            if num_unsat == 0:
                return True, assignment, num_unsat, unsat_indices, clause_sat, lit_sat
            
            # Pick a random unsatisfied clause
            if len(unsat_indices) == 0:
                break
            c_idx = rng.choice(unsat_indices)
            cv = clause_vars[c_idx]
            
            # For each variable in this clause, compute delta
            best_v = -1
            best_score = -1e9
            best_delta = -1e9
            
            for (v, s) in cv:
                delta = compute_delta(v, clause_sat, lit_sat)
                # Add random noise for diversity (WalkSAT style)
                noise = rng.random() * 1.0
                score = delta + noise
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_delta = delta
            
            if best_v == -1:
                break
            
            # Flip best_v
            old_val = assignment[best_v]
            assignment[best_v] = not old_val
            
            # Update affected clauses incrementally
            affected_clauses = set()
            for (ci, pos) in var_to_clauses[best_v]:
                affected_clauses.add(ci)
            
            # Recompute satisfaction for affected clauses only
            for ci in affected_clauses:
                # Check if clause is now satisfied
                sat = False
                for j in range(3):
                    v = clause_vars[ci, j]
                    s = clause_signs[ci, j]
                    val = assignment[v]
                    if (s == 1 and val) or (s == -1 and not val):
                        sat = True
                        break
                clause_sat[ci] = sat
                # Update lit_sat for this clause
                for j in range(3):
                    v = clause_vars[ci, j]
                    s = clause_signs[ci, j]
                    val = assignment[v]
                    lit_sat[ci, j] = (s == 1 and val) or (s == -1 and not val)
            
            # Update num_unsat and unsat_indices
            # Count unsat among affected + keep track of unchanged
            new_unsat_count = 0
            new_unsat_list = []
            for ci in range(C):
                if not clause_sat[ci]:
                    new_unsat_count += 1
                    new_unsat_list.append(ci)
            
            num_unsat = new_unsat_count
            unsat_indices = np.array(new_unsat_list, dtype=np.int32)
            
            if num_unsat == 0:
                return True, assignment, num_unsat, unsat_indices, clause_sat, lit_sat
        
        return False, assignment, num_unsat, unsat_indices, clause_sat, lit_sat
    
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin
    
    rng = random.Random()
    
    best_num_unsat = C + 1
    best_assignment = None
    
    # Multiple restarts with increasing step limits
    restart = 0
    while time.time() < deadline:
        # Random initial assignment
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        # Evaluate
        num_unsat, unsat_indices, clause_sat, lit_sat = evaluate(assignment)
        
        if num_unsat == 0:
            return assignment.tolist()
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment.copy()
        
        # WalkSAT with adaptive step limit
        # Increase steps as we restart more
        base_steps = 3000 + restart * 500
        max_steps = min(base_steps, 50000)
        
        solved, assignment, num_unsat, unsat_indices, clause_sat, lit_sat = walk_sat(
            assignment, num_unsat, unsat_indices, clause_sat, lit_sat, rng, max_steps, deadline
        )
        
        if solved:
            return assignment.tolist()
        
        if num_unsat < best_num_unsat:
            best_num_unsat = num_unsat
            best_assignment = assignment.copy()
        
        restart += 1
    
    # Return best found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
