"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Build clause list with 0-based variables
    # Each clause is a list of (var, is_pos)
    clauses_0 = []
    for clause in clauses:
        clause_0 = []
        for lit in clause:
            var = abs(lit) - 1
            is_pos = lit > 0
            clause_0.append((var, is_pos))
        clauses_0.append(clause_0)
    
    num_clauses = len(clauses_0)
    
    # Efficient DPLL with watched literals approach
    # We'll use a trail-based approach for efficiency
    
    # Assignment: -1 = unassigned, 0 = False, 1 = True
    assignment = [-1] * n
    
    # Watched literals: for each variable, watch which clauses have that variable
    # Actually, let's use a simpler but efficient approach:
    # Maintain a list of active clauses (not satisfied, not empty)
    # Each clause is a list of (var, is_pos) for unassigned variables
    
    def simplify_and_propagate(clauses_list, assignment):
        """
        Simplify clauses and perform unit propagation.
        Returns (new_clauses, contradiction)
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses_list:
                # Simplify this clause
                simplified = []
                satisfied = False
                for (var, is_pos) in clause:
                    if assignment[var] != -1:
                        if assignment[var] == (1 if is_pos else 0):
                            satisfied = True
                            break
                        # else: literal is false, skip
                    else:
                        simplified.append((var, is_pos))
                
                if satisfied:
                    continue
                
                if len(simplified) == 0:
                    return new_clauses, True  # contradiction
                
                new_clauses.append(simplified)
            
            clauses_list = new_clauses
            
            # Check for unit clauses
            for i, clause in enumerate(clauses_list):
                if len(clause) == 1:
                    var, is_pos = clause[0]
                    if assignment[var] == -1:
                        assignment[var] = 1 if is_pos else 0
                        changed = True
                        break  # restart to re-simplify all clauses
                elif len(clause) == 0:
                    return clauses_list, True
        
        return clauses_list, False
    
    def pure_literal_elim(clauses_list, assignment):
        """
        Eliminate pure literals.
        Returns (new_clauses, assignment, contradiction)
        """
        # Find pure literals
        var_polarities = defaultdict(set)
        for clause in clauses_list:
            for (var, is_pos) in clause:
                if assignment[var] == -1:
                    var_polarities[var].add(is_pos)
        
        pure_vars = []
        for var, polarities in var_polarities.items():
            if len(polarities) == 1:
                is_pos = list(polarities)[0]
                pure_vars.append((var, is_pos))
        
        if not pure_vars:
            return clauses_list, assignment, False
        
        # Assign all pure literals
        for var, is_pos in pure_vars:
            assignment[var] = 1 if is_pos else 0
        
        # Simplify
        new_clauses, contradiction = simplify_and_propagate(clauses_list, assignment)
        return new_clauses, assignment, contradiction
    
    def dpll(clauses_list, assignment, depth=0):
        """
        DPLL solver. Returns True if satisfiable.
        """
        if depth > 200:
            return False
        
        # Simplify and propagate
        clauses_list, contradiction = simplify_and_propagate(clauses_list, assignment)
        if contradiction:
            return False
        
        if not clauses_list:
            return True
        
        # Pure literal elimination
        clauses_list, assignment, contradiction = pure_literal_elim(clauses_list, assignment)
        if contradiction:
            return False
        
        if not clauses_list:
            return True
        
        # Choose branching variable: most frequent in remaining clauses
        var_counts = defaultdict(int)
        for clause in clauses_list:
            for (var, is_pos) in clause:
                var_counts[var] += 1
        
        if not var_counts:
            return True
        
        # Pick the variable with highest count
        branch_var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True first
        assignment[branch_var] = 1
        new_clauses, contradiction = simplify_and_propagate(clauses_list, assignment)
        if not contradiction:
            if dpll(new_clauses, assignment, depth + 1):
                return True
        
        # Try False
        assignment[branch_var] = 0
        new_clauses, contradiction = simplify_and_propagate(clauses_list, assignment)
        if not contradiction:
            if dpll(new_clauses, assignment, depth + 1):
                return True
        
        # Backtrack
        assignment[branch_var] = -1
        return False
    
    # Run DPLL
    satisfied = dpll(clauses_0, assignment)
    
    if not satisfied:
        # Fallback: random assignment (shouldn't happen for SAT instances)
        import random
        rng = random.Random(42)
        assignment = [rng.randint(0, 1) for _ in range(n)]
    
    # Convert to list of booleans
    result = [bool(a) for a in assignment]
    
    return result
# EVOLVE-BLOCK-END
