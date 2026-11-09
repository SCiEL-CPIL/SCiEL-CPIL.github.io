"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Convert to 0-based: literal l -> var = abs(l)-1, sign = l > 0
    # We'll work with internal representation where each literal is (var, is_pos)
    # But for efficiency, let's keep signed integers and convert on the fly.
    
    # Build clause list as list of lists of (var, is_pos) tuples for faster lookup
    # Actually, let's use a different approach: 
    # Represent each clause as a list of signed integers (1-based)
    # Maintain:
    #   assignment: list of int, 0=unassigned, 1=true, 2=false
    #   active_clauses: list of lists of signed ints (only unassigned literals)
    #   For each variable, track which clauses contain it (watched literals style is complex, 
    #   but for n=50, simple scanning is fine)
    
    # Let's implement a clean DPLL with:
    # - Unit propagation (iterative)
    # - Pure literal elimination
    # - Branching on variable with highest occurrence count
    
    assignment = [0] * n  # 0=unassigned, 1=true, 2=false
    
    # Convert clauses to internal form: list of (var, is_pos)
    def lit_to_varpos(lit):
        var = abs(lit) - 1
        is_pos = lit > 0
        return var, is_pos
    
    # We'll maintain clauses as lists of (var, is_pos)
    clauses_internal = []
    for clause in clauses:
        c = [lit_to_varpos(lit) for lit in clause]
        clauses_internal.append(c)
    
    def simplify_and_propagate(clauses):
        """
        Applies unit propagation and pure literal elimination.
        Returns (new_clauses, contradiction) where new_clauses has only unassigned literals.
        Updates assignment in place.
        """
        changed = True
        while changed:
            changed = False
            new_clauses = []
            
            for clause in clauses:
                simplified = []
                satisfied = False
                for var, is_pos in clause:
                    val = assignment[var]
                    if val != 0:
                        # Variable is assigned
                        if (val == 1 and is_pos) or (val == 2 and not is_pos):
                            satisfied = True
                            break
                        else:
                            # Literal is false, skip
                            continue
                    else:
                        simplified.append((var, is_pos))
                
                if satisfied:
                    continue
                elif len(simplified) == 0:
                    return None, True  # Contradiction
                else:
                    new_clauses.append(simplified)
            
            clauses = new_clauses
            
            # Check for unit clauses
            units = [clause[0] for clause in clauses if len(clause) == 1]
            if units:
                for var, is_pos in units:
                    if assignment[var] == 0:
                        assignment[var] = 1 if is_pos else 2
                        changed = True
                    else:
                        # Already assigned, check for contradiction
                        if (assignment[var] == 1 and not is_pos) or (assignment[var] == 2 and is_pos):
                            return None, True
            
            # Pure literal elimination
            if not changed:
                var_polarity = defaultdict(set)
                for clause in clauses:
                    for var, is_pos in clause:
                        if assignment[var] == 0:
                            var_polarity[var].add(is_pos)
                
                pure_vars = []
                for var, polarities in var_polarity.items():
                    if len(polarities) == 1:
                        is_pos = list(polarities)[0]
                        pure_vars.append((var, is_pos))
                
                if pure_vars:
                    for var, is_pos in pure_vars:
                        assignment[var] = 1 if is_pos else 2
                    changed = True
                    # Need to re-simplify
                    new_clauses = []
                    for clause in clauses:
                        simplified = []
                        satisfied = False
                        for var, is_pos in clause:
                            val = assignment[var]
                            if val != 0:
                                if (val == 1 and is_pos) or (val == 2 and not is_pos):
                                    satisfied = True
                                    break
                                else:
                                    continue
                            else:
                                simplified.append((var, is_pos))
                        if not satisfied:
                            if len(simplified) == 0:
                                return None, True
                            new_clauses.append(simplified)
                    clauses = new_clauses
        
        return clauses, False
    
    def dpll(clauses):
        """
        Recursive DPLL. Returns True if satisfiable, False otherwise.
        Updates assignment in place.
        """
        clauses, contradiction = simplify_and_propagate(clauses)
        if contradiction:
            return False
        if not clauses:
            return True
        
        # Branching heuristic: pick variable with max occurrence
        var_counts = defaultdict(int)
        for clause in clauses:
            for var, is_pos in clause:
                if assignment[var] == 0:
                    var_counts[var] += 1
        
        if not var_counts:
            return True
        
        branch_var = max(var_counts, key=lambda v: var_counts[v])
        
        # Try True
        assignment[branch_var] = 1
        if dpll(clauses):
            return True
        
        # Try False
        assignment[branch_var] = 2
        if dpll(clauses):
            return True
        
        # Backtrack
        assignment[branch_var] = 0
        return False
    
    satisfied = dpll(clauses_internal)
    
    # Build final assignment
    result = []
    for i in range(n):
        if assignment[i] == 1:
            result.append(True)
        elif assignment[i] == 2:
            result.append(False)
        else:
            # Unassigned, set to True (shouldn't happen if SAT)
            result.append(True)
    
    return result
# EVOLVE-BLOCK-END
