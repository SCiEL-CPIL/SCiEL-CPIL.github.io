"""Deterministic DPLL solver with unit propagation and pure literal elimination."""
# EVOLVE-BLOCK-START
import sys
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Build clause list as tuples of signed literals
    clause_list = [tuple(c) for c in clauses]
    
    # Precompute literal indices
    # We'll use a recursive DPLL with unit propagation and pure literal elimination
    
    def dpll(assignment, clauses):
        """
        assignment: dict mapping variable index (0-based) to bool, or None for unassigned
        clauses: list of tuples of signed literals (1-based)
        Returns a complete assignment dict if satisfiable, None otherwise.
        """
        # Unit propagation
        changed = True
        new_assignment = dict(assignment)
        new_clauses = list(clauses)
        
        while changed:
            changed = False
            i = 0
            while i < len(new_clauses):
                clause = new_clauses[i]
                if len(clause) == 0:
                    return None  # Contradiction
                if len(clause) == 1:
                    lit = clause[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    if var in new_assignment:
                        if new_assignment[var] != val:
                            return None  # Contradiction
                        # Already assigned correctly, remove clause
                        new_clauses.pop(i)
                        changed = True
                        continue
                    else:
                        new_assignment[var] = val
                        changed = True
                        # Remove this clause and simplify others
                        new_clauses.pop(i)
                        # Simplify remaining clauses
                        j = 0
                        while j < len(new_clauses):
                            cl = new_clauses[j]
                            new_cl = []
                            for l in cl:
                                if l == lit:
                                    # Clause satisfied
                                    break
                                elif l == -lit:
                                    # Remove the negated literal
                                    continue
                                else:
                                    new_cl.append(l)
                            if not new_cl:
                                # Clause satisfied
                                new_clauses.pop(j)
                            else:
                                new_clauses[j] = tuple(new_cl)
                                j += 1
                        continue
                i += 1
        
        if not new_clauses:
            # All clauses satisfied
            # Fill in remaining variables arbitrarily
            result = dict(new_assignment)
            for v in range(n):
                if v not in result:
                    result[v] = False
            return result
        
        # Pure literal elimination
        # Find literals that appear only with one sign
        pos_lit_set = set()
        neg_lit_set = set()
        for clause in new_clauses:
            for lit in clause:
                if lit > 0:
                    pos_lit_set.add(lit)
                else:
                    neg_lit_set.add(lit)
        
        pure_lits = []
        for lit in pos_lit_set:
            if -lit not in neg_lit_set:
                pure_lits.append(lit)
        for lit in neg_lit_set:
            if -lit not in pos_lit_set:
                pure_lits.append(lit)
        
        if pure_lits:
            # Assign pure literals
            for lit in pure_lits:
                var = abs(lit) - 1
                if var not in new_assignment:
                    new_assignment[var] = lit > 0
            
            # Remove satisfied clauses and simplify
            i = 0
            while i < len(new_clauses):
                clause = new_clauses[i]
                new_cl = []
                for l in clause:
                    var = abs(l) - 1
                    if var in new_assignment:
                        if new_assignment[var] == (l > 0):
                            new_cl = None  # Satisfied
                            break
                        else:
                            continue  # Remove falsified literal
                    else:
                        new_cl.append(l)
                if new_cl is None:
                    new_clauses.pop(i)
                elif len(new_cl) == 0:
                    return None  # Contradiction
                else:
                    new_clauses[i] = tuple(new_cl)
                    i += 1
            
            # Check if all clauses are satisfied
            if not new_clauses:
                result = dict(new_assignment)
                for v in range(n):
                    if v not in result:
                        result[v] = False
                return result
            
            # Recurse after pure literal assignment
            return dpll(new_assignment, new_clauses)
        
        # Variable selection: choose variable appearing in most clauses
        var_count = {}
        for clause in new_clauses:
            for lit in clause:
                var = abs(lit) - 1
                if var not in var_count:
                    var_count[var] = 0
                var_count[var] += 1
        
        if not var_count:
            result = dict(new_assignment)
            for v in range(n):
                if v not in result:
                    result[v] = False
            return result
        
        # Pick variable with highest count
        chosen_var = max(var_count, key=lambda v: var_count[v])
        
        # Try True first
        new_assign_true = dict(new_assignment)
        new_assign_true[chosen_var] = True
        result = dpll(new_assign_true, new_clauses)
        if result is not None:
            return result
        
        # Try False
        new_assign_false = dict(new_assignment)
        new_assign_false[chosen_var] = False
        result = dpll(new_assign_false, new_clauses)
        if result is not None:
            return result
        
        return None
    
    # Run DPLL
    result = dpll({}, clause_list)
    
    if result is None:
        # Should not happen for satisfiable instances, but return default
        return [False] * n
    
    # Convert dict to list
    assignment_list = [result.get(i, False) for i in range(n)]
    return assignment_list

# EVOLVE-BLOCK-END
