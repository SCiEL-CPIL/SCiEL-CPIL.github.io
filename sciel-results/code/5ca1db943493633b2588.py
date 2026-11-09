"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time


def solve(instance):
    """Deterministic DPLL solver with unit propagation and pure literal elimination."""
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard: 300s wall-clock limit, use 295s for safety margin.
    deadline = time.time() + 295.0
    
    # Convert clauses to list of (var, polarity) for faster access
    # var is 0-based, polarity is 1 for positive, 0 for negative
    clauses_v = [[(abs(l)-1, 1 if l > 0 else 0) for l in c] for c in clauses]
    
    def unit_propagate(clauses, assignment):
        """Apply unit propagation. Returns (new_clauses, new_assignment, contradiction)."""
        assignment = assignment.copy()
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Remove assigned literals
                remaining = []
                for var, pol in clause:
                    if var not in assignment:
                        remaining.append((var, pol))
                    else:
                        if assignment[var] == (pol == 1):
                            # Clause satisfied
                            remaining = None
                            break
                if remaining is None:
                    continue
                elif len(remaining) == 0:
                    return None, None, True  # Contradiction
                elif len(remaining) == 1:
                    # Unit clause: assign the remaining literal
                    var, pol = remaining[0]
                    assignment[var] = (pol == 1)
                    changed = True
                    new_clauses.append(remaining)
                else:
                    new_clauses.append(remaining)
            clauses = new_clauses
        return clauses, assignment, False
    
    def pure_literal_eliminate(clauses, assignment):
        """Apply pure literal elimination. Returns (new_clauses, new_assignment)."""
        assignment = assignment.copy()
        # Count occurrences of each literal
        literal_counts = {}  # (var, pol) -> count
        for clause in clauses:
            for var, pol in clause:
                if var not in assignment:
                    key = (var, pol)
                    literal_counts[key] = literal_counts.get(key, 0) + 1
        
        # Find pure literals
        pure_literals = set()
        for (var, pol), count in literal_counts.items():
            opposite_pol = 1 - pol
            if (var, opposite_pol) not in literal_counts:
                pure_literals.add(var)
        
        # Assign pure literals
        for var in pure_literals:
            # Find the polarity
            for clause in clauses:
                for v, p in clause:
                    if v == var and var not in assignment:
                        assignment[var] = (p == 1)
                        break
                if var in assignment:
                    break
        
        # Remove satisfied clauses and assigned literals
        new_clauses = []
        for clause in clauses:
            remaining = []
            for var, pol in clause:
                if var not in assignment:
                    remaining.append((var, pol))
                else:
                    if assignment[var] == (pol == 1):
                        remaining = None
                        break
            if remaining is None:
                continue
            elif len(remaining) == 0:
                return None, assignment  # Contradiction
            else:
                new_clauses.append(remaining)
        return new_clauses, assignment
    
    def dpll(clauses, assignment, depth=0):
        """DPLL recursive solver."""
        # Check deadline
        if time.time() > deadline - 2.0:
            return None
        
        # Check for contradiction
        if any(len(c) == 0 for c in clauses):
            return None
        
        # Check if all clauses satisfied
        if all(len(c) == 0 for c in clauses) or len(clauses) == 0:
            return assignment
        
        # Apply unit propagation
        clauses, assignment, contradiction = unit_propagate(clauses, assignment)
        if contradiction:
            return None
        if clauses is None:
            return assignment
        
        # Check deadline again after propagation
        if time.time() > deadline - 2.0:
            return None
        
        # Check if all clauses satisfied after propagation
        if len(clauses) == 0:
            return assignment
        
        # Apply pure literal elimination
        clauses, assignment = pure_literal_eliminate(clauses, assignment)
        if clauses is None:
            return None
        
        # Check deadline again
        if time.time() > deadline - 2.0:
            return None
        
        # Check if all clauses satisfied after pure literal elimination
        if len(clauses) == 0:
            return assignment
        
        # Select a variable to branch on
        # Use the variable that appears in the most clauses
        var_counts = {}
        for clause in clauses:
            for var, pol in clause:
                var_counts[var] = var_counts.get(var, 0) + 1
        
        if not var_counts:
            return assignment
        
        # Pick variable with highest count
        var = max(var_counts, key=lambda v: var_counts[v])
        
        # Branch: try var = True first
        assignment_true = assignment.copy()
        assignment_true[var] = True
        clauses_true = []
        for clause in clauses:
            remaining = []
            for v, pol in clause:
                if v == var:
                    if pol == 1:
                        remaining = None
                        break
                    else:
                        continue
                else:
                    remaining.append((v, pol))
            if remaining is None:
                continue
            elif len(remaining) == 0:
                return None  # Contradiction in this branch
            else:
                clauses_true.append(remaining)
        
        result = dpll(clauses_true, assignment_true, depth + 1)
        if result is not None:
            return result
        
        # Branch: try var = False
        assignment_false = assignment.copy()
        assignment_false[var] = False
        clauses_false = []
        for clause in clauses:
            remaining = []
            for v, pol in clause:
                if v == var:
                    if pol == 0:
                        remaining = None
                        break
                    else:
                        continue
                else:
                    remaining.append((v, pol))
            if remaining is None:
                continue
            elif len(remaining) == 0:
                return None  # Contradiction in this branch
            else:
                clauses_false.append(remaining)
        
        result = dpll(clauses_false, assignment_false, depth + 1)
        if result is not None:
            return result
        
        return None
    
    # Start DPLL
    initial_assignment = {}
    result = dpll(clauses_v, initial_assignment)
    
    if result is not None:
        # Convert to list of booleans
        final_assignment = [result.get(i, False) for i in range(n)]
        return final_assignment
    
    # Fallback: return all False (should not happen for satisfiable instances)
    return [False] * n
# EVOLVE-BLOCK-END
