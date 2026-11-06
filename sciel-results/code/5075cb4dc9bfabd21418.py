# EVOLVE-BLOCK-START
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    num_clauses = len(clauses)
    if num_clauses == 0:
        return [True] * n

    # 1. Gaussian Elimination Initialization
    # Build incidence matrix A (num_clauses x n) over GF(2)
    # A[i][j] = 1 if variable j is in clause i
    
    A = np.zeros((num_clauses, n), dtype=np.uint8)
    b = np.zeros(num_clauses, dtype=np.uint8)
    
    # Preprocess clauses
    # Store variable indices for fast access in repair
    clause_vars = []
    
    for i, clause in enumerate(clauses):
        neg_count = 0
        for lit in clause:
            var_idx = abs(lit) - 1
            A[i, var_idx] = 1
            clause_vars.append([]) # placeholder, we'll fill properly
            if lit < 0:
                neg_count += 1
        # Heuristic for b: parity of negative literals
        b[i] = neg_count % 2
        
    # Re-do clause_vars storage properly
    clause_vars = []
    for clause in clauses:
        vars_ = [abs(l) - 1 for l in clause]
        clause_vars.append(vars_)

    # Perform Gaussian Elimination on A x = b over GF(2)
    # We work with a copy of A and b
    A_ge = A.copy()
    b_ge = b.copy()
    
    # Standard GE
    row = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for r in range(row, num_clauses):
            if A_ge[r, col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        
        # Swap rows
        if pivot != row:
            A_ge[[row, pivot]] = A_ge[[pivot, row]]
            b_ge[[row, pivot]] = b_ge[[pivot, row]]
        
        # Eliminate
        for r in range(num_clauses):
            if r != row and A_ge[r, col] == 1:
                A_ge[r] = (A_ge[r] + A_ge[row]) % 2
                b_ge[r] = (b_ge[r] + b_ge[row]) % 2
        
        row += 1
        if row == num_clauses:
            break

    # Extract solution
    # Variables that are pivots are determined.
    # Free variables (columns without a pivot) are set to 0.
    assignment = np.zeros(n, dtype=bool)
    
    # Identify pivot columns
    pivot_cols = []
    row = 0
    for col in range(n):
        found = False
        for r in range(row, num_clauses):
            if A_ge[r, col] == 1:
                # Check if this is the first 1 in the row for this column (reduced row echelon form check is complex, 
                # but standard GE leaves 1s above/below if we don't do full reduction. 
                # Actually, the loop above eliminates ALL other rows, so it IS reduced row echelon form (RREF) essentially, 
                # except we didn't normalize the pivot row to have 1 at the pivot (it's already 1 in GF(2)).
                # So A_ge[row][col] is 1, and A_ge[r][col] is 0 for r != row.
                pivot_cols.append(col)
                assignment[col] = bool(b_ge[row])
                found = True
                break
        if found:
            row += 1
            if row == num_clauses:
                break
        # If not found, it's a free variable, remains 0

    # 2. Targeted Repair Phase
    # Function to get unsatisfied clauses
    def get_unsatisfied(assignment):
        unsat = []
        for i, vars_ in enumerate(clause_vars):
            # Check if clause is satisfied
            # Clause is satisfied if any literal is True
            # Literal lit is True if (assignment[var] == (lit > 0))
            # We need original signs. Let's store them.
            pass # Need to store signs
    
    # Let's re-structure to keep signs for repair
    clause_signs = []
    for clause in clauses:
        signs = [l > 0 for l in clause]
        clause_signs.append(signs)
        
    def count_unsat(assignment):
        count = 0
        unsat_indices = []
        for i in range(num_clauses):
            satisfied = False
            for k in range(len(clause_vars[i])):
                var_idx = clause_vars[i][k]
                sign = clause_signs[i][k]
                if assignment[var_idx] == sign:
                    satisfied = True
                    break
            if not satisfied:
                count += 1
                unsat_indices.append(i)
        return count, unsat_indices

    current_unsat, unsat_indices = count_unsat(assignment)
    
    if current_unsat == 0:
        return assignment.tolist()

    max_iterations = 10000
    stuck_counter = 0
    best_unsat = current_unsat
    best_assignment = assignment.copy()

    for _ in range(max_iterations):
        if current_unsat == 0:
            return assignment.tolist()
            
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment.copy()
            stuck_counter = 0
        else:
            stuck_counter += 1

        # If stuck for too long, restart from best or randomize slightly?
        # For planted instances, best flip usually works.
        # Pick a random unsatisfied clause
        clause_idx = random.choice(unsat_indices)
        vars_ = clause_vars[clause_idx]
        signs = clause_signs[clause_idx]
        
        best_var = -1
        best_delta = float('inf')
        
        # Try flipping each variable in the clause
        for k in range(len(vars_)):
            var_idx = vars_[k]
            
            # Calculate delta if we flip var_idx
            # A flip changes the satisfaction status of all clauses containing var_idx.
            # This is expensive to calculate naively for every flip in every iteration.
            # Optimization: Precompute for each variable, which clauses it affects.
            # But we are in a loop. Let's just do the naive check for the 3 vars in the current clause.
            # Actually, flipping a variable affects ALL clauses it is in, not just the current one.
            
            # To be efficient, we can approximate or just do the full recount for the 3 candidates.
            # Since n=150 and clauses ~ 450 (3x ratio), recounting 450 clauses is fast (microseconds).
            
            assignment[var_idx] = not assignment[var_idx]
            new_unsat, _ = count_unsat(assignment)
            assignment[var_idx] = not assignment[var_idx] # Undo
            
            delta = new_unsat - current_unsat
            if delta < best_delta:
                best_delta = delta
                best_var = var_idx

        # Apply the best flip
        if best_var != -1:
            assignment[best_var] = not assignment[best_var]
            current_unsat += best_delta
            # Update unsat_indices? We can just recompute in next loop or keep it stale?
            # Recomputing unsat_indices is cheap enough.
            _, unsat_indices = count_unsat(assignment)
        else:
            # Should not happen
            break

    return best_assignment.tolist()
# EVOLVE-BLOCK-END
