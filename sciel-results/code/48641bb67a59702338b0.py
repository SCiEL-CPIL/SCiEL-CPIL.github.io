import numpy as np
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess
    num_clauses = len(clauses)
    
    # M: (num_clauses, 3) variable indices (0-based)
    # S: (num_clauses, 3) signs (True if positive)
    M_list = []
    S_list = []
    for c in clauses:
        M_list.append([abs(l) - 1 for l in c])
        S_list.append([l > 0 for l in c])
        
    M = np.array(M_list, dtype=np.int32)
    S = np.array(S_list, dtype=bool)
    
    # Create VarClauseMask: (n, num_clauses) boolean matrix
    # VarClauseMask[v, c] is True if variable v is in clause c
    VarClauseMask = np.zeros((n, num_clauses), dtype=bool)
    for c_idx in range(num_clauses):
        for v in M[c_idx]:
            VarClauseMask[v, c_idx] = True
            
    # Initialize random assignment
    rng = np.random.default_rng(0)
    assignment = rng.random(n) < 0.5
    
    start_time = time.time()
    timeout = 290
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        # Vectorized satisfaction check
        vals = assignment[M]  # (num_clauses, 3)
        lit_sat = (vals == S)  # (num_clauses, 3)
        clause_sat = np.any(lit_sat, axis=1)  # (num_clauses,)
        
        if np.all(clause_sat):
            return assignment.tolist()
            
        unsat_mask = ~clause_sat
        unsat_idx = np.where(unsat_mask)[0]
        
        if len(unsat_idx) == 0:
            return assignment.tolist()
            
        # Pick a random unsatisfied clause
        c_idx = unsat_idx[rng.integers(len(unsat_idx))]
        
        # Get variables in this clause
        vars_in_clause = M[c_idx]
        
        # Calculate unsat count for each variable in the clause
        # unsat_count[v] = sum(unsat_mask[VarClauseMask[v, :]])
        # We can compute unsat_count for all vars at once:
        # unsat_count = np.sum(unsat_mask[:, None] * VarClauseMask, axis=1)
        # But we only need it for 3 vars.
        # However, computing it for all is O(n*C) which is 150*500 = 75000 ops, very fast in NumPy.
        unsat_count = np.sum(unsat_mask[:, None] * VarClauseMask, axis=1)
        
        # Find the variable in vars_in_clause with max unsat_count
        # Add small random noise for tie-breaking
        scores = unsat_count[vars_in_clause] + rng.random(3) * 0.1
        best_pos = np.argmax(scores)
        best_var = vars_in_clause[best_pos]
        
        assignment[best_var] = not assignment[best_var]
        
    return assignment.tolist()
