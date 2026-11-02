import numpy as np
import time
import itertools
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Generate Type A vectors (2 e_i, -2 e_i)
    type_a_vectors = []
    for i in range(D):
        v1 = np.zeros(D, dtype=np.int64)
        v1[i] = 2
        type_a_vectors.append(v1)
        v2 = np.zeros(D, dtype=np.int64)
        v2[i] = -2
        type_a_vectors.append(v2)
    
    # 2. Generate Type B vectors (4 non-zeros +/- 1)
    # We need to map (Support, Signs) to an index for ILP
    supports = list(itertools.combinations(range(D), 4))
    sign_patterns = list(itertools.product([1, -1], repeat=4))
    
    # Map (support, sign_pattern) to variable index
    # Total variables: len(supports) * len(sign_patterns) = 330 * 16 = 5280
    var_index = {}
    vec_list = []
    idx = 0
    for S in supports:
        for sig in sign_patterns:
            v = np.zeros(D, dtype=np.int64)
            for k, i in enumerate(S):
                v[i] = sig[k]
            vec_list.append(v)
            var_index[(S, sig)] = idx
            idx += 1
            
    n_vars = idx
    n_constraints = 0
    # Constraints: For each 3-subset C and sign pattern s (3 bits), sum of vars containing C with signs s <= 1
    subsets_3 = list(itertools.combinations(range(D), 3))
    sign_patterns_3 = list(itertools.product([1, -1], repeat=3))
    
    # Build constraint matrix A
    # A will be (n_constraints, n_vars)
    # Using sparse matrix for efficiency
    from scipy.sparse import lil_matrix
    A = lil_matrix((len(subsets_3) * len(sign_patterns_3), n_vars), dtype=np.int64)
    
    cons_idx = 0
    for C in subsets_3:
        for s in sign_patterns_3:
            # For each variable (S, sig), check if C subset of S and sig|C == s
            # This is O(n_vars * n_constraints) if done naively, which is 5280 * 1320 ~ 7M ops.
            # This is fine in Python/NumPy if vectorized or simple loop.
            # But we can optimize by precomputing which vars contain which 3-subsets.
            pass
            
    # Better approach: Iterate variables, add to relevant constraints.
    # For each variable (S, sig):
    #   It contributes to constraints (C, s) where C is a 3-subset of S.
    #   There are 4 such C.
    #   For each C, the sign pattern s is sig restricted to C.
    #   So we increment A[cons_idx(C, s), var_idx] by 1.
    
    # Precompute mapping from (C, s) to constraint row index
    c_s_to_row = {}
    row_idx = 0
    for C in subsets_3:
        for s in sign_patterns_3:
            c_s_to_row[(C, s)] = row_idx
            row_idx += 1
            
    A = lil_matrix((row_idx, n_vars), dtype=np.int64)
    
    # Iterate all variables
    for (S, sig), v_idx in var_index.items():
        # S is a tuple of 4 indices
        # Generate all 3-subsets of S
        # S = (i1, i2, i3, i4)
        # 3-subsets: (i1,i2,i3), (i1,i2,i4), (i1,i3,i4), (i2,i3,i4)
        # Corresponding sign patterns are sig with one element removed.
        
        # We need to map the indices in S to the positions in sig
        # sig is aligned with S.
        
        # For each 3-subset C of S:
        # Find the position in S that is missing.
        # The sign pattern s is sig with that position removed.
        
        # Example: S=(0,1,2,3), sig=(1,-1,1,1)
        # C=(0,1,2) -> missing index 3 in S -> s=(1,-1,1)
        # C=(0,1,3) -> missing index 2 in S -> s=(1,-1,1) ... wait.
        # s is defined on indices in C.
        # sig[k] is the sign for S[k].
        # If C = S \ {S[j]}, then s = sig with j-th element removed.
        
        for j in range(4):
            # C is S without S[j]
            C = tuple(S[k] for k in range(4) if k != j)
            # s is sig without sig[j]
            s = tuple(sig[k] for k in range(4) if k != j)
            
            r = c_s_to_row[(C, s)]
            A[r, v_idx] = 1
            
    A = A.tocsr()
    
    # Objective: maximize sum of y
    c = np.ones(n_vars, dtype=np.float64)
    
    # Constraints: A @ y <= 1
    # milp expects: A_ub @ x <= b_ub
    # Here A_ub = A, b_ub = 1
    
    # Bounds: 0 <= y <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    integrality = np.ones(n_vars) # all integer
    
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A, -np.inf, 1)],
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:
            y = res.x
            # Select vectors
            selected_b = []
            for i in range(n_vars):
                if y[i] > 0.5:
                    selected_b.append(vec_list[i])
        else:
            # Fallback to greedy if ILP fails
            selected_b = greedy_solve(A, n_vars, vec_list, time.time(), deadline)
    except Exception:
        selected_b = greedy_solve(A, n_vars, vec_list, time.time(), deadline)
        
    # Combine Type A and selected Type B
    all_vectors = type_a_vectors + selected_b
    
    if not all_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(all_vectors, dtype=np.int64)

def greedy_solve(A, n_vars, vec_list, current_time, deadline):
    # Simple greedy: pick variable with fewest conflicts, remove conflicts.
    # A is sparse matrix. A.T[i] gives constraints for variable i.
    # Conflicts: if we pick i, we cannot pick j if A[r, i]==1 and A[r, j]==1 for some r.
    # This is equivalent to: if they share a constraint row.
    # We can maintain a set of available constraints.
    
    # Simpler greedy:
    # 1. Start with all vars available.
    # 2. Pick a var, add to solution.
    # 3. Remove all vars that share a constraint with it.
    # 4. Repeat.
    
    # To speed up, we can precompute conflict lists.
    # But n_vars=5280, conflicts might be dense.
    # Let's just use the matrix.
    
    available = set(range(n_vars))
    selected = []
    
    # Heuristic: pick variable with minimum degree (fewest conflicts) first?
    # Or just iterate.
    # Given time limit, a simple random or sequential might be enough.
    # Let's try sequential with conflict removal.
    
    # A is CSR.
    # For each var i, the constraints it hits are A.indices[A.indptr[i]:A.indptr[i+1]]
    
    # We need to know which vars hit the same constraints.
    # This is the dual graph.
    # Building it is expensive.
    
    # Alternative: Local Search on the binary vector.
    # Start with random feasible solution?
    # Or start empty, add vars greedily.
    
    # Greedy Add:
    # While time < deadline:
    #   Pick a var that doesn't violate any constraint with currently selected vars.
    #   Add it.
    
    selected_mask = np.zeros(n_vars, dtype=bool)
    # active_constraints: count of selected vars per constraint
    # If count == 1, constraint is "full".
    # We can pick var i if for all r in A[:, i], count[r] == 0.
    
    counts = np.zeros(A.shape[0], dtype=np.int64)
    
    # To make it faster, we can shuffle variables.
    order = np.random.permutation(n_vars)
    
    for v_idx in order:
        if time.time() > deadline:
            break
        # Check if v_idx can be added
        # Get constraints for v_idx
        start = A.indptr[v_idx]
        end = A.indptr[v_idx + 1]
        cons = A.indices[start:end]
        
        # Check if any of these constraints are already full (count >= 1)
        if np.any(counts[cons] >= 1):
            continue
            
        # Add it
        selected_mask[v_idx] = True
        counts[cons] += 1
        
    return [vec_list[i] for i in range(n_vars) if selected_mask[i]]
