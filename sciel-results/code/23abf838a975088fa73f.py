import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n # 9, 24, 24
    
    # 1. Construct Target Tensor T
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1
                
    T_flat = T.flatten()
    N_eq = len(T_flat)
    
    # 2. Generate Candidate Dictionary
    # We want to find a sparse linear combination of these candidates.
    # Candidate: (u, v, w) vectors.
    candidates = []
    
    # Add Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
                
    # Add Random Sparse Candidates
    # We add both + and - versions implicitly by allowing integer x, 
    # but for milp with x>=0, we should add both C and -C.
    # However, milp can handle integer variables with lower/upper bounds.
    # Let's allow x in [-10, 10] to be safe, or just [-1, 1] if we expect small coeffs.
    # Actually, minimal rank decompositions often use coeffs in {-1, 0, 1}.
    
    rng = np.random.default_rng(42)
    num_random = 200
    for _ in range(num_random):
        # Generate sparse vectors
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        
        # Support size 2 or 3
        sup_u = rng.choice(2, 3) # 2 or 3
        sup_v = rng.choice(2, 3)
        sup_w = rng.choice(2, 3)
        
        idx_u = rng.choice(nm, sup_u, replace=False)
        idx_v = rng.choice(mp, sup_v, replace=False)
        idx_w = rng.choice(pn, sup_w, replace=False)
        
        u[idx_u] = rng.choice([-1, 1], sup_u)
        v[idx_v] = rng.choice([-1, 1], sup_v)
        w[idx_w] = rng.choice([-1, 1], sup_w)
        
        candidates.append((u, v, w))
        
    # Also add some "Strassen-like" structured candidates?
    # For 3x3x3, Strassen uses 7 terms. 
    # We can map 3x3x3 Strassen terms into the 3x3x8 space by fixing k or summing over k?
    # It's easier to just rely on the random + schoolbook mix for a general search.
    # If we want to be smarter, we can add terms that are linear combinations of schoolbook terms.
    
    M = len(candidates)
    
    # Build Constraint Matrix A
    # A[:, r] is the flattened candidate r
    A = np.zeros((N_eq, M), dtype=int)
    for r, (u, v, w) in enumerate(candidates):
        A[:, r] = np.outer(np.outer(u, v), w).flatten()
        
    # We want min sum |x_r|
    # Let x_r be integers.
    # Objective: c = [1, 1, ..., 1]
    c = np.ones(M)
    
    # Constraints: A @ x == T_flat
    # Bounds: x_r in [-10, 10] (integers)
    # Using milp:
    # We need to handle integer constraints.
    
    try:
        # scipy.optimize.milp
        # variables: x
        # objective: min c @ x
        # constraints: A @ x == T_flat
        # bounds: [-10, 10] for all x, integrality: 1 for all
        
        constraints = [
            LinearConstraint(A, T_flat, T_flat)
        ]
        
        # milp requires c, A_ub, b_ub, A_eq, b_eq, integrality, bounds
        # Our problem is equality constrained.
        # milp(c, A_ub, b_ub, A_eq, b_eq, integrality, bounds)
        
        # A_ub, b_ub: None (no inequality constraints other than bounds)
        # A_eq, b_eq: A, T_flat
        
        integrality = np.ones(M)
        bounds = Bounds(lb=-10, ub=10)
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:
            x = res.x.astype(int)
            # Check if solution is valid
            # Reconstruct U, V, W
            U = np.zeros((nm, M))
            V = np.zeros((mp, M))
            W = np.zeros((pn, M))
            
            # We only keep non-zero terms to reduce rank
            active = np.nonzero(x)[0]
            
            if len(active) == 0:
                 # Fallback to schoolbook
                 pass # handled below
                
            # Build the final matrices with only active terms
            # Note: The rank is the number of active terms.
            # We need to return U, V, W of shape (dim, rank)
            
            U_final = np.zeros((nm, len(active)))
            V_final = np.zeros((mp, len(active)))
            W_final = np.zeros((pn, len(active)))
            
            for idx, r in enumerate(active):
                u, v, w = candidates[r]
                coeff = x[r]
                U_final[:, idx] = u * coeff
                V_final[:, idx] = v * coeff
                W_final[:, idx] = w * coeff
                
            # Convert to lists of lists with integer/half-integer entries
            # Entries are integers here.
            
            return {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
        else:
            # Fallback to schoolbook
            pass
            
    except Exception:
        # Fallback to schoolbook
        pass
        
    # Fallback: Schoolbook Rank 72
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
