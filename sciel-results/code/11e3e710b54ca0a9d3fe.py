"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the tensor for matrix multiplication
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Compute flattening ranks to get lower bound
    T1 = tensor.reshape(n * m, m * p * p * n)
    T2 = tensor.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = tensor.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # For 2x3 x 3x5, the flattening ranks are:
    # T1: shape (6, 75) -> rank 6
    # T2: shape (15, 60) -> rank 15  
    # T3: shape (10, 90) -> rank 10
    # So lower_bound = 15
    
    # The schoolbook rank is 30. We need to check if we can do better.
    # For matrix multiplication tensors, it's known that the rank is often
    # equal to the schoolbook rank for small dimensions, but let's try to
    # find a lower rank decomposition.
    
    # Strategy: Try ranks from lower_bound up to 29
    # For each rank r, we attempt to find a decomposition using the
    # flattening constraint approach.
    
    # For a rank-r decomposition, we need U (6 x r), V (15 x r), W (10 x r)
    # such that tensor = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # This means for each (a,b,c): tensor[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
    
    # We can set up this as a system. One approach:
    # 1. Choose U and V randomly (or structured)
    # 2. Solve for W using the linear system
    
    # But this is nonlinear. Instead, let's use the fact that:
    # If we fix U and V, then for each c in W's rows, we have a linear constraint.
    
    # Alternative: Use the border rank / secant variety approach.
    # The tensor lies in the secant variety of order r if and only if
    # there exist U, V, W with the decomposition.
    
    # For practical purposes, let's try a few structured approaches:
    
    # Approach 1: Try rank = lower_bound (15)
    # If that fails, try 16, 17, ..., up to 29
    
    # For each candidate rank r, we can try:
    # - Random initialization with iterative refinement
    # - Structured initialization based on the tensor's symmetries
    
    # Given the time constraint (120s), let's try a limited search.
    
    best_rank = n * m * p  # Start with schoolbook = 30
    best_decomp = None
    
    # Try to find decompositions for ranks from lower_bound to 29
    for r in range(lower_bound, n * m * p):
        # Try to find a rank-r decomposition
        # We'll use a simple approach: parameterize U and V, then solve for W
        
        # For a rank-r decomposition:
        # tensor[a,b,c] = sum_{c=0}^{r-1} U[a,c] * V[b,c] * W[c,c]
        
        # If we fix U and V, this is linear in W:
        # For each (a,b): tensor[a,b,:] = sum_c U[a,c]*V[b,c]*W[:,c]
        # This is: tensor[a,b,:] = (U[a,:] ⊙ V[b,:]) @ W
        # where ⊙ is element-wise product
        
        # So: tensor[a,b,:] = (U[a,:] * V[b,:]) @ W
        # This is a linear system in W for each (a,b), but W is shared.
        
        # Let's stack all equations:
        # For each (a,b,c): tensor[a,b,c] = sum_{d=0}^{r-1} U[a,d]*V[b,d]*W[c,d]
        
        # This is linear in W if U and V are fixed.
        
        # Let's try random U and V, then solve for W.
        
        success = False
        for trial in range(50):  # Try 50 random initializations
            np.random.seed(trial + r)
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            
            # Build the linear system for W
            # For each (a,b,c): tensor[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            
            # Let's organize: for each c (row of W), we have:
            # sum_d U[a,d]*V[b,d]*W[c,d] = tensor[a,b,c] for all (a,b)
            
            # This is: for each c, W[c,:] is a vector such that
            # (U * V^T)[a,b] weighted by W[c,:] ... 
            
            # Actually, let me reformulate:
            # tensor[a,b,c] = sum_d (U[a,d] * V[b,d]) * W[c,d]
            
            # Let M[a,b,d] = U[a,d] * V[b,d]
            # Then tensor[a,b,c] = sum_d M[a,b,d] * W[c,d]
            
            # For fixed c: tensor[a,b,c] = sum_d M[a,b,d] * W[c,d]
            # This is a linear system in W[c,:] for each c.
            
            # But the same W[c,d] appears in all equations for that c.
            
            # Let's stack all equations:
            # For each (a,b,c): tensor[a,b,c] = sum_d M[a,b,d] * W[c,d]
            
            # Number of equations: n*m * m*p * p*n = 6*15*10 = 900
            # Number of unknowns: r * p*n = r * 10
            
            # If r*10 < 900, the system is overdetermined.
            # We can solve it in least squares sense and check residual.
            
            # Build the matrix for the linear system
            # Unknowns: W[0,0], W[0,1], ..., W[0,9], W[1,0], ..., W[9,9]
            # Total unknowns: r * 10
            
            num_unknowns = r * p * n
            num_equations = n * m * m * p * p * n
            
            # Build A matrix where A[e, u] such that A @ W_flat = tensor_flat
            # e indexes (a,b,c), u indexes (c', d) where c' is row of W, d is column
            
            A = np.zeros((num_equations, num_unknowns))
            rhs = np.zeros(num_equations)
            
            e = 0
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        rhs[e] = tensor[a, b, c]
                        for d in range(r):
                            u = c * r + d  # W[c, d]
                            A[e, u] = U_rand[a, d] * V_rand[b, d]
                        e += 1
            
            # Solve least squares
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
            
            # Check if the solution is good
            residual = np.linalg.norm(A @ W_flat - rhs)
            rel_residual = residual / (np.linalg.norm(rhs) + 1e-10)
            
            if rel_residual < 1e-8:
                # Found a valid decomposition!
                W_sol = W_flat.reshape(p * n, r)
                
                # Verify the decomposition
                tensor_recon = np.zeros((n * m, m * p, p * n))
                for d in range(r):
                    tensor_recon += np.outer(np.outer(U_rand[:, d], V_rand[:, d]), W_sol[:, d])
                
                if np.allclose(tensor_recon, tensor, atol=1e-8):
                    success = True
                    best_rank = r
                    best_decomp = (U_rand, V_rand, W_sol)
                    break
        
        if success:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        rank = best_rank
        # Convert to list of lists with rational-like values
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
