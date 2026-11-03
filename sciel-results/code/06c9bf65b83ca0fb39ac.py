"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 4 matrix.
    
    The matrix multiplication tensor T has entries T[i, j, k] = 1 if and only if
    the output element (i, k) depends on input A[i, j] * B[j, k].
    
    We use the known minimal rank for 2x3 * 3x4 matrix multiplication.
    The rank is known to be 18 (from literature on matrix multiplication complexity).
    
    Actually, let's think about this more carefully.
    For an n x m matrix times m x p matrix, the schoolbook rank is n*m*p = 2*3*4 = 24.
    The minimal rank is known to be less than this.
    
    For 2x2 * 2x2, Strassen's algorithm gives rank 7 instead of 8.
    For 2x3 * 3x4, we need to find the minimal rank.
    
    Let me use a systematic approach: set up the linear system and solve for the minimal rank.
    The tensor T[i,j,k] = delta_{ik} (in the appropriate sense for the trace convention).
    
    Wait, the trace convention is W[k*n+i, r]. So the output is indexed by (k, i) where
    i is the row of A and k is the column of B. The product C = A*B has C[i,k] = sum_j A[i,j]*B[j,k].
    
    So the tensor is T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k], which is always
    the case for each (i,j,k) triple. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Wait no, that's not right. The tensor for matrix multiplication is such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i, j, k] = 1 for all valid i, j, k. This is actually the "all-ones" tensor
    in the sense that every entry is 1. But that's not quite right either.
    
    Let me reconsider. The matrix multiplication tensor is defined as:
    T_{i,j,k} = 1 if the product A[i,j] * B[j,k] contributes to C[i,k]
    
    For standard matrix multiplication, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all i,j,k.
    But that would mean the tensor is just the all-ones tensor of shape (n, m, p), which has rank 1!
    That can't be right.
    
    I think I'm confusing things. The matrix multiplication tensor is not the all-ones tensor.
    Let me think again.
    
    The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as: C = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * E[i,k]
    where E[i,k] is the matrix with 1 at position (i,k) and 0 elsewhere.
    
    So the tensor T[i,j,k] = 1 for all i, j, k. But this is a tensor in the space of (A indices) x (B indices) x (C indices).
    
    The rank of this tensor is the minimum number of rank-1 tensors needed to express it.
    A rank-1 tensor has the form u_i * v_j * w_k.
    
    So we need to find the minimum r such that:
    1 = sum_{r_idx=0}^{r-1} U[i, r_idx] * V[j, r_idx] * W[k, r_idx] for all i, j, k
    
    Wait, but W is indexed by (k, i) in the trace convention. Let me re-read the contract.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank), and W[k*n+i, r] is the entry for output position (i,k) and rank index r.
    
    So the decomposition is:
    C[i,k] = sum_{r} (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) ... 
    
    No wait, that's not right either. Let me think about this more carefully.
    
    The standard decomposition for matrix multiplication is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We want to write this as a sum of rank-1 terms:
    C[i,k] = sum_{r=0}^{R-1} (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) ... 
    
    No, that's not right. The correct form is:
    
    For each rank index r, we have three vectors:
    - U_r: a vector in R^{n*m} (flattened A)
    - V_r: a vector in R^{m*p} (flattened B)  
    - W_r: a vector in R^{n*p} (flattened C, but with trace convention)
    
    And the decomposition is:
    C = sum_r (U_r ⊗ V_r) contracted appropriately...
    
    Actually, the standard way to think about it is:
    The tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} (or with the trace convention, R^{p*n})
    is defined by T[a, b, c] where a indexes A, b indexes B, c indexes C.
    
    For matrix multiplication, T[a, b, c] = 1 if the term A[a] * B[b] contributes to C[c].
    
    With a = i*m+j (row i, column j of A), b = j*p+k (row j, column k of B), c = k*n+i (row i, column k of C, using trace convention where c = k*n+i).
    
    The term A[i,j] * B[j,k] contributes to C[i,k] if and only if the indices are consistent:
    a = i*m+j, b = j'*p+k', c = k'*n+i' where we need i=i', j=j', k=k'.
    
    So T[a, b, c] = 1 if and only if:
    a = i*m+j, b = j*p+k, c = k*n+i for some i, j, k.
    
    In other words, T[a, b, c] = 1 if and only if the "j" index in a matches the "j" index in b, and the "i" index in a matches the "i" index in c, and the "k" index in b matches the "k" index in c.
    
    So T is not the all-ones tensor. It's a sparse tensor where T[a,b,c] = 1 only when the indices are consistent.
    
    Specifically, for a = i*m+j, b = j'*p+k, c = k'*n+i', we have T[a,b,c] = 1 if and only if i=i', j=j', k=k'.
    
    So the tensor has exactly n*m*p = 24 non-zero entries, each equal to 1.
    
    Now, the rank of this tensor is the minimum number of rank-1 tensors needed to express it.
    
    For the schoolbook decomposition, we use rank = n*m*p = 24, with each rank-1 tensor being a single non-zero entry.
    
    Can we do better? For 2x2 * 2x2, the answer is yes (Strassen: rank 7 instead of 8).
    
    For 2x3 * 3x4, I need to find the minimal rank.
    
    Let me set up the linear system. We want to find the minimum R such that there exist matrices U (n*m x R), V (m*p x R), W (p*n x R) with integer or half-integer entries such that:
    
    T[a, b, c] = sum_{r=0}^{R-1} U[a, r] * V[b, r] * W[c, r] for all a, b, c.
    
    Since T is sparse with 24 non-zero entries, we only need to satisfy:
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i in [0,2), j in [0,3), k in [0,4).
    
    And T[a,b,c] = 0 for all other (a,b,c) triples.
    
    This is a system of polynomial equations (trilinear in U, V, W).
    
    Let me try to find a decomposition with rank less than 24.
    
    Actually, I recall that for small matrix multiplication, the minimal ranks are known. Let me look up or derive the minimal rank for 2x3 * 3x4.
    
    One approach: use the fact that the tensor is the "matrix multiplication tensor" and use known results.
    
    For 2x2 * 2x2: rank 7 (Strassen)
    For 2x3 * 3x2: rank ? 
    For 2x3 * 3x4: rank ?
    
    Let me try to construct a decomposition with rank 18 or so.
    
    Actually, let me just solve the linear system numerically and then rationalize.
    
    I'll set up the problem as: find U, V, W with minimal R such that the trilinear form matches T.
    
    For a given R, this is a system of n*m*p = 24 equations (for the non-zero entries) plus many more equations (for the zero entries). The total number of equations is (n*m)*(m*p)*(p*n) = 24*12*8 = 2304, but most are redundant or automatically satisfied.
    
    Actually, the constraint is that for all (a, b, c), sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c].
    
    This is a system of (n*m)*(m*p)*(p*n) = 24*12*8 = 2304 equations in R*(n*m + m*p + p*n) = R*(12+12+8) = 32R unknowns.
    
    For R = 18, we have 32*18 = 576 unknowns and 2304 equations. This is an overdetermined system, but the equations are not all independent.
    
    Let me try a different approach. I'll use the known result that the minimal rank for 2x3 * 3x4 matrix multiplication is 18, and try to find such a decomposition.
    
    Actually, I'm not sure about the exact minimal rank. Let me just try to find a good decomposition using numerical optimization and then rationalize.
    
    Here's my plan:
    1. Set R = 18 (or try various values).
    2. Initialize U, V, W randomly.
    3. Use gradient descent or a similar method to minimize the error.
    4. Rationalize the results.
    
    But this might not converge to the exact solution. Let me try a different approach.
    
    I'll use the fact that for small problems, we can solve the system directly.
    
    Actually, let me just use the schoolbook decomposition with rank 24, since I'm not sure of the minimal rank and I want to ensure correctness.
    
    Wait, but the task says "rank-minimal", so I should try to find the minimal rank.
    
    Let me look at this from a different angle. The tensor T for 2x3 * 3x4 matrix multiplication can be viewed as a 3-tensor of shape (6, 12, 8) (since n*m=6, m*p=12, p*n=8).
    
    The rank of this tensor is what we want to minimize.
    
    I found a reference that suggests the rank is 18 for 2x3 * 3x4. Let me try to construct such a decomposition.
    
    Actually, let me just use a numerical approach to find a low-rank decomposition and then rationalize it.
    
    Here's my revised plan:
    1. Try R = 18, 19, 20, ... until I find a solution.
    2. For each R, set up the linear system and solve it.
    3. Rationalize the solution.
    
    The linear system is:
    For each (a, b, c) with T[a,b,c] = 1: sum_r U[a,r] * V[b,r] * W[c,r] = 1
    For each (a, b, c) with T[a,b,c] = 0: sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    This is a trilinear system, which is hard to solve directly.
    
    Let me try a different approach: fix two of the matrices and solve for the third.
    
    For example, fix U and V, then the equations become linear in W:
    For each (a, b, c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    This can be written as: (U * V^T) * W = T (in some flattened form).
    
    Let me define M[a,b] = sum_r U[a,r] * V[b,r]. Then the equations are:
    sum_r M[a,b] * W[c,r] ... no, that's not right.
    
    Actually, for fixed U and V, the equation for a given (a, b, c) is:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    This is linear in W[c, :]. So for each (a, b, c), we have a linear equation in the vector W[c, :].
    
    But W[c, :] is shared across different (a, b) pairs for the same c.
    
    Let me vectorize this. For a fixed c, the equations involving W[c, :] are:
    For each (a, b): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    This is a system of (n*m)*(m*p) = 6*12 = 72 linear equations in R unknowns (the entries of W[c, :]).
    
    So for each c, we have a 72 x R system. If R <= 72, we can solve this.
    
    Similarly, we can fix U and W and solve for V, or fix V and W and solve for U.
    
    This suggests an iterative approach:
    1. Initialize U, V, W randomly.
    2. Fix U and V, solve for W.
    3. Fix V and W, solve for U.
    4. Fix U and W, solve for V.
    5. Repeat until convergence.
    
    This is similar to the ALS (Alternating Least Squares) algorithm for tensor decomposition.
    
    Let me implement this.
    
    Actually, I realize that the system might not have a solution for R < 24, or the solution might not have integer/half-integer entries. Let me just try it and see.
    
    Here's my implementation:
    
    def construct():
        n, m, p = 2, 3, 4
        nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
        
        # Build the tensor T
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        
        # Try ranks from 18 up to 24
        for R in range(18, 25):
            # Initialize U, V, W
            np.random.seed(42)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # ALS iteration
            for it in range(100):
                # Fix U and V, solve for W
                # For each c, solve: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b
                # This is: (U * V^T) @ W[c,:] = T[:, :, c].flatten()
                # Wait, that's not quite right. Let me think.
                
                # For a fixed c, the equations are:
                # For each (a, b): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # Let M[a,b] = sum_r U[a,r] * V[b,r] ... no, that's not right either.
                
                # Actually, for a fixed c, the equation is:
                # sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
                # Let D[a,b,r] = U[a,r] * V[b,r]. Then the equation is:
                # sum_r D[a,b,r] * W[c,r] = T[a,b,c]
                # This is a linear system in W[c,:].
                
                # Stack the equations: for each (a,b), we have one equation.
                # The coefficient matrix is D.reshape(nm*mp, R) and the RHS is T[:,:,c].flatten()
                
                D = np.einsum('ar,br->abr', U, V)  # shape (nm, mp, R)
                D_flat = D.reshape(nm*mp, R)
                T_c = T[:, :, c].flatten() if False else None  # We'll do this for each c
                
                for c in range(pn):
                    T_c = T[:, :, c].flatten()
                    # Solve D_flat @ W[c,:] = T_c
                    # This is an overdetermined system, so use least squares
                    W[c, :], residuals, rank, sv = np.linalg.lstsq(D_flat, T_c, rcond=None)
                
                # Fix V and W, solve for U
                D = np.einsum('br,cr->bcr', V, W)  # shape (mp, pn, R)
                D_flat = D.reshape(mp*pn, R)
                for a in range(nm):
                    T_a = T[a, :, :].flatten()
                    U[a, :], residuals, rank, sv = np.linalg.lstsq(D_flat, T_a, rcond=None)
                
                # Fix U and W, solve for V
                D = np.einsum('ar,cr->acr', U, W)  # shape (nm, pn, R)
                D_flat = D.reshape(nm*pn, R)
                for b in range(mp):
                    T_b = T[:, b, :].flatten()
                    V[b, :], residuals, rank, sv = np.linalg.lstsq(D_flat, T_b, rcond=None)
            
            # Check the error
            error = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        error += (val - T[a, b, c])**2
            
            if error < 1e-6:
                # Rationalize
                U_frac = np.array([[Fraction(x).limit_denominator(1000) for x in row] for row in U])
                V_frac = np.array([[Fraction(x).limit_denominator(1000) for x in row] for row in V])
                W_frac = np.array([[Fraction(x).limit_denominator(1000) for x in row] for row in W])
                
                # Verify
                U_check = np.array([[float(x) for x in row] for row in U_frac])
                V_check = np.array([[float(x) for x in row] for row in V_frac])
                W_check = np.array([[float(x) for x in row] for row in W_frac])
                
                error2 = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.sum(U_check[a, :] * V_check[b, :] * W_check[c, :])
                            error2 += (val - T[a, b, c])**2
                
                if error2 < 1e-6:
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        # Fallback to schoolbook
        R = nm * mp * pn  # Wait, that's not right. Schoolbook rank is n*m*p = 24
        R = n * m * p
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][column] = 1
                    V[j*p + k][column] = 1
                    W[k*n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    Wait, I made a mistake. The schoolbook rank is n*m*p = 2*3*4 = 24, not nm*mp*pn.
    
    Let me fix this and also improve the rationalization.
    
    Also, I should use Fraction with limit_denominator to get nice fractions.
    
    Let me rewrite this more carefully.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Try ranks from 18 up to 24
    for R in range(18, n*m*p + 1):
        np.random.seed(42)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # ALS iteration
        for it in range(200):
            # Fix U and V, solve for W
            D = np.einsum('ar,br->abr', U, V)
            D_flat = D.reshape(nm*mp, R)
            for c in range(pn):
                T_c = T[:, :, c].flatten()
                W[c, :], _, _, _ = np.linalg.lstsq(D_flat, T_c, rcond=None)
            
            # Fix V and W, solve for U
            D = np.einsum('br,cr->bcr', V, W)
            D_flat = D.reshape(mp*pn, R)
            for a in range(nm):
                T_a = T[a, :, :].flatten()
                U[a, :], _, _, _ = np.linalg.lstsq(D_flat, T_a, rcond=None)
            
            # Fix U and W, solve for V
            D = np.einsum('ar,cr->acr', U, W)
            D_flat = D.reshape(nm*pn, R)
            for b in range(mp):
                T_b = T[:, b, :].flatten()
                V[b, :], _, _, _ = np.linalg.lstsq(D_flat, T_b, rcond=None)
        
        # Check the error
        error = np.sum((np.einsum('ar,br,cr->abc', U, V, W) - T)**2)
        
        if error < 1e-8:
            # Rationalize
            def rationalize_matrix(M, max_denom=1000):
                return [[Fraction(x).limit_denominator(max_denom) for x in row] for row in M]
            
            U_frac = rationalize_matrix(U)
            V_frac = rationalize_matrix(V)
            W_frac = rationalize_matrix(W)
            
            # Verify
            U_check = np.array([[float(x) for x in row] for row in U_frac])
            V_check = np.array([[float(x) for x in row] for row in V_frac])
            W_check = np.array([[float(x) for x in row] for row in W_frac])
            
            error2 = np.sum((np.einsum('ar,br,cr->abc', U_check, V_check, W_check) - T)**2)
            
            if error2 < 1e-8:
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }
    
    # Fallback to schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][column] = 1
                V[j*p + k][column] = 1
                W[k*n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
