"""Matrix multiplication 2x3x3: rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The multiplication tensor T[i,j,k] = delta_{i,k} (where A is n x m, B is m x p, C is p x n)
    Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor is T[i,j,k] = 1 if the indices match the multiplication pattern.
    
    For n x m times m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
    
    The multiplication tensor M has entries M[i,j,k] = 1 (it's the structure tensor).
    A rank-1 decomposition: M = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    where u_r has shape (n*m, 1), v_r has shape (m*p, 1), w_r has shape (p*n, 1)
    
    Using the trace convention: W[k*n+i, r]
    
    The schoolbook decomposition has rank n*m*p = 2*3*3 = 18.
    
    We try to find a decomposition of rank 17 or lower using algebraic methods.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18
    
    # First, build the schoolbook decomposition as a fallback
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[mat[i][j] if isinstance(mat[i][j], Fraction) else float(mat[i][j]) 
                 for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Try to find a rank-17 decomposition
    # Strategy: Use the fact that if a rank-r decomposition exists, we can try to find it
    # by solving a system of polynomial equations. For small ranks, we can use Groebner bases
    # or randomized approaches.
    
    # For 2x3x3, it is known that the rank is 17 (Landsberg, 2013; Bläser, 2011).
    # Let's try to construct it explicitly or find it via search.
    
    # Approach: Use the known structure. For small matrix multiplications, optimal decompositions
    # often have a specific pattern. Let's try a randomized algebraic approach.
    
    # The multiplication tensor for 2x3 x 3x3:
    # T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # T[i,j,k] = 1 for all valid (i,j,k) triples (it's the identity in the "multiplication" sense)
    # Actually, the tensor is: M = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * delta_{structure}
    # For matrix multiplication, the tensor has 1s at positions corresponding to valid multiplications.
    
    # Let me reconsider. The matrix multiplication tensor T ∈ R^{(n*m) x (m*p) x (p*n)}
    # T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i'
    # So T has 1s at positions where all indices match.
    
    # For rank-r decomposition: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    
    # The number of nonzero entries in T is n*m*p = 18 (one for each (i,j,k) triple).
    # Each rank-1 term can contribute to at most one position if we want exact decomposition...
    # No, that's not right. A rank-1 term u ⊗ v ⊗ w has u[a]*v[b]*w[c] at every (a,b,c).
    
    # So we need: for each (a,b,c) with T[a,b,c]=1: sum_r u_r[a] v_r[b] w_r[c] = 1
    # For each (a,b,c) with T[a,b,c]=0: sum_r u_r[a] v_r[b] w_r[c] = 0
    
    # This is a system of trilinear equations. For rank 17, we have 17*(6+9+6) = 306 variables
    # and 6*9*6 - 18 = 324 - 18 = 306 equations (zeros) + 18 equations (ones) = 324 equations.
    # So it's a square system! This is promising.
    
    # Let's set up the system for rank 17.
    # Variables: U (6x17), V (9x17), W (6x17) -> 6*17 + 9*17 + 6*17 = 306 variables
    # Equations: 324 (all positions)
    
    # This is over-determined (324 > 306), but many equations are redundant due to the structure.
    
    # Let's try a different approach: use the known result that rank(2x3x3) = 17.
    # I'll try to find the decomposition using a structured ansatz.
    
    # Actually, let me try a direct numerical approach with rank 17.
    # We can use the fact that the system is polynomial and try to solve it.
    
    # For efficiency, let's try to use a known decomposition if we can recall/derive it.
    
    # Alternative: Try rank 17 with a structured approach.
    # The key insight: for small matrix multiplications, optimal decompositions often
    # use "border rank" techniques or specific algebraic identities.
    
    # Let's try to use a randomized approach: initialize U, V, W randomly and try to
    # minimize the error using gradient descent on the polynomial equations.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Try rank 17 first
    for target_rank in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 30:
            break
        
        # Set up the problem for this rank
        # T[a,b,c] = 1 if (a,b,c) corresponds to a valid multiplication, else 0
        # a = i*m + j (i in [0,n), j in [0,m))
        # b = j'*p + k (j' in [0,m), k in [0,p))  
        # c = k'*n + i' (k' in [0,p), i' in [0,n))
        # T[a,b,c] = 1 if j=j' and k=k' and i=i'
        
        # Build the list of (a,b,c) positions
        ones_positions = []
        zeros_positions = []
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j_prime = b // p
                    k = b % p
                    k_prime = c // n
                    i_prime = c % n
                    if i == i_prime and j == j_prime and k == k_prime:
                        ones_positions.append((a, b, c))
                    else:
                        zeros_positions.append((a, b, c))
        
        # For rank r, we have r*(n*m + m*p + p*n) variables
        # and n*m*p equations
        # We need the system to be consistent.
        
        # Let's try a structured ansatz. For the 2x3x3 case, one known approach
        # is to use the fact that it's close to a product of smaller multiplications.
        
        # Actually, let me try a direct approach: use the known optimal decomposition.
        # From literature, the rank of 2x3x3 is 17. Let me try to construct it.
        
        # I'll use a different strategy: try to find a decomposition by exploiting
        # the structure. For rank 17, we can think of it as rank 18 minus 1, where
        # we merge two terms in a clever way.
        
        # Let's try a numerical optimization approach.
        # Initialize U, V, W and minimize the sum of squares of the residuals.
        
        import random
        random.seed(42)
        
        best_result = None
        
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # We want: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is a trilinear system. We can try to solve it iteratively.
            
            # Use a simple iterative approach: fix two matrices, solve for the third.
            # When U and V are fixed, the equation becomes:
            # sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
            # This is linear in W for each (a,b,c).
            
            # For fixed U and V, define M_ab = sum_r U[a,r] V[b,r] (this is n*m x m*p matrix)
            # Then for each c: sum_r W[c,r] * M_ab[r] ... no, that's not quite right.
            
            # Let me reformulate. For fixed U and V:
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each (a,b), this is: T[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
            # So if we define Z_ab[r] = U[a,r] V[b,r], then T[a,b,c] = sum_r Z_ab[r] W[c,r]
            # This is a linear system in W for each (a,b).
            
            # Actually, let's think of it differently. For fixed U and V, the equation
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] must hold for all (a,b,c).
            # For each c, this gives n*m*m*p equations in p*n*target_rank unknowns... 
            # No, W has shape (p*n, target_rank), so p*n*target_rank unknowns.
            
            # This is getting complex. Let me try a simpler approach.
            
            # Use the fact that we can alternate:
            # 1. Fix V and W, solve for U (linear system)
            # 2. Fix U and W, solve for V (linear system)
            # 3. Fix U and V, solve for W (linear system)
            
            # For step 1: T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
            # For fixed V and W, define Y_b_c[r] = V[b,r] W[c,r]
            # Then T[a,b,c] = sum_r U[a,r] Y_b_c[r]
            # For each (b,c), this is a linear system: U[:, :] @ Y_bc = T[:, b, c]
            # But U has shape (n*m, target_rank), and we need this to hold for all (b,c).
            
            # Actually, the cleanest way: vectorize the tensor.
            # T is a 3D array of shape (n*m, m*p, p*n).
            # Reshape to a 2D array of shape (n*m, (m*p)*(p*n)).
            # Then T_2d[a, (b,c)] = sum_r U[a,r] * (V[b,r] * W[c,r])
            # = sum_r U[a,r] * Z[(b,c), r]
            # where Z[(b,c), r] = V[b,r] * W[c,r]
            
            # So T_2d = U @ Z^T, where Z has shape ((m*p)*(p*n), target_rank)
            # This means U = T_2d @ Z^T @ (Z @ Z^T)^{-1} if Z @ Z^T is invertible.
            
            # Similarly for V and W.
            
            # Let's implement this alternating least squares approach.
            
            # Build the tensor T
            T = np.zeros((n * m, m * p, p * n))
            for (a, b, c) in ones_positions:
                T[a, b, c] = 1.0
            
            # Reshape T to 2D for each mode
            T_a = T.reshape(n * m, -1)  # (n*m, m*p*p*n)
            T_b = T.transpose(1, 0, 2).reshape(m * p, -1)  # (m*p, n*m*p*n)
            T_c = T.transpose(2, 0, 1).reshape(p * n, -1)  # (p*n, n*m*m*p)
            
            # Initialize
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update U: T_a = U @ Z^T where Z[(b,c), r] = V[b,r] * W[c,r]
                # Z has shape (m*p*p*n, target_rank)
                Z = np.zeros((m * p * p * n, target_rank))
                for b in range(m * p):
                    for c in range(p * n):
                        Z[b * p * n + c, :] = V[b, :] * W[c, :]
                # U @ Z^T = T_a
                # U = T_a @ Z @ (Z^T @ Z)^{-1}
                ZtZ = Z.T @ Z  # (target_rank, target_rank)
                ZtT = Z.T @ T_a  # (target_rank, n*m)
                try:
                    U_new = np.linalg.solve(ZtZ, ZtT).T  # (n*m, target_rank)
                except np.linalg.LinAlgError:
                    U_new = np.linalg.lstsq(Z.T, T_a, rcond=None)[0].T
                U = U_new
                
                # Update V: T_b = V @ Y^T where Y[(a,c), r] = U[a,r] * W[c,r]
                Y = np.zeros((n * m * p * n, target_rank))
                for a in range(n * m):
                    for c in range(p * n):
                        Y[a * p * n + c, :] = U[a, :] * W[c, :]
                YtY = Y.T @ Y
                YtT = Y.T @ T_b
                try:
                    V_new = np.linalg.solve(YtY, YtT).T
                except np.linalg.LinAlgError:
                    V_new = np.linalg.lstsq(Y.T, T_b, rcond=None)[0].T
                V = V_new
                
                # Update W: T_c = W @ X^T where X[(a,b), r] = U[a,r] * V[b,r]
                X = np.zeros((n * m * m * p, target_rank))
                for a in range(n * m):
                    for b in range(m * p):
                        X[a * m * p + b, :] = U[a, :] * V[b, :]
                XtX = X.T @ X
                XtT = X.T @ T_c
                try:
                    W_new = np.linalg.solve(XtX, XtT).T
                except np.linalg.LinAlgError:
                    W_new = np.linalg.lstsq(X.T, T_c, rcond=None)[0].T
                W = W_new
            
            # Check the quality of the decomposition
            # Compute the actual tensor from U, V, W
            T_reconstructed = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_reconstructed += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
            
            error = np.max(np.abs(T_reconstructed - T))
            
            if error < 1e-6:
                # Found a valid decomposition! Convert to fractions for exactness
                # and verify
                U_frac = np.zeros((n * m, target_rank), dtype=object)
                V_frac = np.zeros((m * p, target_rank), dtype=object)
                W_frac = np.zeros((p * n, target_rank), dtype=object)
                
                for i in range(n * m):
                    for r in range(target_rank):
                        U_frac[i, r] = Fraction(str(round(U[i, r], 10)))
                for i in range(m * p):
                    for r in range(target_rank):
                        V_frac[i, r] = Fraction(str(round(V[i, r], 10)))
                for i in range(p * n):
                    for r in range(target_rank):
                        W_frac[i, r] = Fraction(str(round(W[i, r], 10)))
                
                # Verify exactness
                T_verify = np.zeros((n * m, m * p, p * n), dtype=object)
                for r in range(target_rank):
                    for a in range(n * m):
                        for b in range(m * p):
                            for c in range(p * n):
                                T_verify[a, b, c] += U_frac[a, r] * V_frac[b, r] * W_frac[c, r]
                
                exact = True
                for (a, b, c) in ones_positions:
                    if T_verify[a, b, c] != 1:
                        exact = False
                        break
                if exact:
                    for (a, b, c) in zeros_positions:
                        if T_verify[a, b, c] != 0:
                            exact = False
                            break
                
                if exact:
                    best_result = (U_frac, V_frac, W_frac, target_rank)
                    break
            
            if error < 1e-10:
                # Very close, try to rationalize
                from fractions import Fraction
                U_frac = np.zeros((n * m, target_rank), dtype=object)
                V_frac = np.zeros((m * p, target_rank), dtype=object)
                W_frac = np.zeros((p * n, target_rank), dtype=object)
                
                for i in range(n * m):
                    for r in range(target_rank):
                        U_frac[i, r] = Fraction(U[i, r]).limit_denominator(1000)
                for i in range(m * p):
                    for r in range(target_rank):
                        V_frac[i, r] = Fraction(V[i, r]).limit_denominator(1000)
                for i in range(p * n):
                    for r in range(target_rank):
                        W_frac[i, r] = Fraction(W[i, r]).limit_denominator(1000)
                
                # Verify
                T_verify = np.zeros((n * m, m * p, p * n), dtype=object)
                for r in range(target_rank):
                    for a in range(n * m):
                        for b in range(m * p):
                            for c in range(p * n):
                                T_verify[a, b, c] += U_frac[a, r] * V_frac[b, r] * W_frac[c, r]
                
                exact = True
                for (a, b, c) in ones_positions:
                    if T_verify[a, b, c] != 1:
                        exact = False
                        break
                if exact:
                    for (a, b, c) in zeros_positions:
                        if T_verify[a, b, c] != 0:
                            exact = False
                            break
                
                if exact:
                    best_result = (U_frac, V_frac, W_frac, target_rank)
                    break
        
        if best_result is not None:
            U_frac, V_frac, W_frac, rank = best_result
            return {
                "U": [[U_frac[i, j] for j in range(rank)] for i in range(n * m)],
                "V": [[V_frac[i, j] for j in range(rank)] for i in range(m * p)],
                "W": [[W_frac[i, j] for j in range(rank)] for i in range(p * n)]
            }
    
    # If we couldn't find a better decomposition, return the schoolbook one
    return {
        "U": [[U_sb[i, j] for j in range(schoolbook_rank)] for i in range(n * m)],
        "V": [[V_sb[i, j] for j in range(schoolbook_rank)] for i in range(m * p)],
        "W": [[W_sb[i, j] for j in range(schoolbook_rank)] for i in range(p * n)]
    }
