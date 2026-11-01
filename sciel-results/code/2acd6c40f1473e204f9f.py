"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 28
    
    # Build the tensor T[a, b, c] where:
    # a = i*m + j, b = jp*p + kp, c = kp*n + ip
    # T[a, b, c] = 1 if i==ip and j==jp and k==kp else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                a = i*m + j
                                b = jp*p + k
                                c = kp*n + ip
                                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # STRUCTURAL CHANGE: Use a Strassen-inspired construction for rank 25
    # The key insight: M_{2,2,7} can be decomposed using 25 bilinear terms
    # by exploiting the structure that A is only 2x2 (4 entries) while B is 2x7 (14 entries)
    
    # Strategy: Use a "column-sharing" approach where we group columns and
    # use Strassen-like products within groups, then combine.
    
    # For rank 25: We can think of it as 5 groups of... no.
    # Alternative: Direct structured search for rank 25 with small integer coefficients.
    
    # Let's try a different structural approach: 
    # Decompose M_{2,2,7} = sum of 25 rank-1 tensors
    # Each rank-1 tensor is u ⊗ v ⊗ w where:
    #   u ∈ R^4 (indices i,j)
    #   v ∈ R^14 (indices j',k)  
    #   w ∈ R^14 (indices k',i')
    
    # The constraint: for all (i,j,j',k,k',i'):
    #   sum_r u_r[i,j] * v_r[j',k] * w_r[k',i'] = delta(i,i')*delta(j,j')*delta(k,k')
    
    # Structural approach: Use the "substitution" method.
    # View B as having 7 columns. For each column, the 2x2 multiplication has rank <= 7 (Strassen).
    # But we can share the A-side linear combinations across columns.
    
    # Key construction for rank 25:
    # Use 7 Strassen products for the "diagonal" structure, plus 18 additional terms.
    # Actually, let's use a known explicit construction.
    
    # M_{2,2,7} rank 25 construction:
    # We use the fact that we can write:
    # C = A * B where C[i,k] = sum_j A[i,j] B[j,k]
    
    # Define linear combinations:
    # For A (2x2): 7 Strassen combinations
    # For B (2x7): we need to be clever about column grouping
    
    # Let me try: partition the 7 columns into groups and use different strategies.
    # 7 = 3 + 2 + 2. Use rank-4 for groups of 1 column, rank-7 for pairs...
    
    # Actually, the cleanest structural approach: 
    # Use a random search but with STRUCTURED initialization based on Strassen patterns.
    
    np.random.seed(12345)
    
    # Try ranks from 25 to 27 (skip 28 since we have schoolbook)
    for target_rank in [25, 26, 27]:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Multiple trials with structured initialization
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            r = target_rank
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Structured initialization: 
            # First 7 columns: Strassen-like pattern for each "virtual" column group
            # Use the 7 Strassen products as a template
            
            # Strassen coefficients for 2x2 multiplication
            # M1 = (a11+a22)(b11+b22)
            # M2 = (a21+a22)b11
            # M3 = a11(b21-b11)
            # M4 = a22(b11-b21)
            # M5 = (a11+a12)b21
            # M6 = (a21-a11)(b11+b21)
            # M7 = (a12-a22)(b21+b11)
            # c11 = M1 - M4 + M3 + M7
            # c21 = M1 + M2 - M5 - M3
            # c12 = M4 + M5
            # c22 = M1 - M2 + M3 + M6
            
            # For our case: A is 2x2, B is 2x7
            # C[i,k] = sum_j A[i,j] B[j,k]
            
            # Strassen products for column k:
            # M1_k = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])
            # M2_k = (A[1,0]+A[1,1]) * B[0,k]
            # M3_k = A[0,0] * (B[1,k]-B[0,k])
            # M4_k = A[1,1] * (B[0,k]-B[1,k])
            # M5_k = (A[0,0]+A[0,1]) * B[1,k]
            # M6_k = (A[1,0]-A[0,0]) * (B[0,k]+B[1,k])
            # M7_k = (A[0,1]-A[1,1]) * (B[1,k]+B[0,k])
            
            # C[0,k] = M1_k - M4_k + M3_k + M7_k
            # C[1,k] = M1_k + M2_k - M5_k - M3_k
            
            # For 7 columns, this gives 49 products. We need to reduce to 25.
            
            # Idea: Share some of the B-side combinations across columns.
            # If we use the same B-side linear combination for multiple columns,
            # we can reduce the count.
            
            # For example, use (B[0,k]+B[1,k]) for all k: that's 7 terms.
            # Use B[0,k] for all k: that's 7 terms.
            # Use B[1,k] for all k: that's 7 terms.
            # Use (B[1,k]-B[0,k]) for all k: that's 7 terms.
            # Total B-side: 28 terms, but with A-side sharing...
            
            # This is getting complex. Let me try a direct numerical approach:
            # Use ALS but with a better initialization and more iterations.
            
            # Initialize with random values scaled appropriately
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Add Strassen-inspired structure to some columns
            if r >= 7:
                # Set first 7 columns to Strassen pattern for column 0
                strassen_A = [
                    [1, 1, 0, 0, 1, 0, 0],   # A[0,0] coefficients
                    [0, 0, 1, 0, 0, -1, -1],  # A[0,1] coefficients  
                    [1, 1, 0, 0, 0, 1, 0],   # A[1,0] coefficients
                    [1, 1, 0, 0, 0, 0, 1],   # A[1,1] coefficients
                ]
                strassen_B_col0 = [
                    [1, 0, 0, 0, 0, 1, 1],   # B[0,0] coefficients
                    [1, 1, 1, 1, 1, 1, 1],   # B[1,0] coefficients
                ]
                
                for rr in range(7):
                    U[0, rr] = strassen_A[0][rr]  # A[0,0]
                    U[1, rr] = strassen_A[1][rr]  # A[0,1]
                    U[2, rr] = strassen_A[2][rr]  # A[1,0]
                    U[3, rr] = strassen_A[3][rr]  # A[1,1]
                    V[0, rr] = strassen_B_col0[0][rr]  # B[0,0]
                    V[1, rr] = strassen_B_col0[1][rr]  # B[1,0]
                    W[0, rr] = 1  # C[0,0]
                    W[1, rr] = 1  # C[1,0]
            
            # ALS iterations
            for iteration in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Update W
                A_mat = np.zeros((n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        for rr in range(r):
                            A_mat[a * m*p + b, rr] = U[a, rr] * V[b, rr]
                
                B_mat = np.zeros((n*m * m*p, p*n))
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            B_mat[a * m*p + b, c] = T[a, b, c]
                
                try:
                    W = np.linalg.lstsq(A_mat, B_mat, rcond=None)[0].T
                except:
                    break
                
                # Update U
                A_mat = np.zeros((m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        for rr in range(r):
                            A_mat[b * p*n + c, rr] = V[b, rr] * W[c, rr]
                
                B_mat = np.zeros((m*p * p*n, n*m))
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            B_mat[b * p*n + c, a] = T[a, b, c]
                
                try:
                    U = np.linalg.lstsq(A_mat, B_mat, rcond=None)[0].T
                except:
                    break
                
                # Update V
                A_mat = np.zeros((n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        for rr in range(r):
                            A_mat[a * p*n + c, rr] = U[a, rr] * W[c, rr]
                
                B_mat = np.zeros((n*m * p*n, m*p))
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            B_mat[a * p*n + c, b] = T[a, b, c]
                
                try:
                    V = np.linalg.lstsq(A_mat, B_mat, rcond=None)[0].T
                except:
                    break
            
            # Check if the decomposition works
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
