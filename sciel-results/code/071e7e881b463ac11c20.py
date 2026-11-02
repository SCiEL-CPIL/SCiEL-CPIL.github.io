"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify rank-r decomposition of matrix multiplication tensor.
    Condition: for all i,k,j1,j2:
        sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] = delta_{j1,j2}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = np.dot(U[i*m+j1, :] * V[j2*p+k, :], W[k*n+i, :])
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _kronecker_decomposition(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
    """
    Combine decompositions of M_{n,m,p1} and M_{1,1,p2} to get M_{n,m,p1*p2}.
    
    M_{n,m,p1*p2}: C[i, k1*p2+k2] = sum_j A[i,j] * B[j, k1*p2+k2]
    
    We can write B[j, k1*p2+k2] as part of a block structure.
    
    For M_{1,1,p2}: computes D[0, k2] = sum_{j2} E[0, j2] * F[j2, k2]
    where E is 1×p2, F is p2×p2.
    
    The Kronecker product approach:
    - U_combined has shape (n*m, r1*r2)
    - V_combined has shape (m*p1*p2, r1*r2)  
    - W_combined has shape (p1*p2*n, r1*r2)
    
    For column c = c1*r2 + c2:
    - U_combined[i*m+j, c] = U1[i*m+j, c1] * U2[0, c2]
    - V_combined[j*p1*p2 + k1*p2 + k2, c] = V1[j*p1+k1, c1] * V2[k2, c2]
    - W_combined[(k1*p2+k2)*n + i, c] = W1[k1*n+i, c1] * W2[k2, c2]
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    U_comb = np.zeros((n * m, r))
    V_comb = np.zeros((m * p1 * p2, r))
    W_comb = np.zeros((p1 * p2 * n, r))
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            
            # U: (n*m, r1) x (1, r2) -> (n*m, r)
            U_comb[:, c] = U1[:, c1] * U2[0, c2]
            
            # V: (m*p1, r1) x (p2, r2) -> (m*p1*p2, r)
            for j in range(m):
                for k1 in range(p1):
                    for k2 in range(p2):
                        V_comb[j * p1 * p2 + k1 * p2 + k2, c] = \
                            V1[j * p1 + k1, c1] * V2[k2, c2]
            
            # W: (p1*n, r1) x (p2, r2) -> (p1*p2*n, r)
            for k1 in range(p1):
                for i in range(n):
                    for k2 in range(p2):
                        W_comb[(k1 * p2 + k2) * n + i, c] = \
                            W1[k1 * n + i, c1] * W2[k2, c2]
    
    return U_comb, V_comb, W_comb

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Strategy: Use Kronecker product with p = p1 * p2
    # For 2x3x9: p1=3, p2=3
    # Need M_{2,3,3} decomposition and M_{1,1,3} decomposition
    
    # M_{1,1,3}: schoolbook rank 3
    # U2: (1*1, 3) = (1, 3), V2: (1*3, 3) = (3, 3), W2: (3*1, 3) = (3, 3)
    U2 = np.ones((1, 3))
    V2 = np.eye(3)
    W2 = np.ones((3, 3))
    # Verify: for i=0, k=0, j1=0, j2=0: sum_c U2[0,c]*V2[j2*1+0,c]*W2[0*1+0,c]
    # = sum_c 1 * V2[j2,c] * 1 = sum_c V2[j2,c] = 1 if j2=0 else 0. Correct!
    
    # M_{2,3,3}: Let's try schoolbook first (rank 18)
    # Then try to find better
    n1, m1, p1 = 2, 3, 3
    rank1_sb = n1 * m1 * p1  # 18
    U1_sb = np.zeros((n1 * m1, rank1_sb))
    V1_sb = np.zeros((m1 * p1, rank1_sb))
    W1_sb = np.zeros((p1 * n1, rank1_sb))
    
    c = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1_sb[i * m1 + j, c] = 1.0
                V1_sb[j * p1 + k, c] = 1.0
                W1_sb[k * n1 + i, c] = 1.0
                c += 1
    
    # Try Kronecker with schoolbook M_{2,3,3}: rank = 18 * 3 = 54 (same as schoolbook)
    # Not helpful. Need better M_{2,3,3} decomposition.
    
    # Let's try to find a rank-9 decomposition for M_{2,3,3}
    # This would give rank 9*3 = 27 for M_{2,3,9}
    
    # For M_{2,3,3}, condition: sum_c U1[i*3+j1,c]*V1[j2*3+k,c]*W1[k*2+i,c] = delta_{j1,j2}
    # for i in {0,1}, k in {0,1,2}, j1,j2 in {0,1,2}
    
    # Try a structured rank-9 decomposition
    # Idea: For each (i,k), we need a 3x3 identity-like structure across j1,j2
    
    # Let's try: for each c in 0..8, assign to (i, k, j) pattern
    # c = i*3 + k for i in {0,1}, k in {0,1,2} gives 6 columns
    # Plus 3 more columns for the "off-diagonal" corrections?
    
    # Alternative: try random search for rank 9
    np.random.seed(42)
    
    found_rank9 = False
    for trial in range(50):
        if time.time() > deadline - 20.0:
            break
        
        U1 = np.random.randn(n1 * m1, 9) * 0.5
        V1 = np.random.randn(m1 * p1, 9) * 0.5
        W1 = np.random.randn(p1 * n1, 9) * 0.5
        
        if _verify_decomposition(U1, V1, W1, n1, m1, p1, tol=1e-6):
            found_rank9 = True
            break
    
    if found_rank9:
        U_comb, V_comb, W_comb = _kronecker_decomposition(
            n, m, p1, p // p1, U1, V1, W1, U2, V2, W2
        )
        if _verify_decomposition(U_comb, V_comb, W_comb, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U_comb, V_comb, W_comb
            best_rank = 27
    
    # If we didn't find rank 9 for M_{2,3,3}, try rank 12, 15, etc.
    if best_rank == rank_sb:
        for target_r1 in [12, 15]:
            if time.time() > deadline - 15.0:
                break
            
            for trial in range(30):
                if time.time() > deadline - 10.0:
                    break
                
                U1 = np.random.randn(n1 * m1, target_r1) * 0.5
                V1 = np.random.randn(m1 * p1, target_r1) * 0.5
                W1 = np.random.randn(p1 * n1, target_r1) * 0.5
                
                if _verify_decomposition(U1, V1, W1, n1, m1, p1, tol=1e-6):
                    U_comb, V_comb, W_comb = _kronecker_decomposition(
                        n, m, p1, p // p1, U1, V1, W1, U2, V2, W2
                    )
                    if _verify_decomposition(U_comb, V_comb, W_comb, n, m, p, tol=1e-6):
                        best_U, best_V, best_W = U_comb, V_comb, W_comb
                        best_rank = target_r1 * 3
                    break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
