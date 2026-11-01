"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    Uses a known rank-11 decomposition (Strassen for 2x2 block + schoolbook for 3rd column)
    and attempts to find a lower rank via SVD-based subspace identification and lattice
    projection (LLL-style rounding), falling back to the rank-11 solution if not found.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    _deadline = time.time() + 130  # safety margin before 150s kill
    
    # Known rank-11 decomposition: Strassen (7) for cols 0,1 + schoolbook (4) for col 2
    rank_11 = 11
    U = [[0] * rank_11 for _ in range(n * m)]
    V = [[0] * rank_11 for _ in range(m * p)]
    W = [[0] * rank_11 for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_11
    
    # Attempt to find lower rank (10, 9, 8) via SVD + lattice projection
    for r in [10, 9, 8]:
        if time.time() > _deadline:
            break
        
        result = _try_lower_rank_svd(n, m, p, r, _deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result


def _try_lower_rank_svd(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition using SVD to identify the optimal subspace,
    then project onto the integer/half-integer lattice.
    
    The tensor T[i,j,k] = 1 for all valid (i,j,k).
    We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Strategy:
    1. Use the known rank-11 decomposition to get a starting point.
    2. Construct the "flattened" tensor matrix and use SVD to find the best rank-r approximation.
    3. Project the SVD solution onto the half-integer lattice and verify.
    """
    np.random.seed(42 + r)
    
    # Get the known rank-11 solution as initial reference
    U11 = np.array(_get_rank11_U(n, m, p), dtype=float)
    V11 = np.array(_get_rank11_V(n, m, p), dtype=float)
    W11 = np.array(_get_rank11_W(n, m, p), dtype=float)
    
    # Build the tensor in matrix form for SVD
    # The tensor T can be "flattened" in different ways.
    # One approach: for each output index (k*n+i), the contribution is:
    #   sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # Let's use a different strategy: random initialization + ALS with half-integer rounding
    # This is more likely to find exact solutions than pure SVD.
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Initialize near the rank-11 solution, keeping only r columns
        # Use a random subset or projection
        if r < 11:
            # Start from a random perturbation of a subset of the rank-11 columns
            idx = np.random.choice(11, size=min(r, 11), replace=False)
            if r > 11:
                idx = np.concatenate([idx, np.random.choice(11, size=r-11, replace=True)])
            U = U11[:, idx].copy()
            V = V11[:, idx].copy()
            W = W11[:, idx].copy()
            # Add small perturbation
            U += np.random.randn(n*m, r) * 0.1
            V += np.random.randn(m*p, r) * 0.1
            W += np.random.randn(p*n, r) * 0.1
        else:
            U = U11[:, :r].copy() + np.random.randn(n*m, r) * 0.1
            V = V11[:, :r].copy() + np.random.randn(m*p, r) * 0.1
            W = W11[:, :r].copy() + np.random.randn(p*n, r) * 0.1
        
        # ALS with half-integer projection
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # Compute current values
            vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        vals[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            
            loss = np.sum((vals - 1.0)**2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Update W: for each (k,i), solve linear system
            # sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
            # For fixed (k,i): sum_j [sum_c U[i*m+j,c] V[j*p+k,c]] * W[k*n+i,c] ... 
            # Actually this is: sum_c (sum_j U[i*m+j,c] V[j*p+k,c]) * W[k*n+i,c] = 1
            # This is linear in W[k*n+i, :]: A_W @ W[k*n+i, :] = 1 where A_W[c] = sum_j U[i*m+j,c] V[j*p+k,c]
            # But we have multiple (j) terms... Let me reconsider.
            
            # The equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] summed over j = 1
            # = sum_c [sum_j U[i*m+j,c] * V[j*p+k,c]] * W[k*n+i,c] = 1
            # So for each (i,k), we have a linear system in W[k*n+i, :]:
            #   (sum_j U[i*m+j,:] * V[j*p+k,:]) @ W[k*n+i, :] = 1
            # This is a scalar equation with r unknowns, so it's underdetermined.
            # We use least squares: W[k*n+i, :] = A^T (A A^T)^{-1} * 1
            
            for i in range(n):
                for k in range(p):
                    # A[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                    A_vec = np.zeros(r)
                    for j in range(m):
                        A_vec += U[i*m+j, :] * V[j*p+k, :]
                    # Solve A_vec @ w = 1 in least squares
                    if np.linalg.norm(A_vec) > 1e-12:
                        W[k*n+i, :] = A_vec / (np.dot(A_vec, A_vec))
            
            # Similarly update U
            for i in range(n):
                for j in range(m):
                    A_vec = np.zeros(r)
                    for k in range(p):
                        A_vec += V[j*p+k, :] * W[k*n+i, :]
                    if np.linalg.norm(A_vec) > 1e-12:
                        U[i*m+j, :] = A_vec / (np.dot(A_vec, A_vec))
            
            # Similarly update V
            for j in range(m):
                for k in range(p):
                    A_vec = np.zeros(r)
                    for i in range(n):
                        A_vec += U[i*m+j, :] * W[k*n+i, :]
                    if np.linalg.norm(A_vec) > 1e-12:
                        V[j*p+k, :] = A_vec / (np.dot(A_vec, A_vec))
            
            # Project to half-integers periodically
            if it % 10 == 9:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
        
        # Final verification with half-integer rounding
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        loss_rounded = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                    loss_rounded += (val - 1.0)**2
        
        if loss_rounded < 1e-6:
            U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n*m)]
            V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m*p)]
            W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try the unrounded best solution
        loss_unrounded = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(best_U[i*m+j, :], best_V[j*p+k, :] * best_W[k*n+i, :])
                    loss_unrounded += (val - 1.0)**2
        if loss_unrounded < 1e-6:
            # Try rounding
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            lr = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        lr += (val - 1.0)**2
            if lr < 1e-6:
                U_list = [[float(U_r[i, c]) for c in range(r)] for i in range(n*m)]
                V_list = [[float(V_r[i, c]) for c in range(r)] for i in range(m*p)]
                W_list = [[float(W_r[i, c]) for c in range(r)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _get_rank11_U(n, m, p):
    U = [[0] * 11 for _ in range(n * m)]
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    U[0][7] = 1
    U[1][8] = 1
    U[2][9] = 1
    U[3][10] = 1
    return U

def _get_rank11_V(n, m, p):
    V = [[0] * 11 for _ in range(m * p)]
    V[0][0] = 1; V[4][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[4][2] = -1
    V[3][3] = 1; V[0][3] = -1
    V[4][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[3][6] = 1; V[4][6] = 1
    V[2][7] = 1
    V[5][8] = 1
    V[2][9] = 1
    V[5][10] = 1
    return V

def _get_rank11_W(n, m, p):
    W = [[0] * 11 for _ in range(p * n)]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1
    return W
# EVOLVE-BLOCK-END
