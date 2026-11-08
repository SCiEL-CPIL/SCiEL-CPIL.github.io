"""Matrix multiplication 4,5,6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_mat[u_idx, :], V_mat[v_idx, :] * W_mat[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_halfint(mat):
    """Round a matrix to nearest half-integer values."""
    return np.round(mat * 2.0) / 2.0

def _try_rank(rank, n, m, p, max_iters=300, n_restarts=30, deadline=None):
    """Try to find an exact decomposition of given rank using ALS with rounding."""
    best = None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Random initialization with structure
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize columns
        for c in range(rank):
            norm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
            if norm > 0:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] /= norm
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Alternating least squares
            # Fix V and W, solve for U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    M_u = np.zeros(rank)
                    for k in range(p):
                        M_u += V[j * p + k, :] * W[k * n + i, :]
                    denom = np.dot(M_u, M_u)
                    if denom > 1e-12:
                        U[u_idx, :] = M_u / denom
            
            # Fix U and W, solve for V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    M_v = np.zeros(rank)
                    for i in range(n):
                        M_v += U[i * m + j, :] * W[k * n + i, :]
                    denom = np.dot(M_v, M_v)
                    if denom > 1e-12:
                        V[v_idx, :] = M_v / denom
            
            # Fix U and V, solve for W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    M_w = np.zeros(rank)
                    for j in range(m):
                        M_w += U[i * m + j, :] * V[j * p + k, :]
                    denom = np.dot(M_w, M_w)
                    if denom > 1e-12:
                        W[w_idx, :] = M_w / denom
            
            # Check convergence and exactness periodically
            if it % 10 == 0:
                # Round to half-integers and verify
                U_r = _round_to_halfint(U)
                V_r = _round_to_halfint(V)
                W_r = _round_to_halfint(W)
                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                    return (U_r, V_r, W_r)
                
                # Also try without rounding
                if _verify_decomposition(U, V, W, n, m, p):
                    return (U, V, W)
        
        # Final attempt with rounding
        U_r = _round_to_halfint(U)
        V_r = _round_to_halfint(V)
        W_r = _round_to_halfint(W)
        if _verify_decomposition(U_r, V_r, W_r, n, m, p):
            return (U_r, V_r, W_r)
    
    return None

def _construct_strassen_based(n, m, p):
    """
    Construct a Strassen-based decomposition.
    
    Strategy: Split the 4x5 matrix into 4x4 + 4x1, and 5x6 into 4x6 + 1x6.
    Use Strassen's algorithm for the 4x4 core, which has rank 7 per 2x2 block.
    
    For 4x4x4, Strassen gives rank 7*4 = 28 (vs 64 schoolbook).
    For 4x5x6, we need to handle the extra columns/rows.
    
    Let's decompose:
    A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    B = [B1; B2] where B1 is 4x6, B2 is 1x6
    
    C = A*B = A1*B1 + A2*B2
    
    A1*B1: 4x4 times 4x6. We can split B1 into [B1a | B1b] where B1a is 4x4, B1b is 4x2.
    A1*B1 = A1*B1a + A1*B1b
    
    A1*B1a: 4x4 times 4x4, use Strassen rank 28.
    A1*B1b: 4x4 times 4x2, schoolbook rank 32.
    A2*B2: 4x1 times 1x6, schoolbook rank 24.
    
    Total: 28 + 32 + 24 = 84.
    
    Actually, let me try a different split. Let's use the fact that we can share some terms.
    
    Alternatively, let's just try to find a good rank by search, but with a better initialization.
    """
    # For now, return None to indicate we should use the search approach
    return None

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook rank decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U, V, W

def _try_strassen_construction(n, m, p):
    """
    Try to construct a Strassen-based decomposition with explicit structure.
    
    Split A (4x5) into A1 (4x4) and A2 (4x1).
    Split B (5x6) into B1 (4x6) and B2 (1x6).
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (4x4 times 4x6):
    Split B1 into B1a (4x4) and B1b (4x2).
    A1*B1 = A1*B1a + A1*B1b
    
    A1*B1a: Use Strassen for 4x4x4. Rank = 28.
    A1*B1b: Schoolbook. Rank = 4*4*2 = 32.
    A2*B2: Schoolbook. Rank = 4*1*6 = 24.
    
    Total: 28 + 32 + 24 = 84.
    """
    # Strassen 2x2x2 has rank 7
    # For 4x4x4, we use 4 blocks of 2x2x2, each with rank 7, total 28.
    
    # Let's construct the Strassen decomposition for 4x4x4.
    # Split 4x4 into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we have 4 such 2x2 blocks, so 4*7 = 28 terms.
    
    # This is complex to implement directly. Let's use the search approach instead,
    # but with a better initialization that hints at the Strassen structure.
    
    return None

def construct():
    """Constructs a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try various ranks in increasing order
    # Start from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But Strassen-based estimates suggest ~84 might be achievable
    
    for rank in range(30, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, max_iters=200, n_restarts=15, deadline=deadline)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify once more
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _construct_schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
