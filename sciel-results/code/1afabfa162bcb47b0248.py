"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, least_squares

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3
    U[0, 2] = 1.0
    V[2, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4
    U[3, 3] = 1.0
    V[1, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = -1.0; W[2, 3] = 1.0
    
    # M5
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = 1.0; W[1, 4] = 1.0
    
    # M6
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[2, 5] = 1.0
    W[2, 5] = 1.0
    
    # M7
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[1, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0; W[3, 6] = 1.0
    
    return U, V, W

def _block_decomposition(n, m, p):
    """
    Construct a block decomposition for M_{2,2,p}.
    """
    if n != 2 or m != 2:
        return None
        
    block_sizes = []
    remaining = p
    while remaining > 0:
        if remaining >= 2:
            block_sizes.append(2)
            remaining -= 2
        else:
            block_sizes.append(1)
            remaining -= 1
            
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U_1 = np.zeros((4, 4), dtype=np.float64)
    V_1 = np.zeros((2, 4), dtype=np.float64)
    W_1 = np.zeros((2, 4), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(2):
            U_1[i * 2 + j, col] = 1.0
            V_1[j, col] = 1.0
            W_1[i, col] = 1.0
            col += 1
            
    total_rank = 0
    for bs in block_sizes:
        if bs == 2:
            total_rank += 7
        elif bs == 1:
            total_rank += 4
            
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    k_offset = 0
    
    for bs in block_sizes:
        if bs == 2:
            for r in range(7):
                U[0, col_offset] = U_s[0, r]
                U[1, col_offset] = U_s[1, r]
                U[2, col_offset] = U_s[2, r]
                U[3, col_offset] = U_s[3, r]
                
                V[0 * p + k_offset, col_offset] = V_s[0, r]
                V[0 * p + k_offset + 1, col_offset] = V_s[1, r]
                V[1 * p + k_offset, col_offset] = V_s[2, r]
                V[1 * p + k_offset + 1, col_offset] = V_s[3, r]
                
                W[k_offset * n + 0, col_offset] = W_s[0, r]
                W[(k_offset + 1) * n + 0, col_offset] = W_s[1, r]
                W[k_offset * n + 1, col_offset] = W_s[2, r]
                W[(k_offset + 1) * n + 1, col_offset] = W_s[3, r]
                
                col_offset += 1
            k_offset += 2
        elif bs == 1:
            for r in range(4):
                U[0, col_offset] = U_1[0, r]
                U[1, col_offset] = U_1[1, r]
                U[2, col_offset] = U_1[2, r]
                U[3, col_offset] = U_1[3, r]
                
                V[0 * p + k_offset, col_offset] = V_1[0, r]
                V[1 * p + k_offset, col_offset] = V_1[1, r]
                
                W[k_offset * n + 0, col_offset] = W_1[0, r]
                W[k_offset * n + 1, col_offset] = W_1[1, r]
                
                col_offset += 1
            k_offset += 1
            
    return U, V, W

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    u_val = U[i * m + j, c]
                    v_val = V[j * p + k, c]
                    w_val = W[k * n + i, c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_low_rank_2x2x3(deadline):
    """
    Find a low-rank decomposition for M_{2,2,3}.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 2, 3
    # Try ranks from 6 upwards
    for rank in range(6, 12):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        rng = np.random.RandomState(rank * 100)
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        def residuals(params):
            U_r = params[0:n*m*rank].reshape(n*m, rank)
            V_r = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_r = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            res = []
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for r in range(rank):
                            val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                    res.append(val - 1.0)
            return res
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use least_squares for better handling of underdetermined systems
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            
            if np.allclose(result.fun, 0, atol=1e-8):
                U_sol = result.x[0:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                if _is_valid(U_sol, V_sol, W_sol, n, m, p):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
    return None

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Try Kronecker product approach
    # M_{2,2,15} = M_{2,2,3} \otimes M_{1,1,5}
    # If we find M_{2,2,3} with rank R, total rank is 5*R.
    # Current best is 53. So we need 5*R < 53 => R <= 10.
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Try to find M_{2,2,3} decomposition
    if time.time() < deadline - 60:
        res_3 = _find_low_rank_2x2x3(deadline - 60)
        if res_3 is not None:
            U3, V3, W3 = res_3
            rank3 = U3.shape[1]
            
            # Construct M_{1,1,5}
            n5, m5, p5 = 1, 1, 5
            U5 = np.ones((n5 * m5, 5), dtype=np.float64)
            V5 = np.ones((m5 * p5, 5), dtype=np.float64)
            W5 = np.eye(p5 * n5, dtype=np.float64) # W is 5x1, so I5
            
            # Kronecker Product
            # U_total = U3 \otimes U5 ?
            # Dimensions:
            # U3: (4, R). U5: (1, 5).
            # np.kron(U3, U5) will be (4*1, R*5) = (4, 5R).
            # V3: (6, R). V5: (1, 5).
            # np.kron(V3, V5) will be (6*1, R*5) = (6, 5R).
            # W3: (6, R). W5: (5, 1).
            # np.kron(W3, W5) will be (6*5, R*1) = (30, R).
            
            # Wait, the order of Kronecker product matters for the indices.
            # T_{i,j,k} for M_{2,2,15}.
            # k = k3 * 5 + k5? No.
            # M_{2,2,15} indices: i (0..1), j (0..1), k (0..14).
            # We map k to (k3, k5) where k3 in 0..2, k5 in 0..4.
            # k = k3 * 5 + k5?
            # If we use standard Kronecker:
            # A (2x2) \otimes A' (1x1) -> 2x2.
            # B (2x3) \otimes B' (1x5) -> 2x15.
            # C (3x2) \otimes C' (5x1) -> 15x2.
            
            # The tensor decomposition:
            # C_{i, k} = \sum_{j, r} A_{i,j} B_{j,k} C_{k,i}
            # In Kronecker:
            # C_{i, k3, k5} = \sum_{j, r3, r5} A_{i,j, r3} B_{j,k3, r3} C_{k3,i, r3} \times A'_{i',j',r5} B'_{j',k5,r5} C'_{k5,i',r5}
            # Since A', B', C' are for 1x1x5, i'=0, j'=0.
            # So it becomes:
            # C_{i, k3, k5} = \sum_{j, r3, r5} U3_{i,j,r3} V3_{j,k3,r3} W3_{k3,i,r3} U5_{0,0,r5} V5_{0,k5,r5} W5_{k5,0,r5}
            # This matches the structure if we flatten indices correctly.
            
            U_tot = np.kron(U3, U5)
            V_tot = np.kron(V3, V5)
            W_tot = np.kron(W3, W5)
            
            # Check shapes
            # U_tot: (4, 5R)
            # V_tot: (6, 5R) -> Wait, V3 is (6, R). V5 is (1, 5).
            # np.kron(V3, V5) is (6, 5R).
            # But V for M_{2,2,15} should be (m*p, rank) = (2*15, rank) = (30, rank).
            # My V3 is for M_{2,2,3}, so V3 is (m*p, rank) = (2*3, R) = (6, R).
            # V5 is (1*5, 5) = (5, 5)?
            # No, V5 for M_{1,1,5} is (m*p, rank) = (1*5, 5) = (5, 5).
            # I defined V5 as np.ones((1, 5)). That's wrong.
            # V5 should be (5, 5) if rank is 5?
            # For M_{1,1,5}, U is (1, 5), V is (5, 5), W is (5, 1).
            # U5 = [1,1,1,1,1] (1x5)
            # V5 = I5 (5x5)
            # W5 = I5 (5x1)
            
            U5 = np.ones((1, 5), dtype=np.float64)
            V5 = np.eye(5, dtype=np.float64)
            W5 = np.eye(5, dtype=np.float64)
            
            U_tot = np.kron(U3, U5) # (4, 5R)
            V_tot = np.kron(V3, V5) # (6, R) x (5, 5) -> (30, 5R)
            W_tot = np.kron(W3, W5) # (6, R) x (5, 1) -> (30, R)
            
            # Wait, W_tot shape is (30, R). But U_tot and V_tot have 5R columns.
            # The ranks must match.
            # The Kronecker product of decompositions of rank R1 and R2 has rank R1*R2.
            # So W_tot should have 5R columns.
            # W3 is (6, R). W5 is (5, 5)? No, W5 is (5, 1) with rank 5?
            # No, W5 is a matrix. The rank is the number of columns.
            # For M_{1,1,5}, rank is 5.
            # W5 should be (5, 5) ?
            # W_{k,i} for M_{1,1,5}: k in 0..4, i=0.
            # W is (5, 5).
            # W5 = I5.
            # Then np.kron(W3, W5) is (6*5, R*5) = (30, 5R).
            
            U5 = np.ones((1, 5), dtype=np.float64)
            V5 = np.eye(5, dtype=np.float64)
            W5 = np.eye(5, dtype=np.float64)
            
            U_tot = np.kron(U3, U5)
            V_tot = np.kron(V3, V5)
            W_tot = np.kron(W3, W5)
            
            if U_tot.shape[1] < best_rank and _is_valid(U_tot, V_tot, W_tot, n, m, p):
                best_U, best_V, best_W = U_tot, V_tot, W_tot
                best_rank = U_tot.shape[1]
    
    # 2. Fall back to block decomposition
    if best_U is None or best_rank >= 53:
        U, V, W = _block_decomposition(n, m, p)
        if U is not None and _is_valid(U, V, W, n, m, p):
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
                
    # 3. Fallback to schoolbook
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def construct():
    return _construct()
# EVOLVE-BLOCK-END
