"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 115

def _build_strassen_21():
    """Build the rank-21 Strassen decomposition for 2x2 * 2x6."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=2, p=6, tol=1e-9):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matrix multiplication tensor."""
    rank = len(U[0])
    # Tensor T[i,j,k] = sum_{a,b} A[i,a] * B[a,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me check the index convention.
    # U has shape (n*m, rank): index i*m + j? or i + j*n?
    # V has shape (m*p, rank): index j*p + k
    # W has shape (p*n, rank): index k*n + i
    
    # The tensor for matrix mult: T[i,j,k] = sum_a A[i,a] B[a,k]
    # where i in [0,n), j in [0,m), k in [0,p)
    # U[i, j, r] should pick out A[i,j] coefficients
    # V[j, k, r] should pick out B[j,k] coefficients  
    # W[k, i, r] should pick out C[k,i]... 
    
    # Actually the standard tensor decomposition for matmul:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = sum_a delta_{i,a}... no.
    
    # T[i,j,k] = (A B)[i,k] * delta... no.
    # The matrix multiplication tensor is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T[i,j,k] = 1 for all valid (i,j,k) in the standard tensor.
    # The decomposition requires:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    
    # With U indexed by (i,j) flattened as i*m+j (row-major) or i + j*n?
    # The task says U shape is (n*m, rank). Let's assume U[i*m+j, r].
    # V shape (m*p, rank): V[j*p+k, r]
    # W shape (p*n, rank): W[k*n+i, r]
    
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # (This is the standard matrix multiplication tensor)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_val = U[i * m + j][r]
                    v_val = V[j * p + k][r]
                    w_val = W[k * n + i][r]
                    val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_20_search(max_time=60):
    """Try to find a rank-20 decomposition by perturbing Strassen-21 and optimizing."""
    if time.time() > _start_time + max_time:
        return None
    
    n, m, p = 2, 2, 6
    U, V, W = _build_strassen_21()
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to nullify one of the 21 product terms
    # We'll try to set one column to zero by adjusting the others
    # This is a hard problem; instead, try random rank-20 initializations
    
    random.seed(42)
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > _start_time + max_time:
            break
        
        # Start from Strassen-21, drop one random product
        drop_col = random.randint(0, 20)
        
        U20 = U_np.copy()
        V20 = V_np.copy()
        W20 = W_np.copy()
        
        # Zero out the dropped column
        U20[:, drop_col] = 0
        V20[:, drop_col] = 0
        W20[:, drop_col] = 0
        
        # Remove the dropped column
        mask = [i for i in range(21) if i != drop_col]
        U20 = U20[:, mask]
        V20 = V20[:, mask]
        W20 = W20[:, mask]
        
        # Now try to optimize U20, V20, W20 to satisfy the tensor equations
        # This is a bilinear system; use gradient descent on the residual
        
        def compute_residual(Uc, Vc, Wc):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(20):
                            val += Uc[i*m+j, r] * Vc[j*p+k, r] * Wc[k*n+i, r]
                        res += (val - 1.0) ** 2
            return res
        
        # Simple gradient-free local search: perturb and accept if better
        best_res = compute_residual(U20, V20, W20)
        
        for step in range(200):
            if time.time() > _start_time + max_time:
                break
            
            # Random perturbation
            scale = 0.1 / (1 + step * 0.01)
            U20_p = U20 + np.random.randn(*U20.shape) * scale
            V20_p = V20 + np.random.randn(*V20.shape) * scale
            W20_p = W20 + np.random.randn(*W20.shape) * scale
            
            new_res = compute_residual(U20_p, V20_p, W20_p)
            if new_res < best_res:
                U20, V20, W20 = U20_p, V20_p, W20_p
                best_res = new_res
        
        if best_res < 1e-6:
            # Check for integer/half-integer solution
            # Round to nearest half-integer
            U_round = np.round(U20 * 2) / 2
            V_round = np.round(V20 * 2) / 2
            W_round = np.round(W20 * 2) / 2
            
            res_round = compute_residual(U_round, V_round, W_round)
            if res_round < 1e-9:
                return (U_round.tolist(), V_round.tolist(), W_round.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find rank-20
    result_20 = _try_rank_20_search(max_time=80)
    if result_20 is not None:
        U, V, W = result_20
        return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-21 Strassen
    U, V, W = _build_strassen_21()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
