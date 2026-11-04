"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p)
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]
        T_computed += tmp2
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _test_strassen_2x2():
    """Test that the Strassen 2x2x2x2 decomposition is correct."""
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()
    n = m = p = 2
    return _verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-10)


def _construct_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _strassen_2x2_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 = 2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all i,j,k in {0,1}
    
    Standard Strassen formulas:
    P1 = (A11+A22)*(B11+B22)
    P2 = (A21+A22)*B11
    P3 = A11*(B12-B22)
    P4 = A22*(B21-B11)
    P5 = (A11+A12)*B22
    P6 = (A21-A11)*(B11+B12)
    P7 = (A12-A22)*(B21+B22)
    
    C11 = P1+P4-P5+P7
    C12 = P3+P5
    C21 = P2+P4
    C22 = P1-P2+P3+P6
    """
    # U[i*2+j, c] corresponds to coefficient of A[i,j] in P_c
    # V[j*2+k, c] corresponds to coefficient of B[j,k] in P_c
    # W[k*2+i, c] corresponds to coefficient of C[i,k] from P_c
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # P2 = (A21+A22)*B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1  # C22 (negative)
    
    # P3 = A11*(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1  # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # P4 = A22*(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1  # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # P5 = (A11+A12)*B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1  # C11 (negative)
    W[1, 4] = 1  # C12
    
    # P6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1  # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # P7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1  # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build a rank-56 decomposition for the 4x4x4 all-ones tensor using Strassen.
    
    For each (ib, kb, jb) triple where ib,kb,jb in {0,1}, we have a 2x2x2 sub-tensor.
    There are 8 such sub-tensors, each with rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    
    This is correct but not optimal. We'll use this as a baseline.
    """
    n, m, p = 4, 4, 4
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()
    
    total_rank = 8 * 7  # 8 (ib,kb,jb) blocks * 7 Strassen terms
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
                for c in range(7):
                    col = (ib * 4 + kb * 2 + jb) * 7 + c
                    
                    # U: for i in {ib*2, ib*2+1}, j in {jb*2, jb*2+1}
                    for li in range(2):
                        for lj in range(2):
                            i = ib * 2 + li
                            j = jb * 2 + lj
                            U_full[i * 4 + j, col] = U_s[li * 2 + lj, c]
                    
                    # V: for j in {jb*2, jb*2+1}, k in {kb*2, kb*2+1}
                    for lj in range(2):
                        for lk in range(2):
                            j = jb * 2 + lj
                            k = kb * 2 + lk
                            V_full[j * 4 + k, col] = V_s[lj * 2 + lk, c]
                    
                    # W: for k in {kb*2, kb*2+1}, i in {ib*2, ib*2+1}
                    for lk in range(2):
                        for li in range(2):
                            i = ib * 2 + li
                            k = kb * 2 + lk
                            W_full[k * 4 + i, col] = W_s[lk * 2 + li, c]
    
    return U_full, V_full, W_full


def _build_4x4x5_decomposition():
    """
    Build a rank-56+1=57 decomposition for 4x4x5.
    Use Strassen for the 4x4x4 core (rank 56) and rank-1 all-ones for k=4 (rank 1).
    """
    n, m, p = 4, 4, 5
    
    # Build 4x4x4 core
    U_core, V_core, W_core = _build_4x4x4_strassen()
    core_rank = U_core.shape[1]
    
    # Verify core
    T_core = np.zeros((4, 4, 4))
    for c in range(core_rank):
        Uc = U_core[:, c].reshape(4, 4)
        Vc = V_core[:, c].reshape(4, 4)
        Wc = W_core[:, c].reshape(4, 4)
        T_core += np.einsum('ij,jk->ijk', Uc, Vc) * Wc.T[:, np.newaxis, :]
    
    if np.max(np.abs(T_core - 1.0)) > 1e-8:
        return None  # Core verification failed
    
    # Build k=4 boundary (rank-1 all-ones slice)
    U_bound = np.zeros((16, 1))
    V_bound = np.zeros((20, 1))
    W_bound = np.zeros((20, 1))
    
    # T[i,j,4] = 1 for all i,j
    # We need: sum_c U[i*4+j, c] * V[j*5+4, c] * W[4*4+i, c] = 1
    # With rank 1: U[i*4+j, 0] * V[j*5+4, 0] * W[16+i, 0] = 1
    # Set U[i*4+j, 0] = 1, V[j*5+4, 0] = 1, W[16+i, 0] = 1
    for i in range(4):
        for j in range(4):
            U_bound[i * 4 + j, 0] = 1
            V_bound[j * 5 + 4, 0] = 1
            W_bound[16 + i, 0] = 1
    
    # Combine
    total_rank = core_rank + 1
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((20, total_rank))
    
    U_full[:, :core_rank] = U_core
    V_full[:, :core_rank] = V_core
    W_full[:, :core_rank] = W_core
    
    U_full[:, core_rank:] = U_bound
    V_full[:, core_rank:] = V_bound
    W_full[:, core_rank:] = W_bound
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist(),
        "rank": total_rank
    }


def _try_direct_low_rank(n, m, p, max_rank=60, max_iters=1000, tol=1e-6):
    """
    Try to find a low-rank decomposition using iterative optimization.
    Initialize with random matrices and use gradient descent.
    """
    import numpy as np
    
    best_result = None
    best_rank = max_rank + 1
    
    for rank in range(max(1, max_rank // 2), max_rank + 1):
        if time.time() > 110:  # Safety margin
            break
        
        # Initialize with small random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Gradient descent
        lr = 0.01
        for it in range(max_iters):
            if time.time() > 115:
                break
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T  # (n, p)
                
                # T_computed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Gradient w.r.t U[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] for all k
                # But we need to match T_target[i,j,k] = 1
                
                # For each k, the contribution to T[i,j,k] from term c is:
                # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # So grad_U[i*m+j,c] = sum_k (T_target[i,j,k] - T_computed[i,j,k]) * V[j*p+k,c] * W[k*n+i,c]
                
                # This is complex; use finite differences or analytical gradient
                pass
            
            # Skip complex gradient; use a simpler approach
            break
        
        # Try a structured approach: use Kronecker products
        # For all-ones tensor, try U = u1 ⊗ u2, V = v1 ⊗ v2, W = w1 ⊗ w2
        pass
    
    return best_result


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118  # Leave safety margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Test Strassen 2x2 first
    if time.time() < deadline:
        strassen_ok = _test_strassen_2x2()
        if strassen_ok:
            # Try Strassen-based 4x4x4 core + rank-1 boundary
            result_strassen = _build_4x4x5_decomposition()
            if result_strassen is not None:
                if _verify_decomposition(result_strassen["U"], result_strassen["V"], result_strassen["W"], n, m, p, tol=1e-8):
                    best_result = result_strassen
                    best_rank = result_strassen["rank"]
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
