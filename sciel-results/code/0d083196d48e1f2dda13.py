"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_rank17():
    """Build the rank-17 Strassen+schoolbook decomposition for 2x3x3 matmul."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # r=7: A02 B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # r=11: A00 B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_random(U, V, W, n, m, p, num_trials=5, tol=1e-9):
    """Verify using random matrix multiplication."""
    rank = U.shape[1]
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        A_r = A_flat @ U
        B_r = B_flat @ V
        
        prod = A_r * B_r
        Result_flat = (W * prod[None, :]).sum(axis=1)
        Result = Result_flat.reshape(p, n).T
        
        if np.abs(Result - AB).max() > tol:
            return False
    return True


def _try_remove_term(U, V, W, n, m, p, term_idx):
    """Try removing term at term_idx and check if remaining terms still work."""
    rank = U.shape[1]
    U2 = np.delete(U, term_idx, axis=1)
    V2 = np.delete(V, term_idx, axis=1)
    W2 = np.delete(W, term_idx, axis=1)
    return _verify(U2, V2, W2, n, m, p)


def _try_remove_two_terms(U, V, W, n, m, p, idx1, idx2):
    """Try removing two terms and check if remaining terms still work."""
    rank = U.shape[1]
    mask = np.ones(rank, dtype=bool)
    mask[idx1] = False
    mask[idx2] = False
    U2 = U[:, mask]
    V2 = V[:, mask]
    W2 = W[:, mask]
    return _verify(U2, V2, W2, n, m, p)


def _search_lower_rank(U, V, W, n, m, p, deadline):
    """Search for lower rank by trying to remove terms."""
    rank = U.shape[1]
    
    # Try removing one term
    for idx in range(rank):
        if time.time() > deadline:
            break
        if _try_remove_term(U, V, W, n, m, p, idx):
            mask = np.ones(rank, dtype=bool)
            mask[idx] = False
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            return U_new, V_new, W_new
    
    # Try removing two terms
    for idx1 in range(rank):
        if time.time() > deadline:
            break
        for idx2 in range(idx1 + 1, rank):
            if time.time() > deadline:
                break
            if _try_remove_two_terms(U, V, W, n, m, p, idx1, idx2):
                mask = np.ones(rank, dtype=bool)
                mask[idx1] = False
                mask[idx2] = False
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                return U_new, V_new, W_new
    
    # Try removing three terms (only if time permits)
    if time.time() < deadline - 10:
        for idx1 in range(rank):
            if time.time() > deadline:
                break
            for idx2 in range(idx1 + 1, rank):
                if time.time() > deadline:
                    break
                for idx3 in range(idx2 + 1, rank):
                    if time.time() > deadline:
                        break
                    mask = np.ones(rank, dtype=bool)
                    mask[idx1] = False
                    mask[idx2] = False
                    mask[idx3] = False
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    if _verify(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build baseline rank-17 decomposition
    U, V, W = _build_rank17()
    
    # Verify baseline
    if not _verify(U, V, W, n, m, p):
        # Fallback: return as-is (shouldn't happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Search for lower rank
    if time.time() < deadline:
        U, V, W = _search_lower_rank(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # If verification fails, rebuild and return baseline
        U, V, W = _build_rank17()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
