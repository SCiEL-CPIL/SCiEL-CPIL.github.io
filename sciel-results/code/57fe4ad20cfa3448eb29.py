import time
import numpy as np
import random

DIMENSIONS = [2, 3, 7]

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x3x7():
    """
    Construct a Winograd-style decomposition for 2x3x7.
    We split the 7 columns into 3x3 + 3x3 + 3x1.
    Rank <= 11 + 11 + 6 = 28.
    """
    n, m, p = 2, 3, 7
    
    # 1. Decompose 2x3x3 (Rank 11)
    # Standard Winograd for 2x3x3:
    # F_2 = [[1,0,0,1,0,0,0,0,0,1,0],
    #        [0,1,0,1,0,0,0,0,0,1,1]]
    # F_3 = [[1,0,0,0,0,0,1,0,0,1,1],
    #        [0,1,0,0,0,0,0,1,0,1,1],
    #        [0,0,1,0,0,0,0,0,1,1,1]]
    # M_11 = diag(1,1,1,1,1,1,1,1,1,1,1)  # Actually M is diag of multipliers
    # Let's use the specific 11-point scheme.
    # A = F_2 A F_3^T
    # B = F_3 B F_2^T
    # C = F_2 (A*B) F_3^T
    
    F_2 = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
    ])
    F_3 = np.array([
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    ])
    
    # For 2x3x3, the rank is 11.
    # U_2x3x3: 6x11
    # V_2x3x3: 9x11
    # W_2x3x3: 6x11
    
    # Let's generate the trivial 27 rank first and then compress? No, just use the known 11.
    # Actually, let's just use the trivial 27 for the 3x3 blocks if we can't easily code the 11.
    # But 11 is known.
    # Let's try to construct the 11-rank decomposition explicitly.
    
    # The 11-point scheme for 2x3x3 is unique up to equivalence.
    # Let's use the one from the literature.
    # U (input transform for A)
    # V (input transform for B)
    # W (output transform for C)
    
    # Actually, it's easier to just use the trivial 27 for 3x3 and 6 for 1x3?
    # 27 + 27 + 6 = 60. That's worse than 42.
    # We need the 11.
    
    # Let's hardcode the 11-rank decomposition for 2x3x3.
    # Source: "Fast Matrix Multiplication" or similar.
    
    # U_11 (6x11)
    # V_11 (9x11)
    # W_11 (6x11)
    
    # Let's try to find it or use a simpler approach.
    # If I can't get 11, I can use 15 (2x3x3 rank is 11, but maybe I can find 15 easily?).
    # Actually, 2x3x3 rank is 11.
    
    # Let's use the trivial 27 for 2x3x3.
    # 27 + 27 + 6 = 60.
    # This is worse than 42.
    
    # So I MUST use the 11-rank scheme for 2x3x3.
    
    # Here is a 11-rank decomposition for 2x3x3:
    # U:
    u_11 = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
    ])
    # Wait, this is F_2? No, U is the transform for A.
    # In Winograd, C = F_C ( (F_A A) * (F_B B) ) F_C^T
    # The rank is the number of columns in F_A, F_B, F_C.
    
    # Let's use the specific 11-point scheme.
    # F_A (2x3 -> 11)
    # F_B (3x3 -> 11)
    # F_C (2x3 -> 11)
    
    # F_A:
    # [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    # [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
    # [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]  -- Wait, F_A is 2x3 to 11? No, A is 2x3.
    # F_A should be 11x6.
    
    # Let's just use the trivial 42 for now if I can't easily get 28.
    # But 28 is a big improvement.
    
    # Let's try to construct 28.
    # 2x3x3 rank 11.
    # 2x3x1 rank 6.
    
    # I will implement the 11-rank scheme for 2x3x3.
    # Reference: https://en.wikipedia.org/wiki/Winograd_algorithm
    # For 2x3x3, the rank is 11.
    
    # F_2 (11x2) ? No, F_2 is 2x11?
    # Let's assume the standard form.
    
    # Let's just return the trivial 42 for now to be safe, and then improve.
    # Or, let's try to use ALS to find 28?
    
    return None

def _verify_exact(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact for the matrix multiplication tensor.
    T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    Should be 1 if i,j,k are valid indices for multiplication, 0 otherwise?
    No, the tensor M has M[i,j,k] = 1 for all i,j,k in the multiplication tensor.
    The multiplication tensor T_mult has T_mult[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait, the tensor is defined as sum_r U_{ij,r} V_{jk,r} W_{ki,r} = delta_{...}?
    No, the tensor for matrix multiplication is:
    T_{i, j, k} = 1 for all i, j, k.
    Because (AB)_{ik} = sum_j A_{ij} B_{jk}.
    The tensor T has entries T_{i, j, k} = 1.
    So the sum over r must be 1 for ALL i, j, k.
    
    Let's re-read the contract.
    "exactly decompose the matrix multiplication tensor"
    The matrix multiplication tensor T has T[i,j,k] = 1.
    So yes, T[i,j,k] must be 1 for all i,j,k.
    """
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
    
    if np.allclose(T, 1.0, atol=1e-9):
        return True
    return False

def _als_search(n, m, p, target_rank, max_iters=200, seed=42):
    """
    Use Alternating Least Squares to find a rank-target_rank decomposition.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, target_rank))
                for k in range(p):
                    for r in range(target_rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                b = np.ones(p)
                try:
                    u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank))
                for i in range(n):
                    for r in range(target_rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                b = np.ones(n)
                try:
                    v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for r in range(target_rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                b = np.ones(m)
                try:
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w
                except:
                    pass
        
        # Check convergence
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)
        err = np.linalg.norm(T - 1.0)
        
        if err < 1e-9:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    trivial = _trivial_decomposition(n, m, p)
    best = trivial
    best_rank = n * m * p  # 42
    
    # Try to find lower ranks
    # Lower bound is 21.
    # We know 2x3x3 has rank 11.
    # 2x3x7 can be split into 2x3x3 + 2x3x3 + 2x3x1.
    # Rank <= 11 + 11 + 6 = 28.
    # Let's try ranks from 21 to 41.
    
    for r in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random seeds for ALS
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            result = _als_search(n, m, p, r, max_iters=150, seed=seed)
            if result is not None:
                # Verify strictly
                if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
            
    return best
