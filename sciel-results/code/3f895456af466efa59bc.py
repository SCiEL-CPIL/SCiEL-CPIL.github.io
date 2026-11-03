"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    # Test with random matrices
    rng = np.random.default_rng(12345)
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        # Compute via decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Rearrange: for each c, define A_c = diag? No.
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j] * ... )
        # Actually: sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' V[j*p+k',c] * B[j,k']) ... this is not separable that way.
        # The correct formula: C[i,k] = sum_c U_col_i * (A * V_col_j) ... 
        # Let me just directly compute:
        C_dec = np.zeros((n, p))
        for c in range(rank):
            # For each c, we have vectors u_c (length m), v_c (length p), w_c scalar per (i,k)
            # Actually U[i*m+j, c] for fixed c is a vector of length n*m, but we need it per i.
            # C[i,k] += sum_j U[i*m+j, c] * A[i,j] * ... no.
            # The tensor is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # And C[i,k] = sum_j A[i,j] * B[j,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] / (A[i,j]*B[j,k]) ... no that's wrong.
            # Actually the decomposition means: the bilinear map (A,B) -> C is given by
            # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k]) ... NO.
            # The standard form: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
            # Wait, that's not right either. Let me think again.
            # The multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
            # A rank-1 term is u ⊗ v ⊗ w where u is over (i,j), v over (j,k), w over (k,i).
            # The product is: C[i,k] = sum_c w[k*n+i, c] * (sum_j u[i*m+j, c] * A[i,j]) * (sum_j' v[j'*p+k, c] * B[j', k])
            # Hmm, but u depends on both i and j, so sum_j u[i*m+j,c] * A[i,j] is a scalar for each (i,c).
            # Similarly sum_j' v[j'*p+k,c] * B[j',k] is a scalar for each (k,c).
            # So C[i,k] = sum_c w[k*n+i,c] * (sum_j u[i*m+j,c] * A[i,j]) * (sum_j' v[j'*p+k,c] * B[j',k])
            # This is NOT the standard form. The standard form for matrix multiplication tensor decomposition is:
            # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j' V[j',k,c] * B[j',k]) * W[i,k,c]
            # But here U is indexed by (i,j) = (i*m+j), V by (j,k) = (j*p+k), W by (k,i) = (k*n+i).
            # So yes: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j' V[j'*p+k,c] * B[j',k])
            # Let me compute this:
            for i in range(n):
                uA = 0.0
                for j in range(m):
                    uA += U[i * m + j][c] * A[i, j]
                for k in range(p):
                    vB = 0.0
                    for j2 in range(m):
                        vB += V[j2 * p + k][c] * B[j2, k]
                    C_dec[i, k] += W[k * n + i][c] * uA * vB
        if np.max(np.abs(C - C_dec)) > tol:
            return False
    return True

def _to_list(mat):
    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else (round(v * 2) / 2 if abs(v * 2 - round(v * 2)) < 1e-9 else v) for v in row] for row in mat]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank using random search
    # Strategy: for a given rank r, randomly initialize U, V, W and try to refine.
    # Alternatively, use the linear structure: fix U, V, solve for W.
    
    rng = np.random.default_rng(42)
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(nm, mp, pn) = max(12, 32, 24) = 32
    # Also, rank >= max(n*p, m*p, n*m) isn't quite right for tensor rank.
    # For matrix multiplication, rank >= max(n*m, m*p, n*p) is not a standard bound.
    # A simple bound: rank >= n*p = 24 (since the output has 24 entries)
    # Let's try from 24 up to 95
    
    for r in range(24, min(96, best_rank)):
        if time.time() > deadline - 5:
            break
        
        # Try several random initializations
        max_iters = 5
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_rand = rng.standard_normal((nm, r))
            V_rand = rng.standard_normal((mp, r))
            W_rand = rng.standard_normal((pn, r))
            
            # Convert to lists and verify
            U_list = _to_list(U_rand)
            V_list = _to_list(V_rand)
            W_list = _to_list(W_rand)
            
            if _verify(U_list, V_list, W_list, n, m, p):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
        
        if best_rank == r:
            # Found at this rank, can stop
            break
    
    # Convert final result
    U_final = best["U"]
    V_final = best["V"]
    W_final = best["W"]
    
    # Ensure integer/half-integer entries
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    U_final = clean_matrix(U_final)
    V_final = clean_matrix(V_final)
    W_final = clean_matrix(W_final)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
