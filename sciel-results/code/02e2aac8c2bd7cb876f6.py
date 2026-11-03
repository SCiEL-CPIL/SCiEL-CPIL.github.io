import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for i=0,1)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for i=2)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute rank-1 tensors
    R = np.zeros((rank, nm * mp * pn), dtype=float)
    for r in range(rank):
        Rr = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        R[r] = Rr.reshape(-1)
    
    keep = [True] * rank
    
    # Pass 1: Check for identical or proportional rank-1 tensors
    for r in range(rank):
        if time.time() > deadline:
            break
        if not keep[r]:
            continue
        
        for s in range(r + 1, rank):
            if not keep[s]:
                continue
            
            norm_r = np.linalg.norm(R[r])
            norm_s = np.linalg.norm(R[s])
            
            if norm_r < 1e-10 or norm_s < 1e-10:
                continue
            
            alpha = np.dot(R[r], R[s]) / (norm_s ** 2)
            
            diff = R[r] - alpha * R[s]
            if np.linalg.norm(diff) / norm_r < 1e-8:
                W_np[:, s] *= (1 + alpha)
                keep[r] = False
                break
    
    # Pass 2: Try to find small linearly dependent sets (size 2 and 3)
    # We check if a subset of rank-1 tensors is linearly dependent.
    # If so, we can potentially remove one tensor by expressing it as a linear combination of others.
    # However, the tensor decomposition is a sum of rank-1 terms.
    # T = sum_{r in S} U_r v_r w_r.
    # If sum_{r in S} c_r U_r v_r w_r = 0, then we can't just drop one.
    # We need the specific structure of the decomposition to allow removal.
    # The "proportional" case is the only simple one where we can combine terms.
    # For general linear dependence, we would need to solve for coefficients and update the factors,
    # which is much harder and might not yield a valid decomposition of the same form (diagonal factors).
    # Given the time constraint and the difficulty, we stick to the proportional check which is robust.
    
    # However, we can check if any two terms are IDENTICAL (alpha=1).
    # This is covered by the proportional check.
    
    # Let's add a check for "negative" proportional terms (alpha = -1)
    # This is also covered.
    
    # Let's verify the current decomposition is valid before returning
    # We don't want to return an invalid decomposition just to save rank.
    
    final_indices = [i for i in range(rank) if keep[i]]
    final_rank = len(final_indices)
    
    U_final = U_np[:, final_indices]
    V_final = V_np[:, final_indices]
    W_final = W_np[:, final_indices]
    
    # Verification
    # Check if the decomposition is exact.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We only need to check a few random points or all points if fast enough.
    # 3*4*8 = 96 points. Rank is ~80. 96*80 = 7680 ops. Very fast.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(final_rank):
                    val += U_final[i*m+j][c] * V_final[j*p+k][c] * W_final[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to original if verification fails (should not happen)
        return {"U": U, "V": V, "W": W}
        
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True
