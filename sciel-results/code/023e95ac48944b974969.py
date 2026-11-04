import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0

    # Build target tensor
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    schoolbook_rank = n * m * p  # 112
    best_result = None
    best_rank = schoolbook_rank

    # Strategy: Structured decomposition using Strassen for sub-blocks
    # Partition the 4x4 matrices into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02, B03, B04, B05, B06],
    #      [B10, B11, B12, B13, B14, B15, B16]] (each 2x7)
    # C = A * B = [[C00, C01], [C10, C11]] (each 2x7)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # ... etc
    #
    # Each "block multiplication" is 2x2 * 2x7 = 2x7
    # We can use Strassen-like decomposition for the 2x2 * 2x2 sub-problems
    # and handle the extra columns separately.
    #
    # Actually, let's use a cleaner approach:
    # The tensor M_{4,4,7} can be viewed as having 7 "columns" in the B matrix.
    # For each pair of columns (k, k'), we can potentially share structure.
    #
    # Better structural approach: Use the fact that M_{n,m,p} tensor rank
    # can be bounded by structured decompositions. For 4x4x7:
    #
    # Approach: Split B into B_left (4x4) and B_right (4x3)
    # C = A * B = A * B_left + A * B_right
    #
    # For A * B_left (4x4 * 4x4 = 4x4): use Strassen, rank 49
    # For A * B_right (4x4 * 4x3 = 4x3): need decomposition
    #
    # The key challenge: the indices in the full tensor are NOT simply
    # a direct sum because V index = j*7+k, not j*4+k for the left part.
    #
    # So we need to "embed" the 4x4x4 Strassen decomposition into the
    # correct positions in the 16x28x28 tensor.

    # Build the 4x4x4 Strassen decomposition and embed it
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    # Kronecker product for 4x4x4: rank 49
    U_444 = np.zeros((16, 49))
    V_444_std = np.zeros((16, 49))  # standard 4x4 indexing: j*4+k
    W_444_std = np.zeros((16, 49))  # standard 4x4 indexing: k*4+i
    
    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444_std[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444_std[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])
    
    # Now embed into full tensor:
    # In the full problem, for the "left" 4 columns (k=0..3):
    # V index = j*7 + k, W index = k*4 + i
    # In the standard 4x4x4: V index = j*4 + k, W index = k*4 + i
    # So W indices match! But V indices differ: j*7+k vs j*4+k
    #
    # We need to create V_444_full of shape (28, 49) where:
    # V_444_full[j*7+k, idx] = V_444_std[j*4+k, idx] for k=0..3
    V_444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_444_full[j*7+k, :] = V_444_std[j*4+k, :]
    
    W_444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_444_full[k*4+i, :] = W_444_std[k*4+i, :]
    
    # Verify the 4x4x4 part
    recon_444 = np.einsum('ak,bk,ck->abc', U_444, V_444_full, W_444_full)
    T_444_part = np.zeros((16, 28, 28), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_444_part[a, b, c] = 1
    
    if not np.allclose(recon_444, T_444_part, atol=1e-10):
        # Fallback: try greedy peeling
        result = _greedy_peel(T, dim_u, dim_v, dim_w, deadline, schoolbook_rank)
        if result is not None:
            return result
        return _schoolbook(n, m, p)
    
    # Now handle the right part: k=4,5,6 (3 columns)
    # For these, V index = j*7+k, W index = k*4+i
    # Schoolbook for this part: rank 4*4*3 = 48
    # Total so far: 49 + 48 = 97
    
    U_right = np.zeros((16, 48))
    V_right = np.zeros((28, 48))
    W_right = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):  # k = 4, 5, 6
                U_right[i*4+j, col] = 1
                V_right[j*7+k, col] = 1
                W_right[k*4+i, col] = 1
                col += 1
    
    # Combine: total rank = 49 + 48 = 97
    U_total = np.hstack([U_444, U_right])  # (16, 97)
    V_total = np.hstack([V_444_full, V_right])  # (28, 97)
    W_total = np.hstack([W_444_full, W_right])  # (28, 97)
    
    # Verify full reconstruction
    recon_full = np.einsum('ak,bk,ck->abc', U_total, V_total, W_total)
    if np.allclose(recon_full, T, atol=1e-10):
        best_rank = 97
        best_result = {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
    
    # Try to improve further with greedy peeling on the residual
    # or try alternative decompositions for the right part
    
    # Try greedy peeling on the right part (4x4x3) to see if we can do better than 48
    if time.time() < deadline - 5.0:
        T_right = np.zeros((16, 28, 28), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    a = i * 4 + j
                    b = j * 7 + k
                    c = k * 4 + i
                    T_right[a, b, c] = 1
        
        right_result = _greedy_peel(T_right, 16, 28, 28, min(deadline - 5.0, time.time() + 60.0), 48)
        if right_result is not None:
            # right_result has U:(16,r), V:(28,r), W:(28,r)
            r_right = len(right_result["U"][0])
            U_r = np.array(right_result["U"]).T  # (16, r_right)
            V_r = np.array(right_result["V"]).T  # (28, r_right)
            W_r = np.array(right_result["W"]).T  # (28, r_right)
            
            U_combined = np.hstack([U_444, U_r])
            V_combined = np.hstack([V_444_full, V_r])
            W_combined = np.hstack([W_444_full, W_r])
            
            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + r_right
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }
    
    # Also try full greedy peeling as alternative
    if time.time() < deadline - 3.0 and best_rank >= 97:
        full_result = _greedy_peel(T, dim_u, dim_v, dim_w, deadline - 1.0, best_rank)
        if full_result is not None:
            r_full = len(full_result["U"][0])
            if r_full < best_rank:
                best_rank = r_full
                best_result = full_result
    
    if best_result is not None and best_rank < schoolbook_rank:
        return best_result
    
    return _schoolbook(n, m, p)


def _greedy_peel(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling with multiple restarts. Returns dict or None."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break
        
        best_score = 0
        best_u, best_v, best_w = None, None, None
        
        # Multiple restarts with different initializations
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            else:
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            
            # Alternating minimization
            n_iters = 15
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)
                
                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)
                
                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)
                
                cu, cv, cw = new_u, new_v, new_w
            
            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()
        
        if best_score <= 0 or best_u is None:
            break
        
        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        
        if np.allclose(R, 0, atol=1e-10):
            break
    
    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    
    # Strassen products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (for A indices: a11=0, a12=1, a21=2, a22=3)
    # V vectors (for B indices: b11=0, b12=1, b21=2, b22=3)
    # W vectors (for C indices: c11=0, c12=1, c21=2, c22=3)
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 += M2, C22 -= M2
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 += M3, C22 += M3
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 += M4, C21 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += M7
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
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

# EVOLVE-BLOCK-END
