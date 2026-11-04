"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    # Define terms as a list of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Index mapping:
    # a: A[i,j] -> i*3+j
    # b: B[j,k] -> j*3+k
    # c: C[i,k] -> k*2+i
    
    terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    # M1 = (a0+a4)*(b0+b4)
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    # M2 = (a3+a4)*b0
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    # M3 = a0*(b1-b4)
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    # M4 = a4*(b3-b0)
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    # M5 = (a0+a1)*b4
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    # M6 = (a3-a0)*(b0+b1)
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # M7 = (a1-a4)*(b3+b4)
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    # C[0,2]=c4, C[1,2]=c5
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Term 8: A[0,0]*B[0,2] -> C[0,2]
    terms.append(([0], [1], [2], [1], [4], [1]))
    # Term 9: A[0,1]*B[1,2] -> C[0,2]
    terms.append(([1], [1], [5], [1], [4], [1]))
    # Term 10: A[1,0]*B[0,2] -> C[1,2]
    terms.append(([3], [1], [2], [1], [5], [1]))
    # Term 11: A[1,1]*B[1,2] -> C[1,2]
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    # C[i,k] += A[i,2]*B[2,k]
    # Term 12: A[0,2]*B[2,0] -> C[0,0]
    terms.append(([2], [1], [6], [1], [0], [1]))
    # Term 13: A[0,2]*B[2,1] -> C[0,1]
    terms.append(([2], [1], [7], [1], [2], [1]))
    # Term 14: A[0,2]*B[2,2] -> C[0,2]
    terms.append(([2], [1], [8], [1], [4], [1]))
    # Term 15: A[1,2]*B[2,0] -> C[1,0]
    terms.append(([5], [1], [6], [1], [1], [1]))
    # Term 16: A[1,2]*B[2,1] -> C[1,1]
    terms.append(([5], [1], [7], [1], [3], [1]))
    # Term 17: A[1,2]*B[2,2] -> C[1,2]
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    # Populate matrices
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
            
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
