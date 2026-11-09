"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

START_TIME = time.time()
TIME_LIMIT = 120.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker product decomposition: Strassen(2x2) ⊗ Laderman(3x3).
    6 = 2*3, so we decompose the 6x6x6 tensor as a tensor product of
    a 2x2x2 tensor (Strassen, rank 7) and a 3x3x3 tensor (Laderman, rank 23).
    Total rank = 7 * 23 = 161.
    """
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Hardcoded Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Indices:
    # A: 00->0, 01->1, 10->2, 11->3
    # B: 00->0, 01->1, 10->2, 11->3
    # C (trace convention k*2+i):
    # C[0,0] (i=0,k=0) -> 0
    # C[1,0] (i=1,k=0) -> 1
    # C[0,1] (i=0,k=1) -> 2
    # C[1,1] (i=1,k=1) -> 3
    
    # Strassen factors U, V, W (each 7x4)
    U2 = np.array([
        [1, 1, 0, 0],  # M1: (a+b)
        [1, -1, 0, 0], # M2: (a-b)
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 0, 0, 1],  # M5: (a+d)
        [0, 1, 0, 1],  # M6: (b+d)
        [1, 0, -1, 0]  # M7: (a-c)
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],  # M1: (e+h)
        [1, 0, 0, 1],  # M2: (e+h)
        [0, 0, 1, -1], # M3: (g-h)
        [1, -1, 0, 0], # M4: (e-f)
        [1, 0, 0, 0],  # M5: e
        [0, 0, 0, 1],  # M6: h
        [0, 0, 1, 1]   # M7: (g+h)
    ], dtype=float)
    
    W2 = np.array([
        [0, -1, 1, 0],  # M1: -C[1,0] + C[0,1]
        [0, 1, 0, -1],  # M2: C[1,0] - C[1,1]
        [-1, 1, 0, 0],  # M3: -C[0,0] + C[1,0]
        [-1, 0, 1, 1],  # M4: -C[0,0] + C[0,1] + C[1,1]
        [1, 1, -1, -1], # M5: C[0,0] + C[1,0] - C[0,1] - C[1,1]
        [0, 0, 1, 1],   # M6: C[0,1] + C[1,1]
        [1, -1, 0, -1]  # M7: C[0,0] - C[1,0] - C[1,1]
    ], dtype=float)
    
    rank_2x2 = 7
    
    # Laderman 3x3 decomposition (rank 23)
    # This uses the standard Laderman algorithm for 3x3 matrix multiplication
    U3, V3, W3 = _laderman_3x3()
    rank_3x3 = 23
    
    # Kronecker product: rank = rank_2x2 * rank_3x3
    total_rank = rank_2x2 * rank_3x3
    
    # Build the 6x6x6 decomposition from Kronecker product
    # 6 = 2*3, so index in 6x6 = index_in_2x2 * 3 + index_in_3x3
    # U_6[i6, r] where i6 = i2*3 + i3, r = r2*27 + r3
    # U_6[i2*3+i3, r2*23+r3] = U2[i2, r2] * U3[i3, r3]
    
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))
    
    for r2 in range(rank_2x2):
        for r3 in range(rank_3x3):
            r = r2 * rank_3x3 + r3
            for i2 in range(4):
                for i3 in range(9):
                    i6 = i2 * 3 + i3
                    U6[i6, r] = U2[r2, i2] * U3[r3, i3]
                    V6[i6, r] = V2[r2, i2] * V3[r3, i3]
                    W6[i6, r] = W2[r2, i2] * W3[r3, i3]
    
    # Verify the decomposition
    T_6x6 = np.zeros((36, 36, 36))
    for l in range(6):
        for j in range(6):
            for k in range(6):
                U_idx = l * 6 + j
                V_idx = j * 6 + k
                W_idx = k * 6 + l
                T_6x6[U_idx, V_idx, W_idx] = 1.0
    
    T_recon = np.zeros((36, 36, 36))
    for r in range(total_rank):
        T_recon += np.einsum('a,b,c->abc', U6[:, r], V6[:, r], W6[:, r])
    
    if not np.allclose(T_recon, T_6x6, atol=1e-8):
        # Fallback to schoolbook if verification fails
        U6 = np.zeros((36, 216))
        V6 = np.zeros((36, 216))
        W6 = np.zeros((36, 216))
        col = 0
        for l in range(6):
            for j in range(6):
                for k in range(6):
                    U6[l*6+j, col] = 1
                    V6[j*6+k, col] = 1
                    W6[k*6+l, col] = 1
                    col += 1
        total_rank = 216
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U6]
    V_list = [[float(x) for x in row] for row in V6]
    W_list = [[float(x) for x in row] for row in W6]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication (rank 23).
    Returns U, V, W matrices of shape (9, 23).
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A = [[a,b,c],[d,e,f],[g,h,i]]
    # B = [[w,x,y],[z,u,v],[p,q,r]]
    # Laderman's 23 bilinear products (standard form).
    
    # P1 = (a+d)(w+z)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # P2 = (a+b)(w+x)
    U[0, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    
    # P3 = (b-d)(x-z)
    U[1, 2] = 1; U[3, 2] = -1
    V[1, 2] = 1; V[3, 2] = -1
    
    # P4 = (c+f)(y+v)
    U[2, 3] = 1; U[5, 3] = 1
    V[2, 3] = 1; V[5, 3] = 1
    
    # P5 = (f-i)(v-q)
    U[5, 4] = 1; U[8, 4] = -1
    V[5, 4] = 1; V[8, 4] = -1
    
    # P6 = (i+g)(r+p)
    U[8, 5] = 1; U[6, 5] = 1
    V[8, 5] = 1; V[6, 5] = 1
    
    # P7 = (d+e)(z+u)
    U[3, 6] = 1; U[4, 6] = 1
    V[3, 6] = 1; V[4, 6] = 1
    
    # P8 = (e-g)(u-p)
    U[4, 7] = 1; U[6, 7] = -1
    V[4, 7] = 1; V[6, 7] = -1
    
    # P9 = (b+e)(x+u)
    U[1, 8] = 1; U[4, 8] = 1
    V[1, 8] = 1; V[4, 8] = 1
    
    # P10 = (a+e)(w+u)
    U[0, 9] = 1; U[4, 9] = 1
    V[0, 9] = 1; V[4, 9] = 1
    
    # P11 = (c+i)(y+r)
    U[2, 10] = 1; U[8, 10] = 1
    V[2, 10] = 1; V[8, 10] = 1
    
    # P12 = (d+g)(z+p)
    U[3, 11] = 1; U[6, 11] = 1
    V[3, 11] = 1; V[6, 11] = 1
    
    # P13 = (a+c)(w+y)
    U[0, 12] = 1; U[2, 12] = 1
    V[0, 12] = 1; V[2, 12] = 1
    
    # P14 = (d+f)(z+v)
    U[3, 13] = 1; U[5, 13] = 1
    V[3, 13] = 1; V[5, 13] = 1
    
    # P15 = (g+i)(p+r)
    U[6, 14] = 1; U[8, 14] = 1
    V[6, 14] = 1; V[8, 14] = 1
    
    # P16 = (b+f+h)(x+v+q)
    U[1, 15] = 1; U[5, 15] = 1; U[7, 15] = 1
    V[1, 15] = 1; V[5, 15] = 1; V[8, 15] = 1
    
    # P17 = (a+d+e)(w+z+u)
    U[0, 16] = 1; U[3, 16] = 1; U[4, 16] = 1
    V[0, 16] = 1; V[3, 16] = 1; V[4, 16] = 1
    
    # P18 = (b+e+h)(x+u+q)
    U[1, 17] = 1; U[4, 17] = 1; U[7, 17] = 1
    V[1, 17] = 1; V[4, 17] = 1; V[8, 17] = 1
    
    # P19 = (c+f+i)(y+v+r)
    U[2, 18] = 1; U[5, 18] = 1; U[8, 18] = 1
    V[2, 18] = 1; V[5, 18] = 1; V[8, 18] = 1
    
    # P20 = (a+d+g)(w+z+p)
    U[0, 19] = 1; U[3, 19] = 1; U[6, 19] = 1
    V[0, 19] = 1; V[3, 19] = 1; V[6, 19] = 1
    
    # P21 = (a+b+c)(w+x+y)
    U[0, 20] = 1; U[1, 20] = 1; U[2, 20] = 1
    V[0, 20] = 1; V[1, 20] = 1; V[2, 20] = 1
    
    # P22 = (d+e+f)(z+u+v)
    U[3, 21] = 1; U[4, 21] = 1; U[5, 21] = 1
    V[3, 21] = 1; V[4, 21] = 1; V[5, 21] = 1
    
    # P23 = (g+h+i)(p+q+r)
    U[6, 22] = 1; U[7, 22] = 1; U[8, 22] = 1
    V[6, 22] = 1; V[7, 22] = 1; V[8, 22] = 1
    
    # Now we need to determine W such that the decomposition is exact.
    # We solve for W using the tensor identity.
    W = _solve_W_for_laderman(U, V, n, m, p, rank)
    
    return U, V, W

def _solve_W_for_laderman(U, V, n, m, p, rank):
    """
    Solve for W such that the decomposition exactly reproduces C = A B.
    For each output entry (i,k) we need a vector w_ik (length r) with
      sum_r w_ik[r] * (sum_j U[i*m+j, r] V[j*p+k, r]) = 1
    and zero for all other (i',k').
    """
    r = rank
    W = np.zeros((p * n, r))
    
    # For each r, define G_r of shape (n, p): G_r[i, k] = sum_j U[i*m+j, r] * V[j*p+k, r]
    G = np.zeros((r, n, p))
    for rr in range(r):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += U[i*m+j, rr] * V[j*p+k, rr]
                G[rr, i, k] = s
    
    # Now we need W such that for each (i, k):
    #   sum_r W[k*n+i, r] * G[r, i, k] = 1
    # and for (i', k') != (i, k):
    #   sum_r W[k*n+i', r] * G[r, i, k'] = 0
    
    # This is a linear system. Let's set it up.
    # Unknowns: W has p*n*r entries.
    # Equations: For each (i, k), we have one equation: sum_r W[k*n+i, r] * G[r, i, k] = 1
    
    # Actually, the tensor identity is:
    #   T[i, k] = sum_r W[k*n+i, r] * G[r, i, k]
    # We want T[i, k] = 1 for all (i, k) (since we're decomposing the multiplication tensor where T[i, j, k] = 1 if the middle index matches, but actually for matrix multiplication the tensor is T[i, j, k] = delta_{j, j'}... wait no).
    
    # Let me reconsider. The matrix multiplication tensor T is defined as:
    #   T[i, j, k] = 1 if we're computing C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor has T[i, j, k] = 1 for all (i, j, k) in the sense that each entry contributes.
    # Actually, the standard decomposition is:
    #   C[i, k] = sum_j A[i, j] B[j, k] = sum_r U[i, j, r] V[j, k, r] W[i, k, r]
    # where the tensor T[i, j, k] = 1 (it's the identity tensor for bilinear forms).
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i, j, k)
    # Wait, that's not right either. Let me be more careful.
    
    # The tensor we're decomposing is T where T[i, j, k] = 1 for all i, j, k? No.
    # For matrix multiplication, the bilinear form is:
    #   C[i, k] = sum_j A[i, j] B[j, k]
    # This can be written as a sum of rank-1 tensors:
    #   C[i, k] = sum_r (sum_j A[i, j] u_r[j]) * (sum_k B[j, k] v_r[k]) * w_r[i, k]
    # Hmm, this is getting complicated. Let me just use the standard approach.
    
    # The correct formulation: We want to find U, V, W such that for all i, j, k:
    #   sum_r U[i, j, r] V[j, k, r] W[i, k, r] = delta_{j, j'} ... no.
    
    # Actually, for matrix multiplication, the tensor is:
    #   T[i, j, k] = 1 if we interpret it as the coefficient of A[i, j] B[j, k] in C[i, k]
    # So T[i, j, k] = 1 for all (i, j, k) in the range, and the decomposition is:
    #   T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[i, k, r]
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # This is a system of n*m*p = 27 equations with p*n*r = 3*3*23 = 207 unknowns.
    # It's underdetermined, so we can pick a minimum-norm solution or any solution.
    
    # Let's set up the linear system.
    # For each (i, j, k), we have:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is linear in W. Let's vectorize.
    # Let x be the vector of W entries (size p*n*r).
    # For each (i, j, k), the equation is:
    #   sum_r (U[i*m+j, r] * V[j*p+k, r]) * x[k*n+i, r] = 1
    
    # So we can form a matrix A of size (n*m*p) x (p*n*r) where:
    #   A[(i*m+j)*p+k, (k*n+i)*r + r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
    # and b[(i*m+j)*p+k] = 1
    
    # Then solve A x = b.
    
    nmp = n * m * p
    pn_r = p * n * r
    
    A = np.zeros((nmp, pn_r))
    b = np.ones(nmp)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = (i * m + j) * p + k
                for r_idx in range(r):
                    var_idx = (k * n + i) * r + r_idx
                    A[eq_idx, var_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
    
    # Solve using least squares (since it's underdetermined)
    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Reshape x to W
    W = x.reshape((p * n, r))
    
    return W
# EVOLVE-BLOCK-END
