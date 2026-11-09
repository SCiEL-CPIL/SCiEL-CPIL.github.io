import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8

    # Verify current rank-56 Strassen decomposition
    # Then try to find a lower rank decomposition

    num_groups = 4  # k-blocks of size 2
    num_ablocks = 2  # j-blocks of size 2  
    num_strassen = 7

    rank = num_groups * num_ablocks * num_strassen  # 56

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    X_forms = np.array([
        [1, 0, 0, 1],    # m1: a + d
        [0, 0, 1, 1],    # m2: c + d
        [1, 0, 0, 0],    # m3: a
        [0, 0, 0, 1],    # m4: d
        [1, 1, 0, 0],    # m5: a + b
        [-1, 0, 1, 0],   # m6: c - a
        [0, 1, 0, -1],   # m7: b - d
    ], dtype=np.int64)

    Y_forms = np.array([
        [1, 0, 0, 1],    # m1: e + h
        [1, 0, 0, 0],    # m2: e
        [0, 1, 0, -1],   # m3: f - h
        [-1, 0, 1, 0],   # m4: g - e
        [0, 0, 0, 1],    # m5: h
        [1, 1, 0, 0],    # m6: e + f
        [0, 0, 1, 1],    # m7: g + h
    ], dtype=np.int64)

    Z_combos = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # c01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # c10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # c11 = m1 - m2 + m3 + m6
    ], dtype=np.int64)

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = X_forms[s, i * 2 + jp]

                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = Y_forms[s, jp * 2 + kp]

                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = Z_combos[i * 2 + kp, s]

                r += 1

    # Verify the decomposition
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the convention is W[k*n+i, r]
    
    # Let me verify with random matrices
    A = np.random.randint(-5, 5, size=(n, m)).astype(np.int64)
    B = np.random.randint(-5, 5, size=(m, p)).astype(np.int64)
    C_expected = A @ B
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p), dtype=np.int64)
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    for r_idx in range(rank):
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # C[i,k] += sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But this isn't quite right. The tensor decomposition is:
        # T = sum_r U_r x V_r x W_r
        # where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]... no.
        
        # Actually, the matrix multiplication tensor M satisfies:
        # M(A, B) = C where C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor M has components M[i,j,k] such that:
        # sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (something)
        
        # I think the correct interpretation is:
        # The tensor T[i,j,k] = 1 for all i,j,k is WRONG.
        # 
        # Let me reconsider. The matrix multiplication can be written as:
        # C = A @ B
        # In tensor form: C_{ik} = sum_j A_{ij} B_{jk}
        # This is a bilinear map from (A, B) to C.
        # 
        # A rank-r decomposition of this bilinear map means:
        # C_{ik} = sum_{r=1}^{R} sum_j (sum_j U_{ij,r} A_{ij}) * (sum_j V_{jk,r} B_{jk}) * W_{ik,r}
        # 
        # Wait, that's not right either. Let me think more carefully.
        #
        # The bilinear map is: (A, B) -> C where C_{ik} = sum_j A_{ij} B_{jk}
        # A rank-1 term is: (A, B) -> u(A) * v(B) * w where u: R^{nm} -> R, v: R^{mp} -> R, w in R^{np}
        # So a rank-R decomposition is: C = sum_r u_r(A) * v_r(B) * w_r
        # where u_r(A) = sum_{i,j} U_{ij,r} A_{ij}, v_r(B) = sum_{j,k} V_{jk,r} B_{jk}, w_r = W_r (a vector in R^{np})
        #
        # So C_{ik} = sum_r (sum_{i',j} U_{i'j,r} A_{i'j}) * (sum_{j',k'} V_{j'k',r} B_{j'k'}) * W_{ik,r}
        #
        # For this to equal sum_j A_{ij} B_{jk} for all A, B, we need:
        # sum_r U_{i'j,r} V_{j'k',r} W_{ik,r} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... no.
        #
        # Actually, we need: for all A, B: sum_{i,k} C_{ik} * 1 = sum_{i,k} (sum_j A_{ij} B_{jk})
        # This is automatically satisfied if the bilinear maps match.
        #
        # The correct condition is: the tensor M_{ijk} (where M is the matrix multiplication tensor)
        # satisfies M_{ijk} = 1 if we're looking at it as a 3-index tensor where
        # M(A,B) = sum_{i,j,k} M_{ijk} A_{ij} B_{jk} gives the "trace" of the product.
        #
        # I'm getting confused. Let me just verify numerically.
        # The correct verification:
        # Given U, V, W of shapes (nm, R), (mp, R), (np, R)
        # The decomposition says: for all A (n x m), B (m x p):
        # C = A @ B = sum_r (U^T A_vec) * (V^T B_vec) * W_r (reshaped to n x p)
        # where A_vec is A flattened, B_vec is B flattened, W_r is the r-th column of W reshaped to n x p.
        
        A_vec = A.flatten()
        B_vec = B.flatten()
        
        C_from_decomp = np.zeros((n, p), dtype=np.float64)
        for r_idx in range(rank):
            u_val = np.dot(U[:, r_idx], A_vec)
            v_val = np.dot(V[:, r_idx], B_vec)
            w_vec = W[:, r_idx].reshape(p, n).T  # W[k*n+i, r] -> C[i,k]
            C_from_decomp += u_val * v_val * w_vec
    
    # Check
    max_err = np.max(np.abs(C_from_decomp - C_expected.astype(np.float64)))
    print(f"Max error: {max_err}")
    
    if max_err > 1e-9:
        print("DECOMPOSITION IS INCORRECT!")
        # Try to debug
        for r_idx in range(min(rank, 10)):
            u_val = np.dot(U[:, r_idx], A_vec)
            v_val = np.dot(V[:, r_idx], B_vec)
            w_vec = W[:, r_idx].reshape(p, n).T
            print(f"r={r_idx}: u={u_val}, v={v_val}, w_nonzero={np.nonzero(w_vec)[0][:5]}")
    else:
        print("Decomposition verified correct!")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
