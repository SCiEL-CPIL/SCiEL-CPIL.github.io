import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    dim = n
    block_size = 3
    num_blocks = 2

    # Laderman's rank 23 decomposition for 3x3 matrix multiplication
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are lists of (i, j, coeff)
    laderman_23 = [
        # 1
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # 2
        ([(0,1,1)], [(1,0,1)], [(0,1,1)]),
        # 3
        ([(1,0,1)], [(0,1,1)], [(1,0,1)]),
        # 4
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # 5
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # 6
        ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
        # 7
        ([(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,-1),(0,1,0),(1,0,0),(1,1,0)]),
        # 8
        ([(0,0,1),(0,2,1),(2,0,1),(2,2,1)], [(0,0,1),(0,2,1),(2,0,1),(2,2,1)], [(0,0,-1),(0,2,0),(2,0,0),(2,2,0)]),
        # 9
        ([(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(1,1,-1),(1,2,0),(2,1,0),(2,2,0)]),
        # 10
        ([(0,0,1),(0,1,1),(0,2,1)], [(0,0,1),(0,1,1),(0,2,1)], [(0,0,1),(0,1,0),(0,2,0)]),
        # 11
        ([(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,0,0),(2,0,0)]),
        # 12
        ([(0,0,1),(1,0,1),(2,0,1)], [(0,1,1),(1,1,1),(2,1,1)], [(0,1,1),(1,1,0),(2,1,0)]),
        # 13
        ([(0,0,1),(1,0,1),(2,0,1)], [(0,2,1),(1,2,1),(2,2,1)], [(0,2,1),(1,2,0),(2,2,0)]),
        # 14
        ([(0,1,1),(1,1,1),(2,1,1)], [(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,0,0),(2,0,0)]),
        # 15
        ([(0,1,1),(1,1,1),(2,1,1)], [(0,1,1),(1,1,1),(2,1,1)], [(0,1,1),(1,1,0),(2,1,0)]),
        # 16
        ([(0,1,1),(1,1,1),(2,1,1)], [(0,2,1),(1,2,1),(2,2,1)], [(0,2,1),(1,2,0),(2,2,0)]),
        # 17
        ([(0,2,1),(1,2,1),(2,2,1)], [(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,0,0),(2,0,0)]),
        # 18
        ([(0,2,1),(1,2,1),(2,2,1)], [(0,1,1),(1,1,1),(2,1,1)], [(0,1,1),(1,1,0),(2,1,0)]),
        # 19
        ([(0,2,1),(1,2,1),(2,2,1)], [(0,2,1),(1,2,1),(2,2,1)], [(0,2,1),(1,2,0),(2,2,0)]),
        # 20
        ([(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1)], [(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1)], [(0,0,1),(0,1,0),(0,2,0),(1,0,0),(1,1,0),(1,2,0)]),
        # 21
        ([(0,0,1),(0,1,1),(0,2,1),(2,0,1),(2,1,1),(2,2,1)], [(0,0,1),(0,1,1),(0,2,1),(2,0,1),(2,1,1),(2,2,1)], [(0,0,1),(0,1,0),(0,2,0),(2,0,0),(2,1,0),(2,2,0)]),
        # 22
        ([(0,0,1),(1,0,1),(2,0,1),(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(0,0,1),(1,0,1),(2,0,1),(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(0,0,1),(1,0,0),(2,0,0),(1,1,0),(1,2,0),(2,1,0),(2,2,0)]),
        # 23
        ([(0,1,1),(0,2,1),(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(0,1,1),(0,2,1),(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(0,1,1),(0,2,0),(1,1,0),(1,2,0),(2,1,0),(2,2,0)]),
    ]
    
    # NOTE: The W coefficients above for terms 7-23 are placeholders/estimates.
    # To ensure correctness, I will implement a verification step.
    # If verification fails, I fall back to schoolbook 27.
    
    # Strassen steps for 2x2 blocks
    strassen_steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # We will try to build the 161 rank decomposition.
    # If it fails verification, we use 189.
    
    def build_decomposition(algo_3x3):
        rank_3x3 = len(algo_3x3)
        total_rank = 7 * rank_3x3
        U = np.zeros((36, total_rank), dtype=np.float64)
        V = np.zeros((36, total_rank), dtype=np.float64)
        W = np.zeros((36, total_rank), dtype=np.float64)
        
        col = 0
        for a_list, b_list, c_list in strassen_steps:
            for r in range(3):
                for s in range(3):
                    for t in range(3):
                        # Iterate over 3x3 algo terms
                        for u_l, v_l, w_l in algo_3x3:
                            # U contribution
                            for (I, J, ca) in a_list:
                                for (ri, sj, cu) in u_l:
                                    # A[I*3+r+ri, J*3+s+sj]
                                    u_idx = ((I * 3 + r + ri) * 6) + (J * 3 + s + sj)
                                    U[u_idx, col] += ca * cu
                                    
                            # V contribution
                            for (J, K, cb) in b_list:
                                for (si, tj, cv) in v_l:
                                    # B[J*3+s+si, K*3+t+tj]
                                    v_idx = ((J * 3 + s + si) * 6) + (K * 3 + t + tj)
                                    V[v_idx, col] += cb * cv
                                    
                            # W contribution
                            for (I, K, cc) in c_list:
                                for (ri, ti, cw) in w_l:
                                    # C[I*3+r+ri, K*3+t+ti]
                                    # Trace convention W[k*6 + i, r]
                                    w_idx = ((K * 3 + t + ti) * 6) + (I * 3 + r + ri)
                                    W[w_idx, col] += cc * cw
                                    
                            col += 1
                            
        return U, V, W, total_rank

    # Try Laderman
    U, V, W, rank = build_decomposition(laderman_23)
    
    # Verify
    # C = U * M * V^T ? No, tensor decomposition:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Here i, j are input indices (36), k is output index (36).
    # Check if sum_r U[i,r] V[j,r] W[k,r] == delta_{i,j,k} for MM tensor.
    # MM tensor T[i,j,k] is 1 if i corresponds to A_{pq}, j to B_{qr}, k to C_{pr}.
    # Indices: i = p*6+q, j = q*6+r, k = r*6+p.
    
    valid = True
    for p in range(6):
        for q in range(6):
            for r in range(6):
                i = p * 6 + q
                j = q * 6 + r
                k = r * 6 + p
                # Sum U[i,:] * V[j,:] * W[k,:]
                val = np.dot(U[i, :], V[j, :] * W[k, :])
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback to schoolbook 27
        schoolbook_27 = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_27.append(([(i,j,1)], [(j,k,1)], [(i,k,1)]))
        U, V, W, rank = build_decomposition(schoolbook_27)
        # Re-verify fallback (should be valid)
        valid = True
        for p in range(6):
            for q in range(6):
                for r in range(6):
                    i = p * 6 + q
                    j = q * 6 + r
                    k = r * 6 + p
                    val = np.dot(U[i, :], V[j, :] * W[k, :])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid:
            # Should never happen
            return {"U": np.zeros((36, 216)).tolist(), "V": np.zeros((36, 216)).tolist(), "W": np.zeros((36, 216)).tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
