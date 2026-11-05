import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
    # C = A*B where A,B,C are 2x2
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = [[c11,c12],[c21,c22]]
    # Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1+M4-M5+M7
    # c12 = M3+M5
    # c21 = M2+M4
    # c22 = M1-M2+M3+M6
    #
    # In tensor decomposition form, each M_k is a rank-1 term:
    # U_factor * V_factor * W_factor
    # where U_factor is a vector in R^4 (for 2x2 A), V_factor in R^4 (for 2x2 B), W_factor in R^4 (for 2x2 C)
    
    def get_strassen_factors():
        """Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
        U_vec has length 4: indices [0,1,2,3] = [a11,a12,a21,a22]
        V_vec has length 4: indices [0,1,2,3] = [b11,b12,b21,b22]
        W_vec has length 4: indices [0,1,2,3] = [c11,c12,c21,c22]
        """
        factors = []
        # M1 = (a11+a22)(b11+b22) -> contributes to c11 (+1) and c22 (+1)
        # U: [1, 0, 0, 1], V: [1, 0, 0, 1], W: [1, 0, 0, 1]
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a21+a22)b11 -> contributes to c21 (+1) and c22 (-1)
        # U: [0, 0, 1, 1], V: [1, 0, 0, 0], W: [0, 0, 1, -1]
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a11(b12-b22) -> contributes to c12 (+1) and c22 (+1)
        # U: [1, 0, 0, 0], V: [0, 1, 0, -1], W: [0, 1, 0, 1]
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = a22(b21-b11) -> contributes to c11 (+1) and c21 (+1)
        # U: [0, 0, 0, 1], V: [0, 0, 1, 0], W: [1, 0, 1, 0]
        # Wait, let me recheck. M4 = a22*(b21-b11)
        # U: [0, 0, 0, 1], V: [-1, 0, 1, 0], W: [1, 0, 1, 0]
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (a11+a12)b22 -> contributes to c11 (-1) and c12 (+1)
        # U: [1, 1, 0, 0], V: [0, 0, 0, 1], W: [-1, 1, 0, 0]
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (a21-a11)(b11+b12) -> contributes to c22 (+1)
        # U: [-1, 0, 1, 0], V: [1, 1, 0, 0], W: [0, 0, 0, 1]
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (a12-a22)(b21+b22) -> contributes to c11 (+1)
        # U: [0, 1, 0, -1], V: [0, 0, 1, 1], W: [1, 0, 0, 0]
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return factors
    
    strassen_factors = get_strassen_factors()
    
    # Total rank calculation
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate matrices
    U = [[0.0] * total_rank for _ in range(n * m)]  # 25 x total_rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 25 x total_rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 25 x total_rank
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                # Block A[i,j]: rows offsets[i]:offsets[i+1], cols offsets[j]:offsets[j+1]
                # Block B[j,k]: rows offsets[j]:offsets[j+1], cols offsets[k]:offsets[k+1]
                # Block C[i,k]: rows offsets[i]:offsets[i+1], cols offsets[k]:offsets[k+1]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    for (uf, vf, wf) in strassen_factors:
                        # uf: [a11, a12, a21, a22] -> indices in A block
                        # In global A: row = offsets[i] + local_row, col = offsets[j] + local_col
                        # local indices: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        # U index: (offsets[i] + local_row) * m + (offsets[j] + local_col)
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),  # a11
                            (offsets[i] + 0) * m + (offsets[j] + 1),  # a12
                            (offsets[i] + 1) * m + (offsets[j] + 0),  # a21
                            (offsets[i] + 1) * m + (offsets[j] + 1),  # a22
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),  # b11
                            (offsets[j] + 0) * p + (offsets[k] + 1),  # b12
                            (offsets[j] + 1) * p + (offsets[k] + 0),  # b21
                            (offsets[j] + 1) * p + (offsets[k] + 1),  # b22
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),  # c11
                            (offsets[k] + 1) * n + (offsets[i] + 0),  # c12
                            (offsets[k] + 0) * n + (offsets[i] + 1),  # c21
                            (offsets[k] + 1) * n + (offsets[i] + 1),  # c22
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                # A block element: local (li, lj) -> global (offsets[i]+li, offsets[j]+lj)
                                # B block element: local (lj, lk) -> global (offsets[j]+lj, offsets[k]+lk)
                                # C block element: local (li, lk) -> global (offsets[i]+li, offsets[k]+lk)
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Deadline guard
    if time.time() - start_time > DEADLINE - 5:
        raise TimeoutError("Deadline exceeded")
    
    return {"U": U, "V": V, "W": W}
