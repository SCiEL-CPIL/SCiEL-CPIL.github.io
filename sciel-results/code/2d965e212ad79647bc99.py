import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    # BCR23 decomposition for 3x3x3 matrix multiplication
    # Position mapping: (r,c) -> r*3+c
    # 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2) 6:(2,0) 7:(2,1) 8:(2,2)
    def get_bcr23_factors():
        """Return 23 (u,v,w) triples for 3x3x3 matmul, each vector of length 9."""
        factors = []
        
        # M1 = (a00+a11+a22)(b00+b11+b22)
        u = [1,0,0, 0,1,0, 0,0,1]
        v = [1,0,0, 0,1,0, 0,0,1]
        w = [1,0,0, 0,1,0, 0,0,1]
        factors.append((u,v,w))
        
        # M2 = (a00+a01+a10+a11)(b00+b02+b20+b22)
        u = [1,1,0, 1,1,0, 0,0,0]
        v = [1,0,1, 0,0,0, 1,0,1]
        w = [1,0,0, 0,0,0, 0,0,1]
        factors.append((u,v,w))
        
        # M3 = (a00+a02+a20+a22)(b00+b01+b10+b11)
        u = [1,0,1, 0,0,0, 1,0,1]
        v = [1,1,0, 1,1,0, 0,0,0]
        w = [1,0,0, 0,1,0, 0,0,0]
        factors.append((u,v,w))
        
        # M4 = (a01+a10+a12+a21)(b01+b10+b12+b21)
        u = [0,1,0, 1,0,1, 0,1,0]
        v = [0,1,0, 1,0,1, 0,1,0]
        w = [0,1,0, 1,0,1, 0,1,0]
        factors.append((u,v,w))
        
        # M5 = (a02+a11+a12+a21)(b02+b11+b12+b21)
        u = [0,0,1, 0,1,1, 0,1,0]
        v = [0,0,1, 0,1,1, 0,1,0]
        w = [0,0,1, 0,1,1, 0,1,0]
        factors.append((u,v,w))
        
        # M6 = (a01+a02+a10+a12+a20+a22)(b01+b02+b10+b12+b20+b22)
        u = [0,1,1, 1,0,1, 1,0,1]
        v = [0,1,1, 1,0,1, 1,0,1]
        w = [0,1,1, 1,0,1, 1,0,1]
        factors.append((u,v,w))
        
        # M7 = (a00+a01+a10+a11)(b02+b12+b22)
        u = [1,1,0, 1,1,0, 0,0,0]
        v = [0,0,1, 0,0,1, 0,0,1]
        w = [0,0,1, 0,0,1, 0,0,1]
        factors.append((u,v,w))
        
        # M8 = (a00+a02+a10+a20)(b01+b11+b21)
        u = [1,0,1, 1,0,0, 1,0,0]
        v = [0,1,0, 0,1,0, 0,1,0]
        w = [0,1,0, 0,1,0, 0,1,0]
        factors.append((u,v,w))
        
        # M9 = (a01+a10+a12+a21)(b00+b02+b20+b22)
        u = [0,1,0, 1,0,1, 0,1,0]
        v = [1,0,1, 0,0,0, 1,0,1]
        w = [1,0,0, 0,0,0, 0,0,1]
        factors.append((u,v,w))
        
        # M10 = (a02+a11+a12+a21)(b00+b10+b20+b22)
        u = [0,0,1, 0,1,1, 0,1,0]
        v = [1,0,0, 1,0,0, 1,0,1]
        w = [1,0,0, 0,0,0, 0,0,1]
        factors.append((u,v,w))
        
        # M11 = (a00+a11+a22)(b01+b12+b21)
        u = [1,0,0, 0,1,0, 0,0,1]
        v = [0,1,0, 0,0,1, 0,1,0]
        w = [0,1,0, 0,0,1, 0,1,0]
        factors.append((u,v,w))
        
        # M12 = (a01+a02+a10+a20)(b00+b11+b22)
        u = [0,1,1, 1,0,0, 1,0,0]
        v = [1,0,0, 0,1,0, 0,0,1]
        w = [1,0,0, 0,1,0, 0,0,1]
        factors.append((u,v,w))
        
        # M13 = (a02+a11+a20+a22)(b00+b01+b10+b12+b21+b22)
        u = [0,0,1, 0,1,0, 1,0,1]
        v = [1,1,0, 1,0,1, 0,1,1]
        w = [1,0,0, 0,1,0, 0,0,1]
        factors.append((u,v,w))
        
        # M14 = (a00+a01+a11+a12+a20+a22)(b00+b02+b10+b12+b20+b22)
        u = [1,1,0, 1,1,1, 1,0,1]
        v = [1,0,1, 1,0,1, 1,0,1]
        w = [1,0,0, 0,0,0, 0,0,1]
        factors.append((u,v,w))
        
        # M15 = (a00+a02+a10+a12+a20+a21)(b01+b02+b10+b11+b20+b21)
        u = [1,0,1, 1,0,1, 1,1,0]
        v = [0,1,1, 1,1,0, 1,1,0]
        w = [0,1,0, 0,0,0, 0,0,0]
        factors.append((u,v,w))
        
        # M16 = (a01+a02+a11+a12+a20+a22)(b01+b02+b10+b12+b20+b22)
        u = [0,1,1, 0,1,1, 1,0,1]
        v = [0,1,1, 0,0,1, 1,0,1]
        w = [0,1,1, 0,0,1, 1,0,1]
        factors.append((u,v,w))
        
        # M17 = (a00+a01+a02+a10+a11+a12)(b00+b01+b02+b10+b11+b12)
        u = [1,1,1, 1,1,1, 0,0,0]
        v = [1,1,1, 1,1,1, 0,0,0]
        w = [1,1,1, 1,1,1, 0,0,0]
        factors.append((u,v,w))
        
        # M18 = (a00+a01+a02+a10+a11+a12+a20+a21+a22)(b00+b01+b02+b10+b11+b12+b20+b21+b22)
        u = [1,1,1, 1,1,1, 1,1,1]
        v = [1,1,1, 1,1,1, 1,1,1]
        w = [0,0,0, 0,0,0, 1,1,1]
        factors.append((u,v,w))
        
        # M19 = (a01+a10+a20+a21+a22)(b00+b02+b10+b12+b20+b22)
        u = [0,1,0, 1,0,0, 1,1,1]
        v = [1,0,1, 1,0,1, 1,0,1]
        w = [1,0,0, 0,0,0, 0,0,1]
        factors.append((u,v,w))
        
        # M20 = (a00+a02+a10+a12+a20+a21+a22)(b00+b01+b10+b11+b20+b21+b22)
        u = [1,0,1, 1,0,1, 1,1,1]
        v = [1,1,0, 1,1,0, 1,1,1]
        w = [1,0,0, 0,1,0, 0,0,1]
        factors.append((u,v,w))
        
        # M21 = (a01+a02+a11+a12+a21+a22)(b01+b02+b11+b12+b21+b22)
        u = [0,1,1, 0,1,1, 0,1,1]
        v = [0,1,1, 0,1,1, 0,1,1]
        w = [0,1,1, 0,1,1, 0,1,1]
        factors.append((u,v,w))
        
        # M22 = (a00+a01+a10+a11+a20+a21+a22)(b00+b01+b10+b11+b20+b21+b22)
        u = [1,1,0, 1,1,0, 1,1,1]
        v = [1,1,0, 1,1,0, 1,1,1]
        w = [1,1,0, 1,1,0, 1,1,1]
        factors.append((u,v,w))
        
        # M23 = (a02+a12+a22)(b02+b12+b22)
        u = [0,0,1, 0,0,1, 0,0,1]
        v = [0,0,1, 0,0,1, 0,0,1]
        w = [0,0,1, 0,0,1, 0,0,1]
        factors.append((u,v,w))
        
        return factors
    
    # Verify BCR23 decomposition
    def verify_bcr23():
        factors = get_bcr23_factors()
        # Test with random matrices
        for _ in range(3):
            A = np.random.randn(3, 3)
            B = np.random.randn(3, 3)
            C_expected = A @ B
            C_computed = np.zeros((3, 3))
            for (u, v, w) in factors:
                ua = np.array(u).reshape(3, 3) @ A
                vb = B @ np.array(v).reshape(3, 3)
                C_computed += ua * vb * np.array(w).reshape(3, 3)
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    bcr23_valid = verify_bcr23()
    
    if not bcr23_valid:
        # Fallback to 2+2+1 partition with Strassen
        return _construct_221()
    
    # Use 3+2 partition with BCR23
    block_sizes = [3, 2]
    offsets = [0, 3, 5]
    
    bcr23_factors = get_bcr23_factors()
    
    # Total rank: 1 block of 3x3x3 (rank 23) + 6 blocks of mixed size (schoolbook) + 1 block of 2x2x2 (Strassen rank 7)
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 3 and mj == 3 and pk == 3:
                    total_rank += 23
                elif ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # Strassen factors for 2x2x2
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors
    
    strassen_factors = get_strassen_factors()
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 3 and mj == 3 and pk == 3:
                    # BCR23 for 3x3x3
                    for (uf, vf, wf) in bcr23_factors:
                        for idx in range(9):
                            if uf[idx] != 0:
                                lr = idx // 3
                                lc = idx % 3
                                U[(offsets[i] + lr) * m + (offsets[j] + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr = idx // 3
                                lc = idx % 3
                                V[(offsets[j] + lr) * p + (offsets[k] + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr = idx // 3
                                lc = idx % 3
                                W[(offsets[k] + lc) * n + (offsets[i] + lr), col] += wf[idx]
                        col += 1
                elif ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                U[(offsets[i] + lr) * m + (offsets[j] + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                V[(offsets[j] + lr) * p + (offsets[k] + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                W[(offsets[k] + lc) * n + (offsets[i] + lr), col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for mixed-size blocks
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(offsets[i] + li) * m + (offsets[j] + lj), col] = 1.0
                                V[(offsets[j] + lj) * p + (offsets[k] + lk), col] = 1.0
                                W[(offsets[k] + lk) * n + (offsets[i] + li), col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Final verification
    U_arr = U
    V_arr = V
    W_arr = W
    # Verify: sum_r U[i,r] * V[j,r] * W[k,r] should give C[i,k] where i indexes A rows*cols, j indexes B rows*cols, k indexes C rows*cols
    # Actually, the tensor decomposition means: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T[i,j,k] = 1 if j == k (i.e., A[i,j] * B[j,k] contributes to C[i,k])
    # Wait, the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 if this term contributes, i.e., always 1 for valid i,j,k
    # So T[i,j,k] = 1 for all i in [0,5), j in [0,5), k in [0,5)
    # And U has shape (n*m, rank) = (25, rank), V has shape (m*p, rank) = (25, rank), W has shape (p*n, rank) = (25, rank)
    # The decomposition: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1 for all i,j,k
    
    # Verify with a sample
    A_test = np.random.randn(5, 5)
    B_test = np.random.randn(5, 5)
    C_expected = A_test @ B_test
    C_computed = np.zeros((5, 5))
    for r in range(total_rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        # u_col[i*5+j] * v_col[j*5+k] * w_col[k*5+i]
        U_mat = u_col.reshape(5, 5)
        V_mat = v_col.reshape(5, 5)
        W_mat = w_col.reshape(5, 5)
        for i in range(5):
            for k in range(5):
                C_computed[i, k] += np.sum(U_mat[i, :] * V_mat[:, k] * W_mat[k, i])
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback
        return _construct_221()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_221():
    """Fallback: 2+2+1 partition with Strassen for 2x2x2 blocks."""
    n, m, p = 5, 5, 5
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors
    
    strassen_factors = get_strassen_factors()
    
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
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                U[(offsets[i] + lr) * m + (offsets[j] + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                V[(offsets[j] + lr) * p + (offsets[k] + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                W[(offsets[k] + lc) * n + (offsets[i] + lr), col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(offsets[i] + li) * m + (offsets[j] + lj), col] = 1.0
                                V[(offsets[j] + lj) * p + (offsets[k] + lk), col] = 1.0
                                W[(offsets[k] + lk) * n + (offsets[i] + li), col] = 1.0
                                col += 1
    
    assert col == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
