import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 5, 5, 5
    
    # Use 5 = 4 + 1 partition
    # Block sizes: [4, 1]
    block_sizes = [4, 1]
    offsets = [0, 4, 5]
    
    # Winograd F(4) algorithm for 4x4 matrix multiplication
    # This uses 12 multiplications instead of 16
    def get_f4_factors():
        """Return list of (U_vec, V_vec, W_vec) for F(4) 4x4x4.
        U_vec has length 16: indices [0..15] = row-major A
        V_vec has length 16: indices [0..15] = row-major B
        W_vec has length 16: indices [0..15] = row-major C
        """
        factors = []
        
        # F(4) algorithm: 12 products
        # The 12 products are combinations of sums/differences
        
        # Product 1: (a00+a01+a10+a11) * (b00+b01+b10+b11)
        factors.append(([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]))
        
        # Product 2: (a00-a01+a10-a11) * (b00-b01+b10-b11)
        factors.append(([1,-1,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,-1,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0]))
        
        # Product 3: (a00+a01-a10-a11) * (b00+b01-b10-b11)
        factors.append(([1,1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0]))
        
        # Product 4: (a00-a01-a10+a11) * (b00-b01-b10+b11)
        factors.append(([1,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [1,-1,1,-1,0,0,0,0,0,0,0,0,0,0,0,0]))
        
        # Products 5-8: for the 2x2 block at positions (0,2),(1,2),(2,0),(3,0) etc.
        # This is getting complex. Let me use a different approach.
        
        return factors
    
    # I'm not confident in my F(4) implementation. Let me use a safer approach.
    # Use 5 = 3 + 2 partition with Strassen for 2x2x2 blocks.
    
    block_sizes = [3, 2]
    offsets = [0, 3, 5]
    
    # Strassen decomposition for 2x2x2
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
    
    # Calculate total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
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
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U, "V": V, "W": W}
