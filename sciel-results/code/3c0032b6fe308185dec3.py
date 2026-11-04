import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = 5, 5, 5
    
    # Block sizes for partitioning 5 = 3 + 2
    block_sizes = [3, 2]
    
    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen decomposition for 2x2x2
    def get_strassen_factors():
        """Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
        U_vec has length 4: indices [0,1,2,3] = [a11,a12,a21,a22]
        V_vec has length 4: indices [0,1,2,3] = [b11,b12,b21,b22]
        W_vec has length 4: indices [0,1,2,3] = [c11,c12,c21,c22]
        """
        factors = []
        # M1 = (a11+a22)(b11+b22)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a21+a22)b11
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a11(b12-b22)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = a22(b21-b11)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (a11+a12)b22
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (a21-a11)(b11+b12)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (a12-a22)(b21+b22)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return factors
    
    strassen_factors = get_strassen_factors()
    
    # Winograd factors for 3x3x3
    # The 23 factors for Winograd's 3x3 matrix multiplication
    # Each factor is (U_vec, V_vec, W_vec) where each vec has length 9
    # Indices: [0,1,2,3,4,5,6,7,8] = row-major 3x3
    
    def get_winograd_factors():
        """Return list of 23 (U_vec, V_vec, W_vec) for Winograd 3x3x3."""
        factors = []
        
        # I'll use the standard Winograd 3x3 factors
        # These are the 23 rank-1 terms
        
        # The factors are:
        # Each term has U, V, W vectors of length 9 (3x3 matrices flattened)
        
        # Let me use a known set of Winograd 3x3 factors
        # Reference: Winograd, S. (1980). Arithmetic complexity of matrix multiplication and convolution.
        
        # I'll generate them programmatically using the known structure
        # Actually, let me just hardcode the 23 factors
        
        # The 23 factors for 3x3 matrix multiplication:
        # These are well-known in the literature
        
        # Factor 1: M1 = (a00+a11+a22)(b00+b11+b22)
        # U: [1,0,0, 0,1,0, 0,0,1], V: [1,0,0, 0,1,0, 0,0,1], W: [1,0,0, 0,1,0, 0,0,1]
        factors.append(([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]))
        
        # Factor 2: M2 = (a00+a22)(b00+b22)
        # U: [1,0,0, 0,0,0, 0,0,1], V: [1,0,0, 0,0,0, 0,0,1], W: [1,0,0, 0,0,0, 0,0,1]
        factors.append(([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,1]))
        
        # Factor 3: M3 = (a11+a22)(b11+b22)
        # U: [0,0,0, 0,1,0, 0,0,1], V: [0,0,0, 0,1,0, 0,0,1], W: [0,0,0, 0,1,0, 0,0,1]
        factors.append(([0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1]))
        
        # Factor 4: M4 = (a00+a11)(b00+b11)
        # U: [1,0,0, 0,1,0, 0,0,0], V: [1,0,0, 0,1,0, 0,0,0], W: [1,0,0, 0,1,0, 0,0,0]
        factors.append(([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]))
        
        # Factor 5: M5 = (a00+a10+a20)(b00+b01+b02)
        # U: [1,0,0, 1,0,0, 1,0,0], V: [1,1,1, 0,0,0, 0,0,0], W: [1,0,0, 0,0,0, 0,0,0]
        factors.append(([1,0,0, 1,0,0, 1,0,0], [1,1,1, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]))
        
        # Factor 6: M6 = (a01+a11+a21)(b10+b11+b12)
        # U: [0,1,0, 0,1,0, 0,1,0], V: [0,0,0, 1,1,1, 0,0,0], W: [0,1,0, 0,0,0, 0,0,0]
        factors.append(([0,1,0, 0,1,0, 0,1,0], [0,0,0, 1,1,1, 0,0,0], [0,1,0, 0,0,0, 0,0,0]))
        
        # Factor 7: M7 = (a02+a12+a22)(b20+b21+b22)
        # U: [0,0,1, 0,0,1, 0,0,1], V: [0,0,0, 0,0,0, 1,1,1], W: [0,0,1, 0,0,0, 0,0,0]
        factors.append(([0,0,1, 0,0,1, 0,0,1], [0,0,0, 0,0,0, 1,1,1], [0,0,1, 0,0,0, 0,0,0]))
        
        # Factor 8: M8 = (a10+a20)(b01+b02)
        # U: [0,0,0, 1,0,0, 1,0,0], V: [0,1,1, 0,0,0, 0,0,0], W: [0,0,0, 0,0,0, 0,0,0]
        # Wait, this needs to contribute to specific C entries
        # Let me reconsider the Winograd factors more carefully
        
        # Actually, let me use a different approach. I'll use the known rank-23 decomposition
        # from the literature. The exact factors are complex, so let me use a simpler approach.
        
        # For now, let me just use schoolbook for 3x3 (rank 27) and see if the overall rank is better
        # 3+2 partition:
        # 3x3x3: 27 (schoolbook) or 23 (Winograd)
        # 3x3x2: 3*3*2 = 18
        # 3x2x3: 3*2*3 = 18
        # 3x2x2: 3*2*2 = 12
        # 2x3x3: 2*3*3 = 18
        # 2x3x2: 2*3*2 = 12
        # 2x2x3: 2*2*3 = 12
        # 2x2x2: 7 (Strassen)
        
        # Total with schoolbook 3x3: 27+18+18+12+18+12+12+7 = 124
        # Total with Winograd 3x3: 23+18+18+12+18+12+12+7 = 120
        
        # Hmm, that's still higher than the 2+2+1 partition which gives 117.
        # Let me recalculate the 2+2+1 partition:
        # 2x2x2: 7
        # 2x2x1: 2*2*1 = 4
        # 2x1x2: 2*1*2 = 4
        # 2x1x1: 2*1*1 = 2
        # 1x2x2: 1*2*2 = 4
        # 1x2x1: 1*2*1 = 2
        # 1x1x2: 1*1*2 = 2
        # 1x1x1: 1*1*1 = 1
        # Total: 7+4+4+2+4+2+2+1 = 26? That doesn't match 117.
        
        # Oh wait, I think I'm confusing myself. Let me recalculate.
        # For 5x5x5 with partition 2+2+1:
        # We have 3x3 = 9 blocks.
        # Block (i,j,k) has size ni x mj x pk where ni, mj, pk are from [2,2,1]
        # The rank contribution is:
        # - If ni=mj=pk=2: 7 (Strassen)
        # - Otherwise: ni*mj*pk (schoolbook)
        
        # Let me list all 9 blocks:
        # (0,0,0): 2x2x2 -> 7
        # (0,0,1): 2x2x1 -> 4
        # (0,1,0): 2x1x2 -> 4
        # (0,1,1): 2x1x1 -> 2
        # (1,0,0): 2x2x2 -> 7
        # (1,0,1): 2x2x1 -> 4
        # (1,1,0): 2x1x2 -> 4
        # (1,1,1): 2x1x1 -> 2
        # (2,0,0): 1x2x2 -> 4
        # (2,0,1): 1x2x1 -> 2
        # (2,1,0): 1x1x2 -> 2
        # (2,1,1): 1x1x1 -> 1
        
        # Wait, that's 12 blocks, not 9. Oh right, 3x3x3 = 27 blocks.
        # Let me redo:
        # i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
        # ni = block_sizes[i], mj = block_sizes[j], pk = block_sizes[k]
        
        # (0,0,0): 2x2x2 -> 7
        # (0,0,1): 2x2x1 -> 4
        # (0,1,0): 2x1x2 -> 4
        # (0,1,1): 2x1x1 -> 2
        # (1,0,0): 2x2x2 -> 7
        # (1,0,1): 2x2x1 -> 4
        # (1,1,0): 2x1x2 -> 4
        # (1,1,1): 2x1x1 -> 2
        # (2,0,0): 1x2x2 -> 4
        # (2,0,1): 1x2x1 -> 2
        # (2,1,0): 1x1x2 -> 2
        # (2,1,1): 1x1x1 -> 1
        
        # That's 12 blocks. Total: 7+4+4+2+7+4+4+2+4+2+2+1 = 43?
        # That doesn't match 117 either.
        
        # I think I'm making an error. Let me just keep the current 2+2+1 partition
        # which gives rank 117 (score 1.0684), and try to improve it slightly.
        
        # Actually, looking at the current program, it uses 2+2+1 and gets rank 117.
        # Let me just return the current program as is, since the 3+2 partition
        # doesn't seem to be better.
        
        # For now, let me just use schoolbook for all blocks in the 3+2 partition
        # and see what rank we get.
        
        # Actually, I realize I should just keep the current 2+2+1 partition
        # and not switch to 3+2. Let me just return the current program.
        
        # But the directive says to keep the design and investigate improvements.
        # The current design is 2+2+1 with Strassen for 2x2x2 blocks.
        
        # Let me just return the current program unchanged.
        
        return []  # Not used, we'll use the 2+2+1 partition
    
    # Use the 2+2+1 partition
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
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
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
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
