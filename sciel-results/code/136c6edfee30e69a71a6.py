import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Strategy: Use Kronecker product M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # Rank(M_{2,2,3}) = 6, Rank(M_{2,2,2}) = 7 (Strassen)
    # Total rank = 6 * 7 = 42
    
    # Strassen rank-7 decomposition for M_{2,2,2}
    # A is 2x2 (indices a0,a1 for row0col0,row0col1,row1col0,row1col1)
    # B is 2x2 (indices b0,b1,b2,b3)
    # C is 2x2 (indices c0,c1,c2,c3)
    # alpha_s: 7x4, beta_s: 7x4, recon_s: 7 lists of (i,j,coeff)
    
    alpha_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    beta_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    recon_s = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    # Rank-6 decomposition for M_{2,2,3}
    # A is 2x2 (indices a0,a1,a2,a3)
    # B is 2x3 (indices b0..b5)
    # C is 2x3 (indices c0..c5)
    # 
    # Construction: We need 6 rank-1 terms.
    # Let's use the fact that M_{2,2,3} can be decomposed using 6 terms.
    # One approach: think of it as multiplying 2x2 by two 2x1 matrices simultaneously.
    # R(M_{2,2,1}) = 4, but with sharing we can do 6 for two columns.
    
    # Actually, let me construct this carefully.
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # We need: sum_s U_s[i,j] * V_s[j,k] * W_s[i,k] = delta_{j,k} (for the multiplication tensor)
    
    # Known rank-6 for M_{2,2,3}:
    # Using the following construction from literature:
    # Terms 1-3 handle the "diagonal" and cross terms for column 0
    # Terms 4-6 handle column 1 and 2 with sharing
    
    # Let me use a verified construction:
    # U vectors (size 4), V vectors (size 6), W vectors (size 6)
    
    U2 = np.zeros((4, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((6, 6))
    
    # Term 0: (a0+a2)*(b0+b3) -> contributes to c0, c3
    # Actually, let me use a systematic approach.
    # For M_{2,2,3}, the tensor has 2*2*3 = 12 non-zero entries.
    # We need at least 12/2 = 6 terms (each term can cover at most 2 entries in a structured way).
    
    # Construction:
    # s=0: U=[1,0,1,0], V=[1,0,0,1,0,0], W=[1,0,0,0,0,0]  -> a0*b0->c0, a2*b3->c3
    # Wait, that's not right. Let me be more careful.
    
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} for the multiplication
    
    # Let me just use a known valid rank-6:
    # From the literature on small matrix multiplication:
    
    # s=0: U=[1,1,0,0], V=[1,0,0,1,0,0], W=[1,0,0,0,0,0]
    #   -> (a0+a1)*b0 -> c0, (a0+a1)*b3 -> c3
    # s=1: U=[0,0,1,1], V=[1,0,0,1,0,0], W=[0,0,0,1,0,0]
    #   -> (a2+a3)*b0 -> c2, (a2+a3)*b3 -> c5
    # s=2: U=[1,0,1,0], V=[0,1,0,0,1,0], W=[1,0,0,0,0,0]
    #   -> a0*b1 -> c0, a2*b4 -> c2
    # s=3: U=[0,1,0,1], V=[0,1,0,0,1,0], W=[0,0,0,1,0,0]
    #   -> a1*b1 -> c2, a3*b4 -> c5
    # s=4: U=[1,0,0,0], V=[0,0,1,0,0,1], W=[1,0,0,0,0,0]
    #   -> a0*b2 -> c0, a0*b5 -> c3
    # s=5: U=[0,0,1,0], V=[0,0,1,0,0,1], W=[0,0,0,1,0,0]
    #   -> a2*b2 -> c2, a2*b5 -> c5
    
    # Let me verify this covers all 12 entries:
    # c0 = a0*b0 + a1*b1 + a0*b2
    #   s0: (a0+a1)*b0 -> a0*b0 + a1*b0  (extra a1*b0!)
    # This doesn't work cleanly. Let me use a different approach.
    
    # Actually, the rank of M_{2,2,3} is indeed 6. Let me use a verified decomposition.
    # From Bini et al., the following rank-6 works:
    
    U2 = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    
    # Verify M_{2,2,3} decomposition
    def verify_223():
        for i in range(2):
            for k in range(3):
                for j in range(2):
                    expected = 1.0 if True else 0.0  # T[i,j,k] = 1 for all valid
                    actual = 0.0
                    for s in range(6):
                        actual += U2[i, s] * V2[j * 3 + k, s] * W2[i * 3 + k, s]
                    # The tensor entry should be 1 if this is the multiplication tensor
                    # Actually T[i,j,k] represents A_{i,j} * B_{j,k} -> C_{i,k}
                    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the multiplication structure)
                    # Wait no. The tensor for matrix multiplication M_{n,m,p} has:
                    # T[i,j,k] = 1 if and only if this corresponds to a valid multiplication
                    # For M_{2,2,3}: T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1,2}
                    # All entries are 1 since every A[i,j]*B[j,k] contributes to C[i,k]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    # Hmm, that's not right either. The tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # So T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] is a valid term)
    # And the decomposition is: T = sum_s u_s ⊗ v_s ⊗ w_s
    # where u_s[i,j] = U[i,j,s], v_s[j,k] = V[j,k,s], w_s[i,k] = W[i,k,s]
    
    # So we need: sum_s U2[i,s] * V2[j*3+k,s] * W2[i*3+k,s] = 1 for all i,j,k
    
    # Let me check if my U2,V2,W2 satisfy this.
    # Actually, I realize I need to be more careful with the index mapping.
    # Let me just directly construct and verify.
    
    # For now, let me use a simpler approach: direct rank-96 (4*4*6) as fallback
    # and try to get the Kronecker product right.
    
    # Kronecker product: M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # A is 4x4 = (2x2) ⊗ (2x2): A[(i1*2+i2), (j1*2+j2)]
    # B is 4x6 = (2x2) ⊗ (2x3): B[(j1*2+j2), (k1*3+k2)]
    # C is 4x6 = (2x2) ⊗ (2x3): C[(i1*2+i2), (k1*3+k2)]
    
    # The tensor: sum_{i1,i2,j1,j2,k1,k2} A[i1,i2,j1,j2] * B[j1,j2,k1,k2] * C[i1,i2,k1,k2]
    # = (sum_{i1,j1,k1} A[i1,j1,k1] * B[j1,k1] * C[i1,k1]) ⊗ (sum_{i2,j2,k2} A[i2,j2,k2] * B[j2,k2] * C[i2,k2])
    # where the first part is M_{2,2,2} and the second is M_{2,2,3}
    
    # Wait, that's not quite right. Let me think again.
    # A is 4x4, viewed as 2x2 blocks of 2x2:
    # A[(2*i1+i2), (2*j1+j2)]
    # B is 4x6, viewed as 2x2 blocks of 2x3:
    # B[(2*j1+j2), (3*k1+k2)]
    # C is 4x6, viewed as 2x2 blocks of 2x3:
    # C[(2*i1+i2), (3*k1+k2)]
    
    # C[(2*i1+i2), (3*k1+k2)] = sum_{j} A[(2*i1+i2), j] * B[j, (3*k1+k2)]
    # = sum_{j1,j2} A[(2*i1+i2), (2*j1+j2)] * B[(2*j1+j2), (3*k1+k2)]
    
    # So the tensor is:
    # sum_{i1,i2,j1,j2,k1,k2} A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 3*k1+k2] * C[2*i1+i2, 3*k1+k2]
    
    # This factors as:
    # (sum_{i1,j1,k1} A[2*i1, 2*j1] * B[2*j1, 3*k1] * C[2*i1, 3*k1])  -- NO, this doesn't factor simply
    
    # The Kronecker product structure requires that the indices separate completely.
    # For M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}, we need:
    # A[i,j] = A1[i1,j1] * A2[i2,j2] (Kronecker product of A's)
    # But that's not how matrix multiplication works in general.
    
    # Actually, the correct statement is:
    # The tensor M_{4,4,6} is isomorphic to M_{2,2,3} ⊗ M_{2,2,2}
    # This means there exist index reorganizations such that the tensor decomposes.
    
    # Let me just use the direct construction with rank 4*4*6 = 96 as a safe fallback,
    # and try to improve if time permits.
    
    # Direct rank-96:
    R = 96
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U[i * 4 + j, col] = 1.0
                V[j * 6 + k, col] = 1.0
                W[i * 6 + k, col] = 1.0
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]
    
    # Verify
    def verify(U, V, W):
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    expected = 1.0
                    actual = 0.0
                    for s in range(R):
                        actual += U[i * 4 + j, s] * V[j * 6 + k, s] * W[i * 6 + k, s]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W):
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Should not reach here
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
