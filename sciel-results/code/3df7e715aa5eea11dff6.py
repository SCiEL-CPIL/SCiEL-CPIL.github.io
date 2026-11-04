"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 60, try to reduce using Strassen-like combinations
    # For 2x2x2 blocks, Strassen reduces from 8 to 7
    # We have 2 rows, 3 middle, 10 cols
    # Group cols into 5 pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # For each col pair, we have a 2x3x2 sub-tensor
    # If we can rank-reduce each 2x3x2 sub-tensor, we save overall
    
    # Let's try: for each pair of cols (k, k+1) and the 3 middle indices,
    # the sub-tensor is 2x3x2. Schoolbook rank for this is 12 (2*3*2).
    # Total schoolbook: 5 * 12 = 60. Correct.
    
    # Can we do better? For a 2x2x2 tensor, rank 7 < 8.
    # For 2x3x2, let's see if we can use Strassen on pairs of middle indices.
    
    # Strategy: For each col pair, use middle index pairs with Strassen
    # Middle indices: 0, 1, 2
    # Use Strassen on (j=0, j=1) for the 2x2x2 part, and handle j=2 separately
    # Strassen for 2x2x2: 7 terms instead of 8
    # Plus 2 terms for j=2 (one for each col in the pair)
    # Total per col pair: 7 + 2 = 9 instead of 12
    # Total: 5 * 9 = 45
    
    # But wait, Strassen combines products across both middle AND output indices
    # Let me be more careful.
    
    # For fixed col pair (k, k+1), the sub-tensor T(i, j, k') for i in [2], j in [3], k' in {k, k+1}
    # Schoolbook: for each i, j, k', one term: A[i,j] * B[j, k+n*k']
    # That's 2*3*2 = 12 terms.
    
    # Strassen on the 2x2 part (i in [2], j in {0,1}, k' in {0,1} relative to the pair):
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])  [but B indices are j, k]
    # This is getting complex. Let me just implement a known rank-55 construction.
    
    # Alternative: use the fact that the 2x3x10 tensor can be decomposed
    # by treating it as a sum of 3 (2x1x10) tensors, each of rank 2*10 = 20
    # Total: 60. No savings.
    
    # Let me try a direct rank-55 construction using Strassen on 2x2x2 sub-blocks.
    
    # For each of 5 col-pairs (k0, k1) where k0=2t, k1=2t+1:
    #   We have sub-tensor with i in [2], j in [3], k in [2] (local)
    #   Use Strassen on the j={0,1} part: 7 terms for the 2x2x2 (i, j in {0,1}, k local)
    #   Use schoolbook for j=2: 2*2 = 4 terms (i in [2], k local in [2])
    #   Total per col-pair: 7 + 4 = 11
    #   Total: 5 * 11 = 55
    
    # Let me verify Strassen's 7 terms for 2x2x2:
    # C[0,0] = M1 - M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 - M7
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For our case, with i in [2], j in {0,1}, k_local in [2]:
    # The "A" matrix is 2x2: A[i, j] for i in [2], j in {0,1}
    # The "B" matrix is 2x2: B[j, k_global] for j in {0,1}, k_global in {k0, k1}
    # So B[0,0] = B[0, k0], B[0,1] = B[0, k1], B[1,0] = B[1, k0], B[1,1] = B[1, k1]
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
    # M2 = (A[1,0]+A[1,1]) * B[0,k0]
    # M3 = A[0,0] * (B[0,k1]-B[1,k1])
    # M4 = A[1,1] * (B[1,k0]-B[0,k0])
    # M5 = (A[0,0]+A[0,1]) * B[1,k1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
    
    # Then for j=2:
    # C[0,k0] += A[0,2]*B[2,k0]
    # C[0,k1] += A[0,2]*B[2,k1]
    # C[1,k0] += A[1,2]*B[2,k0]
    # C[1,k1] += A[1,2]*B[2,k1]
    
    # This gives 7 + 4 = 11 terms per col-pair, total 55.
    
    # Let me build this decomposition.
    
    U = []
    V = []
    W = []
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen terms for j in {0,1}
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: A[0,0] and A[1,1] have coefficient 1
        # V: B[0,k0] and B[1,k1] have coefficient 1
        # W: C[0,k0] gets +1, C[0,k1] gets +1, C[1,k0] gets +1, C[1,k1] gets -1
        # Wait, let me recheck the Strassen formulas.
        
        # C[0,0] = M1 - M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M3 + M6 - M7
        
        # So:
        # M1: C[0,0]+=1, C[1,1]+=1
        # M2: C[1,0]+=1
        # M3: C[0,1]+=1, C[1,1]+=-1
        # M4: C[0,0]+=-1, C[1,0]+=1
        # M5: C[0,0]+=-1, C[0,1]+=1
        # M6: C[1,1]+=1
        # M7: C[0,0]+=1, C[1,1]+=-1
        
        strassen_terms = [
            # (U_coeffs for i,j in {0,1}x{0,1}, V_coeffs for j in {0,1}x{k0,k1}, W_coeffs for i in [2], k_local in [2])
            # M1
            ([(0,0,1), (1,1,1)], [(0,k0,1), (1,k1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,k0,1)], [(1,0,1)]),
            # M3
            ([(0,0,1)], [(0,k1,1), (1,k1,-1)], [(0,1,1), (1,1,-1)]),
            # M4
            ([(1,1,1)], [(1,k0,1), (0,k0,-1)], [(0,0,-1), (1,0,1)]),
            # M5
            ([(0,0,1), (0,1,1)], [(1,k1,1)], [(0,0,-1), (0,1,1)]),
            # M6
            ([(1,0,1), (0,0,-1)], [(0,k0,1), (0,k1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1), (1,1,-1)], [(1,k0,1), (1,k1,1)], [(0,0,1), (1,1,-1)]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            u_row = {}
            for i, j, c in u_coeffs:
                idx = i * m + j
                u_row[idx] = u_row.get(idx, 0) + c
            
            v_row = {}
            for j, k, c in v_coeffs:
                idx = j * p + k
                v_row[idx] = v_row.get(idx, 0) + c
            
            w_row = {}
            for i, k_local, c in w_coeffs:
                k_global = k0 + k_local
                idx = k_global * n + i
                w_row[idx] = w_row.get(idx, 0) + c
            
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
        
        # Schoolbook for j=2
        for i in range(2):
            for k_local in range(2):
                k_global = k0 + k_local
                u_row = {i * m + 2: 1}
                v_row = {2 * p + k_global: 1}
                w_row = {k_global * n + i: 1}
                U.append(u_row)
                V.append(v_row)
                W.append(w_row)
    
    # Convert to dense matrices
    rank = len(U)
    U_dense = [[0] * rank for _ in range(n * m)]
    V_dense = [[0] * rank for _ in range(m * p)]
    W_dense = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in U[c].items():
            U_dense[idx][c] = val
        for idx, val in V[c].items():
            V_dense[idx][c] = val
        for idx, val in W[c].items():
            W_dense[idx][c] = val
    
    return {"U": U_dense, "V": V_dense, "W": W_dense}
# EVOLVE-BLOCK-END
