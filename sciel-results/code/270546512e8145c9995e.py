import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    block_sizes = [4, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
        
    def get_strassen_2x2x2_factors():
        # Returns list of (u, v, w) for 2x2x2
        # u, v, w are lists of 4 floats
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]

    def get_strassen_4x4x4_factors():
        s2 = get_strassen_2x2x2_factors()
        
        # Sub-block indices for 4x4
        # 0: 11, 1: 12, 2: 21, 3: 22
        # Indices in 4x4 matrix (row-major)
        # 11: 0,1,4,5
        # 12: 2,3,6,7
        # 21: 8,9,12,13
        # 22: 10,11,14,15
        sub_idx = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        # Strassen steps for 4x4x4
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        steps = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
            ({2: 1, 3: 1}, {0: 1}),       # M2
            ({0: 1}, {1: 1, 3: -1}),      # M3
            ({3: 1}, {2: 1, 0: -1}),      # M4
            ({0: 1, 1: 1}, {3: 1}),       # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
        ]
        
        factors = []
        for a_coeffs, b_coeffs in steps:
            for u, v, w in s2:
                u_global = [0.0] * 16
                v_global = [0.0] * 16
                w_global = [0.0] * 16
                
                # Expand u
                for sub_i, coeff in a_coeffs.items():
                    if coeff == 0: continue
                    # u[j] is coeff for A'_sub_j
                    # A'_sub_j is formed by a_coeffs
                    # Wait, the logic is:
                    # A' is a 2x2 matrix where each entry is a linear combo of A sub-blocks.
                    # u is a vector of 4 coeffs for the 4 entries of A'.
                    # So for each entry j of A' (0..3), if u[j] != 0,
                    # we add u[j] * (linear combo of A sub-blocks for A'_j)
                    # But a_coeffs maps A'_j to A_sub_k?
                    # No, a_coeffs is the definition of A'.
                    # A'_0 = sum_{k} a_coeffs[k] A_k ?
                    # No, in Strassen, A' is formed by ADDING sub-blocks.
                    # So A'_0 (top-left of A') is A_11 + A_22.
                    # So a_coeffs should map the "slot" in A' to the A sub-blocks.
                    # My `steps` definition: `{0: 1, 3: 1}` means A'_0 = A_0 + A_3.
                    # This is correct for M1.
                    
                    for k, c in a_coeffs.items():
                        if c == 0: continue
                        # u[j] is the coeff for A'_j in the Strassen-2x2 factor.
                        # We are iterating over u[j] implicitly?
                        # No, u is the vector for the Strassen-2x2 factor.
                        # u[0] corresponds to A'_11.
                        # So if u[0] != 0, it contributes to A'_11.
                        # A'_11 is defined by a_coeffs[0]?
                        # No, a_coeffs is a dict {slot: coeff}.
                        # So A'_slot = sum_{k} a_coeffs[slot]? No.
                        # My dict is {k: coeff} where k is A_sub index.
                        # This defines A'_slot?
                        # For M1: A'_11 = A_11 + A_22.
                        # So a_coeffs should be {0: 1, 3: 1} for slot 0?
                        # But a_coeffs is the same for all slots?
                        # No, in Strassen, A' is a matrix.
                        # A'_11 = A_11 + A_22.
                        # A'_12 = 0? No, A' is not a matrix of sub-blocks.
                        # A' IS a 2x2 matrix.
                        # The entries of A' are linear combos of A's sub-blocks.
                        # For M1:
                        # A'_11 = A_11 + A_22
                        # A'_12 = 0
                        # A'_21 = 0
                        # A'_22 = 0
                        # So a_coeffs should be a list of 4 dicts?
                        # Or just 4 vectors?
                        
                        # Let's redefine steps.
                        # Each step defines 4 vectors for A' and 4 vectors for B'.
                        # vec_A[0] is coeff vector for A'_11.
                        # vec_A[1] for A'_12.
                        # etc.
                        
                        pass # Placeholder, will rewrite below

        # Rewrite steps definition
        # vec_A is list of 4 vectors of length 4 (coeffs for A_11..A_22)
        # vec_B is list of 4 vectors of length 4
        
        def make_vec(idx, val):
            v = [0.0]*4
            v[idx] = val
            return v
            
        steps = [
            # M1
            ([make_vec(0,1)+make_vec(3,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]],
             [make_vec(0,1)+make_vec(3,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]]),
            # M2
            ([make_vec(0,0), [0,0,0,0], make_vec(2,1)+make_vec(3,1), [0,0,0,0]],
             [make_vec(0,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]]),
            # M3
            ([make_vec(0,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]],
             [[0,0,0,0], make_vec(1,1), [0,0,0,0], make_vec(3,-1)]),
            # M4
            ([[0,0,0,0], [0,0,0,0], [0,0,0,0], make_vec(3,1)],
             [[0,0,0,0], [0,0,0,0], make_vec(2,1), make_vec(0,-1)]),
            # M5
            ([make_vec(0,1)+make_vec(1,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]],
             [[0,0,0,0], [0,0,0,0], [0,0,0,0], make_vec(3,1)]),
            # M6
            ([make_vec(0,0), [0,0,0,0], make_vec(2,1)-make_vec(0,1), [0,0,0,0]],
             [make_vec(0,1)+make_vec(1,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]]),
            # M7
            ([make_vec(0,0), make_vec(1,1), [0,0,0,0], make_vec(3,-1)],
             [[0,0,0,0], [0,0,0,0], make_vec(2,1)+make_vec(3,1), [0,0,0,0]])
        ]
        
        # Note: list subtraction `make_vec(2,1)-make_vec(0,1)` works element-wise?
        # No, lists don't support `-`.
        # Need helper.
        
        def add_vec(a, b):
            return [x+y for x,y in zip(a,b)]
        def sub_vec(a, b):
            return [x-y for x,y in zip(a,b)]
            
        # Re-do steps with add/sub
        steps = [
            ([add_vec(make_vec(0,1), make_vec(3,1)), [0,0,0,0], [0,0,0,0], [0,0,0,0]],
             [add_vec(make_vec(0,1), make_vec(3,1)), [0,0,0,0], [0,0,0,0], [0,0,0,0]]),
            ([make_vec(0,0), [0,0,0,0], add_vec(make_vec(2,1), make_vec(3,1)), [0,0,0,0]],
             [make_vec(0,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]]),
            ([make_vec(0,1), [0,0,0,0], [0,0,0,0], [0,0,0,0]],
             [[0,0,0,0], make_vec(1,1), [0,0,0,0], make_vec(3,-1)]),
            ([[0,0,0,0], [0,0,0,0], [0,0,0,0], make_vec(3,1)],
             [[0,0,0,0], [0,0,0,0], make_vec(2,1), make_vec(0,-1)]),
            ([add_vec(make_vec(0,1), make_vec(1,1)), [0,0,0,0], [0,0,0,0], [0,0,0,0]],
             [[0,0,0,0], [0,0,0,0], [0,0,0,0], make_vec(3,1)]),
            ([make_vec(0,0), [0,0,0,0], sub_vec(make_vec(2,1), make_vec(0,1)), [0,0,0,0]],
             [add_vec(make_vec(0,1), make_vec(1,1)), [0,0,0,0], [0,0,0,0], [0,0,0,0]]),
            ([make_vec(0,0), make_vec(1,1), [0,0,0,0], make_vec(3,-1)],
             [[0,0,0,0], [0,0,0,0], add_vec(make_vec(2,1), make_vec(3,1)), [0,0,0,0]])
        ]
        
        factors = []
        for vec_A, vec_B in steps:
            for u, v, w in s2:
                u_global = [0.0] * 16
                v_global = [0.0] * 16
                w_global = [0.0] * 16
                
                for j in range(4):
                    if u[j] != 0:
                        # u[j] is coeff for A'_j
                        # A'_j is linear combo of A_sub_k defined by vec_A[j]
                        for k in range(4):
                            if vec_A[j][k] != 0:
                                coeff = u[j] * vec_A[j][k]
                                for idx in sub_idx[k]:
                                    u_global[idx] += coeff
                                    
                    if v[j] != 0:
                        for k in range(4):
                            if vec_B[j][k] != 0:
                                coeff = v[j] * vec_B[j][k]
                                for idx in sub_idx[k]:
                                    v_global[idx] += coeff
                                    
                    if w[j] != 0:
                        # w[j] is coeff for C'_j
                        # C'_j is just C_sub_j (no linear combo for C in Strassen)
                        # So w[j] * 1 for C_sub_j
                        for idx in sub_idx[j]:
                            w_global[idx] += w[j]
                            
                factors.append((u_global, v_global, w_global))
                
        return factors

    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 4 and mj == 4 and pk == 4:
                    total_rank += 49
                else:
                    total_rank += ni * mj * pk
                    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    break
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 4 and mj == 4 and pk == 4:
                    factors = get_strassen_4x4x4_factors()
                    for (uf, vf, wf) in factors:
                        local_to_global_U = [offsets[i]*m + offsets[j] + idx for idx in range(16)]
                        local_to_global_V = [offsets[j]*p + offsets[k] + idx for idx in range(16)]
                        local_to_global_W = [offsets[k]*n + offsets[i] + idx for idx in range(16)]
                        
                        for idx in range(16):
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
                                
    assert col == total_rank
    return {"U": U, "V": V, "W": W}
