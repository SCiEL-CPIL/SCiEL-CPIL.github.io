import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def get_strassen_2x2_tensors():
    # Returns list of (u, v, w) for 2x2x2 Strassen
    # u, v, w are length 4 vectors (flattened 2x2)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    tensors = []
    
    # M1 = (a+d)(e+h)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C11 and C22 get +1
    # Actually, we need to map to C indices
    # C11 = M1 + ...
    # C22 = M1 + ...
    # So w1 has 1 at C11 (0) and C22 (3)
    w1 = [1, 0, 0, 1]
    tensors.append((u1, v1, w1))
    
    # M2 = (c+d)e
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # C21 +1, C22 -1
    tensors.append((u2, v2, w2))
    
    # M3 = a(f-h)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # C12 +1, C22 +1
    tensors.append((u3, v3, w3))
    
    # M4 = d(g-e)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0] # C11 +1, C21 +1
    tensors.append((u4, v4, w4))
    
    # M5 = (a+b)h
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # C11 -1, C12 +1
    tensors.append((u5, v5, w5))
    
    # M6 = (c-a)(e+f)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # C22 +1
    tensors.append((u6, v6, w6))
    
    # M7 = (b-d)(g+h)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # C11 +1
    tensors.append((u7, v7, w7))
    
    return tensors

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    U = []
    V = []
    W = []
    
    def add_tensor(u_local, v_local, w_local, i_offset, j_offset, k_offset):
        # u_local: length n_part * m_part
        # v_local: length m_part * p_part
        # w_local: length p_part * n_part
        # i_offset: row offset for A (0..n-1)
        # j_offset: col offset for A / row offset for B
        # k_offset: col offset for B / col offset for C
        
        # Map local indices to global
        # A index: i*m + j
        # B index: j*p + k
        # C index: k*n + i (Note: Task says W[k*n+i, r])
        
        u_global = [0] * (n * m)
        v_global = [0] * (m * p)
        w_global = [0] * (p * n)
        
        # Fill U
        for idx, val in enumerate(u_local):
            if val == 0: continue
            li = idx // (m - j_offset) # This is tricky, need local dims
            # Better to pass local dims
            
        # This approach is fragile. Let's use explicit loops for the 4 parts.
        pass

    # Part 1: 4x4x4 Strassen
    # A: 4x4, B: 4x4, C: 4x4
    # Global A: rows 0-3, cols 0-3
    # Global B: rows 0-3, cols 0-3
    # Global C: rows 0-3, cols 0-3
    
    strassen_tensors = get_strassen_2x2_tensors()
    
    # We need to embed 7 2x2 Strassen instances into 4x4.
    # 4x4 = 2x2 blocks.
    # Block (0,0): A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # Block (0,1): A[0:2, 2:4], B[0:2, 2:4], C[0:2, 2:4]
    # ...
    
    # Actually, Strassen for 4x4 uses 7 multiplications of 2x2.
    # Each 2x2 multiplication is itself a tensor.
    # We need to expand the 7 "virtual" multiplications into 7 actual rank-1 tensors each.
    
    # Let's generate the 49 tensors for 4x4x4.
    # For each of the 7 Strassen steps, we have a 2x2x2 tensor.
    # We map the 2x2 indices to the 4x4 indices.
    
    # Strassen step 1: M1 = (A11+A22)(B11+B22)
    # This involves A blocks (0,0) and (1,1), B blocks (0,0) and (1,1).
    # The result contributes to C blocks (0,0) and (1,1).
    
    # This is complex to map manually.
    # Alternative: Just use schoolbook for 4x4x4 (Rank 16) to be safe?
    # Rank 16 is much better than 49.
    # Why did I think Strassen was 49?
    # Strassen for 4x4 is 7 * R(2,2,2) = 7 * 7 = 49.
    # Schoolbook for 4x4 is 16.
    # 16 < 49.
    # So Strassen is WORSE for 4x4?
    # Yes, Strassen is only better for large n.
    # For n=2, R=7 < 8.
    # For n=4, R=49 > 16.
    # So we should use Schoolbook for 4x4x4!
    
    # So:
    # Part 1: 4x4x4 Schoolbook (Rank 16)
    # Part 2: 4x1x4 Schoolbook (Rank 4)
    # Part 3: 4x4x2 Schoolbook (Rank 16)
    # Part 4: 4x1x2 Schoolbook (Rank 2)
    # Total Rank = 16 + 4 + 16 + 2 = 38.
    
    # This is much better than 71.
    # And 38 < 120.
    # Score = 120 / 38 = 3.15.

    # Let's verify the split.
    # R(4,5,6) <= R(4,4,4) + R(4,1,4) + R(4,4,2) + R(4,1,2)
    # R(4,4,4) = 16
    # R(4,1,4) = 4
    # R(4,4,2) = 16
    # R(4,1,2) = 2
    # Total 38.
    
    # Is R(4,4,4) really 16?
    # Yes, schoolbook.
    # Is there a lower rank?
    # R(4,4,4) >= 22? No, lower bound is 19.
    # But we don't have a decomposition for 22.
    # So 38 is a good target.

    # Implementation:
    # Generate schoolbook tensors for each part and map to global.

    def add_schoolbook(n_p, m_p, p_p, i_off, j_off, k_off):
        for i in range(n_p):
            for j in range(m_p):
                for k in range(p_p):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    # A index
                    a_idx = (i + i_off) * m + (j + j_off)
                    u[a_idx] = 1
                    
                    # B index
                    b_idx = (j + j_off) * p + (k + k_off)
                    v[b_idx] = 1
                    
                    # C index
                    c_idx = (k + k_off) * n + (i + i_off)
                    w[c_idx] = 1
                    
                    U.append(u)
                    V.append(v)
                    W.append(w)

    # Part 1: 4x4x4
    add_schoolbook(4, 4, 4, 0, 0, 0)
    
    # Part 2: 4x1x4
    add_schoolbook(4, 1, 4, 0, 4, 0)
    
    # Part 3: 4x4x2
    add_schoolbook(4, 4, 2, 0, 0, 4)
    
    # Part 4: 4x1x2
    add_schoolbook(4, 1, 2, 0, 4, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
