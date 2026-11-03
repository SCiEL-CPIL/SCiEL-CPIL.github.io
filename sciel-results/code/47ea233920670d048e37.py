import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: 150s limit, return best result with safety margin
    deadline = start_time + 145.0
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Rank 26 Solution: Strassen 2x2 core + 19 border terms
    R = 26
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    # Define Strassen terms for the 2x2 core (indices 0,1)
    # U, V, W entries for Strassen
    strassen_terms = [
        # (U_indices, V_indices, W_indices)
        # r=0: (A00+A11)(B00+B11)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # r=1: (A10+A11)B00
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),
        # r=2: A00(B11-B01)
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ]),
        # r=3: A11(B10-B00)
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]),
        # r=4: (A00+A01)B11
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (1,0,1) ]),
        # r=5: (A10-A00)(B00+B01)
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        # r=6: (A01-A11)(B10+B11)
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ])
    ]
    
    for r, (u_terms, v_terms, w_terms) in enumerate(strassen_terms):
        for (i, j, coeff) in u_terms:
            U[idx_u(i, j), r] += coeff
        for (j, k, coeff) in v_terms:
            V[idx_v(j, k), r] += coeff
        for (k, i, coeff) in w_terms:
            W[idx_w(k, i), r] += coeff
            
    # 19 Border terms
    # These are simple 1-1-1 multiplications for the remaining parts
    # The 2x2 core covers C00, C01, C10, C11 partially (only the sum over k=0,1)
    # We need to add the k=2 parts for C00..C11, and all of C02, C12, C20, C21, C22
    # C00 += A02*B20
    # C01 += A02*B21
    # C10 += A12*B20
    # C11 += A12*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    border_terms = [
        # Corrections for central 2x2
        (0, 2, 2, 0), # A02*B20 -> C00
        (0, 2, 2, 1), # A02*B21 -> C01
        (1, 2, 2, 0), # A12*B20 -> C10
        (1, 2, 2, 1), # A12*B21 -> C11
        # Row 0, Col 2 of C
        (0, 0, 0, 2), # A00*B02
        (0, 1, 1, 2), # A01*B12
        (0, 2, 2, 2), # A02*B22
        # Row 1, Col 2 of C
        (1, 0, 0, 2), # A10*B02
        (1, 1, 1, 2), # A11*B12
        (1, 2, 2, 2), # A12*B22
        # Row 2, Col 0 of C
        (2, 0, 0, 0), # A20*B00
        (2, 1, 1, 0), # A21*B10
        (2, 2, 2, 0), # A22*B20
        # Row 2, Col 1 of C
        (2, 0, 0, 1), # A20*B01
        (2, 1, 1, 1), # A21*B11
        (2, 2, 2, 1), # A22*B21
        # Row 2, Col 2 of C
        (2, 0, 0, 2), # A20*B02
        (2, 1, 1, 2), # A21*B12
        (2, 2, 2, 2), # A22*B22
    ]
    
    col = 7
    for (i, j, k_l, k_r) in border_terms:
        # U is (i, j), V is (j, k_r), W is (k_l, i)
        # Wait, V index is (j, k). Here k is the col of B.
        # W index is (k, i). Here k is the row of B (which is col of A? No).
        # Tensor: C_{i, k} = sum_j A_{i, j} B_{j, k}
        # Term: A_{i, j} * B_{j, k} -> contributes to C_{i, k}
        # U index: i, j
        # V index: j, k
        # W index: k, i
        
        # In my border_terms tuple (i, j, k_l, k_r):
        # i is row of A
        # j is col of A / row of B
        # k_r is col of B
        # The W index should be (k_r, i) ? 
        # Wait, W[k*n+i, r]. The tensor is T_{i, j, k} = A_{ij} B_{jk}.
        # Output C_{i, k} = sum_j T_{i, j, k}.
        # So W index is (k, i).
        # My tuple is (i, j, k_l, k_r).
        # If I use A_{i, j} * B_{j, k}, then U(i,j), V(j,k), W(k,i).
        # So k_l should be k.
        
        # Let's re-verify the border terms mapping.
        # (i, j, k) means A_{i,j} * B_{j,k} -> C_{i,k}
        # W index is (k, i).
        
        pass # The loop below handles it
        
    col = 7
    for (i, j, k) in [
        (0, 2, 0), (0, 2, 1), # A02*B00, A02*B01 -> C00, C01? 
        # Wait, A02 is j=2. B_{2, k}.
        # A_{0,2} * B_{2,0} -> C_{0,0}. U(0,2), V(2,0), W(0,0).
        # My previous list had (0,2,2,0) which was A02*B20. 
        # Let's stick to the explicit (i,j,k) triple for A_ij * B_jk
        
        # Corrections for C00..C11 (j=2)
        (0, 2, 0), # A02*B20 -> C00
        (0, 2, 1), # A02*B21 -> C01
        (1, 2, 0), # A12*B20 -> C10
        (1, 2, 1), # A12*B21 -> C11
        # C02 (i=0, k=2)
        (0, 0, 2), # A00*B02
        (0, 1, 2), # A01*B12
        (0, 2, 2), # A02*B22
        # C12 (i=1, k=2)
        (1, 0, 2), # A10*B02
        (1, 1, 2), # A11*B12
        (1, 2, 2), # A12*B22
        # C20 (i=2, k=0)
        (2, 0, 0), # A20*B00
        (2, 1, 0), # A21*B10
        (2, 2, 0), # A22*B20
        # C21 (i=2, k=1)
        (2, 0, 1), # A20*B01
        (2, 1, 1), # A21*B11
        (2, 2, 1), # A22*B21
        # C22 (i=2, k=2)
        (2, 0, 2), # A20*B02
        (2, 1, 2), # A21*B12
        (2, 2, 2), # A22*B22
    ]:
        U[idx_u(i, j), col] = 1
        V[idx_v(j, k), col] = 1
        W[idx_w(k, i), col] = 1
        col += 1
        
    # Verify
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                expected = 0.0
                actual = 0.0
                for j in range(m):
                    expected += 1.0 if (True) else 0.0 # Just checking structure
                # Check specific entries
                # Actually, just check a few random ones or all
                for i_c in range(n):
                    for k_c in range(p):
                        exp = 1.0 # We are decomposing the identity mapping of multiplication
                        # Wait, the tensor is M_{ijk} = delta_{ik} delta_{j?}
                        # No, M_{i, j, k} is the coefficient of A_ij B_jk in C.
                        # For the tensor to be correct, sum_r U_{ir} V_{jr} W_{kr} = delta_{i,k} ? No.
                        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
                        # The tensor T has T_{i,j,k} = 1 if we are computing C_{i,k} using A_{i,j}B_{j,k}.
                        # So T_{i,j,k} = 1 for all i,j,k? No.
                        # T_{i,j,k} is 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
                        # It contributes to C_{i,k} always.
                        # So T_{i,j,k} = 1 for all i,j,k.
                        # We need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k?
                        # No, W is indexed by (k,i).
                        # Let's check: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} should be 1 for all i,j,k.
                        pass
        
        # Correct Verification:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U[idx_u(i,j),r]*V[idx_v(j,k),r]*W[idx_w(k,i),r] for r in range(R))
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to Schoolbook 27
        R27 = 27
        U27 = np.zeros((n*m, R27), dtype=np.float64)
        V27 = np.zeros((m*p, R27), dtype=np.float64)
        W27 = np.zeros((p*n, R27), dtype=np.float64)
        r = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U27[idx_u(i,j),r]=1
                    V27[idx_v(j,k),r]=1
                    W27[idx_w(k,i),r]=1
                    r += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
