import time

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is a tuple (U_dict, V_dict, W_dict) where:
    U_dict: {(i, j): coeff}
    V_dict: {(j, k): coeff}
    W_dict: {(k, i): coeff}
    
    The decomposition satisfies:
    sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{i,k} * (product of A[i,j] and B[j,k] structure)
    Specifically for standard matrix multiplication tensor M_{i,j,k}:
    M_{i,j,k} = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    """
    terms = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00, A11 -> 1
    # V: B00, B11 -> 1
    # W: C00 gets +1, C10 gets +1? 
    # Let's re-derive carefully.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4
    # C10 = M1 + M2
    # C11 = M3 - M2 + M5 + M7
    
    # For each M_c, we define U_c, V_c, W_c such that:
    # sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 if (i,j,k) is a valid matmul index, else 0.
    # Actually, the tensor T_{i,j,k} is 1 for all i,j,k in {0,1}^3.
    # So we need sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 for all i,j,k.
    
    # Let's verify the standard Strassen decomposition in this form.
    # Term 1 (M1):
    # U1: {(0,0):1, (1,1):1}
    # V1: {(0,0):1, (1,1):1}
    # W1: {(0,0):1, (0,1):1}  -- Wait, W is indexed by (k,i).
    # C00 corresponds to i=0, k=0. W1[0,0] should be 1.
    # C10 corresponds to i=1, k=0. W1[0,1] should be 1.
    # Let's check contribution to T_{i,j,k}:
    # If i=0, j=0, k=0: U1[0,0]=1, V1[0,0]=1, W1[0,0]=1 -> 1. Correct.
    # If i=1, j=1, k=0: U1[1,1]=1, V1[1,0]=0 -> 0.
    # If i=0, j=1, k=0: U1[0,1]=0 -> 0.
    # If i=1, j=0, k=0: U1[1,0]=0 -> 0.
    # If i=0, j=0, k=1: V1[0,1]=0 -> 0.
    # If i=1, j=1, k=1: U1[1,1]=1, V1[1,1]=1, W1[1,1]=0 -> 0.
    # Wait, M1 should contribute to C00 and C10.
    # C00 is i=0, k=0. C10 is i=1, k=0.
    # So W1 should have 1 at (k=0, i=0) and (k=0, i=1).
    # W1: {(0,0):1, (0,1):1}
    
    terms.append((
        {(0,0):1, (1,1):1},  # U
        {(0,0):1, (1,1):1},  # V
        {(0,0):1, (0,1):1}   # W
    ))
    
    # Term 2 (M2): (A10+A11)B00
    # U2: {(1,0):1, (1,1):1}
    # V2: {(0,0):1}
    # W2: {(0,1):1}  # Contributes to C10 (i=1, k=0)
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1}
    ))
    
    # Term 3 (M3): A00(B01-B11)
    # U3: {(0,0):1}
    # V3: {(0,1):1, (1,1):-1}
    # W3: {(1,0):1, (1,1):-1} # Contributes to C01 (i=0, k=1) and C11 (i=1, k=1)
    # Check C01: i=0, k=1. W3[1,0]=1.
    # Check C11: i=1, k=1. W3[1,1]=-1.
    terms.append((
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):-1}
    ))
    
    # Term 4 (M4): A11(B10-B00)
    # U4: {(1,1):1}
    # V4: {(1,0):1, (0,0):-1}
    # W4: {(0,0):-1, (1,0):1} # Contributes -1 to C00, +1 to C01
    terms.append((
        {(1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,0):-1, (1,0):1}
    ))
    
    # Term 5 (M5): (A00+A01)B11
    # U5: {(0,0):1, (0,1):1}
    # V5: {(1,1):1}
    # W5: {(1,1):1} # Contributes to C11
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(1,1):1}
    ))
    
    # Term 6 (M6): (A10-A00)(B00+B01)
    # U6: {(1,0):1, (0,0):-1}
    # V6: {(0,0):1, (0,1):1}
    # W6: {(0,0):1} # Contributes to C00
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(0,0):1}
    ))
    
    # Term 7 (M7): (A01-A11)(B10+B11)
    # U7: {(0,1):1, (1,1):-1}
    # V7: {(1,0):1, (1,1):1}
    # W7: {(0,0):1, (1,1):1} # Contributes to C00 and C11
    terms.append((
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    ))
    
    return terms

def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 120

    # Fallback: Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    try:
        strassen_terms = _strassen_2x2x2_terms()
        
        all_U = []
        all_V = []
        all_W = []
        
        # Decompose into 3 blocks of size 3x3 * 3x2
        # For each block b (0, 1, 2), columns k are 2*b and 2*b+1
        for b in range(3):
            k_offset = 2 * b
            
            # Part 1: i in {0,1}, j in {0,1}, k in {0,1} (local indices)
            # This is a 2x2x2 multiplication, rank 7
            for u_dict, v_dict, w_dict in strassen_terms:
                U_term = [0] * (n * m)
                V_term = [0] * (m * p)
                W_term = [0] * (p * n)
                
                for (i, j), coeff in u_dict.items():
                    U_term[i * m + j] = coeff
                for (j, k), coeff in v_dict.items():
                    V_term[j * p + (k_offset + k)] = coeff
                for (k, i), coeff in w_dict.items():
                    W_term[(k_offset + k) * n + i] = coeff
                    
                all_U.append(U_term)
                all_V.append(V_term)
                all_W.append(W_term)
                
            # Part 2: i=2, j in {0,1}, k in {0,1}
            # C[2, k] = A[2,0]B[0,k] + A[2,1]B[1,k]
            # This is rank 2 (one for each j)
            for j in range(2):
                for k in range(2):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[2 * m + j] = 1
                    V_term[j * p + (k_offset + k)] = 1
                    W_term[(k_offset + k) * n + 2] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
                    
            # Part 3: i in {0,1,2}, j=2, k in {0,1}
            # C[i, k] = A[i,2]B[2,k]
            # This is rank 2 (one for each k)
            for k in range(2):
                for i in range(3):
                    U_term = [0] * (n * m)
                    V_term = [0] * (m * p)
                    W_term = [0] * (p * n)
                    
                    U_term[i * m + 2] = 1
                    V_term[2 * p + (k_offset + k)] = 1
                    W_term[(k_offset + k) * n + i] = 1
                    
                    all_U.append(U_term)
                    all_V.append(V_term)
                    all_W.append(W_term)
        
        rank = len(all_U)
        
        # Verify correctness
        def verify(U, V, W, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(len(U[0])):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True

        if verify(all_U, all_V, all_W, n, m, p):
            if rank < rank_sb:
                return {"U": all_U, "V": all_V, "W": all_W}
                
    except Exception:
        pass

    return {"U": U_sb, "V": V_sb, "W": W_sb}
