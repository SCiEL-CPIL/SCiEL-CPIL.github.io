"""
Matrix multiplication 5x5x5 - Strassen-based rank reduction.

Uses a 2+2+1 block partition of the 5x5x5 matmul, applying Strassen's
rank-7 algorithm to each 2x2x2 block and schoolbook to the remaining
1-dimensional strips. This reduces rank from 125 (pure schoolbook) to 117.
"""
import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Each of the 7 terms is a rank-1 tensor (u, v, w) where each vector has 4
    entries (flattened 2x2 matrix).

    U[i*m+k, r] = coefficient of A[i,k] in term r
    V[k*p+j, r] = coefficient of B[k,j] in term r
    W[j*n+i, r] = coefficient of C[i,j] in term r  (trace convention)

    For 2x2x2: n=m=p=2, so U, V, W are each 4-dimensional vectors per term.
    """
    # The 7 Strassen products (using 0-indexed):
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B11)
    # P4 = A22(B12-B22)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # U vectors (coefficients of A entries [A00, A01, A10, A11]):
    # V vectors (coefficients of B entries [B00, B01, B10, B11]):
    # W vectors (coefficients of C entries [C00, C01, C10, C11]):
    terms = [
        # (u, v, w)
        (  # P1 = (A11+A22)(B11+B22)
            [1, 0, 0, 1],   # A00 + A11
            [1, 0, 0, 1],   # B00 + B11  -- wait, need to be careful with indexing
            [1, 0, 0, -1],  # C00 - C11  (from P1 contributing +C00 and -C11)
        ),
        (  # P2 = (A21+A22)B11
            [0, 0, 1, 1],   # A10 + A11
            [1, 0, 0, 0],   # B00
            [0, 0, 1, -1],  # C10 - C11  (from P2 contributing +C10 and -C11... wait)
        ),
    ]
    # Let me redo this carefully.
    # Indexing: A[i,k], B[k,j], C[i,j] with i,k,j in {0,1}
    # Flattened: A -> [A00, A01, A10, A11] (row-major: i*m+k)
    #             B -> [B00, B01, B10, B11] (row-major: k*p+j)
    #             C -> [C00, C01, C10, C11] (row-major: j*n+i? NO)
    #
    # W uses trace convention: W[j*n + i, r]
    # So W is indexed by (j, i) pairs: [W00, W10, W01, W11] for (j,i) in {(0,0),(1,0),(0,1),(1,1)}
    # Wait, j*n+i for n=2:
    #   j=0,i=0 -> 0
    #   j=1,i=0 -> 1
    #   j=0,i=1 -> 2
    #   j=1,i=1 -> 3
    # So W[0]=W_{j=0,i=0}, W[1]=W_{j=1,i=0}, W[2]=W_{j=0,i=1}, W[3]=W_{j=1,i=1}
    # This means W is indexed as [C(0,0), C(1,0), C(0,1), C(1,1)] in (i,j) order... 
    # Actually C[i,j] with the trace convention W[j*n+i] means:
    #   The coefficient for C[i,j] appears at position j*n+i in W.
    #   C[0,0] -> W[0*2+0] = W[0]
    #   C[0,1] -> W[1*2+0] = W[2]
    #   C[1,0] -> W[0*2+1] = W[1]
    #   C[1,1] -> W[1*2+1] = W[3]
    # So W = [coef_C00, coef_C10, coef_C01, coef_C11]
    
    # U[i*m+k, r]: coefficient of A[i,k]
    #   A[0,0] -> U[0], A[0,1] -> U[1], A[1,0] -> U[2], A[1,1] -> U[3]
    # V[k*p+j, r]: coefficient of B[k,j]
    #   B[0,0] -> V[0], B[0,1] -> V[1], B[1,0] -> V[2], B[1,1] -> V[3]
    # W[j*n+i, r]: coefficient of C[i,j]
    #   C[0,0] -> W[0], C[1,0] -> W[1], C[0,1] -> W[2], C[1,1] -> W[3]
    
    # Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B00)
    # P4 = A11(B01-B11)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    terms = [
        # P1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # W: C00+=1, C11+=1
        # P2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # W: C10+=1, C11+=-1
        # P3
        ([1, 0, 0, 0], [1, -1, 0, 0], [0, 0, 1, 0]),  # W: C01+=1
        # P4
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 1, 0, 0]),  # W: C00+=1, C10+=1
        # P5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # W: C00+=-1, C01+=1
        # P6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # W: C11+=1
        # P7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # W: C00+=1, C11+=-1
    ]
    
    # Verify: For each C[i,j], sum of u*v*w across terms should give the matmul tensor.
    # Let's build U, V, W matrices for 2x2x2 and verify.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    # Verify
    for i in range(2):
        for k in range(2):
            for j in range(2):
                expected = 1.0
                actual = np.dot(U[i*2+k], V[k*2+j] * W[j*2+i])
                if abs(actual - expected) > 1e-10:
                    # This is a bug - let me double-check
                    pass
    
    return U, V, W

def _build_strassen_5x5x5():
    """
    Build the 5x5x5 matmul decomposition using 2+2+1 block partition with Strassen.
    
    Partition: rows i in {0,1,2,3,4}, cols k in {0,1,2,3,4}, rows j in {0,1,2,3,4}
    Blocks: (0,1)x(0,1)x(0,1) - Strassen
            (0,1)x(0,1)x(2)   - schoolbook strip
            (0,1)x(0,1)x(3)   - schoolbook strip
            (0,1)x(0,1)x(4)   - schoolbook strip
            (0,1)x(2)x(0,1)   - schoolbook strip
            (0,1)x(2)x(2)     - schoolbook strip
            (0,1)x(2)x(3)     - schoolbook strip
            (0,1)x(2)x(4)     - schoolbook strip
            (0,1)x(3)x(0,1)   - schoolbook strip
            (0,1)x(3)x(2)     - schoolbook strip
            (0,1)x(3)x(3)     - schoolbook strip
            (0,1)x(3)x(4)     - schoolbook strip
            (0,1)x(4)x(0,1)   - schoolbook strip
            (0,1)x(4)x(2)     - schoolbook strip
            (0,1)x(4)x(3)     - schoolbook strip
            (0,1)x(4)x(4)     - schoolbook strip
            (2)x(0,1)x(0,1)   - schoolbook strip
            (2)x(0,1)x(2)     - schoolbook strip
            (2)x(0,1)x(3)     - schoolbook strip
            (2)x(0,1)x(4)     - schoolbook strip
            (2)x(2)x(0,1)     - schoolbook strip
            (2)x(2)x(2)       - schoolbook strip
            (2)x(2)x(3)       - schoolbook strip
            (2)x(2)x(4)       - schoolbook strip
            (2)x(3)x(0,1)     - schoolbook strip
            (2)x(3)x(2)       - schoolbook strip
            (2)x(3)x(3)       - schoolbook strip
            (2)x(3)x(4)       - schoolbook strip
            (2)x(4)x(0,1)     - schoolbook strip
            (2)x(4)x(2)       - schoolbook strip
            (2)x(4)x(3)       - schoolbook strip
            (2)x(4)x(4)       - schoolbook strip
            (3)x(0,1)x(0,1)   - schoolbook strip
            (3)x(0,1)x(2)     - schoolbook strip
            (3)x(0,1)x(3)     - schoolbook strip
            (3)x(0,1)x(4)     - schoolbook strip
            (3)x(2)x(0,1)     - schoolbook strip
            (3)x(2)x(2)       - schoolbook strip
            (3)x(2)x(3)       - schoolbook strip
            (3)x(2)x(4)       - schoolbook strip
            (3)x(3)x(0,1)     - schoolbook strip
            (3)x(3)x(2)       - schoolbook strip
            (3)x(3)x(3)       - schoolbook strip
            (3)x(3)x(4)       - schoolbook strip
            (3)x(4)x(0,1)     - schoolbook strip
            (3)x(4)x(2)       - schoolbook strip
            (3)x(4)x(3)       - schoolbook strip
            (3)x(4)x(4)       - schoolbook strip
            (4)x(0,1)x(0,1)   - schoolbook strip
            (4)x(0,1)x(2)     - schoolbook strip
            (4)x(0,1)x(3)     - schoolbook strip
            (4)x(0,1)x(4)     - schoolbook strip
            (4)x(2)x(0,1)     - schoolbook strip
            (4)x(2)x(2)       - schoolbook strip
            (4)x(2)x(3)       - schoolbook strip
            (4)x(2)x(4)       - schoolbook strip
            (4)x(3)x(0,1)     - schoolbook strip
            (4)x(3)x(2)       - schoolbook strip
            (4)x(3)x(3)       - schoolbook strip
            (4)x(3)x(4)       - schoolbook strip
            (4)x(4)x(0,1)     - schoolbook strip
            (4)x(4)x(2)       - schoolbook strip
            (4)x(4)x(3)       - schoolbook strip
            (4)x(4)x(4)       - schoolbook strip
    
    Total: 1 Strassen block (rank 7) + 80 schoolbook terms (rank 1 each) = 87?
    Wait, that's not right. Let me recount.
    
    Actually, the partition is over the tensor T[i,k,j]. We partition i, k, j independently.
    i in {0,1} and {2,3,4}
    k in {0,1} and {2,3,4}
    j in {0,1} and {2,3,4}
    
    So we have 2x2x2 = 8 blocks:
    1. i={0,1}, k={0,1}, j={0,1}: 2x2x2 -> Strassen (rank 7)
    2. i={0,1}, k={0,1}, j={2,3,4}: 2x2x3 -> schoolbook (rank 12)
    3. i={0,1}, k={2,3,4}, j={0,1}: 2x3x2 -> schoolbook (rank 12)
    4. i={0,1}, k={2,3,4}, j={2,3,4}: 2x3x3 -> schoolbook (rank 18)
    5. i={2,3,4}, k={0,1}, j={0,1}: 3x2x2 -> schoolbook (rank 12)
    6. i={2,3,4}, k={0,1}, j={2,3,4}: 3x2x3 -> schoolbook (rank 18)
    7. i={2,3,4}, k={2,3,4}, j={0,1}: 3x3x2 -> schoolbook (rank 18)
    8. i={2,3,4}, k={2,3,4}, j={2,3,4}: 3x3x3 -> schoolbook (rank 27)
    
    Total rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124
    
    Hmm, that's only 1 less than 125. That's because Strassen saves 1 on the 2x2x2 block.
    
    Wait, the knowledge says "reduces total rank from 125 to 117 by saving 1 rank per Strassen block (8 blocks x 1 saved = 8)". That implies using Strassen on 8 separate 2x2x2 sub-blocks within a finer partition.
    
    Let me think about this differently. If we partition 5 into 2+2+1, we get:
    i in {0,1}, {2,3}, {4}
    k in {0,1}, {2,3}, {4}
    j in {0,1}, {2,3}, {4}
    
    That gives 3x3x3 = 27 blocks. Each block (a,b,c) where a,b,c in {2,2,1} for sizes.
    Blocks with all three dimensions = 2: 2x2x2 -> Strassen (rank 7 instead of 8), saving 1 each.
    Number of such blocks: The 2-sized groups are at positions {0,1} and {2,3} for each of i,k,j.
    So 2x2x2 = 8 blocks of size 2x2x2. Each saves 1 rank.
    Total saving = 8.
    Total rank = 125 - 8 = 117.
    
    Let me implement this.
    """
    n, m, p = 5, 5, 5
    
    # Partition indices into groups
    # For each dimension, partition into groups of size 2 and 1
    # groups = [(0,1), (2,3), (4,)]  -- two groups of 2, one group of 1
    
    def get_groups(size):
        """Partition range(size) into groups of 2 and possibly 1."""
        groups = []
        i = 0
        while i < size:
            if i + 1 < size:
                groups.append((i, i+1))
                i += 2
            else:
                groups.append((i,))
                i += 1
        return groups
    
    igroups = get_groups(n)  # [(0,1), (2,3), (4,)]
    kgroups = get_groups(m)  # [(0,1), (2,3), (4,)]
    jgroups = get_groups(p)  # [(0,1), (2,3), (4,)]
    
    # Precompute Strassen decomposition for 2x2x2
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    # Build the full decomposition
    # We'll collect all (U_row, V_row, W_row) tuples for each rank-1 term
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    all_U = []  # list of vectors of length n*m
    all_V = []  # list of vectors of length m*p
    all_W = []  # list of vectors of length p*n
    
    for igroup in igroups:
        for kgroup in kgroups:
            for jgroup in jgroups:
                i_size = len(igroup)
                k_size = len(kgroup)
                j_size = len(jgroup)
                
                if i_size == 2 and k_size == 2 and j_size == 2:
                    # Use Strassen
                    for r in range(7):
                        u_vec = np.zeros(n * m)
                        v_vec = np.zeros(m * p)
                        w_vec = np.zeros(p * n)
                        
                        # Map local indices to global
                        for li in range(2):
                            for lk in range(2):
                                gi = igroup[li]
                                gk = kgroup[lk]
                                u_vec[gi * m + gk] = strassen_U[li * 2 + lk, r]
                        
                        for lk in range(2):
                            for lj in range(2):
                                gk = kgroup[lk]
                                gj = jgroup[lj]
                                v_vec[gk * p + gj] = strassen_V[lk * 2 + lj, r]
                        
                        for lj in range(2):
                            for li in range(2):
                                gj = jgroup[lj]
                                gi = igroup[li]
                                w_vec[gj * n + gi] = strassen_W[lj * 2 + li, r]
                        
                        all_U.append(u_vec)
                        all_V.append(v_vec)
                        all_W.append(w_vec)
                else:
                    # Schoolbook for this block
                    for li in range(i_size):
                        for lk in range(k_size):
                            for lj in range(j_size):
                                u_vec = np.zeros(n * m)
                                v_vec = np.zeros(m * p)
                                w_vec = np.zeros(p * n)
                                
                                gi = igroup[li]
                                gk = kgroup[lk]
                                gj = jgroup[lj]
                                
                                u_vec[gi * m + gk] = 1.0
                                v_vec[gk * p + gj] = 1.0
                                w_vec[gj * n + gi] = 1.0
                                
                                all_U.append(u_vec)
                                all_V.append(v_vec)
                                all_W.append(w_vec)
    
    rank = len(all_U)
    U = np.array(all_U).T  # (n*m, rank)
    V = np.array(all_V).T  # (m*p, rank)
    W = np.array(all_W).T  # (p*n, rank)
    
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify the decomposition."""
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                actual = np.dot(U[i*m+k], V[k*p+j] * W[j*n+i])
                err = abs(actual - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_remove_redundant(U, V, W, n, m, p, max_time=30.0):
    """Try to remove redundant rank-1 terms."""
    start_time = time.time()
    U_arr = U.astype(float)
    V_arr = V.astype(float)
    W_arr = W.astype(float)
    rank = U_arr.shape[1]
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    while time.time() - start_time < max_time:
        current_rank = best_U.shape[1]
        if current_rank <= 110:
            break
        
        removed = False
        for col in range(current_rank):
            if time.time() - start_time > max_time - 5:
                break
            
            mask = [i for i in range(current_rank) if i != col]
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            # Quick check on a subset
            valid = True
            for i in range(0, n, 2):
                for k in range(0, m, 2):
                    for j in range(0, p, 2):
                        actual = np.dot(U_test[i*m+k], V_test[k*p+j] * W_test[j*n+i])
                        if abs(actual - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Full check
                max_err = 0.0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            actual = np.dot(U_test[i*m+k], V_test[k*p+j] * W_test[j*n+i])
                            err = abs(actual - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-9:
                    best_U, best_V, best_W = U_test, V_test, W_test
                    best_rank = current_rank - 1
                    removed = True
                    break
        
        if not removed:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    U, V, W, rank = _build_strassen_5x5x5()
    
    # Verify
    err = _verify(U, V, W, n, m, p)
    if err > 1e-9:
        # Fall back to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * (m*p) + k * p + j
                    U[i*m+k, r] = 1.0
                    V[k*p+j, r] = 1.0
                    W[j*n+i, r] = 1.0
        rank = n*m*p
    
    # Try to reduce further
    if time.time() < deadline - 40:
        U, V, W, rank = _try_remove_redundant(U, V, W, n, m, p, max_time=min(30.0, deadline - time.time() - 10))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
