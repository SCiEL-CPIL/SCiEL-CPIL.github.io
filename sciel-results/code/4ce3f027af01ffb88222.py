"""Find rank-minimal tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    rank = U.shape[1]
    i_idx = np.repeat(np.arange(n), m*p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n*m)
    
    U_idx = i_idx * m + j_idx
    V_idx = j_idx * p + k_idx
    W_idx = k_idx * n + i_idx
    
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    W_sel = W[W_idx, :]
    
    result = np.sum(U_sel * V_sel * W_sel, axis=1)
    return np.all(np.abs(result - 1.0) < tol)

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 multiplications for 2x2 matrix mult
    # C = A*B where A, B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # U rows indexed by [i*m+j] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    # V rows indexed by [j*p+k] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    # W rows indexed by [k*n+i] = [0,1,2,3] for (0,0),(0,1),(1,0),(1,1)
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: contributes to C[0,0]=ae+bg and C[1,1]=cf+dh
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: contributes to C[1,0]=ce+dg
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: contributes to C[0,1]=af+bh
    # M4 = (b-c)h: U: b=1,c=-1; V: h=1; W: contributes to C[0,1]=af+bh
    # M5 = (a+b)g: U: a=1,b=1; V: g=1; W: contributes to C[1,1]=cf+dh
    # M6 = (c-d)(f+h): U: c=1,d=-1; V: f=1,h=1; W: contributes to C[1,0]=ce+dg
    # M7 = (a-c)(e+g): U: a=1,c=-1; V: e=1,g=1; W: contributes to C[0,0]=ae+bg
    
    # Column 0: M1
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Column 1: M2
    U[2, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    
    # Column 2: M3
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f, h
    W[1, 2] = 1  # C[0,1]
    
    # Column 3: M4
    U[1, 3] = 1; U[2, 3] = -1  # b, c
    V[3, 3] = 1  # h
    W[1, 3] = 1  # C[0,1]
    
    # Column 4: M5
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[2, 4] = 1  # g
    W[3, 4] = 1  # C[1,1]
    
    # Column 5: M6
    U[2, 5] = 1; U[3, 5] = -1  # c, d
    V[1, 5] = 1; V[3, 5] = 1  # f, h
    W[2, 5] = 1  # C[1,0]
    
    # Column 6: M7
    U[0, 6] = 1; U[2, 6] = -1  # a, c
    V[0, 6] = 1; V[2, 6] = 1  # e, g
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions by horizontal concatenation."""
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((decomps[0][0].shape[0], total_rank))
    V = np.zeros((decomps[0][1].shape[0], total_rank))
    W = np.zeros((decomps[0][2].shape[0], total_rank))
    col = 0
    for d in decomps:
        r = d[0].shape[1]
        U[:, col:col+r] = d[0]
        V[:, col:col+r] = d[1]
        W[:, col:col+r] = d[2]
        col += r
    return U, V, W

def _build_3x2x2():
    """Build 3x2 * 2x2 decomposition using Strassen(2x2x2) + schoolbook(1x2x2)."""
    n, m, p = 3, 2, 2
    # A is 3x2, B is 2x2, C is 3x2
    # Split A into A1 (2x2) and A2 (1x2)
    # C = A1*B + A2*B
    # Use Strassen for A1*B (rank 7), schoolbook for A2*B (rank 4)
    
    U1, V1, W1 = _build_strassen_2x2x2()
    # U1 is (4,7), V1 is (4,7), W1 is (4,7)
    # Need to extend W1 to shape (p*n, rank) = (6, 7)
    # W1 corresponds to C[0,0],C[0,1],C[1,0],C[1,1] for the 2x2 result
    # But C is 3x2, so C has rows 0,1,2 and cols 0,1
    # W index: k*n+i where k in {0,1}, i in {0,1,2}
    # For the 2x2 part (rows 0,1 of C): W indices are (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W1 has shape (4,7) with indices 0,1,2,3 which match
    # But we need W to have shape (6,7), with rows 4,5 (for k=1,i=2) being zero
    
    U1_ext = np.zeros((n*m, 7))  # (6, 7)
    U1_ext[:4, :] = U1
    # U1_ext[4, :] and U1_ext[5, :] remain 0
    
    V1_ext = np.zeros((m*p, 7))  # (4, 7) - same as V1
    V1_ext = V1.copy()
    
    W1_ext = np.zeros((p*n, 7))  # (6, 7)
    W1_ext[:4, :] = W1
    # W1_ext[4, :] and W1_ext[5, :] remain 0
    
    # Schoolbook for A2 (1x2) * B (2x2): rank = 1*2*2 = 4
    # A2 is row 2 of A (indices 4,5 in U), B is 2x2, C2 is row 2 of C
    # U indices for A2: i=2, j=0,1 -> 2*2+0=4, 2*2+1=5
    # V indices: j*p+k
    # W indices: k*n+i = k*3+2
    U2, V2, W2 = _schoolbook(1, 2, 2)
    # U2 is (2,4), V2 is (4,4), W2 is (2,4)
    # U2 row 0 -> A2[0,0] -> U index 4
    # U2 row 1 -> A2[0,1] -> U index 5
    # W2 row 0 -> C[2,0] -> W index 0*3+2=2
    # W2 row 1 -> C[2,1] -> W index 1*3+2=5
    
    U2_ext = np.zeros((n*m, 4))
    U2_ext[4:6, :] = U2
    
    V2_ext = np.zeros((m*p, 4))
    V2_ext = V2.copy()
    
    W2_ext = np.zeros((p*n, 4))
    W2_ext[2, 0] = W2[0, 0]
    W2_ext[5, 1] = W2[1, 1]
    
    return _combine_decompositions([(U1_ext, V1_ext, W1_ext), (U2_ext, V2_ext, W2_ext)])

def _build_poly_interp_3x5x7():
    """
    Polynomial interpolation embedding for 3x5 * 5x7.
    
    Key idea: C[i,k] = sum_j A[i,j]*B[j,k] is the x^4 coefficient of
    P_i(x) * Q_k(x) where P_i(x) = sum_j A[i,j] x^j and Q_k(x) = sum_j B[j,k] x^(4-j).
    
    To extract the x^4 coefficient of a degree-8 polynomial, we evaluate at 9 points
    and use inverse DFT. We share evaluation points across all (i,k) pairs.
    
    For each evaluation point t_l (l=0..8):
      - Compute a_{i,l} = P_i(t_l) for i=0,1,2  (these are "U" values)
      - Compute b_{k,l} = Q_k(t_l) for k=0..6  (these are "V" values)  
      - The product a_{i,l} * b_{k,l} contributes to R_{i,k}(t_l)
      - C[i,k] = (1/9) sum_l R_{i,k}(t_l) * omega^{-4l}
    
    This gives a tensor decomposition where each term c corresponds to an evaluation point l,
    and we need to map this to the (U,V,W) format.
    
    Actually, this gives rank 9*3*7 = 189 which is worse. Let me think more carefully.
    
    The issue: in the tensor decomposition, each rank-1 term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    The polynomial approach computes C[i,k] = sum_c U_part[i,c] * V_part[k,c] * W_part[i,k,c] which doesn't
    match the required structure where U depends on (i,j), V on (j,k), W on (k,i).
    
    Let me try a different approach: use the polynomial structure to find a smarter block decomposition.
    """
    pass  # Placeholder - this approach doesn't directly give a valid tensor decomposition

def _build_strassen_block_2x2x2_in_3x5x7():
    """
    Build 3x5*5x7 by identifying 2x2*2x2 subproblems and applying Strassen.
    
    Strategy: 
    - A is 3x5, B is 5x7, C is 3x7
    - For each pair of output rows (i1,i2) and output columns (k1,k2), 
      the submatrix C[i1:i2+1, k1:k2+1] involves A[i1:i2+1, :] * B[:, k1:k2+1]
    - If we can find 2x2*2x2 blocks, apply Strassen (rank 7 instead of 8)
    
    More concretely:
    - Split A's 5 columns into groups, B's 5 rows into matching groups
    - For a 2-column group of A and 2-row group of B, we have a 2x2 * 2x2 problem
    - But the full computation is more entangled
    
    Alternative: use the "laser method" or direct construction.
    
    Let me try: decompose 3x5*5x7 as sum of simpler tensors.
    Split the 5-dim inner product: j = 0,1,2,3,4
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k]
    
    Group terms: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k]) + A[i,4]*B[4,k]
    
    First group: 3x2*2x7 tensor, rank <= 3*2*7 = 42 schoolbook
    Second group: 3x2*2x7 tensor, rank <= 42 schoolbook  
    Third group: 3x1*1x7 tensor, rank = 21 schoolbook
    Total: 42+42+21 = 105 (no improvement)
    
    For the 3x2*2x7 subproblem: can we do better than 42?
    3x2*2x7: A is 3x2, B is 2x7, C is 3x7
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    This is like: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    Rank of M_{3x2x7}: at most 3*2*7=42. Can we do better?
    The tensor M_{3x2x7} = sum_{i,k,j} e_i^U e_j^V e_k^W ... 
    Actually M_{3x2x7} = M_{1x2x1} tensor M_{3x1x7} in some sense.
    The rank of M_{3x2x7} is at most 3*2=6 times rank of something? No.
    
    M_{3x2x7}: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    = sum_j A[i,j]*B[j,k] where j=0,1
    This is a 3x2x7 tensor. Its rank is at most 3*2*7=42.
    Lower bound: at least max(3*7, 2*7, 3*2) = 21? No, that's not right either.
    
    Actually, M_{3x2x7} can be seen as: for each j (2 values), we have a rank-1 term
    that is outer product of A[:,j] (3-dim), B[j,:] (7-dim), and... 
    Wait, in the tensor decomposition: U is (3*2, r) = (6, r), V is (2*7, r) = (14, r), W is (7*3, r) = (21, r)
    Each term c: U[i*2+j, c] * V[j*7+k, c] * W[k*3+i, c]
    
    For M_{3x2x7}, the "natural" decomposition has r=42 with one term per (i,j,k).
    Can we share? If U[i*2+0, c] and U[i*2+1, c] are related...
    
    This is getting too theoretical. Let me just try several block decompositions and pick the best.
    """
    pass  # Placeholder

def _build_best_block_decomp():
    """
    Try multiple block decompositions for 3x5*5x7 and return the best one.
    
    Decomposition strategies:
    1. Split A rows: (2x5*5x7) + (1x5*5x7) = 70 + 35 = 105
    2. Split A rows and B cols: 
       (2x4*4x6) + (2x1*1x6) + (2x4*4x1) + (2x1*1x1) + (1x5*5x7)
       = 48 + 12 + 8 + 2 + 35 = 105
    3. Use Strassen on 2x2*2x2 subblocks where possible
    
    Let me try: split into blocks that are 2x2*2x2 (Strassen rank 7) plus remainder.
    
    A is 3x5, B is 5x7.
    Split A columns: 2+2+1, B rows: 2+2+1
    Split A rows: 2+1, B cols: 2+2+2+1 or 3+2+2
    
    Block (i-block, j-block, k-block) contributes A[i-block, j-block] * B[j-block, k-block] to C[i-block, k-block]
    
    Let's use: A rows split as [2,1], A cols split as [2,2,1], B rows split as [2,2,1], B cols split as [2,2,2,1]
    
    That's 2*3*4 = 24 blocks, each of size (a_i x b_j) * (b_j x c_k)
    For blocks where a_i=2, b_j=2, c_k=2: use Strassen rank 7
    For others: schoolbook
    
    This is complex but potentially gives savings.
    """
    n, m, p = 3, 5, 7
    # A row splits
    a_row_splits = [2, 1]  # sums to 3
    # A col / B row splits  
    b_splits = [2, 2, 1]   # sums to 5
    # B col splits
    c_col_splits = [2, 2, 2, 1]  # sums to 7
    
    decomps = []
    a_offset = 0
    for ai, a_size in enumerate(a_row_splits):
        b_offset = 0
        for bi, b_size in enumerate(b_splits):
            c_offset = 0
            for ci, c_size in enumerate(c_col_splits):
                # Block: A[a_offset:a_offset+a_size, b_offset:b_offset+b_size] * B[b_offset:b_offset+b_size, c_offset:c_offset+c_size]
                # Result: C[a_offset:a_offset+a_size, c_offset:c_offset+c_size]
                block_n = a_size
                block_m = b_size
                block_p = c_size
                
                if block_n == 2 and block_m == 2 and block_p == 2:
                    # Use Strassen: rank 7
                    U_blk, V_blk, W_blk = _build_strassen_2x2x2()
                else:
                    # Schoolbook
                    U_blk, V_blk, W_blk = _schoolbook(block_n, block_m, block_p)
                
                # Extend to full size
                rank_blk = U_blk.shape[1]
                U_full = np.zeros((n*m, rank_blk))
                V_full = np.zeros((m*p, rank_blk))
                W_full = np.zeros((p*n, rank_blk))
                
                # Fill U: rows a_offset*m + b_offset to (a_offset+a_size)*m + b_offset + b_size - 1
                for i in range(block_n):
                    for j in range(block_m):
                        U_full[(a_offset+i)*m + (b_offset+j), :] = U_blk[i*block_m+j, :]
                
                # Fill V: rows b_offset*p + c_offset to ...
                for j in range(block_m):
                    for k in range(block_p):
                        V_full[(b_offset+j)*p + (c_offset+k), :] = V_blk[j*block_p+k, :]
                
                # Fill W: rows c_offset*n + a_offset to ...
                for k in range(block_p):
                    for i in range(block_n):
                        W_full[(c_offset+k)*n + (a_offset+i), :] = W_blk[k*block_n+i, :]
                
                decomps.append((U_full, V_full, W_full))
                c_offset += c_size
            b_offset += b_size
        a_offset += a_size
    
    return _combine_decompositions(decomps)

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try block decomposition with Strassen on 2x2x2 subblocks
    if time.time() - start_time < deadline - 10:
        try:
            U_block, V_block, W_block = _build_best_block_decomp()
            block_rank = U_block.shape[1]
            if _verify_fast(U_block, V_block, W_block, n, m, p, tol=1e-6):
                if block_rank < best_rank:
                    best_rank = block_rank
                    best_result = {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
        except Exception:
            pass
    
    # Try simple row split: (2x5*5x7) + (1x5*5x7)
    if time.time() - start_time < deadline - 10:
        try:
            n1, m1, p1 = 2, 5, 7
            U1, V1, W1 = _schoolbook(n1, m1, p1)
            U1_full = np.zeros((n*m, U1.shape[1]))
            U1_full[:n1*m1, :] = U1
            V1_full = V1.copy()
            W1_full = np.zeros((p*n, W1.shape[1]))
            for k in range(p1):
                for i in range(n1):
                    W1_full[k*n+i, :] = W1[k*n1+i, :]
            
            n2, m2, p2 = 1, 5, 7
            U2, V2, W2 = _schoolbook(n2, m2, p2)
            U2_full = np.zeros((n*m, U2.shape[1]))
            U2_full[n1*m1:, :] = U2
            V2_full = V2.copy()
            W2_full = np.zeros((p*n, W2.shape[1]))
            for k in range(p2):
                W2_full[k*n+2, :] = W2[k*n2+0, :]
            
            U_comb, V_comb, W_comb = _combine_decompositions([(U1_full, V1_full, W1_full), (U2_full, V2_full, W2_full)])
            comb_rank = U_comb.shape[1]
            if _verify_fast(U_comb, V_comb, W_comb, n, m, p, tol=1e-6):
                if comb_rank < best_rank:
                    best_rank = comb_rank
                    best_result = {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}
        except Exception:
            pass
    
    # Try column split on B: (3x5*5x4) + (3x5*5x3)
    if time.time() - start_time < deadline - 10:
        try:
            # Part 1: 3x5 * 5x4
            n1, m1, p1 = 3, 5, 4
            U1, V1, W1 = _schoolbook(n1, m1, p1)
            U1_full = np.zeros((n*m, U1.shape[1]))
            U1_full[:n1*m1, :] = U1
            V1_full = np.zeros((m*p, V1.shape[1]))
            V1_full[:m1*p1, :] = V1
            W1_full = np.zeros((p*n, W1.shape[1]))
            for k in range(p1):
                for i in range(n1):
                    W1_full[k*n+i, :] = W1[k*n1+i, :]
            
            # Part 2: 3x5 * 5x3
            n2, m2, p2 = 3, 5, 3
            U2, V2, W2 = _schoolbook(n2, m2, p2)
            U2_full = np.zeros((n*m, U2.shape[1]))
            U2_full[:n2*m2, :] = U2
            V2_full = np.zeros((m*p, V2.shape[1]))
            for j in range(m2):
                for k in range(p2):
                    V2_full[j*p + (p1+k), :] = V2[j*p2+k, :]
            W2_full = np.zeros((p*n, W2.shape[1]))
            for k in range(p2):
                for i in range(n2):
                    W2_full[(p1+k)*n+i, :] = W2[k*n2+i, :]
            
            U_comb, V_comb, W_comb = _combine_decompositions([(U1_full, V1_full, W1_full), (U2_full, V2_full, W2_full)])
            comb_rank = U_comb.shape[1]
            if _verify_fast(U_comb, V_comb, W_comb, n, m, p, tol=1e-6):
                if comb_rank < best_rank:
                    best_rank = comb_rank
                    best_result = {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}
        except Exception:
            pass
    
    return best_result
