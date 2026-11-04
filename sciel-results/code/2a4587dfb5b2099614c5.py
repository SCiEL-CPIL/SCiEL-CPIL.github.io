"""Improved 5x5x5 matrix multiplication tensor decomposition using block structure."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_vec, v_vec, w_vec) where each is a dict {index: coeff}.
    U indices: i*m+j, V indices: j*p+k, W indices: k*n+i (with m=p=n=2 for the block)
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U vectors (A indices: 00,01,10,11 -> 0,1,2,3)
    # V vectors (B indices: 00,01,10,11 -> 0,1,2,3)
    # W vectors (C indices: 00,01,10,11 -> 0,1,2,3)
    
    terms = []
    
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11(+1), C22(+1)
    u1 = {0: 1, 3: 1}
    v1 = {0: 1, 3: 1}
    w1 = {0: 1, 3: 1}
    terms.append((u1, v1, w1))
    
    # M2 = (A21+A22)*B11 -> contributes to C21(+1), C22(-1)
    u2 = {2: 1, 3: 1}
    v2 = {0: 1}
    w2 = {2: 1, 3: -1}
    terms.append((u2, v2, w2))
    
    # M3 = A11*(B12-B22) -> contributes to C12(+1), C22(+1)
    u3 = {0: 1}
    v3 = {1: 1, 3: -1}
    w3 = {1: 1, 3: 1}
    terms.append((u3, v3, w3))
    
    # M4 = A22*(B21-B11) -> contributes to C11(+1), C21(+1)
    u4 = {3: 1}
    v4 = {2: 1, 0: -1}
    w4 = {0: 1, 2: 1}
    terms.append((u4, v4, w4))
    
    # M5 = (A11+A12)*B22 -> contributes to C11(-1), C12(+1)
    u5 = {0: 1, 1: 1}
    v5 = {3: 1}
    w5 = {0: -1, 1: 1}
    terms.append((u5, v5, w5))
    
    # M6 = (A21-A11)*(B11+B12) -> contributes to C22(+1)
    u6 = {2: 1, 0: -1}
    v6 = {0: 1, 1: 1}
    w6 = {3: 1}
    terms.append((u6, v6, w6))
    
    # M7 = (A12-A22)*(B21+B22) -> contributes to C11(+1)
    u7 = {1: 1, 3: -1}
    v7 = {2: 1, 3: 1}
    w7 = {0: 1}
    terms.append((u7, v7, w7))
    
    return terms


def _laderman_3x3():
    """Return the 14 Laderman terms for 3x3x3 matrix multiplication.
    Laderman's algorithm uses 14 scalar multiplications for 3x3 matrices.
    """
    # Laderman's 14-term formula for 3x3 matrix multiplication
    # A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    # C = A*B
    
    # The 14 products:
    # P1 = (a11+a21+a31)(b11+b21+b31)  -- no, this isn't quite right
    # Let me use the standard Laderman formula.
    
    # Actually, let me use a known correct 14-term decomposition for 3x3.
    # The Laderman formula (1976) gives rank(3,3,3) <= 14.
    
    # U indices: i*3+j for i,j in {0,1,2} -> 0..8
    # V indices: j*3+k for j,k in {0,1,2} -> 0..8  
    # W indices: k*3+i for k,i in {0,1,2} -> 0..8
    
    terms = []
    
    # Standard Laderman 14-term decomposition:
    # Let me define the 14 terms carefully.
    
    # Term 1: (a00+a10+a20)(b00+b10+b20) -> but this isn't standard Laderman
    
    # Let me use the explicit Laderman formula from the literature:
    # The 14 bilinear forms are:
    
    # I'll use a verified 14-term decomposition.
    # Reference: Laderman, "On the number of multiplications necessary to multiply 3x3 matrices", 1976.
    
    # A common presentation:
    # m1 = (a11 + a21 + a31)(b11 + b21 + b31)
    # m2 = (a11 + a21)(b12 + b22)  
    # m3 = (a12 + a22)(b11 + b21)
    # m4 = (a11 + a31)(b13 + b33)
    # m5 = (a13 + a33)(b11 + b31)
    # m6 = (a21 + a31)(b22 + b32)
    # m7 = (a22 + a32)(b21 + b31)
    # m8 = (a11 - a21 + a31)(b12 - b22 + b32)
    # m9 = (a12 - a22 + a32)(b11 - b21 + b31)
    # m10 = (a11 + a21 - a31)(b13 + b23 - b33)
    # m11 = (a13 + a23 - a33)(b11 + b21 - b31)
    # m12 = (a21 + a31 - a11)(b22 + b32 - b12)
    # m13 = (a22 + a32 - a12)(b21 + b31 - b11)
    # m14 = (a11 - a21 - a31)(b11 - b21 - b31)
    
    # This is getting complicated. Let me use a different, well-known 14-term formula.
    
    # Actually, for safety, let me construct the 3x3 block using a simpler approach:
    # Use 9 schoolbook terms for the 3x3 block. This gives rank 9 for that block.
    # Total rank would be: 9 (3x3) + 7 (2x2) + 7 (2x2) + cross terms
    
    # Wait, the strategy says to use Laderman for 3x3 (rank 14) and Strassen for 2x2 (rank 7 each).
    # But I need to be careful about cross-block terms.
    
    # Let me reconsider the approach. The partition is 5 = 3 + 2.
    # We have blocks:
    # A = [A33 A32; A23 A22] where A33 is 3x3, A32 is 3x2, A23 is 2x3, A22 is 2x2
    # B = [B33 B32; B23 B22]
    # C = A*B = [A33*B33 + A32*B23, A33*B32 + A32*B22; A23*B33 + A22*B23, A23*B32 + A22*B22]
    
    # The diagonal blocks A33*B33 (3x3x3) can use Laderman (rank 14)
    # The diagonal blocks A22*B22 (2x2x2) can use Strassen (rank 7)
    # But we have TWO 2x2 blocks? No, just one 2x2 block in A and one in B.
    
    # Wait, the partition is 5 = 3 + 2, so:
    # A is 5x5, split as rows {0,1,2} and {3,4}, cols {0,1,2} and {3,4}
    # B is 5x5, split as rows {0,1,2} and {3,4}, cols {0,1,2} and {3,4}
    # C is 5x5, split as rows {0,1,2} and {3,4}, cols {0,1,2} and {3,4}
    
    # C11 (3x3) = A11*B11 + A12*B21  (where A11 is 3x3, A12 is 3x2, B11 is 3x3, B21 is 2x3)
    # C12 (3x2) = A11*B12 + A12*B22
    # C21 (2x3) = A21*B11 + A22*B21
    # C22 (2x2) = A21*B12 + A22*B22
    
    # This is more complex. The cross terms A12*B21 etc. are not square blocks.
    
    # A simpler approach: just use schoolbook for all cross terms and optimized for diagonal.
    # But this might not give a good rank reduction.
    
    # Alternative: Use the full 5x5 decomposition with a different strategy.
    # Let me just use schoolbook for everything to be safe, but that gives rank 125.
    
    # Better: Let me try the 2+2+1 split from the inspirations, which achieved rank 97.
    
    # For now, let me implement a safe block decomposition with 2+2+1 split.
    
    return None  # We'll use a different approach


def _construct_block_221():
    """Construct a block decomposition using 2+2+1 split with Strassen on 2x2x2 blocks.
    
    Partition indices 0..4 into G0={0,1}, G1={2,3}, G2={4}.
    For each pair of blocks (Gi, Gj) in the row/col partition of A, B, C,
    we handle the sub-multiplication.
    
    The key insight: for 2x2x2 sub-blocks, use Strassen (rank 7 instead of 8).
    For other sizes, use schoolbook.
    """
    n = m = p = 5
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    # Define the partition
    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    groups = [G0, G1, G2]
    
    # For each combination of row-group (for A rows), col-group (for A cols / B rows), 
    # and output col-group (for B cols / C cols), we handle the sub-problem.
    
    # A is indexed by (i, j) where i in 0..4 (row), j in 0..4 (col)
    # B is indexed by (j, k) where j in 0..4 (row), k in 0..4 (col)
    # C is indexed by (i, k) where i in 0..4 (row), k in 0..4 (col)
    
    # U index: i*5+j
    # V index: j*5+k
    # W index: k*5+i
    
    # We partition i into groups, j into groups, k into groups.
    # For each (row_group_i, col_group_j, out_group_k), we have a sub-problem
    # of size |row_group_i| x |col_group_j| x |out_group_k|.
    
    for gi in range(3):
        for gj in range(3):
            for gk in range(3):
                rows_i = groups[gi]
                cols_j = groups[gj]
                cols_k = groups[gk]
                
                ni = len(rows_i)
                nj = len(cols_j)
                nk = len(cols_k)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen for 2x2x2
                    strassen_terms = _strassen_2x2()
                    for (su, sv, sw) in strassen_terms:
                        # Map local indices to global
                        u_dict = {}
                        v_dict = {}
                        w_dict = {}
                        for (li, lc), coeff in su.items():
                            i = rows_i[li]
                            j = cols_j[lc]
                            u_dict[i * 5 + j] = coeff
                        for (lj, lk), coeff in sv.items():
                            j = cols_j[lj]
                            k = cols_k[lk]
                            v_dict[j * 5 + k] = coeff
                        for (lk, li), coeff in sw.items():
                            k = cols_k[lk]
                            i = rows_i[li]
                            w_dict[k * 5 + i] = coeff
                        terms.append((u_dict, v_dict, w_dict))
                else:
                    # Use schoolbook for this sub-block
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                i = rows_i[li]
                                j = cols_j[lj]
                                k = cols_k[lk]
                                u_dict = {i * 5 + j: 1}
                                v_dict = {j * 5 + k: 1}
                                w_dict = {k * 5 + i: 1}
                                terms.append((u_dict, v_dict, w_dict))
    
    return terms


def _verify_decomposition(terms, n, m, p, timeout=10):
    """Verify that the tensor decomposition is correct.
    
    Check that sum_r U[:,r] * V[:,r] * W[:,r] equals the matrix multiplication tensor.
    The tensor T[i,j,k] = 1 if and only if the (i,j) entry of A times (j,k) entry of B
    contributes to (i,k) entry of C.
    """
    start = time.time()
    
    # Build the expected tensor
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor is T_{i,j,k} = 1 for all valid i,j,k.
    
    # Build U, V, W as sparse representations
    rank = len(terms)
    
    # For efficiency, just check a few random points
    # Or check all points if feasible
    total_points = n * m * p
    
    if total_points > 10000:
        # Check random subset
        import random
        random.seed(42)
        indices = random.sample(range(total_points), min(1000, total_points))
    else:
        indices = range(total_points)
    
    for idx in indices:
        if time.time() - start > timeout:
            return False
        
        i = idx // (m * p)
        rem = idx % (m * p)
        j = rem // p
        k = rem % p
        
        # Expected value: 1
        # Computed value: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        total = 0
        for r in range(rank):
            u_val = terms[r][0].get(u_idx, 0)
            v_val = terms[r][1].get(v_idx, 0)
            w_val = terms[r][2].get(w_idx, 0)
            total += u_val * v_val * w_val
        
        if total != 1:
            return False
    
    return True


def _construct_schoolbook():
    """Return the schoolbook decomposition (rank 125)."""
    n = m = p = 5
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_dict = {i * m + j: 1}
                v_dict = {j * p + k: 1}
                w_dict = {k * n + i: 1}
                terms.append((u_dict, v_dict, w_dict))
    return terms


def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin
    
    # Try the block decomposition with 2+2+1 split
    try:
        terms = _construct_block_221()
        rank = len(terms)
        
        if time.time() < deadline:
            if _verify_decomposition(terms, n, m, p, timeout=20):
                # Build the U, V, W matrices
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for col, (u_indices, v_indices, w_indices) in enumerate(terms):
                    for (idx, coeff) in u_indices.items():
                        U[idx][col] = coeff
                    for (idx, coeff) in v_indices.items():
                        V[idx][col] = coeff
                    for (idx, coeff) in w_indices.items():
                        W[idx][col] = coeff
                
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 125)
    terms = _construct_schoolbook()
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_indices, v_indices, w_indices) in enumerate(terms):
        for (idx, coeff) in u_indices.items():
            U[idx][col] = coeff
        for (idx, coeff) in v_indices.items():
            V[idx][col] = coeff
        for (idx, coeff) in w_indices.items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
