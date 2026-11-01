"""Rank-44 decomposition for 2x3 * 3x8 via 4 blocks of (Strassen 2x2 + outer 2x1*1x2)."""
import time
_start_time = time.time()
_DEADLINE = _start_time + 130  # safety margin before 150s kill

def _strassen_2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns list of 7 tuples: (U_row, V_row, W_row, coeff)
    where U is 4x1 (flattened 2x2 A), V is 4x1 (flattened 2x2 B), W is 4x1 (flattened 2x2 C).
    Indexing: A[i,j] -> index i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    """
    # Strassen: M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each term r: U vector (4 entries for A), V vector (4 entries for B), W vector (4 entries for C)
    # We'll store as: for each rank term, which A-index, which B-index, which C-index, and coefficient
    
    terms = []
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    terms.append([(0, 1.0), (2, 1.0)],  # U: A[0,0]=1, A[1,1]=1
                 [(0, 1.0), (3, 1.0)],  # V: B[0,0]=1, B[1,1]=1
                 [(0, 1.0), (3, 1.0)])  # W: C[0,0]=1, C[1,1]=1
    
    # Term 1: M2 = (A10+A11)B00
    # U: A[1,0]=1, A[1,1]=1; V: B[0,0]=1; W: C[1,0]=1, C[1,1]=-1
    terms.append([(2, 1.0), (3, 1.0)],
                 [(0, 1.0)],
                 [(2, 1.0), (3, -1.0)])
    
    # Term 2: M3 = A00(B10-B11)
    # U: A[0,0]=1; V: B[1,0]=1, B[1,1]=-1; W: C[0,1]=1, C[1,0]=1
    terms.append([(0, 1.0)],
                 [(2, 1.0), (3, -1.0)],
                 [(1, 1.0), (2, 1.0)])
    
    # Term 3: M4 = A11(B01-B00)
    # U: A[1,1]=1; V: B[0,1]=1, B[0,0]=-1; W: C[0,0]=1, C[1,0]=1
    terms.append([(3, 1.0)],
                 [(1, 1.0), (0, -1.0)],
                 [(0, 1.0), (2, 1.0)])
    
    # Term 4: M5 = (A00+A01)B11
    # U: A[0,0]=1, A[0,1]=1; V: B[1,1]=1; W: C[0,0]=-1, C[0,1]=1
    terms.append([(0, 1.0), (1, 1.0)],
                 [(3, 1.0)],
                 [(0, -1.0), (1, 1.0)])
    
    # Term 5: M6 = (A10-A00)(B00+B10)
    # U: A[1,0]=1, A[0,0]=-1; V: B[0,0]=1, B[1,0]=1; W: C[1,1]=1
    terms.append([(2, 1.0), (0, -1.0)],
                 [(0, 1.0), (2, 1.0)],
                 [(3, 1.0)])
    
    # Term 6: M7 = (A01-A11)(B01+B11)
    # U: A[0,1]=1, A[1,1]=-1; V: B[0,1]=1, B[1,1]=1; W: C[0,0]=1
    terms.append([(1, 1.0), (3, -1.0)],
                 [(1, 1.0), (3, 1.0)],
                 [(0, 1.0)])
    
    return terms

def _build_block_decomp(k0, k1):
    """
    Build rank-11 decomposition for the 2x3 * 3x2 subproblem using columns k0, k1.
    Returns list of 11 rank-1 terms.
    Each term: (U_entries, V_entries, W_entries)
    U_entries: list of (index_in_6, coeff) for the 2x3 A matrix (flattened: i*3+j)
    V_entries: list of (index_in_24, coeff) for the 3x2 B portion (flattened: j*8+k)
    W_entries: list of (index_in_16, coeff) for the 2x2 C portion (flattened: k*2+i)
    """
    terms = []
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] (2x2 * 2x2)
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=8+k0, B[1,k1]=8+k1
    # C indices: C[k0,0]=k0*2+0, C[k0,1]=k0*2+1, C[k1,0]=k1*2+0, C[k1,1]=k1*2+1
    
    strassen_terms = _strassen_2x2()
    # Map local 2x2 indices to global
    # A local: 0->A[0,0]=0, 1->A[0,1]=1, 2->A[1,0]=3, 3->A[1,1]=4
    a_map = {0: 0, 1: 1, 2: 3, 3: 4}
    # B local: 0->B[0,k0]=k0, 1->B[0,k1]=k1, 2->B[1,k0]=8+k0, 3->B[1,k1]=8+k1
    b_map = {0: k0, 1: k1, 2: 8 + k0, 3: 8 + k1}
    # C local: 0->C[k0,0]=k0*2, 1->C[k0,1]=k0*2+1, 2->C[k1,0]=k1*2, 3->C[k1,1]=k1*2+1
    c_map = {0: k0 * 2, 1: k0 * 2 + 1, 2: k1 * 2, 3: k1 * 2 + 1}
    
    for u_local, v_local, w_local in strassen_terms:
        u_global = [(a_map[idx], coeff) for idx, coeff in u_local]
        v_global = [(b_map[idx], coeff) for idx, coeff in v_local]
        w_global = [(c_map[idx], coeff) for idx, coeff in w_local]
        terms.append((u_global, v_global, w_global))
    
    # Part 2: Outer product A[:, 2] * B[2, :] (2x1 * 1x2)
    # A[0,2]=2, A[1,2]=5
    # B[2,k0]=16+k0, B[2,k1]=16+k1
    # C[k0,0]=k0*2, C[k0,1]=k0*2+1, C[k1,0]=k1*2, C[k1,1]=k1*2+1
    
    # Term 7: A[0,2] * B[2,k0] -> C[k0,0]
    terms.append([(2, 1.0)], [(16 + k0, 1.0)], [(k0 * 2, 1.0)])
    # Term 8: A[0,2] * B[2,k1] -> C[k1,0]
    terms.append([(2, 1.0)], [(16 + k1, 1.0)], [(k1 * 2, 1.0)])
    # Term 9: A[1,2] * B[2,k0] -> C[k0,1]
    terms.append([(5, 1.0)], [(16 + k0, 1.0)], [(k0 * 2 + 1, 1.0)])
    # Term 10: A[1,2] * B[2,k1] -> C[k1,1]
    terms.append([(5, 1.0)], [(16 + k1, 1.0)], [(k1 * 2 + 1, 1.0)])
    
    return terms

def construct():
    """Rank-44 decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]   # 16 x 44
    
    col = 0
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        block_terms = _build_block_decomp(k0, k1)
        for u_entries, v_entries, w_entries in block_terms:
            for idx, coeff in u_entries:
                U[idx][col] += coeff
            for idx, coeff in v_entries:
                V[idx][col] += coeff
            for idx, coeff in w_entries:
                W[idx][col] += coeff
            col += 1
    
    # Verify we used exactly 44 columns
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Time check
    if time.time() > _DEADLINE:
        # Shouldn't happen for this deterministic construction, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
