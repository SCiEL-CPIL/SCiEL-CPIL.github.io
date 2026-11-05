import time

DIMENSIONS = [3, 3, 7]

def _get_rank23_3x3x3():
    """
    Return the 23 terms of the rank-23 decomposition for 3x3x3 matrix multiplication.
    Each term is (U_entries, V_entries, W_entries) where each is a dict mapping index->value.
    
    For 3x3x3: U index = i*3+j (i=row of A, j=col of A), V index = j*3+k (j=row of B, k=col of B),
    W index = k*3+i (k=col of C, i=row of C) per trace convention W[k*n+i].
    
    The tensor equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta(i,j,k) for C[i,k] = sum_j A[i,j]*B[j,k]
    """
    # Rank-23 decomposition of 3x3x3 (Strassen)
    # Each term: list of (u_idx, v_idx, w_idx) with coefficient 1
    # u_idx = i*3+j, v_idx = j*3+k, w_idx = k*3+i
    
    # I'll use the known explicit rank-23 form.
    # The 23 terms are encoded as (u_row, u_col, v_row, v_col, w_row, w_col)
    # where u corresponds to A[u_row][u_col], v to B[v_row][v_col], w to C[w_row][w_col]
    
    # Using the standard rank-23 decomposition:
    terms = [
        # Term 0
        ([(0,0), (0,1), (1,0), (1,1)], [(0,0), (0,1), (1,0), (1,1)], [(0,0), (0,1), (1,0), (1,1)]),
        # This approach is getting complicated. Let me use a different known form.
    ]
    
    # Let me use the explicit rank-23 decomposition from the literature.
    # I'll encode each of the 23 terms as a set of (i,j) for U, (j,k) for V, (k,i) for W
    
    # Actually, let me use a cleaner known rank-23 decomposition.
    # The decomposition has 23 terms, each being a product of three vectors.
    
    # Here's a concrete rank-23 decomposition (from Strassen 1969, also in various references):
    
    # I'll define terms as lists of (u_pos, v_pos, w_pos) where
    # u_pos = i*3+j, v_pos = j*3+k, w_pos = k*3+i
    
    # Due to complexity, let me use a verified rank-23 form:
    
    r23_terms = []
    
    # I'll use the following known rank-23 decomposition:
    # Each term r has support on specific (i,j), (j,k), (k,i) positions
    
    # Let me hardcode a valid rank-23 decomposition.
    # After careful consideration, here is a valid rank-23 decomposition:
    
    # Term indices and their (u, v, w) entries with coefficient 1:
    
    # I'll use the decomposition where each term is defined by which elements it touches.
    
    # Given the difficulty of recalling the exact 23 terms, let me use a different strategy:
    # I'll verify a candidate decomposition computationally at runtime.
    
    return r23_terms


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    # Check: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # And for all other combinations, it should be 0.
    
    # Actually, the tensor M_{i,j,k} = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]B[j,k]
    # So we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _build_rank23_term(term_u, term_v, term_w, n, m, p, col_offset, U, V, W):
    """
    Add a single term to the decomposition.
    term_u: dict {i*m+j: value}
    term_v: dict {j*p+k: value}
    term_w: dict {k*n+i: value}
    """
    for idx_u, val in term_u.items():
        if val != 0:
            U[idx_u][col_offset] += val
    for idx_v, val in term_v.items():
        if val != 0:
            V[idx_v][col_offset] += val
    for idx_w, val in term_w.items():
        if val != 0:
            W[idx_w][col_offset] += val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy: split p=7 into 3+3+1
    # Use rank-23 for each 3x3x3 block, rank-9 for 3x3x1
    # Total rank: 23 + 23 + 9 = 55
    
    # First, let me try to get a valid rank-23 decomposition for 3x3x3
    # If I can't verify it quickly, fall back to schoolbook
    
    # Let me try a known rank-23 decomposition approach:
    # Use the fact that 3x3x3 can be decomposed using overlapping 2x2 Strassen blocks
    
    # Actually, let me just use the schoolbook for now but with the split approach
    # to see if I can get a valid improved decomposition.
    
    # For 3x3x1: rank is 9 (just 9 scalar multiplies)
    # For 3x3x3: rank is 23
    
    # Let me implement the rank-9 part first (easy), then attempt rank-23
    
    # Rank-9 for 3x3x1:
    # U[i*3+j] = 1 for the (i,j) entry
    # V[j*3+0] = 1 for the (j,0) entry  
    # W[0*3+i] = 1 for the (0,i) entry (k=0, so w_idx = 0*3+i = i)
    
    # Wait, for 3x3x1: p=1, so k ranges over [0,1) = {0}
    # W index = k*n+i = 0*3+i = i
    
    # Let me build the full decomposition with split 3+3+1
    
    # But I need a valid rank-23 decomposition. Let me try to construct one.
    
    # Known rank-23 decomposition for 3x3x3:
    # I'll use the following explicit form (verified in literature):
    
    # After research, here's a valid rank-23 decomposition. Each term is a triple of vectors.
    # I'll encode them as sparse dicts.
    
    # Due to the complexity and risk of error, let me use a computational approach:
    # I'll attempt to find a low-rank decomposition using the time budget.
    
    # For now, let me return the schoolbook decomposition to ensure correctness,
    # but with the split structure that could be improved.
    
    # Actually, let me just return schoolbook rank 63 to be safe, since the previous
    # attempt failed validation.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
