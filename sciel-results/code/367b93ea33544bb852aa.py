<<<<<<< SEARCH
def _add_bcl_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add BCL rank-13 decomposition for 4x4 * 4x4 block multiplication.
    Uses the Bini-Capovani-Lotti decomposition with 13 rank-1 terms.
    """
    # BCL 13 terms for 4x4x4 (entries in {-1, 0, 1})
    # Verified: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta_{ik} for all i,j,k
    
    bcl_terms = [
        # Term 1: diagonal pattern
        ({(0,0):1, (1,1):1, (2,2):-1, (3,3):-1},
         {(0,0):1, (1,1):-1, (2,2):1, (3,3):-1},
         {(0,0):1, (1,1):-1, (2,2):-1, (3,3):1}),
        # Term 2
        ({(0,1):1, (1,0):-1, (2,3):-1, (3,2):1},
         {(0,1):1, (1,0):-1, (2,3):-1, (3,2):1},
         {(0,1):1, (1,0):-1, (2,3):-1, (3,2):1}),
        # Term 3
        ({(0,2):1, (2,0):-1, (1,3):1, (3,1):-1},
         {(0,2):1, (2,0):-1, (1,3):1, (3,1):-1},
         {(0,2):1, (2,0):-1, (1,3):1, (3,1):-1}),
        # Term 4
        ({(1,2):1, (2,1):-1, (0,3):-1, (3,0):1},
         {(1,2):1, (2,1):-1, (0,3):-1, (3,0):1},
         {(1,2):1, (2,1):-1, (0,3):-1, (3,0):1}),
        # Term 5
        ({(0,0):1, (0,1):1, (1,0):-1, (1,1):-1, (2,2):1, (2,3):1, (3,2):-1, (3,3):-1},
         {(0,0):1, (0,1):-1, (1,0):1, (1,1):-1, (2,2):1, (2,3):-1, (3,2):1, (3,3):-1},
         {(0,0):1, (0,1):1, (1,0):1, (1,1):-1, (2,2):1, (2,3):1, (3,2):1, (3,3):-1}),
        # Term 6
        ({(0,0):1, (0,2):-1, (2,0):-1, (2,2):1, (1,1):1, (1,3):-1, (3,1):-1, (3,3):1},
         {(0,0):1, (0,2):-1, (2,0):-1, (2,2):1, (1,1):1, (1,3):-1, (3,1):-1, (3,3):1},
         {(0,0):1, (0,2):-1, (2,0):-1, (2,2):1, (1,1):1, (1,3):-1, (3,1):-1, (3,3):1}),
        # Term 7
        ({(0,0):1, (0,3):-1, (3,0):-1, (3,3):1, (1,2):-1, (2,1):1, (1,1):1, (2,2):-1},
         {(0,0):1, (0,3):-1, (3,0):-1, (3,3):1, (1,2):-1, (2,1):1, (1,1):1, (2,2):-1},
         {(0,0):1, (0,3):-1, (3,0):-1, (3,3):1, (1,2):-1, (2,1):1, (1,1):1, (2,2):-1}),
        # Term 8
        ({(0,1):1, (1,0):1, (0,0):-1, (1,1):-1, (2,3):1, (3,2):1, (2,2):-1, (3,3):-1},
         {(0,1):1, (1,0):1, (0,0):-1, (1,1):-1, (2,3):1, (3,2):1, (2,2):-1, (3,3):-1},
         {(0,1):1, (1,0):1, (0,0):-1, (1,1):-1, (2,3):1, (3,2):1, (2,2):-1, (3,3):-1}),
        # Term 9
        ({(0,2):1, (2,0):1, (0,0):-1, (2,2):-1, (1,3):1, (3,1):1, (1,1):-1, (3,3):-1},
         {(0,2):1, (2,0):1, (0,0):-1, (2,2):-1, (1,3):1, (3,1):1, (1,1):-1, (3,3):-1},
         {(0,2):1, (2,0):1, (0,0):-1, (2,2):-1, (1,3):1, (3,1):1, (1,1):-1, (3,3):-1}),
        # Term 10
        ({(1,2):1, (2,1):1, (1,1):-1, (2,2):-1, (0,3):1, (3,0):1, (0,0):-1, (3,3):-1},
         {(1,2):1, (2,1):1, (1,1):-1, (2,2):-1, (0,3):1, (3,0):1, (0,0):-1, (3,3):-1},
         {(1,2):1, (2,1):1, (1,1):-1, (2,2):-1, (0,3):1, (3,0):1, (0,0):-1, (3,3):-1}),
        # Term 11
        ({(0,0):1, (1,1):-1, (2,2):-1, (3,3):1, (0,1):-1, (1,0):1, (2,3):-1, (3,2):1},
         {(0,0):1, (1,1):-1, (2,2):-1, (3,3):1, (0,1):-1, (1,0):1, (2,3):-1, (3,2):1},
         {(0,0):1, (1,1):-1, (2,2):-1, (3,3):1, (0,1):-1, (1,0):1, (2,3):-1, (3,2):1}),
        # Term 12
        ({(0,2):-1, (2,0):1, (1,3):1, (3,1):-1, (0,0):1, (2,2):-1, (1,1):-1, (3,3):1},
         {(0,2):-1, (2,0):1, (1,3):1, (3,1):-1, (0,0):1, (2,2):-1, (1,1):-1, (3,3):1},
         {(0,2):-1, (2,0):1, (1,3):1, (3,1):-1, (0,0):1, (2,2):-1, (1,1):-1, (3,3):1}),
        # Term 13
        ({(1,2):-1, (2,1):1, (0,3):1, (3,0):-1, (1,1):-1, (2,2):1, (0,0):-1, (3,3):1},
         {(1,2):-1, (2,1):1, (0,3):1, (3,0):-1, (1,1):-1, (2,2):1, (0,0):-1, (3,3):1},
         {(1,2):-1, (2,1):1, (0,3):1, (3,0):-1, (1,1):-1, (2,2):1, (0,0):-1, (3,3):1}),
    ]
    
    for u_pat, v_pat, w_pat in bcl_terms:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in u_pat.items():
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff
        
        for (r, c), coeff in v_pat.items():
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff
        
        for (r, c), coeff in w_pat.items():
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
=======
def _add_bcl_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add BCL rank-13 decomposition for 4x4 * 4x4 block multiplication.
    Falls back to recursive Strassen (rank 49) if BCL verification fails.
    """
    # First, try BCL decomposition
    bcl_U = []
    bcl_V = []
    bcl_W = []
    
    # BCL 13 terms (need to be verified)
    bcl_terms = _get_bcl_terms()
    
    if bcl_terms is not None:
        for u_pat, v_pat, w_pat in bcl_terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (r, c), coeff in u_pat.items():
                i = oi + r
                j = oj + c
                if 0 <= i < n and 0 <= j < m:
                    u_col[i * m + j] += coeff
            
            for (r, c), coeff in v_pat.items():
                j = oj + r
                k = ok + c
                if 0 <= j < m and 0 <= k < p:
                    v_col[j * p + k] += coeff
            
            for (r, c), coeff in w_pat.items():
                i = oi + r
                k = ok + c
                if 0 <= i < n and 0 <= k < p:
                    w_col[k * n + i] += coeff
            
            bcl_U.append(u_col)
            bcl_V.append(v_col)
            bcl_W.append(w_col)
        
        # Verify BCL decomposition
        if _verify_bcl_block(bcl_U, bcl_V, bcl_W, oi, oj, ok, n, m, p):
            U_cols.extend(bcl_U)
            V_cols.extend(bcl_V)
            W_cols.extend(bcl_W)
            return
    
    # Fallback: use recursive Strassen
    _add_strassen_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p)


def _get_bcl_terms():
    """
    Return BCL rank-13 decomposition terms for 4x4x4, or None if not available.
    """
    # I'll try to construct a valid BCL decomposition.
    # The BCL decomposition is known to exist with rank 13.
    # For now, return None to use the Strassen fallback.
    # TODO: Implement actual BCL terms
    return None


def _verify_bcl_block(U, V, W, oi, oj, ok, n, m, p):
    """
    Verify that the BCL block decomposition is correct.
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 
    for all i in [oi, oi+4), j in [oj, oj+4), k in [ok, ok+4)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for i in range(oi, min(oi+4, n)):
        for j in range(oj, min(oj+4, m)):
            for k in range(ok, min(ok+4, p)):
                total = 0
                for r in range(rank):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                if total != 1:
                    return False
    return True
>>>>>>> REPLACE
