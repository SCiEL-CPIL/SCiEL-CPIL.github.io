"""Strassen-based decomposition: Rank-71 for 3x3x8 using 2x2x2 Strassen + schoolbook."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8

    # Build a list of terms. Each term is a dict with 'U', 'V', 'W' keys,
    # each mapping row-index to coefficient.
    terms = []

    # ---- Strassen rank-7 for the 2x2x2 sub-block (i,j,k in {0,1}) ----
    # Standard Strassen 2x2x2 decomposition:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In our tensor: T[i,j,k] = 1 for all (i,j,k).
    # The Strassen terms must satisfy:
    # For each (i,j,k) in {0,1}^3: sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    #
    # Let me define the 7 Strassen terms with their U, V, W patterns:
    # U rows: i*m+j, V rows: j*p+k, W rows: k*n+i

    # M1 = (A00+A11)*(B00+B11)
    # U: A[0,0]=1, A[1,1]=1 -> U[0]=1, U[4]=1
    # V: B[0,0]=1, B[1,1]=1 -> V[0]=1, V[4]=1
    # W: C[0,0] gets +1 (k=0,i=0), C[1,1] gets +1 (k=1,i=1)
    #    W[0*3+0]=W[0]=1, W[1*3+1]=W[4]=1
    terms.append({'U': {0: 1, 4: 1}, 'V': {0: 1, 4: 1}, 'W': {0: 1, 4: 1}})

    # M2 = (A10+A11)*B00
    # U: A[1,0]=1, A[1,1]=1 -> U[3]=1, U[4]=1
    # V: B[0,0]=1 -> V[0]=1
    # W: C[1,0] gets +1 (k=0,i=1) -> W[0*3+1]=W[1]=1
    terms.append({'U': {3: 1, 4: 1}, 'V': {0: 1}, 'W': {1: 1}})

    # M3 = A00*(B10-B11)
    # U: A[0,0]=1 -> U[0]=1
    # V: B[1,0]=1, B[1,1]=-1 -> V[3]=1, V[4]=-1
    # W: C[0,1] gets +1 (k=1,i=0) -> W[1*3+0]=W[3]=1
    terms.append({'U': {0: 1}, 'V': {3: 1, 4: -1}, 'W': {3: 1}})

    # M4 = A11*(B01-B00)
    # U: A[1,1]=1 -> U[4]=1
    # V: B[0,1]=1, B[0,0]=-1 -> V[1]=1, V[0]=-1
    # W: C[0,0] gets +1 (k=0,i=0) -> W[0]=1, C[1,0] gets +1 (k=0,i=1) -> W[1]=1
    terms.append({'U': {4: 1}, 'V': {1: 1, 0: -1}, 'W': {0: 1, 1: 1}})

    # M5 = (A00+A01)*B11
    # U: A[0,0]=1, A[0,1]=1 -> U[0]=1, U[1]=1
    # V: B[1,1]=1 -> V[4]=1
    # W: C[0,0] gets -1 (k=0,i=0) -> W[0]=-1, C[0,1] gets +1 (k=1,i=0) -> W[3]=1
    terms.append({'U': {0: 1, 1: 1}, 'V': {4: 1}, 'W': {0: -1, 3: 1}})

    # M6 = (A10-A00)*(B00+B10)
    # U: A[1,0]=1, A[0,0]=-1 -> U[3]=1, U[0]=-1
    # V: B[0,0]=1, B[1,0]=1 -> V[0]=1, V[3]=1
    # W: C[1,1] gets +1 (k=1,i=1) -> W[1*3+1]=W[4]=1
    terms.append({'U': {3: 1, 0: -1}, 'V': {0: 1, 3: 1}, 'W': {4: 1}})

    # M7 = (A01-A11)*(B01+B11)
    # U: A[0,1]=1, A[1,1]=-1 -> U[1]=1, U[4]=-1
    # V: B[0,1]=1, B[1,1]=1 -> V[1]=1, V[4]=1
    # W: C[0,0] gets +1 (k=0,i=0) -> W[0]=1, C[1,1] gets +1 (k=1,i=1) -> W[4]=1
    terms.append({'U': {1: 1, 4: -1}, 'V': {1: 1, 4: 1}, 'W': {0: 1, 4: 1}})

    # ---- Schoolbook for all other (i,j,k) terms ----
    # For each (i,j,k) where NOT (i in {0,1} AND j in {0,1} AND k in {0,1}),
    # add a schoolbook term.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                terms.append({'U': {u_idx: 1}, 'V': {v_idx: 1}, 'W': {w_idx: 1}})

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, term in enumerate(terms):
        for u_idx, coeff in term['U'].items():
            U[u_idx][r] = coeff
        for v_idx, coeff in term['V'].items():
            V[v_idx][r] = coeff
        for w_idx, coeff in term['W'].items():
            W[w_idx][r] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
