"""Matrix multiplication 4,5,6 decomposition via block splitting."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return the 7 rank-1 terms of the 2x2 Strassen decomposition.
    Each term is (U_entry, V_entry, W_entry) where each is a 4-vector
    representing a row in the factor matrix (indexed by i*m+j etc).
    For 2x2: U has 4 rows (i*2+j), V has 4 rows (j*2+k), W has 4 rows (k*2+i).
    """
    # Strassen: M1..M7 with specific patterns
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+d)(h), M6 = (b-c)(g), M7 = (b+d)(e+f-h-g)
    # Result: [[M1+M4-M5+M7, M1+M3-M5-M6], [M2+M4+M5-M7, M2+M3-M4+M6]]
    
    # For each multiplication term, we need to express as sum of rank-1 tensors.
    # The 2x2x2 tensor has entries T[i,j,k] = 1 if the product C[i,k] involves A[i,j]*B[j,k].
    # Standard: T[i,j,k] = 1 for all (i,j,k) in {0,1}^3 (that's schoolbook rank 8).
    # Strassen replaces this with 7 rank-1 terms with +/-1 entries.
    
    # The 7 Strassen rank-1 terms for 2x2x2:
    # Each is (u, v, w) where u, v, w are in {0, 1, -1}^4 (for 4 rows each)
    # u is indexed by (i,j) -> i*2+j, v by (j,k) -> j*2+k, w by (k,i) -> k*2+i
    
    # Term 1: (a+b)(e+h)
    # u: a=1 at (0,0), b=1 at (0,1) -> [1,1,0,0]
    # v: e=1 at (0,0), h=1 at (1,1) -> [1,0,0,1]
    # w: appears as +M1 at C[0,0], +M1 at C[0,1] -> w indexed by (k,i): C[0,0] is k=0,i=0 -> idx 0; C[0,1] is k=1,i=0 -> idx 2
    # Actually w[k,i] for C[i,k]: C[0,0] -> k=0,i=0 -> w[0*2+0]=w[0]; C[0,1] -> k=1,i=0 -> w[1*2+0]=w[2]
    # So w = [1, 0, 1, 0]
    
    # Let me be more careful. The convention is W[k*n+i, r].
    # For n=2: W has 4 entries indexed by k*2+i.
    # C[i,k] gets contribution from w[k*2+i].
    
    # M1 = (a+b)(e+h) contributes to C[0,0] (+1), C[0,1] (+1)
    # So w[0*2+0] = w[0] = +1, w[1*2+0] = w[2] = +1, others 0
    # u: a at (i=0,j=0) -> u[0*2+0]=u[0]=1; b at (i=0,j=1) -> u[0*2+1]=u[1]=1
    # v: e at (j=0,k=0) -> v[0*2+0]=v[0]=1; h at (j=1,k=1) -> v[1*2+1]=v[3]=1
    
    terms = []
    # M1: (a+b)(e+h) -> C[0,0]+=1, C[0,1]+=1
    terms.append(([1,1,0,0], [1,0,0,1], [1,0,1,0]))
    
    # M2: (c+d)(e) -> C[1,0]+=1, C[1,1]+=1
    # u: c at (1,0)->u[2]=1, d at (1,1)->u[3]=1
    # v: e at (0,0)->v[0]=1
    # w: C[1,0]: k=0,i=1 -> w[0*2+1]=w[1]=1; C[1,1]: k=1,i=1 -> w[1*2+1]=w[3]=1
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
    
    # M3: a(f-h) -> C[0,1]+=1, C[0,0]-=1  (a*f at C[0,1], a*(-h) at C[0,0])
    # u: a at (0,0)->u[0]=1
    # v: f at (0,1)->v[1]=1, -h at (1,1)->v[3]=-1
    # w: C[0,1]: k=1,i=0 -> w[2]=1; C[0,0]: k=0,i=0 -> w[0]=-1
    terms.append(([1,0,0,0], [0,1,0,-1], [-1,0,1,0]))
    
    # M4: d(g-e) -> C[1,0]+=1, C[1,1]-=1
    # u: d at (1,1)->u[3]=1
    # v: g at (1,0)->v[2]=1, -e at (0,0)->v[0]=-1
    # w: C[1,0]: k=0,i=1 -> w[1]=1; C[1,1]: k=1,i=1 -> w[3]=-1
    terms.append(([0,0,0,1], [-1,0,1,0], [0,1,0,-1]))
    
    # M5: (a+d)(h) -> C[0,1]-=1, C[1,1]+=1
    # u: a at (0,0)->u[0]=1, d at (1,1)->u[3]=1
    # v: h at (1,1)->v[3]=1
    # w: C[0,1]: k=1,i=0 -> w[2]=-1; C[1,1]: k=1,i=1 -> w[3]=1
    terms.append(([1,0,0,1], [0,0,0,1], [0,0,-1,1]))
    
    # M6: (b-c)(g) -> C[0,1]-=1, C[1,0]+=1
    # u: b at (0,1)->u[1]=1, -c at (1,0)->u[2]=-1
    # v: g at (1,0)->v[2]=1
    # w: C[0,1]: k=1,i=0 -> w[2]=-1; C[1,0]: k=0,i=1 -> w[1]=1
    terms.append(([0,1,-1,0], [0,0,1,0], [0,1,0,-1]))
    
    # M7: (b+d)(e+f-h-g) -> C[0,0]+=1, C[1,1]-=1
    # u: b at (0,1)->u[1]=1, d at (1,1)->u[3]=1
    # v: e at (0,0)->v[0]=1, f at (0,1)->v[1]=1, -h at (1,1)->v[3]=-1, -g at (1,0)->v[2]=-1
    # w: C[0,0]: k=0,i=0 -> w[0]=1; C[1,1]: k=1,i=1 -> w[3]=-1
    terms.append(([0,1,0,1], [1,1,-1,-1], [1,0,0,-1]))
    
    return terms

def _kronecker_product_terms(terms1, terms2, n1, n2):
    """
    Given two sets of rank-1 terms for matrices of sizes n1 and n2,
    produce rank-1 terms for the tensor product (size n1*n2).
    
    If T1 has terms (u1_r, v1_r, w1_r) for r=1..R1 (each vector length n1^2),
    and T2 has terms (u2_s, v2_s, w2_s) for s=1..R2,
    then T1 ⊗ T2 has R1*R2 terms where:
    term (r,s): u = u1_r ⊗ u2_s, v = v1_r ⊗ v2_s, w = w1_r ⊗ w2_s
    (Kronecker product of the vectors)
    """
    R1 = len(terms1)
    R2 = len(terms2)
    result = []
    for r in range(R1):
        u1, v1, w1 = terms1[r]
        for s in range(R2):
            u2, v2, w2 = terms2[s]
            # Kronecker product: (u1 ⊗ u2)[i*n2 + j] = u1[i] * u2[j]
            u = [u1[i] * u2[j] for i in range(n1) for j in range(n2)]
            v = [v1[i] * v2[j] for i in range(n1) for j in range(n2)]
            w = [w1[i] * w2[j] for i in range(n1) for j in range(n2)]
            result.append((u, v, w))
    return result

def _strassen_4x4():
    """Get 49 rank-1 terms for 4x4x4 matrix multiplication via 2x2 Strassen tensor product."""
    s2 = _strassen_2x2()
    return _kronecker_product_terms(s2, s2, 2, 2)

def _schoolbook_block(n, m, p, i_off, j_off, k_off, U, V, W):
    """Add schoolbook rank-1 terms for an n×m×p block at offset (i_off, j_off, k_off)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = len(U)
                U.append([0] * (U[0][0] if False else len(U[0])) if U else [0])
                # Actually let me build differently - collect all terms first
                pass

def construct():
    """
    Decompose 4x5x6 matrix multiplication.
    Split: 4x5 = 4x4 + 4x1 (in j dimension)
           5x6 = 4x4 + 4x2 + 1x4 + 1x2 (in j,k dimensions)
    
    Blocks:
    1. i in [0,4), j in [0,4), k in [0,4): 4x4x4 -> Strassen rank 49
    2. i in [0,4), j in [0,4), k in [4,6): 4x4x2 -> schoolbook rank 8
    3. i in [0,4), j in [4,5), k in [0,4): 4x1x4 -> schoolbook rank 4
    4. i in [0,4), j in [4,5), k in [4,6): 4x1x2 -> schoolbook rank 2
    
    Total rank: 49 + 8 + 4 + 2 = 63
    """
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    all_terms = []  # list of (u_vec, v_vec, w_vec)
    
    # Block 1: 4x4x4 Strassen, offset i=0, j=0, k=0
    s4 = _strassen_4x4()
    for (u, v, w) in s4:
        # u is length 16, needs to be length 20 (pad with 0s)
        # v is length 16, needs to be length 30 (pad with 0s)
        # w is length 16, needs to be length 24 (pad with 0s)
        u_full = list(u) + [0] * (U_rows - len(u))
        v_full = list(v) + [0] * (V_rows - len(v))
        w_full = list(w) + [0] * (W_rows - len(w))
        all_terms.append((u_full, v_full, w_full))
    
    # Block 2: 4x4x2 schoolbook, offset i=0, j=0, k=4
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u = [0] * U_rows
                v = [0] * V_rows
                w = [0] * W_rows
                u[i * m + j] = 1
                v[j * p + (k + 4)] = 1
                w[(k + 4) * n + i] = 1
                all_terms.append((u, v, w))
    
    # Block 3: 4x1x4 schoolbook, offset i=0, j=4, k=0
    for i in range(4):
        for k in range(4):
            u = [0] * U_rows
            v = [0] * V_rows
            w = [0] * W_rows
            u[i * m + 4] = 1
            v[4 * p + k] = 1
            w[k * n + i] = 1
            all_terms.append((u, v, w))
    
    # Block 4: 4x1x2 schoolbook, offset i=0, j=4, k=4
    for i in range(4):
        for k in range(2):
            u = [0] * U_rows
            v = [0] * V_rows
            w = [0] * W_rows
            u[i * m + 4] = 1
            v[4 * p + (k + 4)] = 1
            w[(k + 4) * n + i] = 1
            all_terms.append((u, v, w))
    
    rank = len(all_terms)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for r, (u, v, w) in enumerate(all_terms):
        for i in range(U_rows):
            U[i][r] = u[i]
        for j in range(V_rows):
            V[j][r] = v[j]
        for k in range(W_rows):
            W[k][r] = w[k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

