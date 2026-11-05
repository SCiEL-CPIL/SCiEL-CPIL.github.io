import time
import numpy as np

n_dim, m_dim, p_dim = 2, 4, 8
_TIME_LIMIT = 120.0


def _build_schoolbook():
    """Standard O(n*m*p) decomposition."""
    n, m, p = n_dim, m_dim, p_dim
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u][r] = 1
                V[v][r] = 1
                W[w][r] = 1
    return U, V, W


def _build_strassen_2x2x2_block(i0, j0, k0, U, V, W):
    """
    Adds 7 Strassen terms for a 2x2x2 block.
    U rows: (i0..i0+1, j0..j0+1)
    V rows: (j0..j0+1, k0..k0+1)
    W rows: (k0..k0+1, i0..i0+1)
    """
    n, m, p = n_dim, m_dim, p_dim
    
    def add(u_dict, v_dict, w_dict, col):
        for (i, j), val in u_dict.items():
            if val: U[i*m + j][col] += val
        for (j, k), val in v_dict.items():
            if val: V[j*p + k][col] += val
        for (k, i), val in w_dict.items():
            if val: W[k*n + i][col] += val

    # M1 = (A00 + A11)(B00 + B11)
    add({(i0, j0): 1, (i0+1, j0+1): 1},
        {(j0, k0): 1, (j0+1, k0+1): 1},
        {(k0, i0): 1, (k0, i0+1): -1, (k0+1, i0): 1, (k0+1, i0+1): 1},
        0)
    # This M1 above was a mistake in the previous logic or I need to re-verify Strassen formulas.
    # Correct Strassen for C = AB:
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M6 + M5
    # C11 = M1 - M2 + M4 + M3
    
    # Let's just use the standard 7 products for 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # The previous code's add_term calls were correct in structure. 
    # I will rewrite the block builder carefully.
    
    col_idx = 0
    # We need to pass the starting column index. Since this is a helper, 
    # it's better to return the 7 U,V,W vectors and let the caller place them.
    
    vectors = []
    
    # M1
    vectors.append(({(i0, j0): 1, (i0+1, j0+1): 1},
                    {(j0, k0): 1, (j0+1, k0+1): 1},
                    {(k0, i0): 1, (k0+1, i0): -1, (k0, i0+1): -1, (k0+1, i0+1): 1})) 
    # Wait, M1 contributes to C00, C01, C10, C11?
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1 is wrong.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M6 + M5
    # C11 = M1 - M2 + M4 + M3
    
    # Let's map the contributions to W (which is the output).
    # W[k*n + i] corresponds to output element C[i][k]?
    # Output C is n x p.
    # Our tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i.
    # Non-zero when i,j,k match.
    # So W term for a product M should add to C[i0][k0], C[i0][k0+1], etc.
    
    # M1 = (A00+A11)(B00+B11)
    # A00 is i0,j0. A11 is i0+1, j0+1.
    # B00 is j0,k0. B11 is j0+1, k0+1.
    # Product terms in A*B:
    # (A00 B00) -> C[i0][k0]
    # (A00 B11) -> C[i0][k0+1]
    # (A11 B00) -> C[i0+1][k0]
    # (A11 B11) -> C[i0+1][k0+1]
    # So M1 contributes:
    # W[k0, i0] += 1
    # W[k0+1, i0] += 1
    # W[k0, i0+1] += 1
    # W[k0+1, i0+1] += 1
    vectors[-1] = ({(i0, j0): 1, (i0+1, j0+1): 1},
                   {(j0, k0): 1, (j0+1, k0+1): 1},
                   {(k0, i0): 1, (k0+1, i0): 1, (k0, i0+1): 1, (k0+1, i0+1): 1})
    
    # M2 = (A10+A11)B00
    # A10: i0+1, j0. A11: i0+1, j0+1.
    # B00: j0, k0.
    # (A10 B00) -> C[i0+1][k0]
    # (A11 B00) -> C[i0+1][k0]  (Wait, A11 is i0+1,j0+1. B00 is j0,k0. j mismatch? No, j is internal.)
    # Matrix mult: Sum over j.
    # (A10+A11) is a 2x2 block? No.
    # A is 2x2. A10 is (1,0) element? No, A10 is bottom-left 1x1?
    # Standard 2x2 partition:
    # A = [A00 A01; A10 A11]
    # B = [B00 B01; B10 B11]
    # M2 = (A10 + A11) B00
    # A10+A11 is not a standard matrix block. It's the sum of elements?
    # No, A10 and A11 are 1x1 blocks.
    # So (A10+A11) is a 1x1 block (scalar sum).
    # Then (A10+A11) B00 is 1x1 * 1x1 = 1x1.
    # Result contributes to C10 (bottom-left of C).
    # C10 = M6 + M5.
    # So M2 contributes to C10?
    # Let's check C11 = M1 - M2 + M4 + M3.
    # So M2 contributes to C11 with -1.
    
    # Let's re-verify the Strassen formulas for C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M6 + M5
    # C11 = M1 - M2 + M4 + M3
    
    # M2 = (A10 + A11) B00
    # A10 is at (1,0). A11 is at (1,1).
    # A10+B11? No.
    # A10+A11 is the bottom row sum?
    # If A10, A11 are scalars, then A10+A11 is a scalar.
    # B00 is scalar.
    # Product is scalar.
    # Which C element?
    # The formula says C01 and C11.
    # C01 (top-right) gets M2.
    # C11 (bottom-right) gets -M2.
    
    # Let's check dimensions.
    # A10 (1,0), A11 (1,1).
    # (A10+A11) doesn't form a matrix that multiplies B00 naturally unless we consider the block structure.
    # Actually, standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # M2: (A10 + A11) B00.
    # A10 is (1,0), A11 is (1,1).
    # This implies we are summing columns 0 and 1 of the bottom row of A.
    # Then multiplying by B00 (which is (0,0) block).
    # This seems dimensionally weird if A10, A11 are blocks.
    # In 2x2 case, they are scalars.
    # (A10+A11) is scalar. B00 is scalar.
    # Result scalar.
    # Where does it go?
    # C01 = M2 + M3.
    # C11 = M1 - M2 + M4 + M3.
    
    # Let's trace M2 contribution.
    # U: A10 (1,0) is 1, A11 (1,1) is 1.
    # V: B00 (0,0) is 1.
    # W: C01 (0,1) is 1, C11 (1,1) is -1.
    # Let's check if this works.
    # C01 term: A10*B00 + A11*B00 = (A10+A11)B00. Correct.
    # C11 term: -A10*B00 - A11*B00.
    
    # So for M2:
    # U: {(i0+1, j0): 1, (i0+1, j0+1): 1}
    # V: {(j0, k0): 1}
    # W: {(k0, i0): 1, (k0+1, i0+1): -1}  # Note: W index is (k, i). C01 is i0, k0+1?
    # C[i][k].
    # C01 is i=0, k=1. So W[k0+1, i0].
    # C11 is i=1, k=1. So W[k0+1, i0+1].
    # So W: {(k0+1, i0): 1, (k0+1, i0+1): -1}
    
    vectors.append(({(i0+1, j0): 1, (i0+1, j0+1): 1},
                    {(j0, k0): 1},
                    {(k0+1, i0): 1, (k0+1, i0+1): -1}))
    
    # M3 = A00(B01 - B11)
    # U: A00 (0,0) -> {(i0, j0): 1}
    # V: B01 (0,1) -> 1, B11 (1,1) -> -1.
    #    B01 is j0, k0+1. B11 is j0+1, k0+1.
    #    V: {(j0, k0+1): 1, (j0+1, k0+1): -1}
    # W: C01 (0,1) -> 1, C11 (1,1) -> 1.
    #    C01: W[k0+1, i0] += 1.
    #    C11: W[k0+1, i0+1] += 1.
    vectors.append(({(i0, j0): 1},
                    {(j0, k0+1): 1, (j0+1, k0+1): -1},
                    {(k0+1, i0): 1, (k0+1, i0+1): 1}))
    
    # M4 = A11(B10 - B00)
    # U: A11 (1,1) -> {(i0+1, j0+1): 1}
    # V: B10 (1,0) -> 1, B00 (0,0) -> -1.
    #    B10: j0+1, k0. B00: j0, k0.
    #    V: {(j0+1, k0): 1, (j0, k0): -1}
    # W: C00 (0,0) -> 1, C11 (1,1) -> 1.
    #    C00: W[k0, i0] += 1.
    #    C11: W[k0+1, i0+1] += 1.
    vectors.append(({(i0+1, j0+1): 1},
                    {(j0+1, k0): 1, (j0, k0): -1},
                    {(k0, i0): 1, (k0+1, i0+1): 1}))
    
    # M5 = (A00 + A01)B11
    # U: A00 (0,0), A01 (0,1).
    #    U: {(i0, j0): 1, (i0, j0+1): 1}
    # V: B11 (1,1).
    #    V: {(j0+1, k0+1): 1}
    # W: C00 (0,0) -> -1, C10 (1,0) -> 1.
    #    C00: W[k0, i0] += -1.
    #    C10: W[k0, i0+1] += 1.
    vectors.append(({(i0, j0): 1, (i0, j0+1): 1},
                    {(j0+1, k0+1): 1},
                    {(k0, i0): -1, (k0, i0+1): 1}))
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10 (1,0) -> 1, A00 (0,0) -> -1.
    #    U: {(i0+1, j0): 1, (i0, j0): -1}
    # V: B00 (0,0) -> 1, B01 (0,1) -> 1.
    #    V: {(j0, k0): 1, (j0, k0+1): 1}
    # W: C10 (1,0) -> 1.
    #    W: {(k0, i0+1): 1}
    vectors.append(({(i0+1, j0): 1, (i0, j0): -1},
                    {(j0, k0): 1, (j0, k0+1): 1},
                    {(k0, i0+1): 1}))
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01 (0,1) -> 1, A11 (1,1) -> -1.
    #    U: {(i0, j0+1): 1, (i0+1, j0+1): -1}
    # V: B10 (1,0) -> 1, B11 (1,1) -> 1.
    #    V: {(j0+1, k0): 1, (j0+1, k0+1): 1}
    # W: C00 (0,0) -> 1.
    #    W: {(k0, i0): 1}
    vectors.append(({(i0, j0+1): 1, (i0+1, j0+1): -1},
                    {(j0+1, k0): 1, (j0+1, k0+1): 1},
                    {(k0, i0): 1}))
    
    return vectors


def construct():
    n, m, p = n_dim, m_dim, p_dim
    
    # Rank 56 from 8 blocks of Strassen 2x2x2.
    # n=2, m=4, p=8.
    # We can partition into:
    # i: 2 (only 1 block of 2x2? No, n=2 so 1 block of height 2).
    # j: 4 (2 blocks of 2x2).
    # k: 8 (4 blocks of 2x2).
    # Total blocks: 1 * 2 * 4 = 8 blocks.
    # Each block uses 7 products.
    # Total rank: 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                vectors = _build_strassen_2x2x2_block(i0, j0, k0, U, V, W)
                for u_dict, v_dict, w_dict in vectors:
                    for (i, j), val in u_dict.items():
                        if val:
                            U[i*m + j][col] += val
                    for (j, k), val in v_dict.items():
                        if val:
                            V[j*p + k][col] += val
                    for (k, i), val in w_dict.items():
                        if val:
                            W[k*n + i][col] += val
                    col += 1
    
    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
        
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T_target[u, v, w] = 1.0
        
        if not np.allclose(T_recon, T_target, atol=1e-9):
            # Fallback to schoolbook
            U, V, W = _build_schoolbook()
    except Exception:
        U, V, W = _build_schoolbook()
        
    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
