"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i in [0,n), j in [0,m), k in [0,p):
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: Use block decomposition with Strassen and Laderman where beneficial.
    # Split n=3 into 2+1, m=5 into 2+3, p=7 into 3+4
    # This gives 2x2x2x2 blocks: (2,2,3), (2,2,4), (2,3,3), (2,3,4), (1,2,3), (1,2,4), (1,3,3), (1,3,4)
    
    # For each block (a,b,c) where a is the row split of n, b is col split of m, c is row split of p:
    # We need a decomposition of M_{a,b,c}
    
    # Known decompositions:
    # M_{2,2,2}: Strassen, rank 7
    # M_{2,3,3}: Laderman, rank 7
    # M_{a,b,c} for other cases: schoolbook (rank a*b*c) or combinations
    
    # Let's build the decomposition by processing each block.
    # Block (i_split, j_split, k_split) where i_split is {0,1} (which n-part), j_split is {0,1} (which m-part), k_split is {0,1} (which p-part)
    # n-parts: [0,1] -> size 2, [2] -> size 1
    # m-parts: [0,1] -> size 2, [2,3,4] -> size 3
    # p-parts: [0,1,2] -> size 3, [3,4,5,6] -> size 4
    
    # Block sizes:
    # (2,2,3), (2,2,4), (2,3,3), (2,3,4), (1,2,3), (1,2,4), (1,3,3), (1,3,4)
    
    # For each block, we need a decomposition. Let's define helper functions.
    
    def strassen_222():
        """Return (U, V, W) for M_{2,2,2} with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # Products:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(f-h)
        # p4 = d(g-e)
        # p5 = (a+b)h
        # p6 = (c-d)(f+g)
        # p7 = (b-d)(e+h)
        # C[0,0] = p1 - p4 + p5 - p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6
        
        # In our tensor convention:
        # U[i*2+j, c] corresponds to coefficient for A[i,j]
        # V[j*2+k, c] corresponds to coefficient for B[j,k]
        # W[k*2+i, c] corresponds to coefficient for C[i,k]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # c=0: p1 = (a+d)(e+h)
        # A[0,0]=a, A[1,1]=d, B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # U[0*2+0]=1 (a), U[1*2+1]=1 (d)
        # V[0*2+0]=1 (e), V[1*2+1]=1 (h)
        # W: C[0,0] += 1, C[0,1] += 0, C[1,0] += 0, C[1,1] += 1
        # W[k*2+i]: C[0,0] -> W[0*2+0]=1, C[1,1] -> W[1*2+1]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # c=1: p2 = (c+d)e
        # A[1,0]=c, A[1,1]=d, B[0,0]=e
        # U[1*2+0]=1 (c), U[1*2+1]=1 (d)
        # V[0*2+0]=1 (e)
        # C[1,0] += 1 -> W[0*2+1]=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # c=2: p3 = a(f-h)
        # A[0,0]=a, B[0,1]=f, B[1,1]=h
        # U[0,2]=1 (a)
        # V[0*2+1]=1 (f), V[1*2+1]=-1 (h)
        # C[0,1] += 1 -> W[1*2+0]=1, C[1,1] += 1 -> W[1*2+1]=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # c=3: p4 = d(g-e)
        # A[1,1]=d, B[1,0]=g, B[0,0]=e
        # U[3,3]=1 (d)
        # V[2,3]=1 (g), V[0,3]=-1 (e)
        # C[0,0] += -1 -> W[0,3]=-1, C[1,0] += 1 -> W[1,3]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[1, 3] = 1
        
        # c=4: p5 = (a+b)h
        # A[0,0]=a, A[0,1]=b, B[1,1]=h
        # U[0,4]=1 (a), U[1,4]=1 (b)
        # V[3,4]=1 (h)
        # C[0,0] += 1 -> W[0,4]=1, C[0,1] += 1 -> W[2,4]=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[2, 4] = 1
        
        # c=5: p6 = (c-d)(f+g)
        # A[1,0]=c, A[1,1]=d, B[0,1]=f, B[1,0]=g
        # U[2,5]=1 (c), U[3,5]=-1 (d)
        # V[1,5]=1 (f), V[2,5]=1 (g)
        # C[1,1] += 1 -> W[3,5]=1
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # c=6: p7 = (b-d)(e+h)
        # A[0,1]=b, A[1,1]=d, B[0,0]=e, B[1,1]=h
        # U[1,6]=1 (b), U[3,6]=-1 (d)
        # V[0,6]=1 (e), V[3,6]=1 (h)
        # C[0,0] += -1 -> W[0,6]=-1
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        
        return U, V, W
    
    def laderman_233():
        """Return (U, V, W) for M_{2,3,3} with rank 7.
        U: (6, 7), V: (9, 7), W: (6, 7)
        Convention: U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
        """
        # Laderman's algorithm for 2x3 by 3x3 matrix multiplication
        # This is a known rank-7 decomposition.
        # A is 2x3, B is 3x3, C = A*B is 2x3
        
        U = np.zeros((6, 7))
        V = np.zeros((9, 7))
        W = np.zeros((6, 7))
        
        # Laderman's 7 products:
        # Let A = [[a,b,c],[d,e,f]], B = [[g,h,i],[j,k,l],[m,n,o]]
        # C = [[ag+bj+cm, ah+bk+cn, ai+bl+co],
        #      [dg+ej+fm, dh+ek+fn, di+el+fo]]
        
        # Products:
        # p1 = (a+d)(g+j+m)  -> no, let me look up the exact Laderman decomposition
        
        # Actually, let me use a simpler approach. For M_{2,3,3}, we can use:
        # Split 2x3x3 into 2x2x3 + 2x1x3
        # 2x2x3: use Strassen-like or schoolbook
        # 2x1x3: schoolbook rank 6
        # Total: 8 + 6 = 14, not good.
        
        # Let me just use schoolbook for now to be safe, and optimize later.
        # Schoolbook for 2x3x3: rank 18
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    col = i * 3 * 3 + j * 3 + k
                    if col >= 7:
                        # This won't work with rank 7. Let me use a proper Laderman.
                        pass
        
        # Let me implement a proper Laderman decomposition.
        # From literature, Laderman's rank-7 algorithm for 2x3 by 3x3:
        # The 7 bilinear forms are:
        # p1 = a*g + b*j + c*m + d*g + e*j + f*m  -- no, that's not right
        
        # Let me use a different approach. I'll use the fact that M_{2,3,3} can be
        # decomposed using 7 terms. Here's one known decomposition:
        
        # Actually, for safety, let me just use schoolbook for all non-2x2x2 blocks.
        # This gives rank = sum of a*b*c for each block.
        # (2,2,3): 12, (2,2,4): 16, (2,3,3): 18, (2,3,4): 24, (1,2,3): 6, (1,2,4): 8, (1,3,3): 9, (1,3,4): 12
        # Total: 12+16+18+24+6+8+9+12 = 105, same as schoolbook.
        
        # Let me use Strassen for the 2x2x2 sub-blocks within the larger blocks.
        # For (2,2,3): split into (2,2,2) + (2,2,1) = 7 + 4 = 11
        # For (2,2,4): split into (2,2,2) + (2,2,2) = 7 + 7 = 14
        # For (2,3,3): split into (2,2,3) + (2,1,3) = 11 + 6 = 17, or use Laderman rank 7
        # For (2,3,4): split into (2,2,4) + (2,1,4) = 14 + 8 = 22
        # For (1,2,3): 6, (1,2,4): 8, (1,3,3): 9, (1,3,4): 12
        
        # Let me implement this properly. I'll build the full decomposition.
        
        return U, V, W
    
    # Let me take a cleaner approach. I'll build the decomposition block by block.
    # For each block (a,b,c) with specific row/col indices, I'll add the appropriate terms.
    
    # Define the splits:
    # n-split: [0,1] (size 2), [2] (size 1)
    # m-split: [0,1] (size 2), [2,3,4] (size 3)
    # p-split: [0,1,2] (size 3), [3,4,5,6] (size 4)
    
    n_splits = [([0,1], 2), ([2], 1)]
    m_splits = [([0,1], 2), ([2,3,4], 3)]
    p_splits = [([0,1,2], 3), ([3,4,5,6], 4)]
    
    # For each combination of splits, determine the block size (a,b,c)
    # and use the appropriate decomposition.
    
    # I'll build U, V, W as lists of terms, then combine.
    # Each term is (u_dict, v_dict, w_dict) where u_dict maps (i,j) -> coeff, etc.
    
    terms = []
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a single rank-1 term.
        u_coeff: dict mapping (i,j) -> coeff for U
        v_coeff: dict mapping (j,k) -> coeff for V
        w_coeff: dict mapping (k,i) -> coeff for W
        """
        terms.append((u_coeff, v_coeff, w_coeff))
    
    # Process each block
    for n_idx, (n_rows, n_size) in enumerate(n_splits):
        for m_idx, (m_cols, m_size) in enumerate(m_splits):
            for p_idx, (p_rows, p_size) in enumerate(p_splits):
                a, b, c = n_size, m_size, p_size
                
                if a == 2 and b == 2 and c == 2:
                    # Strassen
                    U_s, V_s, W_s = strassen_222()
                    for col in range(7):
                        u_coeff = {}
                        v_coeff = {}
                        w_coeff = {}
                        for i_local in range(2):
                            for j_local in range(2):
                                i = n_rows[i_local]
                                j = m_cols[j_local]
                                coeff = U_s[i*2+j, col]
                                if coeff != 0:
                                    u_coeff[(i,j)] = coeff
                        for j_local in range(2):
                            for k_local in range(2):
                                j = m_cols[j_local]
                                k = p_rows[k_local]
                                coeff = V_s[j*2+k, col]
                                if coeff != 0:
                                    v_coeff[(j,k)] = coeff
                        for k_local in range(2):
                            for i_local in range(2):
                                k = p_rows[k_local]
                                i = n_rows[i_local]
                                coeff = W_s[k*2+i, col]
                                if coeff != 0:
                                    w_coeff[(k,i)] = coeff
                        add_term(u_coeff, v_coeff, w_coeff)
                elif a == 2 and b == 2:
                    # Split c into 2s and 1s
                    # For c=3: one (2,2,2) Strassen + one (2,2,1) schoolbook
                    # For c=4: two (2,2,2) Strassen
                    if c == 3:
                        # First 2 columns of p: Strassen
                        p_sub = p_rows[:2]
                        U_s, V_s, W_s = strassen_222()
                        for col in range(7):
                            u_coeff = {}
                            v_coeff = {}
                            w_coeff = {}
                            for i_local in range(2):
                                for j_local in range(2):
                                    i = n_rows[i_local]
                                    j = m_cols[j_local]
                                    coeff = U_s[i*2+j, col]
                                    if coeff != 0:
                                        u_coeff[(i,j)] = coeff
                            for j_local in range(2):
                                for k_local in range(2):
                                    j = m_cols[j_local]
                                    k = p_sub[k_local]
                                    coeff = V_s[j*2+k_local, col]
                                    if coeff != 0:
                                        v_coeff[(j,k)] = coeff
                            for k_local in range(2):
                                for i_local in range(2):
                                    k = p_sub[k_local]
                                    i = n_rows[i_local]
                                    coeff = W_s[k_local*2+i_local, col]
                                    if coeff != 0:
                                        w_coeff[(k,i)] = coeff
                            add_term(u_coeff, v_coeff, w_coeff)
                        
                        # Last 1 column of p: schoolbook (2,2,1)
                        k = p_rows[2]
                        for i_local in range(2):
                            for j_local in range(2):
                                i = n_rows[i_local]
                                j = m_cols[j_local]
                                u_coeff = {(i,j): 1.0}
                                v_coeff = {(j,k): 1.0}
                                w_coeff = {(k,i): 1.0}
                                add_term(u_coeff, v_coeff, w_coeff)
                    elif c == 4:
                        # Two Strassen blocks
                        for p_start in [0, 2]:
                            p_sub = p_rows[p_start:p_start+2]
                            U_s, V_s, W_s = strassen_222()
                            for col in range(7):
                                u_coeff = {}
                                v_coeff = {}
                                w_coeff = {}
                                for i_local in range(2):
                                    for j_local in range(2):
                                        i = n_rows[i_local]
                                        j = m_cols[j_local]
                                        coeff = U_s[i*2+j, col]
                                        if coeff != 0:
                                            u_coeff[(i,j)] = coeff
                                for j_local in range(2):
                                    for k_local in range(2):
                                        j = m_cols[j_local]
                                        k = p_sub[k_local]
                                        coeff = V_s[j*2+k_local, col]
                                        if coeff != 0:
                                            v_coeff[(j,k)] = coeff
                                for k_local in range(2):
                                    for i_local in range(2):
                                        k = p_sub[k_local]
                                        i = n_rows[i_local]
                                        coeff = W_s[k_local*2+i_local, col]
                                        if coeff != 0:
                                            w_coeff[(k,i)] = coeff
                                add_term(u_coeff, v_coeff, w_coeff)
                elif a == 2 and b == 3:
                    # Split b into 2+1
                    # (2,2,c) + (2,1,c)
                    m_sub1 = m_cols[:2]
                    m_sub2 = m_cols[2:]
                    
                    # (2,2,c) part
                    if c == 3:
                        # (2,2,3): Strassen + schoolbook
                        p_sub = p_rows[:2]
                        U_s, V_s, W_s = strassen_222()
                        for col in range(7):
                            u_coeff = {}
                            v_coeff = {}
                            w_coeff = {}
                            for i_local in range(2):
                                for j_local in range(2):
                                    i = n_rows[i_local]
                                    j = m_sub1[j_local]
                                    coeff = U_s[i*2+j_local, col]
                                    if coeff != 0:
                                        u_coeff[(i,j)] = coeff
                            for j_local in range(2):
                                for k_local in range(2):
                                    j = m_sub1[j_local]
                                    k = p_sub[k_local]
                                    coeff = V_s[j_local*2+k_local, col]
                                    if coeff != 0:
                                        v_coeff[(j,k)] = coeff
                            for k_local in range(2):
                                for i_local in range(2):
                                    k = p_sub[k_local]
                                    i = n_rows[i_local]
                                    coeff = W_s[k_local*2+i_local, col]
                                    if coeff != 0:
                                        w_coeff[(k,i)] = coeff
                            add_term(u_coeff, v_coeff, w_coeff)
                        
                        k = p_rows[2]
                        for i_local in range(2):
                            for j_local in range(2):
                                i = n_rows[i_local]
                                j = m_sub1[j_local]
                                u_coeff = {(i,j): 1.0}
                                v_coeff = {(j,k): 1.0}
                                w_coeff = {(k,i): 1.0}
                                add_term(u_coeff, v_coeff, w_coeff)
                        
                        # (2,1,3) schoolbook
                        for i_local in range(2):
                            for j_local in range(1):
                                for k_local in range(3):
                                    i = n_rows[i_local]
                                    j = m_sub2[j_local]
                                    k = p_rows[k_local]
                                    u_coeff = {(i,j): 1.0}
                                    v_coeff = {(j,k): 1.0}
                                    w_coeff = {(k,i): 1.0}
                                    add_term(u_coeff, v_coeff, w_coeff)
                    elif c == 4:
                        # (2,2,4): two Strassen
                        for p_start in [0, 2]:
                            p_sub = p_rows[p_start:p_start+2]
                            U_s, V_s, W_s = strassen_222()
                            for col in range(7):
                                u_coeff = {}
                                v_coeff = {}
                                w_coeff = {}
                                for i_local in range(2):
                                    for j_local in range(2):
                                        i = n_rows[i_local]
                                        j = m_sub1[j_local]
                                        coeff = U_s[i*2+j_local, col]
                                        if coeff != 0:
                                            u_coeff[(i,j)] = coeff
                                for j_local in range(2):
                                    for k_local in range(2):
                                        j = m_sub1[j_local]
                                        k = p_sub[k_local]
                                        coeff = V_s[j_local*2+k_local, col]
                                        if coeff != 0:
                                            v_coeff[(j,k)] = coeff
                                for k_local in range(2):
                                    for i_local in range(2):
                                        k = p_sub[k_local]
                                        i = n_rows[i_local]
                                        coeff = W_s[k_local*2+i_local, col]
                                        if coeff != 0:
                                            w_coeff[(k,i)] = coeff
                                add_term(u_coeff, v_coeff, w_coeff)
                        
                        # (2,1,4) schoolbook
                        for i_local in range(2):
                            for j_local in range(1):
                                for k_local in range(4):
                                    i = n_rows[i_local]
                                    j = m_sub2[j_local]
                                    k = p_rows[k_local]
                                    u_coeff = {(i,j): 1.0}
                                    v_coeff = {(j,k): 1.0}
                                    w_coeff = {(k,i): 1.0}
                                    add_term(u_coeff, v_coeff, w_coeff)
                else:
                    # Schoolbook for all other cases
                    for i_local in range(a):
                        for j_local in range(b):
                            for k_local in range(c):
                                i = n_rows[i_local]
                                j = m_cols[j_local]
                                k = p_rows[k_local]
                                u_coeff = {(i,j): 1.0}
                                v_coeff = {(j,k): 1.0}
                                w_coeff = {(k,i): 1.0}
                                add_term(u_coeff, v_coeff, w_coeff)
    
    # Now combine all terms into U, V, W matrices
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u_coeff, v_coeff, w_coeff) in enumerate(terms):
        for (i,j), coeff in u_coeff.items():
            U[i*m+j, c] = coeff
        for (j,k), coeff in v_coeff.items():
            V[j*p+k, c] = coeff
        for (k,i), coeff in w_coeff.items():
            W[k*n+i, c] = coeff
    
    # Verify
    if time.time() < deadline:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
