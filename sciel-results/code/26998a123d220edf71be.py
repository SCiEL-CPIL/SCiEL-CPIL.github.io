import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We aim for Rank 71 using 4 Strassen blocks of size 2x2x2
    # and schoolbook for the remainder.
    
    # Initialize U, V, W as lists of lists (or numpy arrays)
    # Max rank estimate: 75 (schoolbook). We will build up to 71.
    # It's safer to use a list of columns and then convert to numpy array.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vals, v_vals, w_vals):
        """
        u_vals, v_vals, w_vals are dicts mapping index to value.
        Index for U: i*m + j
        Index for V: j*p + k
        Index for W: k*n + i
        """
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for idx, val in u_vals.items():
            u_col[idx] = val
        for idx, val in v_vals.items():
            v_col[idx] = val
        for idx, val in w_vals.items():
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    # Strassen Block Definition
    # Local indices:
    # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C: (0,0)=i, (0,1)=j, (1,0)=k, (1,1)=l
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output:
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 - M6
    
    def add_strassen_block(i0, j0, k0):
        # 1. M1 = (a+d)(e+h)
        add_col(
            {i0*m + j0: 1, (i0+1)*m + (j0+1): 1},
            {j0*p + k0: 1, (j0+1)*p + (k0+1): 1},
            {k0*n + i0: 1, (k0+1)*n + (i0+1): 1} # Wait, W index is k*n+i
            # i is local 0,0 -> global i0, k0. Index k0*n + i0.
            # l is local 1,1 -> global i0+1, k0+1. Index (k0+1)*n + (i0+1).
        )
        
        # 2. M2 = (c+d)e
        add_col(
            {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): 1}, # c is (1,0) local? No.
            # A: a(0,0), b(0,1), c(1,0), d(1,1)
            # c is (1,0) -> i0+1, j0.
            # d is (1,1) -> i0+1, j0+1.
            # Correction:
            { (i0+1)*m + j0: 1, (i0+1)*m + (j0+1): 1 },
            { j0*p + k0: 1 }, # e is (0,0) -> j0, k0
            { (k0+1)*n + (i0+1): 1 } # k is local (1,0) -> i0+1, k0+1
        )
        
        # 3. M3 = a(f-h)
        add_col(
            { i0*m + j0: 1 }, # a is (0,0)
            { (j0+1)*p + k0: 1, (j0+1)*p + (k0+1): -1 }, # f is (0,1), h is (1,1)
            { k0*n + (i0+1): 1, (k0+1)*n + (i0+1): 1 } # j is local (0,1) -> i0, k0+1; l is local (1,1) -> i0+1, k0+1
            # Wait, j is local (0,1) in C?
            # C: i(0,0), j(0,1), k(1,0), l(1,1)
            # j corresponds to local (0,1) in C.
            # Global i = i0, k = k0+1. Index (k0+1)*n + i0.
            # l corresponds to local (1,1) in C.
            # Global i = i0+1, k = k0+1. Index (k0+1)*n + (i0+1).
            # So: { (k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1 }
        )
        # Re-doing M3 W part carefully.
        # M3 contributes to j and l.
        # j (local 0,1) -> global i0, k0+1. Index (k0+1)*n + i0.
        # l (local 1,1) -> global i0+1, k0+1. Index (k0+1)*n + (i0+1).
        # I will rewrite the add_col calls to be safer.
        
    # Let's restart the Strassen block definition with explicit indices to avoid errors.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        for idx, val in u_dict.items(): u_col[idx] = val
        for idx, val in v_dict.items(): v_col[idx] = val
        for idx, val in w_dict.items(): w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def add_strassen(i0, j0, k0):
        # M1 = (a+d)(e+h) -> i + l
        # a: i0, j0. d: i0+1, j0+1.
        # e: j0, k0. h: j0+1, k0+1.
        # i: i0, k0. l: i0+1, k0+1.
        add_term(
            {i0*m + j0: 1, (i0+1)*m + (j0+1): 1},
            {j0*p + k0: 1, (j0+1)*p + (k0+1): 1},
            {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        )
        
        # M2 = (c+d)e -> k
        # c: i0+1, j0. d: i0+1, j0+1.
        # e: j0, k0.
        # k: i0+1, k0+1.
        add_term(
            {(i0+1)*m + j0: 1, (i0+1)*m + (j0+1): 1},
            {j0*p + k0: 1},
            {(k0+1)*n + (i0+1): 1}
        )
        
        # M3 = a(f-h) -> j + l
        # a: i0, j0.
        # f: j0, k0+1. h: j0+1, k0+1.
        # j: i0, k0+1. l: i0+1, k0+1.
        add_term(
            {i0*m + j0: 1},
            {j0*p + (k0+1): 1, (j0+1)*p + (k0+1): -1},
            {(k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1}
        )
        
        # M4 = d(g-e) -> i + k
        # d: i0+1, j0+1.
        # g: j0+1, k0. e: j0, k0.
        # i: i0, k0. k: i0+1, k0+1.
        add_term(
            {(i0+1)*m + (j0+1): 1},
            {(j0+1)*p + k0: 1, j0*p + k0: -1},
            {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        )
        
        # M5 = (a+b)h -> i + j
        # a: i0, j0. b: i0, j0+1.
        # h: j0+1, k0+1.
        # i: i0, k0. j: i0, k0+1.
        add_term(
            {i0*m + j0: 1, i0*m + (j0+1): 1},
            {(j0+1)*p + (k0+1): 1},
            {k0*n + i0: 1, (k0+1)*n + i0: 1}
        )
        
        # M6 = (c-a)(e+f) -> l
        # c: i0+1, j0. a: i0, j0.
        # e: j0, k0. f: j0, k0+1.
        # l: i0+1, k0+1.
        add_term(
            {(i0+1)*m + j0: 1, i0*m + j0: -1},
            {j0*p + k0: 1, j0*p + (k0+1): 1},
            {(k0+1)*n + (i0+1): 1}
        )
        
        # M7 = (b-d)(g+h) -> i + l
        # b: i0, j0+1. d: i0+1, j0+1.
        # g: j0+1, k0. h: j0+1, k0+1.
        # i: i0, k0. l: i0+1, k0+1.
        add_term(
            {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): -1},
            {(j0+1)*p + k0: 1, (j0+1)*p + (k0+1): 1},
            {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        )

    # Apply 4 Strassen blocks
    # i0=0, j0=0, k0=0
    add_strassen(0, 0, 0)
    # i0=0, j0=2, k0=0
    add_strassen(0, 2, 0)
    # i0=0, j0=0, k0=2
    add_strassen(0, 0, 2)
    # i0=0, j0=2, k0=2
    add_strassen(0, 2, 2)
    
    # Schoolbook for the remainder
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0,1] and j in [0,1,2,3] and k in [0,1,2,3]:
                    continue # Already covered by Strassen
                
                # Schoolbook term: A[i,j] * B[j,k] -> C[k,i]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                u_dict = {u_idx: 1}
                v_dict = {v_idx: 1}
                w_dict = {w_idx: 1}
                
                add_term(u_dict, v_dict, w_dict)
                
    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    # Check rank
    # The rank is the number of columns
    rank = U.shape[1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
