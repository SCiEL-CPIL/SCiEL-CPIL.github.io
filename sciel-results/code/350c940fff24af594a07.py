"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-18 decomposition.
    # Split p=5 into 2 + 2 + 1.
    # Use Strassen (rank 7) for the first 2 columns.
    # Use Strassen (rank 7) for the next 2 columns.
    # Use Schoolbook (rank 4) for the last 1 column.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Terms are (u, v, w) where u is vector in A-space (dim 4), v in B-space (dim 4), w in C-space (dim 4)
    # Indices: A[i*m+j], B[j*p+k], W[k*n+i]
    # For a sub-block of B columns [k_start, k_start+2) and A full, C full.
    # We need to map the local indices to global indices.
    
    def strassen_terms(k_start):
        # k_start is the starting column index in B
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices for this block: 
        # local (0,0) -> global j=0, k=k_start
        # local (0,1) -> global j=0, k=k_start+1
        # local (1,0) -> global j=1, k=k_start
        # local (1,1) -> global j=1, k=k_start+1
        
        # 1. (a+d)(e+h)
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1] # e at (0,0), h at (1,1) in local B
        # Map v to global:
        v_global = [0]*10
        v_global[0*p + k_start] = 1     # j=0, k=k_start
        v_global[1*p + k_start+1] = 1   # j=1, k=k_start+1
        # w: C[0,0] +=, C[1,1] +=
        # W index: k*n + i
        # C[0,0] is k=k_start, i=0 -> 2*k_start + 0
        # C[1,1] is k=k_start+1, i=1 -> 2*(k_start+1) + 1
        w_global = [0]*10
        w_global[2*k_start + 0] = 1
        w_global[2*(k_start+1) + 1] = 1
        add_term(u, v_global, w_global)
        
        # 2. (c+d)(e+f)
        u = [0, 0, 1, 1]
        v = [1, 1, 0, 0] # e at (0,0), f at (0,1)
        v_global = [0]*10
        v_global[0*p + k_start] = 1
        v_global[0*p + k_start+1] = 1
        # w: C[1,0] +=, C[1,1] += ? No, (c+d)(e+f) contributes to C[1,0] and C[1,1]?
        # C[1,0] = c*e + d*g. C[1,1] = c*f + d*h.
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e+f)
        # M3 = (a+b)(g-e)
        # M4 = (c+d)(h-g)
        # M5 = (a+b)(f-h)
        # M6 = c(e+h)
        # M7 = d(f+g)
        # C00 = M1 - M4 - M6 + M7
        # C01 = M5 + M4
        # C10 = M2 + M3
        # C11 = M1 - M2 + M6 - M3
        
        # So M2 = (c+d)(e+f) contributes to C10 (+) and C11 (-).
        w_global = [0]*10
        w_global[2*k_start + 1] = 1     # C[1,0] -> k=k_start, i=1
        w_global[2*(k_start+1) + 1] = -1 # C[1,1] -> k=k_start+1, i=1
        add_term(u, v_global, w_global)
        
        # 3. (a+b)(g-e)
        u = [1, 1, 0, 0]
        v = [0, 0, 1, -1] # g at (1,0), -e at (0,0)
        v_global = [0]*10
        v_global[1*p + k_start] = 1
        v_global[0*p + k_start] = -1
        # M3 contributes to C10 (+) and C11 (-).
        w_global = [0]*10
        w_global[2*k_start + 1] = 1
        w_global[2*(k_start+1) + 1] = -1
        add_term(u, v_global, w_global)
        
        # 4. (c+d)(h-g)
        u = [0, 0, 1, 1]
        v = [0, 0, -1, 1] # h at (1,1), -g at (1,0)
        v_global = [0]*10
        v_global[1*p + k_start+1] = 1
        v_global[1*p + k_start] = -1
        # M4 contributes to C00 (-), C01 (+), C11 (-).
        w_global = [0]*10
        w_global[2*k_start + 0] = -1
        w_global[2*(k_start+1) + 0] = 1
        w_global[2*(k_start+1) + 1] = -1
        add_term(u, v_global, w_global)
        
        # 5. (a+b)(f-h)
        u = [1, 1, 0, 0]
        v = [0, 1, 0, -1] # f at (0,1), -h at (1,1)
        v_global = [0]*10
        v_global[0*p + k_start+1] = 1
        v_global[1*p + k_start+1] = -1
        # M5 contributes to C01 (+).
        w_global = [0]*10
        w_global[2*(k_start+1) + 0] = 1
        add_term(u, v_global, w_global)
        
        # 6. c(e+h)
        u = [0, 0, 1, 0]
        v = [1, 0, 0, 1] # e, h
        v_global = [0]*10
        v_global[0*p + k_start] = 1
        v_global[1*p + k_start+1] = 1
        # M6 contributes to C00 (-), C11 (+).
        w_global = [0]*10
        w_global[2*k_start + 0] = -1
        w_global[2*(k_start+1) + 1] = 1
        add_term(u, v_global, w_global)
        
        # 7. d(f+g)
        u = [0, 0, 0, 1]
        v = [0, 1, 1, 0] # f, g
        v_global = [0]*10
        v_global[0*p + k_start+1] = 1
        v_global[1*p + k_start] = 1
        # M7 contributes to C00 (+).
        w_global = [0]*10
        w_global[2*k_start + 0] = 1
        add_term(u, v_global, w_global)
        
    # Schoolbook for 2x2x1
    def schoolbook_term_1col(k_start):
        # A full, B col k_start, C col k_start
        # 4 terms: (0,0), (0,1), (1,0), (1,1)
        for i in range(2):
            for j in range(2):
                u = [0]*4
                u[i*m + j] = 1
                v = [0]*10
                v[j*p + k_start] = 1
                w = [0]*10
                w[k_start*n + i] = 1
                add_term(u, v, w)

    # Execute construction
    strassen_terms(0)
    strassen_terms(2)
    schoolbook_term_1col(4)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
