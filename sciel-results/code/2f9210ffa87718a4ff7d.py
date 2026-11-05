"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by summing sub-decompositions.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_decomp(local_U, local_V, local_W, offset_i, offset_j, offset_k):
        """
        Adds a local decomposition to the global one.
        local_U: list of lists, shape (n_local * m_local, rank_local)
        local_V: list of lists, shape (m_local * p_local, rank_local)
        local_W: list of lists, shape (p_local * n_local, rank_local)
        offset_i, offset_j, offset_k: global indices for the start of this block.
        """
        rank_local = len(local_U[0]) if local_U else 0
        n_local = len(local_U) // m_local if local_U else 0
        m_local = len(local_U) // n_local if local_U else 0
        p_local = len(local_W) // n_local if local_W else 0
        
        # We need to map local indices to global indices.
        # Local U index: i_l * m_local + j_l
        # Global U index: (i_l + offset_i) * m + (j_l + offset_j)
        
        # We will append to global lists.
        # First, we need to know the total rank to allocate space?
        # No, we can just build the lists row by row.
        
        # This approach is tricky because we need to align columns (rank).
        # Better: Build a list of (U_row, V_row, W_row) tuples for each rank term.
        
        pass

    # Alternative approach:
    # Generate a list of terms: (U_vec, V_vec, W_vec)
    # U_vec is a vector of length n*m
    # V_vec is a vector of length m*p
    # W_vec is a vector of length p*n
    
    terms = []
    
    def add_term(U_vec, V_vec, W_vec):
        terms.append((U_vec, V_vec, W_vec))
        
    def add_block(n_l, m_l, p_l, offset_i, offset_j, offset_k, local_decomp_func):
        """
        local_decomp_func(n_l, m_l, p_l) returns a list of (U_vec, V_vec, W_vec)
        where U_vec is length n_l*m_l, etc.
        """
        local_terms = local_decomp_func(n_l, m_l, p_l)
        for U_l, V_l, W_l in local_terms:
            U_g = [0] * (n * m)
            V_g = [0] * (m * p)
            W_g = [0] * (p * n)
            
            for i_l in range(n_l):
                for j_l in range(m_l):
                    idx_l = i_l * m_l + j_l
                    idx_g = (i_l + offset_i) * m + (j_l + offset_j)
                    U_g[idx_g] = U_l[idx_l]
                    
            for j_l in range(m_l):
                for k_l in range(p_l):
                    idx_l = j_l * p_l + k_l
                    idx_g = (j_l + offset_j) * p + (k_l + offset_k)
                    V_g[idx_g] = V_l[idx_l]
                    
            for k_l in range(p_l):
                for i_l in range(n_l):
                    idx_l = k_l * n_l + i_l
                    idx_g = (k_l + offset_k) * n + (i_l + offset_i)
                    W_g[idx_g] = W_l[idx_l]
                    
            add_term(U_g, V_g, W_g)

    def schoolbook(n_l, m_l, p_l):
        res = []
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U = [0] * (n_l * m_l)
                    V = [0] * (m_l * p_l)
                    W = [0] * (p_l * n_l)
                    U[i * m_l + j] = 1
                    V[j * p_l + k] = 1
                    W[k * n_l + i] = 1
                    res.append((U, V, W))
        return res

    def strassen_222(n_l, m_l, p_l):
        # n_l=2, m_l=2, p_l=2
        # Returns 7 terms
        res = []
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00+=1, C11+=1
        # ... this is tedious to write out manually.
        # I will use the standard formulas.
        
        def make_term(u_vals, v_vals, w_vals):
            # u_vals: dict (i,j) -> val
            # v_vals: dict (j,k) -> val
            # w_vals: dict (k,i) -> val
            U = [0] * 4
            V = [0] * 4
            W = [0] * 4
            for (i,j), val in u_vals.items():
                U[i*2+j] = val
            for (j,k), val in v_vals.items():
                V[j*2+k] = val
            for (k,i), val in w_vals.items():
                W[k*2+i] = val
            return (U, V, W)
            
        # M1
        res.append(make_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2
        res.append(make_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1})) # C10 += M2
        # M3
        res.append(make_term({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1})) # C01 += M3
        # M4
        res.append(make_term({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1})) # C00 += M4, C10 += M4
        # M5
        res.append(make_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1})) # C00 -= M5, C01 += M5
        # M6
        res.append(make_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})) # C11 += M6
        # M7
        res.append(make_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})) # C00 += M7
        
        return res

    # Decomposition of 3x2x3 (Rank 17)
    def decomp_323(n_l, m_l, p_l):
        # Split n=1+2, p=1+2
        # 1x2x1 (Rank 2)
        # 1x2x2 (Rank 4)
        # 2x2x1 (Rank 4)
        # 2x2x2 (Rank 7)
        res = []
        # Block 1: n=1, m=2, p=1 at (0,0,0)
        res.extend(add_block_local(1, 2, 1, 0, 0, 0, schoolbook))
        # Block 2: n=1, m=2, p=2 at (0,0,1)
        res.extend(add_block_local(1, 2, 2, 0, 0, 1, schoolbook))
        # Block 3: n=2, m=2, p=1 at (1,0,0)
        res.extend(add_block_local(2, 2, 1, 1, 0, 0, schoolbook))
        # Block 4: n=2, m=2, p=2 at (1,0,1)
        res.extend(add_block_local(2, 2, 2, 1, 0, 1, strassen_222))
        return res

    def add_block_local(n_l, m_l, p_l, off_i, off_j, off_k, func):
        # This is a helper for local decomposition
        # It returns terms in the local coordinate system of the 3x2x3 block
        local_terms = func(n_l, m_l, p_l)
        mapped = []
        for U_l, V_l, W_l in local_terms:
            U_new = [0] * (3*2)
            V_new = [0] * (2*3)
            W_new = [0] * (3*3)
            
            for i_l in range(n_l):
                for j_l in range(m_l):
                    U_new[(i_l+off_i)*2 + (j_l+off_j)] = U_l[i_l*m_l + j_l]
            for j_l in range(m_l):
                for k_l in range(p_l):
                    V_new[(j_l+off_j)*3 + (k_l+off_k)] = V_l[j_l*p_l + k_l]
            for k_l in range(p_l):
                for i_l in range(n_l):
                    W_new[(k_l+off_k)*3 + (i_l+off_i)] = W_l[k_l*n_l + i_l]
            mapped.append((U_new, V_new, W_new))
        return mapped

    # Decomposition of 3x3x3 (Rank 19 - Placeholder with Rank 26 for safety)
    def decomp_333(n_l, m_l, p_l):
        # Split n=1+2, m=1+2, p=1+2
        res = []
        # 1x1x1
        res.extend(add_block_local_333(1, 1, 1, 0, 0, 0, schoolbook))
        # 1x1x2
        res.extend(add_block_local_333(1, 1, 2, 0, 0, 1, schoolbook))
        # 1x2x1
        res.extend(add_block_local_333(1, 2, 1, 0, 1, 0, schoolbook))
        # 1x2x2
        res.extend(add_block_local_333(1, 2, 2, 0, 1, 1, schoolbook))
        # 2x1x1
        res.extend(add_block_local_333(2, 1, 1, 1, 0, 0, schoolbook))
        # 2x1x2
        res.extend(add_block_local_333(2, 1, 2, 1, 0, 1, schoolbook))
        # 2x2x1
        res.extend(add_block_local_333(2, 2, 1, 1, 1, 0, schoolbook))
        # 2x2x2
        res.extend(add_block_local_333(2, 2, 2, 1, 1, 1, strassen_222))
        return res

    def add_block_local_333(n_l, m_l, p_l, off_i, off_j, off_k, func):
        local_terms = func(n_l, m_l, p_l)
        mapped = []
        for U_l, V_l, W_l in local_terms:
            U_new = [0] * 9
            V_new = [0] * 9
            W_new = [0] * 9
            for i_l in range(n_l):
                for j_l in range(m_l):
                    U_new[(i_l+off_i)*3 + (j_l+off_j)] = U_l[i_l*m_l + j_l]
            for j_l in range(m_l):
                for k_l in range(p_l):
                    V_new[(j_l+off_j)*3 + (k_l+off_k)] = V_l[j_l*p_l + k_l]
            for k_l in range(p_l):
                for i_l in range(n_l):
                    W_new[(k_l+off_k)*3 + (i_l+off_i)] = W_l[k_l*n_l + i_l]
            mapped.append((U_new, V_new, W_new))
        return mapped

    # Main construction
    # Split m=5 into 2+3
    # Block 1: 3x2x6 (j=0,1)
    # Split p=6 into 3+3
    # Block 1.1: 3x2x3 (k=0,1,2)
    add_block(3, 2, 3, 0, 0, 0, decomp_323)
    # Block 1.2: 3x2x3 (k=3,4,5)
    add_block(3, 2, 3, 0, 0, 3, decomp_323)
    
    # Block 2: 3x3x6 (j=2,3,4)
    # Split p=6 into 3+3
    # Block 2.1: 3x3x3 (k=0,1,2)
    add_block(3, 3, 3, 0, 2, 0, decomp_333)
    # Block 2.2: 3x3x3 (k=3,4,5)
    add_block(3, 3, 3, 0, 2, 3, decomp_333)

    # Convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        U_vec, V_vec, W_vec = terms[r]
        for i in range(n * m):
            U[i][r] = U_vec[i]
        for j in range(m * p):
            V[j][r] = V_vec[j]
        for k in range(p * n):
            W[k][r] = W_vec[k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
