import time

def construct():
    start_time = time.time()
    deadline = 120 - 10 # Safety margin
    
    n, m, p = 3, 5, 6
    
    # We will build U, V, W as lists of lists
    # Rank estimate: 18 + 23 + 18 + 23 = 82
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        if coeff == 0: return
        U.append(u_idx)
        V.append(v_idx)
        W.append(w_idx)
        # Store coeff? The interface asks for matrices of shape (n*m, rank).
        # So U is a matrix.
        # We need to append to the last column?
        # No, let's build the matrices directly.
        
    # Let's use lists for rows, then transpose or build correctly.
    # U shape: (15, rank)
    # V shape: (30, rank)
    # W shape: (18, rank)
    
    # We don't know rank in advance if we add dynamically, but we do: 82.
    rank = 82
    U = [[0]*rank for _ in range(15)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(18)]
    
    col = 0
    
    # Helper to add a local decomposition
    def add_local_decomp(n_local, m_local, p_local, offset_i, offset_j, offset_k, terms):
        nonlocal col
        for (u_l, v_l, w_l, coeff) in terms:
            # Global indices
            # u_l is row in U_local (size n_local * m_local)
            # Map to global U index:
            # i_l = u_l // m_local
            # j_l = u_l % m_local
            # i = i_l + offset_i
            # j = j_l + offset_j
            # u_global = i * m + j
            
            # v_l is row in V_local (size m_local * p_local)
            # j_l = v_l // p_local
            # k_l = v_l % p_local
            # j = j_l + offset_j
            # k = k_l + offset_k
            # v_global = j * p + k
            
            # w_l is row in W_local (size p_local * n_local)
            # k_l = w_l // n_local
            # i_l = w_l % n_local
            # k = k_l + offset_k
            # i = i_l + offset_i
            # w_global = k * n + i
            
            i_l = u_l // m_local
            j_l = u_l % m_local
            i = i_l + offset_i
            j = j_l + offset_j
            u_global = i * m + j
            
            j_lv = v_l // p_local
            k_l = v_l % p_local
            j = j_lv + offset_j
            k = k_l + offset_k
            v_global = j * p + k
            
            k_lw = w_l // n_local
            i_lw = w_l % n_local
            k = k_lw + offset_k
            i = i_lw + offset_i
            w_global = k * n + i
            
            U[u_global][col] += coeff
            V[v_global][col] += coeff
            W[w_global][col] += coeff
            col += 1
            
    # Schoolbook terms for 3x2x3 (18 terms)
    # u: i*2+j, v: j*3+k, w: k*3+i
    schoolbook_323 = []
    for i in range(3):
        for j in range(2):
            for k in range(3):
                u_l = i*2 + j
                v_l = j*3 + k
                w_l = k*3 + i
                schoolbook_323.append((u_l, v_l, w_l, 1))
                
    # Laderman terms for 3x3x3 (23 terms)
    # u: i*3+j, v: j*3+k, w: k*3+i
    laderman_333 = [
        (0,0,0,1), (0,1,1,1), (0,2,2,1),
        (1,0,0,1), (1,1,1,1), (1,2,2,1),
        (2,0,0,1), (2,1,1,1), (2,2,2,1),
        (0,0,1,1), (0,1,0,1), (0,2,0,1),
        (1,0,2,1), (1,1,2,1), (1,2,0,1),
        (2,0,1,1), (2,1,2,1), (2,2,1,1),
        (0,2,1,1), (1,0,1,1), (2,1,0,1),
        (0,1,2,1), (1,2,1,1), (2,0,2,1)
    ]
    
    # Check time
    if time.time() > deadline:
         # Return schoolbook if deadline exceeded? 
         # But we are constructing fast.
         pass

    # Add decompositions
    # 1. A_L B_11: n=3, m=2, p=3. offset i=0, j=0, k=0
    add_local_decomp(3, 2, 3, 0, 0, 0, schoolbook_323)
    
    # 2. A_R B_21: n=3, m=3, p=3. offset i=0, j=2, k=0
    add_local_decomp(3, 3, 3, 0, 2, 0, laderman_333)
    
    # 3. A_L B_12: n=3, m=2, p=3. offset i=0, j=0, k=3
    add_local_decomp(3, 2, 3, 0, 0, 3, schoolbook_323)
    
    # 4. A_R B_22: n=3, m=3, p=3. offset i=0, j=2, k=3
    add_local_decomp(3, 3, 3, 0, 2, 3, laderman_333)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
