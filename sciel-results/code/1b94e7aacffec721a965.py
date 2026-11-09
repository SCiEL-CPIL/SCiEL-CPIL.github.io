import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """Return the lowest-rank exact decomposition found within the time budget."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    n_m = n * m      # 12
    m_p = m * p      # 24
    n_p = p * n      # 18
    
    # We will build the decomposition block by block.
    # Block structure:
    # i: 3 -> 2 + 1
    # j: 4 -> 2 + 2
    # k: 6 -> 2 + 2 + 2
    
    # We need to add terms to U, V, W.
    # U shape: (12, Rank)
    # V shape: (24, Rank)
    # W shape: (18, Rank)
    
    # We'll accumulate the terms in lists first, then stack.
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_local, v_local, w_local, i_offset, j_offset, k_offset):
        """
        Add a rank-1 term to the global decomposition.
        u_local: vector of length (n_block * m_block)
        v_local: vector of length (m_block * p_block)
        w_local: vector of length (p_block * n_block)
        i_offset, j_offset, k_offset: starting indices for i, j, k
        """
        n_b, m_b, p_b = len(u_local) // m_b, m_b, len(v_local) // p_b
        # Actually, we need to know the block dimensions to place correctly.
        # Let's pass explicit dimensions.
        
    # It's easier to construct the full vectors directly.
    
    def create_block_term(n_b, m_b, p_b, u_vals, v_vals, w_vals, i_off, j_off, k_off):
        """
        Create global U, V, W vectors for a block.
        u_vals: list of (index_in_block, value) or just a vector of length n_b*m_b
        """
        u_global = np.zeros(n * m)
        v_global = np.zeros(m * p)
        w_global = np.zeros(p * n)
        
        # Map local indices to global
        # Local A index: i_l * m_b + j_l
        # Global A index: (i_off + i_l) * m + (j_off + j_l)
        
        # Local B index: j_l * p_b + k_l
        # Global B index: (j_off + j_l) * p + (k_off + k_l)
        
        # Local C index: k_l * n_b + i_l
        # Global C index: (k_off + k_l) * n + (i_off + i_l)
        
        for i_l in range(n_b):
            for j_l in range(m_b):
                val = u_vals[i_l * m_b + j_l]
                if val != 0:
                    u_global[(i_off + i_l) * m + (j_off + j_l)] = val
                    
        for j_l in range(m_b):
            for k_l in range(p_b):
                val = v_vals[j_l * p_b + k_l]
                if val != 0:
                    v_global[(j_off + j_l) * p + (k_off + k_l)] = val
                    
        for k_l in range(p_b):
            for i_l in range(n_b):
                val = w_vals[k_l * n_b + i_l]
                if val != 0:
                    w_global[(k_off + k_l) * n + (i_off + i_l)] = val
                    
        U_terms.append(u_global)
        V_terms.append(v_global)
        W_terms.append(w_global)

    # Strassen 2x2x2 decomposition
    # Returns 7 terms (u, v, w) for 2x2x2 block
    def strassen_2x2x2():
        terms = []
        # M1
        u = np.array([1, 0, 0, 1])
        v = np.array([1, 0, 0, 1])
        w = np.array([1, 0, 0, 1]) # C00, C11
        terms.append((u, v, w))
        
        # M2
        u = np.array([0, 0, 1, 1])
        v = np.array([1, 0, 0, 0])
        w = np.array([0, 0, 1, 0]) # C10
        terms.append((u, v, w))
        
        # M3
        u = np.array([1, 0, 0, 0])
        v = np.array([0, 1, 0, -1])
        w = np.array([0, 1, 0, 0]) # C01
        terms.append((u, v, w))
        
        # M4
        u = np.array([0, 0, 0, 1])
        v = np.array([0, 0, 1, -1])
        w = np.array([1, 0, 1, 0]) # C00, C10
        terms.append((u, v, w))
        
        # M5
        u = np.array([1, 1, 0, 0])
        v = np.array([0, 0, 0, 1])
        w = np.array([-1, 1, 0, 0]) # -C00, C01
        terms.append((u, v, w))
        
        # M6
        u = np.array([-1, 0, 1, 0])
        v = np.array([1, 1, 0, 0])
        w = np.array([0, 0, 0, 1]) # C11
        terms.append((u, v, w))
        
        # M7
        u = np.array([0, 1, -1, 0])
        v = np.array([0, 0, 1, 1])
        w = np.array([1, 0, 0, 0]) # C00
        terms.append((u, v, w))
        
        return terms

    # Add 2x2x2 block at (i_off, j_off, k_off)
    def add_strassen_block(i_off, j_off, k_off):
        terms = strassen_2x2x2()
        for u, v, w in terms:
            create_block_term(2, 2, 2, u, v, w, i_off, j_off, k_off)

    # Add 1x2x6 block (schoolbook)
    # n_b=1, m_b=2, p_b=6
    def add_1x2x6_block(i_off, j_off, k_off):
        # 1 * 2 * 6 = 12 terms
        for j_l in range(2):
            for k_l in range(6):
                u = np.zeros(1 * 2)
                v = np.zeros(2 * 6)
                w = np.zeros(6 * 1)
                
                u[0 * 2 + j_l] = 1.0
                v[j_l * 6 + k_l] = 1.0
                w[k_l * 1 + 0] = 1.0
                
                create_block_term(1, 2, 6, u, v, w, i_off, j_off, k_off)

    # Add 2x2x6 block using 3 Strassen blocks
    def add_2x2x6_block(i_off, j_off, k_off):
        # k splits into 3 pairs of 2
        for k_pair in range(3):
            k_off_local = k_off + k_pair * 2
            add_strassen_block(i_off, j_off, k_off_local)

    # Add 2x4x6 block using two 2x2x6 blocks
    def add_2x4x6_block(i_off, j_off, k_off):
        # j splits into two pairs of 2
        for j_pair in range(2):
            j_off_local = j_off + j_pair * 2
            add_2x2x6_block(i_off, j_off_local, k_off)

    # Add 1x4x6 block using two 1x2x6 blocks
    def add_1x4x6_block(i_off, j_off, k_off):
        for j_pair in range(2):
            j_off_local = j_off + j_pair * 2
            add_1x2x6_block(i_off, j_off_local, k_off)

    # Main decomposition for 3x4x6
    # i splits into 2 (top) and 1 (bottom)
    
    # Top part: i=0,1. j=0..3. k=0..5
    add_2x4x6_block(0, 0, 0)
    
    # Bottom part: i=2. j=0..3. k=0..5
    add_1x4x6_block(2, 0, 0)

    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook if we run out of time (shouldn't happen)
        pass

    U = np.column_stack(U_terms)
    V = np.column_stack(V_terms)
    W = np.column_stack(W_terms)
    
    rank = U.shape[1]
    
    # Verify
    T_rec = np.zeros((n_m, m_p, n_p))
    for s in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
        
    T = np.zeros((n_m, m_p, n_p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    if not np.allclose(T_rec, T, atol=1e-9):
        # If verification fails, return schoolbook
        U_sb = np.zeros((n_m, n*m*p))
        V_sb = np.zeros((m_p, n*m*p))
        W_sb = np.zeros((n_p, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
