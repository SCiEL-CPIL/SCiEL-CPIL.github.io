"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 term using sparse index/value lists."""
        u = {}
        v = {}
        w = {}
        
        for idx, val in zip(u_indices, u_vals):
            u[idx] = u.get(idx, 0) + val
        for idx, val in zip(v_indices, v_vals):
            v[idx] = v.get(idx, 0) + val
        for idx, val in zip(w_indices, w_vals):
            w[idx] = w.get(idx, 0) + val
            
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_terms(n_l, m_l, p_l, i_off, j_off, k_off):
        """
        Recursively generate rank-1 terms for an n_l x m_l matrix times m_l x p_l matrix.
        Uses Strassen for 2x2x2 blocks and schoolbook for base cases/odd dimensions.
        """
        # Base cases: if any dimension is 1, use schoolbook (rank = product of other dims)
        if n_l == 1:
            for jj in range(m_l):
                for kk in range(p_l):
                    i = i_off
                    j = j_off + jj
                    k = k_off + kk
                    add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        if m_l == 1:
            for ii in range(n_l):
                for kk in range(p_l):
                    i = i_off + ii
                    j = j_off
                    k = k_off + kk
                    add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        if p_l == 1:
            for ii in range(n_l):
                for jj in range(m_l):
                    i = i_off + ii
                    j = j_off + jj
                    k = k_off
                    add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])
            return

        # Strassen for 2x2x2
        if n_l == 2 and m_l == 2 and p_l == 2:
            i0 = i_off; i1 = i_off+1
            j0 = j_off; j1 = j_off+1
            k0 = k_off; k1 = k_off+1
            
            # M1 = (a00+a11)(b00+b11) -> c00, c11
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1], 
                           [j0*p+k0, j1*p+k1], [1, 1], 
                           [k0*n+i0, k1*n+i1], [1, 1])
            
            # M2 = (a10+a11)(b00) -> c10, -c11
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1], 
                           [j0*p+k0], [1], 
                           [k0*n+i1, k1*n+i1], [1, -1])
            
            # M3 = (a00)(b01-b11) -> c01, c11
            add_rank1_term([i0*m+j0], [1], 
                           [j0*p+k1, j1*p+k1], [1, -1], 
                           [k1*n+i0, k1*n+i1], [1, 1])
            
            # M4 = (a11)(b10-b00) -> c00, c10
            add_rank1_term([i1*m+j1], [1], 
                           [j1*p+k0, j0*p+k0], [1, -1], 
                           [k0*n+i0, k0*n+i1], [1, 1])
            
            # M5 = (a00+a01)(b11) -> -c00, c01
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1], 
                           [j1*p+k1], [1], 
                           [k0*n+i0, k1*n+i0], [-1, 1])
            
            # M6 = (a10-a00)(b00+b01) -> c11
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1], 
                           [j0*p+k0, j0*p+k1], [1, 1], 
                           [k1*n+i1], [1])
            
            # M7 = (a01-a11)(b10+b11) -> c00
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1], 
                           [j1*p+k0, j1*p+k1], [1, 1], 
                           [k0*n+i0], [1])
            return

        # Recursive decomposition:
        # Split the larger dimension to get 2x2x2 blocks where possible
        
        # If n_l is even, split into two halves of n_l/2
        if n_l % 2 == 0:
            n_half = n_l // 2
            gen_terms(n_half, m_l, p_l, i_off, j_off, k_off)
            gen_terms(n_half, m_l, p_l, i_off + n_half, j_off, k_off)
            return
        
        # If m_l is even, split into two halves of m_l/2
        if m_l % 2 == 0:
            m_half = m_l // 2
            gen_terms(n_l, m_half, p_l, i_off, j_off, k_off)
            gen_terms(n_l, m_half, p_l, i_off, j_off + m_half, k_off)
            return
        
        # If p_l is even, split into two halves of p_l/2
        if p_l % 2 == 0:
            p_half = p_l // 2
            gen_terms(n_l, m_l, p_half, i_off, j_off, k_off)
            gen_terms(n_l, m_l, p_half, i_off, j_off, k_off + p_half)
            return
        
        # Fallback: schoolbook for odd dimensions that can't be split into 2x2x2
        for ii in range(n_l):
            for jj in range(m_l):
                for kk in range(p_l):
                    i = i_off + ii
                    j = j_off + jj
                    k = k_off + kk
                    add_rank1_term([i*m+j], [1.0], [j*p+k], [1.0], [k*n+i], [1.0])

    # Start the recursive decomposition
    gen_terms(n, m, p, 0, 0, 0)
    
    rank = len(U_vecs)
    
    # Convert sparse dicts to dense matrices
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_vecs[c].items():
            U[idx, c] = val
        for idx, val in V_vecs[c].items():
            V[idx, c] = val
        for idx, val in W_vecs[c].items():
            W[idx, c] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
