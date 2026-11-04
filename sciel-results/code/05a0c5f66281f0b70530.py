import numpy as np

def construct():
    n, m, p = 3, 4, 8
    
    # Strassen 2x2x2 base case terms (local coords)
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, coeff)
    # v_entries: list of (j, k, coeff)
    # w_entries: list of (k, i, coeff)
    
    strassen_terms = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    # Build factor matrices directly using a recursive approach
    # that applies Strassen where possible and schoolbook otherwise.
    
    U_list = []
    V_list = []
    W_list = []
    
    def emit_term(u_entries, v_entries, w_entries, i0, j0, k0):
        """Add one rank-1 term to the factor matrices."""
        u_row = {}
        v_row = {}
        w_row = {}
        for (li, lj, c) in u_entries:
            gi = i0 + li
            gj = j0 + lj
            u_row[gi * m + gj] = c
        for (lj, lk, c) in v_entries:
            gj = j0 + lj
            gk = k0 + lk
            v_row[gj * p + gk] = c
        for (lk, li, c) in w_entries:
            gk = k0 + lk
            gi = i0 + li
            w_row[gk * n + gi] = c
        U_list.append(u_row)
        V_list.append(v_row)
        W_list.append(w_row)
    
    def schoolbook_block(i0, j0, k0, ni, mj, pk):
        """Schoolbook decomposition for a sub-block."""
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    emit_term([(i, j, 1)], [(j, k, 1)], [(k, i, 1)], i0, j0, k0)
    
    def decompose(i0, j0, k0, ni, mj, pk):
        """Recursive decomposition using Strassen where applicable."""
        # If all dimensions are 2, use Strassen
        if ni == 2 and mj == 2 and pk == 2:
            for (ue, ve, we) in strassen_terms:
                emit_term(ue, ve, we, i0, j0, k0)
            return
        
        # Try to split along m (inner dimension) if even and > 2
        if mj >= 4 and mj % 2 == 0:
            mh = mj // 2
            decompose(i0, j0, k0, ni, mh, pk)
            decompose(i0, j0 + mh, k0, ni, mj - mh, pk)
            return
        
        # Try to split along p if even and > 2
        if pk >= 4 and pk % 2 == 0:
            ph = pk // 2
            decompose(i0, j0, k0, ni, mj, ph)
            decompose(i0, j0, k0 + ph, ni, mj, pk - ph)
            return
        
        # Try to split along n if > 2
        if ni >= 3:
            # Split into 2 and (ni-2)
            decompose(i0, j0, k0, 2, mj, pk)
            decompose(i0 + 2, j0, k0, ni - 2, mj, pk)
            return
        
        # Fallback: schoolbook
        schoolbook_block(i0, j0, k0, ni, mj, pk)
    
    decompose(0, 0, 0, n, m, p)
    
    rank = len(U_list)
    
    # Convert sparse dicts to dense matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (idx, c) in U_list[r].items():
            U[idx, r] = c
        for (idx, c) in V_list[r].items():
            V[idx, r] = c
        for (idx, c) in W_list[r].items():
            W[idx, r] = c
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
