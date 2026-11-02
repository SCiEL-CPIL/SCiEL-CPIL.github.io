import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 105
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    terms = []
    
    def add_strassen_block(i_off, j_off, k_off):
        """Add Strassen decomposition for a 2x2x2 block at given offsets."""
        # Strassen 7 terms for 2x2x2 tensor decomposition.
        # U indexed by (i,j), V by (j,k), W by (k,i)
        strassen_terms = [
            # M1: U=(00,11), V=(00,11), W=(00,11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2: U=(10,11), V=(00), W=(01,11) with W[0,1]=1, W[1,1]=-1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            # M3: U=(00), V=(00,10) with V[0,0]=-1,V[1,0]=1, W=(00,10) with W[0,0]=-1,W[1,0]=1
            ({(0,0): 1}, {(0,0): -1, (1,0): 1}, {(0,0): -1, (1,0): 1}),
            # M4: U=(11), V=(01,10) with V[0,1]=1,V[1,0]=-1, W=(00,01) with W[0,0]=1,W[0,1]=1
            ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (0,1): 1}),
            # M5: U=(00,01), V=(11), W=(10,11) with W[1,0]=1,W[1,1]=1
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
            # M6: U=(10,00) with U[1,0]=1,U[0,0]=-1, V=(00,01), W=(01,10) with W[0,1]=1,W[1,0]=-1
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1, (1,0): -1}),
            # M7: U=(01,11) with U[0,1]=1,U[1,1]=-1, V=(10,11), W=(00,11) with W[0,0]=1,W[1,1]=-1
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
        ]
        
        # Verify this Strassen decomposition locally
        local_U = np.zeros((4, 7))
        local_V = np.zeros((4, 7))
        local_W = np.zeros((4, 7))
        for r, (u_d, v_d, w_d) in enumerate(strassen_terms):
            for (di, dj), val in u_d.items():
                local_U[di*2+dj, r] = val
            for (dj, dk), val in v_d.items():
                local_V[dj*2+dk, r] = val
            for (dk, di), val in w_d.items():
                local_W[dk*2+di, r] = val
        
        local_ok = True
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    s = sum(local_U[di*2+dj, r] * local_V[dj*2+dk, r] * local_W[dk*2+di, r] for r in range(7))
                    if abs(s - 1) > 1e-9:
                        local_ok = False
                        break
                if not local_ok:
                    break
            if not local_ok:
                break
        
        if not local_ok:
            # Fallback: use schoolbook for this 2x2x2 block (rank 8)
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        u_abs = {(i_off + di, j_off + dj): 1}
                        v_abs = {(j_off + dj, k_off + dk): 1}
                        w_abs = {(k_off + dk, i_off + di): 1}
                        terms.append((u_abs, v_abs, w_abs))
            return
        
        for u_dict, v_dict, w_dict in strassen_terms:
            u_abs = {}
            for (di, dj), val in u_dict.items():
                u_abs[(i_off + di, j_off + dj)] = val
            v_abs = {}
            for (dj, dk), val in v_dict.items():
                v_abs[(j_off + dj, k_off + dk)] = val
            w_abs = {}
            for (dk, di), val in w_dict.items():
                w_abs[(k_off + dk, i_off + di)] = val
            terms.append((u_abs, v_abs, w_abs))
    
    def add_schoolbook_block(i_off, j_off, k_off, n_block, m_block, p_block):
        """Add schoolbook decomposition for n_block x m_block x p_block."""
        for di in range(n_block):
            for dj in range(m_block):
                for dk in range(p_block):
                    u = {(i_off + di, j_off + dj): 1}
                    v = {(j_off + dj, k_off + dk): 1}
                    w = {(k_off + dk, i_off + di): 1}
                    terms.append((u, v, w))
    
    # Partition: j = {0,1}, {2,3}, {4}; k = {0,1}, {2,3}, {4}
    j_groups = [(0, 2), (2, 4), (4, 5)]
    k_groups = [(0, 2), (2, 4), (4, 5)]
    
    for (j_start, j_end) in j_groups:
        for (k_start, k_end) in k_groups:
            m_block = j_end - j_start
            p_block = k_end - k_start
            n_block = n  # always 2
            
            if m_block == 2 and p_block == 2:
                # Use Strassen
                add_strassen_block(0, j_start, k_start)
            else:
                # Use schoolbook
                add_schoolbook_block(0, j_start, k_start, n_block, m_block, p_block)
    
    rank = len(terms)
    
    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    # Verify the decomposition
    max_error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error = abs(val - 1)
                if error > max_error:
                    max_error = error
    
    if max_error > 1e-9:
        # Fallback to the known-good rank 46 decomposition
        terms = []
        
        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))
        
        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))
        
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)
        
        rank = len(terms)
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        
        return {"U": U, "V": V, "W": W}
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Convert numpy floats to int where possible
    def clean_list(lst):
        result = []
        for row in lst:
            cleaned_row = []
            for val in row:
                if isinstance(val, (int, np.integer)):
                    cleaned_row.append(int(val))
                else:
                    fv = float(val)
                    if abs(fv - round(fv)) < 1e-9:
                        cleaned_row.append(int(round(fv)))
                    elif abs(fv*2 - round(fv*2)) < 1e-9:
                        cleaned_row.append(fv)
                    else:
                        cleaned_row.append(fv)
            result.append(cleaned_row)
        return result
    
    return {"U": clean_list(U_list), "V": clean_list(V_list), "W": clean_list(W_list)}
