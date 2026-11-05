"""Program for low discrepancy on all arithmetic progressions.
Uses quadratic minimization with precomputed sparse AP overlap matrix
and efficient gradient-based sign flips via NumPy operations.
"""
import numpy as np
import time
from scipy import sparse

def construct():
    n = 256
    
    # 1. Precompute all APs
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    for d in range(1, n + 1):
        for s in range(0, n):
            ap_indices = []
            current = s
            while current < n:
                ap_indices.append(current)
                pos_to_aps[current].append(len(aps))
                current += d
            if ap_indices:
                aps.append(ap_indices)
    
    num_aps = len(aps)
    
    # 2. Build the overlap matrix M where M[i,j] = number of APs containing both i and j
    row_indices = []
    col_indices = []
    values = []
    
    for ap in aps:
        m = len(ap)
        for a in range(m):
            for b in range(m):
                row_indices.append(ap[a])
                col_indices.append(ap[b])
                values.append(1)
    
    M = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n, n)
    )
    M_csr = M.tocsr()
    M_diag = np.array(M_csr.diagonal()).flatten()
    
    # Precompute columns as dense arrays for fast access
    M_cols = []
    for j in range(n):
        col = np.array(M_csr.getcol(j)).flatten()
        M_cols.append(col)
    
    # 3. Local search
    
    best_signs = None
    best_max_disc = float('inf')
    
    start_time = time.time()
    max_time = 55.0
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Initialize randomly
        signs = np.random.choice([-1, 1], size=n, dtype=np.int32)
        
        # Compute M @ x
        Mx = np.array(M_csr.dot(signs)).flatten()
        
        # Compute AP sums
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for ap_idx, ap in enumerate(aps):
            ap_sums[ap_idx] = int(np.sum(signs[ap]))
        max_disc = int(np.max(np.abs(ap_sums)))
        
        if max_disc < best_max_disc:
            best_max_disc = max_disc
            best_signs = signs.copy()
        
        improved = True
        while improved and (time.time() - start_time < max_time):
            improved = False
            
            # Compute deltas for all positions
            deltas = -4 * signs * Mx + 4 * M_diag
            
            best_pos = int(np.argmin(deltas))
            best_delta = int(deltas[best_pos])
            
            if best_delta < 0:
                # Perform the flip
                i = best_pos
                old_sign = signs[i]
                signs[i] = -old_sign
                
                # Update Mx
                Mx -= 2 * old_sign * M_cols[i]
                
                # Update AP sums incrementally
                for ap_idx in pos_to_aps[i]:
                    ap_sums[ap_idx] -= 2 * old_sign
                
                max_disc = int(np.max(np.abs(ap_sums)))
                
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
                
                improved = True
            else:
                # Random kick to escape local minimum
                kick_pos = int(np.random.randint(n))
                old_sign = signs[kick_pos]
                signs[kick_pos] = -old_sign
                
                Mx -= 2 * old_sign * M_cols[kick_pos]
                
                for ap_idx in pos_to_aps[kick_pos]:
                    ap_sums[ap_idx] -= 2 * old_sign
                
                max_disc = int(np.max(np.abs(ap_sums)))
                
                if max_disc < best_max_disc:
                    best_max_disc = max_disc
                    best_signs = signs.copy()
    
    if best_signs is None:
        return [1] * n
    
    return [int(s) for s in best_signs]
