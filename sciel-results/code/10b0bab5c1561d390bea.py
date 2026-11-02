import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Returns a list of 7 tuples (U_dict, V_dict, W_dict) for Strassen 2x2x2
        applied to the block at top-left (i0, j0, k0).
        U index: i * m_dim + j
        V index: j * p_dim + k
        W index: k * n_dim + i
        """
        cols = []
        
        # M1 = (A11+A22)(B11+B22)
        cols.append((
            {i0*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M2 = (A21+A22)B11
        cols.append((
            {(i0+1)*m_dim+j0: 1, (i0+1)*m_dim+(j0+1): 1},
            {j0*p_dim+k0: 1},
            {k0*n_dim+(i0+1): 1, (k0+1)*n_dim+(i0+1): -1}
        ))
        
        # M3 = A11(B12-B22)
        cols.append((
            {i0*m_dim+j0: 1},
            {j0*p_dim+(k0+1): 1, (j0+1)*p_dim+(k0+1): -1},
            {(k0+1)*n_dim+i0: 1, (k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M4 = A22(B21-B11)
        cols.append((
            {(i0+1)*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+k0: 1, j0*p_dim+k0: -1},
            {k0*n_dim+i0: 1, k0*n_dim+(i0+1): 1}
        ))
        
        # M5 = (A11+A12)B22
        cols.append((
            {i0*m_dim+j0: 1, i0*m_dim+(j0+1): 1},
            {(j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: -1, (k0+1)*n_dim+i0: 1}
        ))
        
        # M6 = (A21-A11)(B11+B12)
        cols.append((
            {(i0+1)*m_dim+j0: 1, i0*m_dim+j0: -1},
            {j0*p_dim+k0: 1, j0*p_dim+(k0+1): 1},
            {(k0+1)*n_dim+(i0+1): 1}
        ))
        
        # M7 = (A12-A22)(B21+B22)
        cols.append((
            {i0*m_dim+(j0+1): 1, (i0+1)*m_dim+(j0+1): -1},
            {(j0+1)*p_dim+k0: 1, (j0+1)*p_dim+(k0+1): 1},
            {k0*n_dim+i0: 1}
        ))
        
        return cols

    def decompose_best(n_curr, m_curr, p_curr, i0, j0, k0, n_dim, m_dim, p_dim):
        """
        Returns the best (minimum rank) list of (U_dict, V_dict, W_dict) for the block
        by trying all valid split strategies.
        """
        best_cols = None
        best_rank = float('inf')
        
        def try_strassen():
            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                if time.time() > deadline:
                    return None
                cols = list(strassen_2x2x2_cols(i0, j0, k0, n_dim, m_dim, p_dim))
                # Try all orderings of remainder recursion
                remainders = []
                if n_curr > 2:
                    remainders.append(('n', n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0))
                if m_curr > 2:
                    remainders.append(('m', 2, m_curr - 2, p_curr, i0, j0 + 2, k0))
                if p_curr > 2:
                    remainders.append(('p', 2, 2, p_curr - 2, i0, j0, k0 + 2))
                
                # Recursively solve each remainder and sum
                rem_cols = []
                for rem in remainders:
                    if time.time() > deadline:
                        return None
                    rem_nc, rem_mc, rem_pc = rem[1], rem[2], rem[3]
                    rem_ii, rem_jj, rem_kk = rem[4], rem[5], rem[6]
                    rem_best = decompose_best(rem_nc, rem_mc, rem_pc, rem_ii, rem_jj, rem_kk, n_dim, m_dim, p_dim)
                    if rem_best is None:
                        return None
                    rem_cols.extend(rem_best)
                
                total = cols + rem_cols
                return total
        
        def try_split(dim, split_size):
            """Split along dim at position split_size"""
            if time.time() > deadline:
                return None
            if dim == 'n':
                if n_curr >= 2:
                    c1 = decompose_best(split_size, m_curr, p_curr, i0, j0, k0, n_dim, m_dim, p_dim)
                    if c1 is None:
                        return None
                    c2 = decompose_best(n_curr - split_size, m_curr, p_curr, i0 + split_size, j0, k0, n_dim, m_dim, p_dim)
                    if c2 is None:
                        return None
                    return c1 + c2
            elif dim == 'm':
                if m_curr >= 2:
                    c1 = decompose_best(n_curr, split_size, p_curr, i0, j0, k0, n_dim, m_dim, p_dim)
                    if c1 is None:
                        return None
                    c2 = decompose_best(n_curr, m_curr - split_size, p_curr, i0, j0 + split_size, k0, n_dim, m_dim, p_dim)
                    if c2 is None:
                        return None
                    return c1 + c2
            elif dim == 'p':
                if p_curr >= 2:
                    c1 = decompose_best(n_curr, m_curr, split_size, i0, j0, k0, n_dim, m_dim, p_dim)
                    if c1 is None:
                        return None
                    c2 = decompose_best(n_curr, m_curr, p_curr - split_size, i0, j0, k0 + split_size, n_dim, m_dim, p_dim)
                    if c2 is None:
                        return None
                    return c1 + c2
            return None
        
        # Base case
        if n_curr == 1 and m_curr == 1 and p_curr == 1:
            u_idx = i0 * m_dim + j0
            v_idx = j0 * p_dim + k0
            w_idx = k0 * n_dim + i0
            return [({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0})]
        
        # Try Strassen 2x2x2
        strassen_result = try_strassen()
        if strassen_result is not None and len(strassen_result) < best_rank:
            best_rank = len(strassen_result)
            best_cols = strassen_result
        
        # Try splitting each dimension at each possible split point
        for dim, total in [('n', n_curr), ('m', m_curr), ('p', p_curr)]:
            for split_size in range(1, total):
                result = try_split(dim, split_size)
                if result is not None and len(result) < best_rank:
                    best_rank = len(result)
                    best_cols = result
        
        if best_cols is None:
            # Fallback: simple split
            if n_curr > 1:
                c1 = decompose_best(1, m_curr, p_curr, i0, j0, k0, n_dim, m_dim, p_dim)
                c2 = decompose_best(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0, n_dim, m_dim, p_dim)
                return (c1 or []) + (c2 or [])
            elif m_curr > 1:
                c1 = decompose_best(n_curr, 1, p_curr, i0, j0, k0, n_dim, m_dim, p_dim)
                c2 = decompose_best(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0, n_dim, m_dim, p_dim)
                return (c1 or []) + (c2 or [])
            elif p_curr > 1:
                c1 = decompose_best(n_curr, m_curr, 1, i0, j0, k0, n_dim, m_dim, p_dim)
                c2 = decompose_best(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1, n_dim, m_dim, p_dim)
                return (c1 or []) + (c2 or [])
            else:
                u_idx = i0 * m_dim + j0
                v_idx = j0 * p_dim + k0
                w_idx = k0 * n_dim + i0
                return [({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0})]
        
        return best_cols

    def eliminate_redundant_cols(cols):
        """Try to eliminate columns that are linear combinations of others."""
        if not cols:
            return cols
        
        rank = len(cols)
        if rank <= 1:
            return cols
        
        # Build matrices
        n_dim, m_dim, p_dim = DIMENSIONS
        U = np.zeros((n_dim * m_dim, rank))
        V = np.zeros((m_dim * p_dim, rank))
        W = np.zeros((p_dim * n_dim, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(cols):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val
        
        # Check if we can remove column r by expressing it as combination of others
        # This is a complex optimization; for now, just return as-is
        return cols

    cols = decompose_best(n, m, p, 0, 0, 0, n, m, p)
    
    if cols is None:
        # Fallback to simple schoolbook
        cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    cols.append(({i * m + j: 1.0}, {j * p + k: 1.0}, {k * n + i: 1.0}))
    
    rank = len(cols)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
