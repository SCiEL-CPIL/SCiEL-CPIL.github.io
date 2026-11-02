import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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

    def direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        """Direct schoolbook decomposition for a block."""
        for di in range(nc):
            for dj in range(mc):
                for dk in range(pc):
                    u_idx = (ii + di) * m_dim + (jj + dj)
                    v_idx = (jj + dj) * p_dim + (kk + dk)
                    w_idx = (kk + dk) * n_dim + (ii + di)
                    cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

    def decompose_recursive(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        """
        Recursive decomposition trying to minimize rank.
        """
        if time.time() > deadline:
            return
        
        # Base case: 1x1x1
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        
        # If any dimension is 1, use direct decomposition (Strassen not applicable)
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        
        # Try Strassen 2x2x2 if all dimensions >= 2
        if nc >= 2 and mc >= 2 and pc >= 2:
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            
            # Recurse on remainders
            # 1. (nc-2) x mc x pc at (ii+2, jj, kk)
            if nc > 2:
                decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            
            # 2. 2 x (mc-2) x pc at (ii, jj+2, kk)
            if mc > 2:
                decompose_recursive(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            
            # 3. 2 x 2 x (pc-2) at (ii, jj, kk+2)
            if pc > 2:
                decompose_recursive(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            
            return
        
        # Fallback: split one dimension
        if nc > 1:
            decompose_recursive(1, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc - 1, mc, pc, ii + 1, jj, kk, n_dim, m_dim, p_dim, cols)
        elif mc > 1:
            decompose_recursive(nc, 1, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc - 1, pc, ii, jj + 1, kk, n_dim, m_dim, p_dim, cols)
        elif pc > 1:
            decompose_recursive(nc, mc, 1, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            decompose_recursive(nc, mc, pc - 1, ii, jj, kk + 1, n_dim, m_dim, p_dim, cols)

    def decompose_split_first(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        """
        Alternative: split the largest dimension first, then apply Strassen.
        """
        if time.time() > deadline:
            return
        
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        
        # Just use the recursive approach but with a twist: prefer splitting the smallest dimension
        min_dim = min(nc, mc, pc)
        if min_dim == nc:
            if nc >= 2:
                decompose_recursive(2, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                if nc > 2:
                    decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            else:
                direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
        elif min_dim == mc:
            if mc >= 2:
                decompose_recursive(nc, 2, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                if mc > 2:
                    decompose_recursive(nc, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            else:
                direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
        else:  # min_dim == pc
            if pc >= 2:
                decompose_recursive(nc, mc, 2, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                if pc > 2:
                    decompose_recursive(nc, mc, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            else:
                direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)

    def decompose_split_largest(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols):
        """
        Alternative: split the largest dimension first.
        """
        if time.time() > deadline:
            return
        
        if nc == 1 and mc == 1 and pc == 1:
            u_idx = ii * m_dim + jj
            v_idx = jj * p_dim + kk
            w_idx = kk * n_dim + ii
            cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
            return
        
        if nc == 1 or mc == 1 or pc == 1:
            direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
            return
        
        # Prefer splitting the largest dimension
        max_dim = max(nc, mc, pc)
        if max_dim == nc:
            if nc >= 2:
                decompose_recursive(2, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                if nc > 2:
                    decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            else:
                direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
        elif max_dim == mc:
            if mc >= 2:
                decompose_recursive(nc, 2, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                if mc > 2:
                    decompose_recursive(nc, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            else:
                direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)
        else:  # max_dim == pc
            if pc >= 2:
                decompose_recursive(nc, mc, 2, ii, jj, kk, n_dim, m_dim, p_dim, cols)
                if pc > 2:
                    decompose_recursive(nc, mc, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            else:
                direct_decompose(nc, mc, pc, ii, jj, kk, n_dim, m_dim, p_dim, cols)

    def try_l1_optimization(cols_input):
        """
        Attempt to reduce rank by solving an L1 optimization problem.
        This is a heuristic attempt to find a sparser representation.
        """
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        rank = len(cols_input)
        
        # Build the linear system
        # Variables: U (n*m*rank), V (m*p*rank), W (p*n*rank)
        # Total variables: (n*m + m*p + p*n) * rank
        
        # For each (i, j, k) in the tensor:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a nonlinear problem (trilinear). 
        # We can linearize by fixing two matrices and solving for the third.
        # Let's try fixing U and V from the input and solving for W using L1.
        
        # Actually, let's just try to see if we can drop some columns.
        # We'll try all subsets of columns of size rank-1, rank-2, etc.
        # This is expensive, so we'll limit it.
        
        # For now, return None to indicate no improvement found.
        return None

    # Strategy 1: Original recursive
    cols1 = []
    decompose_recursive(n, m, p, 0, 0, 0, n, m, p, cols1)
    rank1 = len(cols1)
    
    # Strategy 2: Split smallest first
    cols2 = []
    decompose_split_first(n, m, p, 0, 0, 0, n, m, p, cols2)
    rank2 = len(cols2)
    
    # Strategy 3: Split largest first
    cols3 = []
    decompose_split_largest(n, m, p, 0, 0, 0, n, m, p, cols3)
    rank3 = len(cols3)
    
    # Pick the best
    best_rank = min(rank1, rank2, rank3)
    if rank1 == best_rank:
        cols = cols1
        rank = rank1
    elif rank2 == best_rank:
        cols = cols2
        rank = rank2
    else:
        cols = cols3
        rank = rank3
    
    # Try L1 optimization to potentially reduce rank further
    optimized_cols = try_l1_optimization(cols)
    if optimized_cols is not None and len(optimized_cols) < rank:
        cols = optimized_cols
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
