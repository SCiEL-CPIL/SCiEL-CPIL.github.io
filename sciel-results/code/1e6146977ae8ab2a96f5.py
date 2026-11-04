"""Matrix multiplication 3x5 by 5x6: rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 120.0
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook_2x2x1(a_rows, c_cols, d_cols):
        """2x2x1: A is 2x2, B is 2x1, C is 2x1. Rank 4."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,1), 1)], [(V(1,0), 1)], [(W(0,1), 1)]))
        
    def add_schoolbook_2x1x2(a_rows, c_cols, d_cols):
        """2x1x2: A is 2x1, B is 1x2, C is 2x2. Rank 4."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,1), 1)], [(W(1,1), 1)]))
        
    def add_schoolbook_1x2x2(a_rows, c_cols, d_cols):
        """1x2x2: A is 1x2, B is 2x2, C is 1x2. Rank 4."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,1), 1)], [(W(1,0), 1)]))
        
    def add_schoolbook_2x1x1(a_rows, c_cols, d_cols):
        """2x1x1: A is 2x1, B is 1x1, C is 2x1. Rank 2."""
        def U(la, lc):
            return a_rows[la] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[la]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(1,0), 1)], [(V(0,0), 1)], [(W(0,1), 1)]))
        
    def add_schoolbook_1x2x1(a_rows, c_cols, d_cols):
        """1x2x1: A is 1x2, B is 2x1, C is 1x1. Rank 2."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[lc]
        def V(lc, ld):
            return c_cols[lc] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,1), 1)], [(V(1,0), 1)], [(W(0,0), 1)]))
        
    def add_schoolbook_1x1x2(a_rows, c_cols, d_cols):
        """1x1x2: A is 1x1, B is 1x2, C is 1x2. Rank 2."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[ld]
        def W(ld, la):
            return d_cols[ld] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        terms.append(([(U(0,0), 1)], [(V(0,1), 1)], [(W(1,0), 1)]))
        
    def add_schoolbook_1x1x1(a_rows, c_cols, d_cols):
        """1x1x1: A is 1x1, B is 1x1, C is 1x1. Rank 1."""
        def U(la, lc):
            return a_rows[0] * m + c_cols[0]
        def V(lc, ld):
            return c_cols[0] * p + d_cols[0]
        def W(ld, la):
            return d_cols[0] * n + a_rows[0]
        
        terms.append(([(U(0,0), 1)], [(V(0,0), 1)], [(W(0,0), 1)]))
        
    # Iterate over all block combinations
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                if time.time() - _start_time > TIME_LIMIT - 5:
                    break
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 1 and p_size == 1:
                    add_schoolbook_2x1x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 2 and p_size == 1:
                    add_schoolbook_1x2x1(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(a_rows, c_cols, d_cols)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(a_rows, c_cols, d_cols)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
    # Build U, V, W matrices as lists of entries
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
    
    # Try to find a linear dependency among the rank terms to reduce rank
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We want to find if one term is a linear combination of others
    # Use Gaussian elimination on the system of bilinear equations
    
    # Build the dense tensor representation for verification
    # T[i,j,k] for i in range(n*m), j in range(m*p), k in range(p*n)
    # But we only care about entries where the indices are "compatible"
    # i = a*m + c, j = c'*p + b, k = b'*n + a'
    # Valid when c == c' and b == b' (for matrix multiplication)
    
    # Actually, for the decomposition to be valid, we need:
    # For all a, c, b: sum_r U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r] = 1 if (a,c,b) forms a valid triple
    # And for all other combinations, the sum should be 0.
    
    # Let's try to find a linear dependency by checking if the columns of the 
    # "bilinear form" matrix are linearly dependent.
    
    # For each term r, define a bilinear form B_r(i,j) = U[i,r] * V[j,r]
    # The tensor entry T[i,j,k] = sum_r B_r(i,j) * W[k,r]
    
    # A linear dependency among terms means there exist alpha_r (not all zero) such that
    # for all (i,j,k): sum_r alpha_r * U[i,r]*V[j,r]*W[k,r] = 0
    
    # This is a system of (n*m)*(m*p)*(p*n) equations in rank unknowns, but most are trivially satisfied.
    # The non-trivial constraints are:
    # 1. For valid triples (a,c,b): sum_r alpha_r * U[a*m+c,r]*V[c*p+b,r]*W[b*n+a,r] = 0
    # 2. For invalid triples: sum_r alpha_r * U[i,r]*V[j,r]*W[k,r] = 0
    
    # Let's set up the system for valid triples only (n*m*p = 90 equations)
    # and check if there's a non-trivial solution.
    
    if time.time() - _start_time > TIME_LIMIT - 30:
        return {"U": U, "V": V, "W": W}
    
    # Build coefficient matrix for the system:
    # For each valid triple (a, c, b) and each term r, coefficient is U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r]
    # We want to find alpha such that for all (a,c,b): sum_r alpha_r * coeff[(a,c,b), r] = 0
    
    # But actually, we want to eliminate a term, so we need to check if one term's
    # "pattern" is a linear combination of the others.
    
    # Alternative approach: check if the rank of the 90x84 matrix is < 84
    # If so, there's a dependency.
    
    # Let's build the 90x84 matrix M where M[(a,c,b), r] = U[a*m+c,r] * V[c*p+b,r] * W[b*n+a,r]
    # Then find if there's a vector alpha in the null space.
    
    num_equations = 0
    eq_indices = []
    for a in range(n):
        for c in range(m):
            for b in range(p):
                eq_indices.append((a, c, b))
                num_equations += 1
    
    # Build matrix as list of lists of Fractions
    mat = []
    for (a, c, b) in eq_indices:
        row = []
        for r in range(rank):
            val = U[a*m+c][r] * V[c*p+b][r] * W[b*n+a][r]
            row.append(Fraction(val))
        mat.append(row)
    
    # Gaussian elimination to find null space
    # We want to find a non-trivial alpha such that M @ alpha = 0
    # This is equivalent to finding if the columns are linearly dependent
    
    # Perform Gaussian elimination on the transpose (rank x 90) to find column dependencies
    # Actually, let's just do row reduction on the 90 x 84 matrix and check rank
    
    if time.time() - _start_time > TIME_LIMIT - 20:
        return {"U": U, "V": V, "W": W}
    
    # Copy matrix for elimination
    mat_copy = [row[:] for row in mat]
    nrows = len(mat_copy)
    ncols = rank
    
    pivot_cols = []
    for col in range(ncols):
        # Find pivot
        pivot_row = -1
        for row in range(len(pivot_cols), nrows):
            if mat_copy[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap
        if pivot_row != len(pivot_cols):
            mat_copy[pivot_row], mat_copy[len(pivot_cols)] = mat_copy[len(pivot_cols)], mat_copy[pivot_row]
        # Eliminate
        for row in range(len(pivot_cols) + 1, nrows):
            if mat_copy[row][col] != 0:
                factor = mat_copy[row][col] / mat_copy[len(pivot_cols)][col]
                for c2 in range(col, ncols):
                    mat_copy[row][c2] -= factor * mat_copy[len(pivot_cols)][c2]
        pivot_cols.append(col)
    
    actual_rank = len(pivot_cols)
    
    if actual_rank < rank:
        # There's a dependency. Find which term can be eliminated.
        # The non-pivot columns are linearly dependent on pivot columns.
        non_pivot_cols = [c for c in range(ncols) if c not in pivot_cols]
        
        if non_pivot_cols:
            # Try to eliminate the last non-pivot column
            elim_col = non_pivot_cols[-1]
            
            # Find the coefficients: express column elim_col as linear combination of pivot columns
            # From the reduced matrix, for each pivot row i with pivot in column pivot_cols[i]:
            # mat_copy[i][elim_col] gives the coefficient of pivot column i in the combination
            # (with sign)
            
            # Actually, in row echelon form, the non-pivot columns give the null space vectors.
            # Let's find the null space vector corresponding to eliminating column elim_col.
            
            # Set alpha[elim_col] = 1, alpha[non-pivot others] = 0
            # For pivot columns, alpha[pivot_cols[i]] = -mat_copy[i][elim_col] / mat_copy[i][pivot_cols[i]]
            
            alpha = [Fraction(0)] * rank
            alpha[elim_col] = Fraction(1)
            
            for i, pc in enumerate(pivot_cols):
                if mat_copy[i][pc] != 0:
                    alpha[pc] = -mat_copy[i][elim_col] / mat_copy[i][pc]
            
            # Verify: M @ alpha should be 0
            # (Skip verification for speed, trust the math)
            
            # Now eliminate term elim_col by adding alpha * term_elim_col to other terms
            # New term r (for r != elim_col): U_new[r] = U[r] + alpha[r] * U[elim_col]
            # Wait, that's not right. We need:
            # sum_r alpha_r * U[i,r]*V[j,r]*W[k,r] = 0
            # So if we define new terms by absorbing term elim_col into others:
            # For each r != elim_col, we can't simply add because the product structure doesn't allow it.
            
            # Actually, the correct approach is: if sum_r alpha_r * T_r = 0 where T_r is the rank-1 tensor
            # U_r ⊗ V_r ⊗ W_r, then we can remove term elim_col if we can express it as a combination.
            # But we can't just add rank-1 tensors to get a new rank-1 tensor.
            
            # The dependency means the 84 rank-1 tensors are linearly dependent, but that doesn't
            # directly give us a rank-83 decomposition. We need to find a different set of 83 rank-1
            # tensors that span the same subspace.
            
            # One approach: if alpha has a nonzero entry at elim_col, we can try to "absorb"
            # term elim_col by modifying other terms. But this generally doesn't work for rank-1 tensors.
            
            # Alternative: use the dependency to find a different partition or to verify
            # that the rank is actually 83 by finding an explicit 83-term decomposition.
            
            # For now, let's just report that a dependency exists and try a different approach.
            # We could try to find a 83-term decomposition by solving a least-squares problem
            # with the constraint that one term is eliminated.
            
            # Actually, let's try a simpler approach: check if we can find a different
            # partition that gives 83 terms directly.
            
            pass
    
    # For now, return the current solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
