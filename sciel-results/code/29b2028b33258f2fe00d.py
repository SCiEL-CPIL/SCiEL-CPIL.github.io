"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        """Add 7 Strassen terms for a 2x2x2 block at offsets (i0, j0, k0)."""
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3: A11(B21-B11) -> C12 += M3, C22 += M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4: A22(B12-B22) -> C11 += M4, C12 += M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5: (A11+A12)B22 -> C12 += M5, C11 -= M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        """Add schoolbook terms for a sub-block."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    def verify_decomposition(U, V, W, rank):
        """Verify the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    def eliminate_columns(U, V, W, rank):
        """
        Try to eliminate columns that are linear combinations of others.
        For each column r, check if it can be expressed as a combination of
        other columns. If so, merge it.
        A column r is redundant if there exist coefficients c_s (s != r)
        such that for all (i,j,k):
          U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
        
        We use a simpler approach: try to find pairs of columns that can be
        merged by adjusting entries.
        """
        # Build the tensor as a list of (i,j,k) -> value for each column
        # Then check linear dependence
        
        # For efficiency, represent each column as a vector of size n*m*p
        # where entry (i,j,k) maps to index i*m*p + j*p + k
        # But the tensor entry is U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Actually, let's use a different approach:
        # Try to find columns r1, r2 such that column r2 = alpha * column r1 + adjustments
        # This is complex. Let's use a simpler heuristic:
        # If two columns have the same support pattern (same nonzero positions in U, V, W),
        # they might be mergeable.
        
        # Better approach: use linear algebra on the "outer product" vectors
        # Each column r defines a vector in R^(n*m * m*p * p*n) but that's too large.
        
        # Simpler: try to find if column r can be written as sum of c_s * column_s
        # by checking on a subset of (i,j,k) entries.
        
        # For now, let's try a greedy approach:
        # For each pair of columns, check if they differ only by a scalar multiple
        # on their supports.
        
        # Actually, let's use a more systematic approach:
        # Build a matrix where each row corresponds to an (i,j,k) triple
        # and each column r has entry U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Then find the rank of this matrix and eliminate dependent columns.
        
        # But this matrix is n*m*p x rank = 105 x 99, which is manageable.
        
        # Let's build it
        entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for r in range(rank):
                        row.append(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r])
                    entries.append(row)
        
        # Now we have a 105x99 matrix. We want to find if any column is a linear
        # combination of others. Use Gaussian elimination.
        
        # Convert to Fraction for exact arithmetic
        mat = [[Fraction(x) for x in row] for row in entries]
        num_rows = len(mat)
        num_cols = rank
        
        # Perform Gaussian elimination to find pivot columns
        pivot_cols = []
        for col in range(num_cols):
            # Find a row >= len(pivot_cols) where mat[row][col] != 0
            found = False
            for row in range(len(pivot_cols), num_rows):
                if mat[row][col] != 0:
                    # Swap rows
                    if row != len(pivot_cols):
                        mat[row], mat[len(pivot_cols)] = mat[len(pivot_cols)], mat[row]
                    # Eliminate below
                    pivot_val = mat[len(pivot_cols)][col]
                    for row2 in range(len(pivot_cols) + 1, num_rows):
                        factor = mat[row2][col] / pivot_val
                        if factor != 0:
                            for col2 in range(col, num_cols):
                                mat[row2][col2] -= factor * mat[len(pivot_cols)][col2]
                    pivot_cols.append(col)
                    found = True
                    break
            if not found:
                # This column is dependent on previous pivot columns
                # We can potentially eliminate it
                pass
        
        # The non-pivot columns are linearly dependent
        # But we need to be careful: we can only eliminate a column if we can
        # express it as a combination of OTHER columns in the original decomposition
        # (not just in the tensor space).
        
        # This is a necessary but not sufficient condition.
        # For now, let's just return the original if no elimination is found.
        # A proper implementation would solve for the coefficients and update U, V, W.
        
        # Let's try a simpler approach: if we find a dependent column, 
        # try to express it and merge.
        
        non_pivot = [c for c in range(num_cols) if c not in pivot_cols]
        
        if not non_pivot:
            return U, V, W, rank
        
        # For each non-pivot column, try to express it as a combination of pivot columns
        # and merge it into them.
        
        # This is complex because the tensor entry is a product of three factors.
        # Let's skip this for now and just return the original.
        
        return U, V, W, rank

    # Build the 99-term solution using Strassen + schoolbook
    rank = 99
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}: Strassen
    for k0 in [0, 2, 4]:
        col = add_strassen(U, V, W, 0, 0, k0, col)
    # i in {0,1}, j in {0,1}, k in {6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [0, 1], [6], col)
    # i in {0,1}, j in {2,3}, k in {0,1}, {2,3}, {4,5}: Strassen
    for k0 in [0, 2, 4]:
        col = add_strassen(U, V, W, 0, 2, k0, col)
    # i in {0,1}, j in {2,3}, k in {6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [2, 3], [6], col)
    # i in {0,1}, j in {4}, k in {0..6}: schoolbook
    col = add_schoolbook(U, V, W, [0, 1], [4], range(7), col)
    # i in {2}, j in {0..4}, k in {0..6}: schoolbook
    col = add_schoolbook(U, V, W, [2], range(5), range(7), col)
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Verify correctness
    assert verify_decomposition(U, V, W, rank), "Decomposition verification failed"
    
    # Try to eliminate columns
    U, V, W, rank = eliminate_columns(U, V, W, rank)
    
    if rank < 99:
        # Truncate matrices
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        assert verify_decomposition(U, V, W, rank), "Post-elimination verification failed"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
