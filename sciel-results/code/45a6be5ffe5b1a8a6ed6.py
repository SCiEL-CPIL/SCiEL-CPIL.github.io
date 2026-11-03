"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _to_rational_list(matrix):
    """Convert a numpy array of floats (that are actually rational) to list of lists of Fractions."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = matrix[i][j]
            # Convert float to Fraction exactly
            # We assume the values are exact halves or integers
            frac = Fraction(val).limit_denominator(1000000)
            result[i][j] = frac
    return result

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should be 1 if all indices are valid (which they are) and 0 otherwise?
    No, for matrix multiplication, the tensor is defined such that:
    (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    This can be written as:
    C_{i,k} = sum_{j,c} A_{i,j} B_{j,k} W_{k,n,i,c} ... wait.
    
    Let's check the convention in the task:
    "output factor follows the official trace convention W[k*n+i,r]"
    And U, V, W shapes:
    U: (n*m, rank) -> index (i*m + j, r)
    V: (m*p, rank) -> index (j*p + k, r)
    W: (p*n, rank) -> index (k*n + i, r)
    
    The tensor T_{i,j,k} for matrix multiplication is 1 if we are multiplying A_{i,j} and B_{j,k}.
    Actually, the standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    T_{i,j,k} = delta_{i,j,k} ? No.
    
    The multiplication map is:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    In tensor form, the multiplication tensor T has indices (i, j, k) and entries:
    T_{i,j,k} = 1 if the operation is A_{i,j} * B_{j,k}?
    No, the tensor T defines the bilinear map.
    The map is (A, B) -> C.
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    
    This can be seen as a 3-tensor T_{i,j,k} where:
    T_{i,j,k} = 1 if we are using the term A_{i,j} B_{j,k}.
    But wait, the indices of T must correspond to the inputs A (i,j), B (j,k) and output C (i,k).
    Usually, the tensor is indexed by (input1, input2, output) or similar.
    
    Let's look at the factors:
    U_{i*m+j, r}, V_{j*p+k, r}, W_{k*n+i, r}.
    The product sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    This sum must equal the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    
    In the standard schoolbook algorithm:
    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    For a fixed (i,j,k), the coefficient is 1 if we are summing over j?
    No, for a specific i, k, we sum over j.
    So for a specific triple (i, j, k), the term A_{i,j} B_{j,k} appears in C_{i,k} with coefficient 1.
    So T_{i,j,k} = 1 for all valid i, j, k?
    No, that's not right. The tensor T is defined such that:
    C_{i,k} = sum_{j, r} A_{i,j} B_{j,k} ... no.
    
    Let's look at the rank-1 decomposition.
    A rank-1 term is U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    If we set U, V, W to be 1s at specific indices, we get a term.
    In the schoolbook algorithm, each term is A_{i,j} B_{j,k}.
    So for a fixed (i, j, k), we have a term.
    Wait, the indices of U are (i, j), V are (j, k), W are (k, i).
    The product is A_{i,j} * B_{j,k} * (contribution to C_{i,k}).
    So the sum over r of U_{i,j,r} V_{j,k,r} W_{k,i,r} should be 1 for each (i, j, k)?
    No, that would mean C_{i,k} = sum_{j} A_{i,j} B_{j,k} * 1.
    Yes! That's exactly it.
    So the condition is:
    For all i in [0, n), j in [0, m), k in [0, p):
    sum_{r=0}^{rank-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Let's verify with the schoolbook code:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1
    W[k*n+i][column] = 1
    For a specific (i, j, k), only the column corresponding to (i, j, k) has all three 1s.
    So the sum is 1.
    For any other (i', j', k') != (i, j, k), at least one of the indices is different, so the product is 0.
    So yes, the condition is:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank using random search or structured search
    # Since n=2, m=3, p=5, the schoolbook rank is 30.
    # Known lower bounds: max(nm, mp, pn) = max(6, 15, 10) = 15.
    # Strassen-like algorithms might reduce it, but for small sizes, it's tricky.
    # We can try random integer matrices and solve for W?
    # Or try to find a decomposition of rank R < 30.
    
    # Strategy: Random search for rank R
    # For a given R, pick random U, V with small integer entries.
    # Then for each (i, j, k), we need sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
    # This is a linear system in W.
    # Let A_{(i,j,k), r} = U[i*m+j, r] * V[j*p+k, r].
    # We want A * W_{k*n+i} = 1 for all (i, j, k).
    # Note that W has p*n rows. Each row of W is determined by a specific (k, i).
    # The equations are coupled: for a fixed (k, i), we have m equations (one for each j).
    # So for each (k, i), we have a system of m equations in R unknowns (the entries of W[k*n+i, :]).
    # But the same W row is used for all j.
    # So we need to find W[k*n+i] such that for all j:
    # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1.
    
    # This is a linear system for each (k, i).
    # Let's try to solve this for small R.
    
    def try_rank(R):
        if time.time() > deadline:
            return None
        
        # We will try to find U, V, W of rank R.
        # We can fix U and V randomly and solve for W.
        # Or fix U and W and solve for V.
        # Let's fix U and V with random small integers (-1, 0, 1) and solve for W.
        
        import numpy as np
        np.random.seed(42) # Fixed seed for reproducibility in this attempt
        
        # Try a few random U, V pairs
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U_rand = np.random.randint(-1, 2, size=(n*m, R))
            V_rand = np.random.randint(-1, 2, size=(m*p, R))
            
            # Convert to lists for Fraction arithmetic
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            
            W_list = [[0] * R for _ in range(p*n)]
            
            success = True
            for k in range(p):
                for i in range(n):
                    # System of m equations for W[k*n+i, :]
                    # Equations: for each j in 0..m-1:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    
                    # Build the system matrix B (m x R) and vector b (m x 1)
                    B = np.zeros((m, R), dtype=object)
                    b = np.ones(m, dtype=object)
                    
                    for j in range(m):
                        for r in range(R):
                            B[j, r] = U_list[i*m+j][r] * V_list[j*p+k][r]
                    
                    # Solve B * W_row = b
                    # We need an exact solution.
                    # Use Gaussian elimination with Fractions.
                    W_row = _solve_linear_system(B, b)
                    if W_row is None:
                        success = False
                        break
                    W_list[k*n+i] = W_row
                if not success:
                    break
            
            if success:
                # Verify the solution
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
        
        return None
    
    # Try ranks from 15 to 29
    for R in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = R
            break  # We found a lower rank, stop searching for lower ones?
                    # Or continue to find even lower?
                    # Let's continue to find the minimum.
    
    return best_result

def _solve_linear_system(A, b):
    """
    Solve A x = b using Gaussian elimination with Fractions.
    A is a list of lists (or numpy array of objects) of size m x n.
    b is a list of size m.
    Returns x as a list of size n, or None if no unique solution.
    """
    m = len(A)
    n = len(A[0])
    
    # Convert to Fractions
    A_frac = [[Fraction(A[i][j]) for j in range(n)] for i in range(m)]
    b_frac = [Fraction(b[i]) for i in range(m)]
    
    # Augmented matrix
    M = [A_frac[i] + [b_frac[i]] for i in range(m)]
    
    # Gaussian elimination
    for col in range(min(m, n)):
        # Find pivot
        pivot_row = None
        for row in range(col, m):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Eliminate
        for row in range(m):
            if row == col:
                continue
            factor = M[row][col] / M[col][col]
            for k in range(col, n + 1):
                M[row][k] -= factor * M[col][k]
    
    # Check for consistency and extract solution
    x = [Fraction(0) for _ in range(n)]
    for i in range(m):
        # Check if row is all zeros except last
        if all(M[i][j] == 0 for j in range(n)):
            if M[i][n] != 0:
                return None  # Inconsistent
            continue
        # Find leading variable
        for j in range(n):
            if M[i][j] != 0:
                if j != i:  # Not in reduced form, but we can still extract if it's a unit matrix
                    # This simple implementation assumes we get a unit matrix for square systems
                    # For non-square, it's more complex. Let's assume m <= n and we get a partial identity.
                    pass
                x[j] = M[i][n] / M[i][j]
                break
    
    # Verify solution
    for i in range(m):
        s = sum(A_frac[i][j] * x[j] for j in range(n))
        if s != b_frac[i]:
            return None
    
    return x

# EVOLVE-BLOCK-END
