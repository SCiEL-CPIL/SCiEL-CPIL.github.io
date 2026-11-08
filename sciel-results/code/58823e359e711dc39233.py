"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 if i,j,k are valid, else 0
    # Actually, the tensor T[i,j,k] = delta_ij * delta_jk for matmul... 
    # For matrix multiplication: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T[i,j,k] = 1 if this term contributes, i.e., it's the "multiplication tensor"
    # T[i,j,k] = 1 for all i,j,k (since every term A_ij * B_jk contributes to C_ik)
    # Wait, let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # if m is the shared dimension. Here n=3, m=3, p=5.
    # So T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i in [0,3), j in [0,3), k in [0,5):
    #   sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_r(r, deadline):
    """Try to find a rank-r decomposition using the slice method.
    
    The slice method: fix U and V to have a specific structure, then solve for W.
    For each rank-r candidate, we can try:
    - Fix U to be an (n*m) x r matrix with entries in {0, 1/2, 1}
    - Fix V to be an (m*p) x r matrix with entries in {0, 1/2, 1}  
    - Solve for W: for each (i,k), W[k*n+i, c] must satisfy
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
    
    This is a system of m equations in r unknowns for each (i,k) pair.
    """
    n, m, p = DIMENSIONS
    
    # For small r, we can try structured U and V
    # A common approach: try U and V with entries in {0, 1}
    # For rank r, U is (n*m) x r = 9 x r, V is (m*p) x r = 15 x r
    
    # Strategy: enumerate sparse binary U and V patterns
    # For very small r, the number of possible binary matrices is manageable
    # But 2^(9*r) is too large for r > 4
    
    # Alternative: use the fact that for matmul, there are known lower bounds
    # For 3x3 x 3x5, the rank is at least max(n*p, m*p, n*m) = max(15, 15, 9) = 15
    # Actually, the rank of the matmul tensor for n x m times m x p is at least n*p
    
    # Let's try ranks from 15 upward (since schoolbook is 45, and lower bound is 15)
    # Start from the theoretical lower bound
    
    # For the slice method to work well, we need r >= m = 3 (so the system is determined/overdetermined)
    # Let's try a few structured approaches
    
    # Approach 1: Try rank = 15 (theoretical lower bound)
    # With r = 15, U is 9x15, V is 15x15, W is 15x15
    # This is a lot of unknowns. Let's use a different approach.
    
    # Approach: Use the "embedding" trick
    # For matmul, if we can find rank-r decompositions for smaller cases,
    # we can embed them. But let's try direct search for small improvements.
    
    # Let's try rank 20, 25, 30, 35, 40 with structured search
    # Actually, let's be more systematic.
    
    # For a given rank r, the slice method says:
    # Fix U (9 x r) and V (15 x r). For each (i,k) in [0,3)x[0,5):
    #   We need W[k*3+i, :] such that for all j in [0,3):
    #     sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1
    # This is a 3x15 system (if r=15) or similar.
    
    # Let's try: for each (i,k), the system is:
    #   M_{jk} = sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] = 1 for j=0,1,2
    # This is linear in W[k*3+i, :].
    
    # So: for fixed U, V, we can solve for W row by row!
    # Each row of W (indexed by (k,i)) is determined by solving a linear system.
    
    # The question is: does a solution exist with W entries in {0, 1/2, 1}?
    # Or more generally, rational entries?
    
    # Let's try: enumerate U and V with entries in {0, 1, 2} (small integers)
    # and solve for W, then check if W has nice entries.
    
    # For r=15: U is 9x15, V is 15x15. Too many possibilities.
    
    # Let's try a smarter approach: use the known structure of matmul decompositions.
    # For square matmul, there are known decompositions. For rectangular, we can pad.
    
    # Actually, let's just try to verify if rank 15 is possible by solving the linear system.
    # For r=15, fix U and V to be identity-like structures.
    
    # Simpler approach: try rank values from 15 to 44 and for each, try a few structured U,V
    
    # Let's implement a basic search for rank r:
    # 1. Try U = [I_9 | 0_9x(r-9)] if r >= 9
    # 2. Try to solve for V and W
    
    # Actually, the cleanest approach for the time budget:
    # Return schoolbook if we can't find better quickly.
    
    # Let's try rank 25 with a structured approach
    # U: 9x25, V: 15x25, W: 15x25
    
    # For now, let's just return schoolbook as a safe fallback
    # and add a simple check for a few candidate ranks
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from 15 (lower bound) up to 44 (one less than schoolbook)
    # For each rank, try structured U and V matrices
    
    # The slice method: fix U (n*m x r) and V (m*p x r), solve for W (p*n x r)
    # For each (i,k), we need W[k*n+i, :] to satisfy:
    #   for all j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of m equations in r unknowns.
    
    # For the system to have a unique solution, we need m <= r and the coefficient matrix
    # (formed by rows [U[i*m+j, :] * V[j*p+k, :] for j=0..m-1]) to have full rank.
    
    # Let's try small r values with structured U, V
    
    # Helper: solve linear system A x = b using fractions for exactness
    def _solve_linear(A, b):
        """Solve A x = b exactly using fraction arithmetic.
        A is m x r, b is m x 1. Returns x (r x 1) or None if no unique solution."""
        import numpy as np
        A_frac = [[Fraction(A[i][j]) for j in range(len(A[0]))] for i in range(len(A))]
        b_frac = [Fraction(b[i]) for i in range(len(b))]
        
        # Gaussian elimination
        m_eq = len(A_frac)
        n_var = len(A_frac[0])
        
        # Check if system is consistent and has unique solution
        # We need rank(A) = rank([A|b]) = n_var for unique solution
        # Or rank(A) = rank([A|b]) < n_var for multiple solutions (we'd pick one)
        
        # Augmented matrix
        aug = [row[:] + [b_frac[i]] for i, row in enumerate(A_frac)]
        
        # Forward elimination
        rank = 0
        for col in range(n_var):
            # Find pivot
            pivot_row = None
            for row in range(rank, m_eq):
                if aug[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            # Swap
            aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
            # Eliminate
            for row in range(m_eq):
                if row != rank:
                    factor = aug[row][col] / aug[rank][col]
                    for j in range(col, n_var + 1):
                        aug[row][j] -= factor * aug[rank][j]
            rank += 1
            if rank == n_var:
                break
        
        # Check consistency
        for row in range(rank, m_eq):
            if aug[row][n_var] != 0:
                return None  # Inconsistent
        
        # If rank < n_var, we have free variables. Set them to 0.
        # Extract solution
        x = [Fraction(0)] * n_var
        for i in range(rank):
            # Find the column with pivot in row i
            pivot_col = None
            for col in range(n_var):
                if aug[i][col] != 0:
                    pivot_col = col
                    break
            if pivot_col is not None:
                x[pivot_col] = aug[i][n_var] / aug[i][pivot_col]
        
        # Verify solution
        for i in range(m_eq):
            s = sum(A_frac[i][j] * x[j] for j in range(n_var))
            if s != b_frac[i]:
                return None
        
        return x
    
    # Try structured U, V for each rank
    # For rank r, try U with entries in {0, 1} and V with entries in {0, 1}
    # But the space is huge. Let's try specific structured patterns.
    
    # Pattern 1: U = [I_{min(n*m, r)} | 0] (identity block)
    # Pattern 2: U with each row having exactly one 1
    # etc.
    
    # Given time constraints, let's try a limited search
    best_result = _schoolbook()
    
    # Try rank 15 with structured approach
    # For r=15, U is 9x15. Let's try U = [I_9 | 0_9x6]
    # Then for each (i,k), the system becomes:
    #   for j: sum_{c=0}^{8} delta_{i,j} * V[j*5+k, c] * W[k*3+i, c] + sum_{c=9}^{14} 0 * ... = 1
    #   So: V[i*5+k, 0..8] dot W[k*3+i, 0..8] = 1
    # This is 1 equation in 9 unknowns for each (i,k). Always solvable.
    # But we need W entries to be in {0, 1/2, 1}.
    
    # Let's try: U = [I_9 | 0], V = [I_15 | 0] (if r >= 15)
    # Then for (i,k): V[i*5+k, 0..14] has a 1 at position (i*5+k) mod 15
    # So the equation is: W[k*3+i, i*5+k] = 1
    # This gives W directly!
    
    # Let's check: if U = [I_9 | 0_{9x(r-9)}] and V = [I_15 | 0_{15x(r-15)}] for r >= 15
    # Then U[i*3+j, c] = 1 if c == i*3+j and c < 9, else 0
    # V[j*5+k, c] = 1 if c == j*5+k and c < 15, else 0
    # For the product U[i*3+j, c] * V[j*5+k, c] to be nonzero, we need c = i*3+j = j*5+k and c < min(9,15)=9
    # So i*3+j = j*5+k, which means i*3 = j*5 + k - j = j*4 + k
    # For this to hold for all i,j,k, we need... this doesn't work for all (i,j,k).
    
    # The issue: U[i*3+j, c] and V[j*5+k, c] share index c, but they're indexed differently.
    # U row index is i*m+j = i*3+j, V row index is j*p+k = j*5+k.
    # For the product to be 1, we need U[i*3+j, c] = V[j*5+k, c] = 1 for some c.
    # With U = I_9 block, U[i*3+j, c] = 1 iff c = i*3+j.
    # With V = I_15 block, V[j*5+k, c] = 1 iff c = j*5+k.
    # So we need i*3+j = j*5+k, i.e., i*3 = j*4 + k.
    # This is not true for all (i,j,k). For example, i=0,j=0,k=0: 0=0 OK.
    # i=0,j=0,k=1: 0=1? No.
    
    # So this simple identity approach doesn't work.
    
    # Let me try a different approach: use the fact that we can have U and V with overlapping support.
    
    # For the slice method to work:
    # Fix U (9 x r) and V (15 x r). For each (i,k) in [0,3)x[0,5):
    #   Build matrix A_{jk} = sum_c U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c]
    #   We need A_{jk} = 1 for all j.
    #   This is linear in W[k*3+i, :].
    
    # So for fixed U, V: for each (i,k), solve the linear system:
    #   M_{jk} = sum_c (U[i*3+j, c] * V[j*5+k, c]) * W[k*3+i, c] = 1
    #   where M is a 3 x r matrix with M[j, c] = U[i*3+j, c] * V[j*5+k, c]
    
    # The system is solvable if rank(M) = 3 (for unique solution) or consistent.
    
    # Let's try: for each rank r from 15 to 44, try a few structured U, V
    # and check if W can be solved with entries in {0, 1/2, 1}.
    
    # Structured U, V: try entries in {0, 1} with limited patterns
    # For r=15: U is 9x15, V is 15x15. Too many possibilities to enumerate.
    
    # Alternative: use random structured matrices and check.
    # But we need exact solutions, so random won't help much.
    
    # Given the time budget, let's just return schoolbook.
    # The search space is too large for a quick improvement.
    
    return best_result
# EVOLVE-BLOCK-END
