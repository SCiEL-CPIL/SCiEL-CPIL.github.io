"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

from fractions import Fraction as Fr
import random
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 16 is schoolbook. We aim for lower.
    # Known lower bounds suggest 13 might be possible? Or 14?
    # Let's try to find a decomposition with rank < 16.
    # The task says "rank-minimal".
    # For 2x2 * 2x4, the tensor is T_{i,j,k} = delta_{i,j} delta_{j,k}? No.
    # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # Indices: i in [0,1], j in [0,1], k in [0,3].
    # Tensor T_{i,j,k} is 1 if the multiplication involves A[i,j] and B[j,k].
    # Since A is 2x2 and B is 2x4, j is the shared index.
    # So T_{i,j,k} = 1 for all i, j, k?
    # Wait, the tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} = 1 if the product C[i,k] depends on A[i,j] and B[j,k].
    # In standard matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
    # So the term A[i,j]B[j,k] contributes to C[i,k].
    # Thus the tensor is T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's the "generic" tensor where every combination exists.
    # Let's check the schoolbook code:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
    # This corresponds to the term A[i,j] * B[j,k] contributing to C[i,k].
    # The output index in W is k*n + i.
    # The input indices for U are i*m + j (row i, col j of A).
    # The input indices for V are j*p + k (row j, col k of B).
    # So yes, the tensor is the full tensor of 1s?
    # No, the tensor T is defined by T_{i,j,k} = 1 if the monomial A_{i,j} B_{j,k} appears in C_{i,k}.
    # In standard matmul, it does.
    # So T_{i,j,k} = 1 for all i,j,k.
    # This is the tensor of ones? No, it's the structure tensor.
    # The rank of the all-ones tensor of size n x m x p is not necessarily n*m*p.
    # Actually, the tensor T_{i,j,k} = 1 is the "all-ones" tensor.
    # The rank of the all-ones tensor of size 2x2x4?
    # Let's check small cases.
    # 2x2x2 all-ones tensor rank is 3? (Strassen-like?)
    # Actually, the all-ones tensor of size n x n x n has rank n? No.
    # Let's look at the specific structure.
    # The tensor is T_{i,j,k} = 1.
    # We want to decompose T = sum_r u^{(r)} \otimes v^{(r)} \otimes w^{(r)}.
    # This is equivalent to finding the rank of the tensor.
    
    # Let's try a randomized search for low rank.
    # We will attempt ranks from 8 up to 15.
    # If we find one, we return it.
    # If not, we return schoolbook.
    
    def verify(U, V, W, rank):
        """
        Verify if U, V, W (lists of lists of Fr) decompose the tensor.
        U: (n*m) x rank
        V: (m*p) x rank
        W: (p*n) x rank
        Tensor T_{i,j,k} = 1.
        We check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fr(0)
                    for r in range(rank):
                        s += Fr(U[i*m + j][r]) * Fr(V[j*p + k][r]) * Fr(W[k*n + i][r])
                    if s != 1:
                        return False
        return True

    def try_rank(rank):
        """
        Try to find a decomposition of given rank using randomized bilinear reduction.
        """
        # We will sample random vectors and solve for the third set.
        # Fix U and V randomly, then W is determined by the equations?
        # The equations are linear in W entries if U and V are fixed.
        # Number of equations: n*m*p = 16.
        # Number of unknowns in W: p*n*rank = 4*2*rank = 8*rank.
        # If rank >= 2, 8*rank >= 16, so it's underdetermined or square.
        # We can pick random U, V and solve for W.
        # But U, V, W must have rational entries.
        # Random rational numbers.
        
        # Let's try a structured approach.
        # The all-ones tensor is symmetric in a sense?
        # Actually, for 2x2x4, let's try rank 8, 9, ...
        
        # Strategy:
        # 1. Generate random rational U (2*2 x rank) and V (2*4 x rank).
        # 2. Set up linear system for W.
        #    For each (i,j,k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
        #    This is a linear system in the variables W[k,i,r].
        #    There are 16 equations and 8*rank variables.
        #    We can solve this using Gaussian elimination over rationals.
        
        for attempt in range(500):
            # Generate random rational U and V
            # Use small integers to keep numbers manageable
            U = [[random.randint(-10, 10) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-10, 10) for _ in range(rank)] for _ in range(m*p)]
            
            # Convert to Fraction for exact arithmetic
            U_fr = [[Fr(x) for x in row] for row in U]
            V_fr = [[Fr(x) for x in row] for row in V]
            
            # Build linear system A * W_vec = b
            # Variables: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,rank)
            # Total vars: p*n*rank
            num_vars = p * n * rank
            num_eqs = n * m * p
            
            A = [[Fr(0)] * num_vars for _ in range(num_eqs)]
            b = [Fr(1)] * num_eqs
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                        for r in range(rank):
                            var_idx = (k * n + i) * rank + r
                            A[eq_idx][var_idx] = U_fr[i*m + j][r] * V_fr[j*p + k][r]
                        eq_idx += 1
            
            # Solve A * x = b using Gaussian elimination
            # We need to find a solution x (which is W_vec)
            # Since it's underdetermined (usually), we can set some variables to 0 and solve for others.
            # Or use a rank-revealing decomposition.
            # Let's try to find a particular solution.
            
            # Augmented matrix
            M = [row + [b[i]] for i, row in enumerate(A)]
            
            # Gaussian elimination
            rows = len(M)
            cols = num_vars
            pivot = 0
            for r in range(rows):
                if pivot >= cols:
                    break
                # Find pivot
                found = False
                for i in range(r, rows):
                    if M[i][pivot] != 0:
                        found = True
                        break
                if not found:
                    pivot += 1
                    continue
                
                # Swap
                M[r], M[found] = M[found], M[r]
                
                # Normalize pivot row
                pv = M[r][pivot]
                for j in range(pivot, cols + 1):
                    M[r][j] /= pv
                
                # Eliminate
                for i in range(rows):
                    if i != r:
                        factor = M[i][pivot]
                        if factor != 0:
                            for j in range(pivot, cols + 1):
                                M[i][j] -= factor * M[r][j]
                
                pivot += 1
            
            # Check for inconsistency
            for i in range(rows):
                # Check if row is [0 ... 0 | non-zero]
                if all(M[i][j] == 0 for j in range(cols)):
                    if M[i][cols] != 0:
                        return False  # Inconsistent
            
            # Construct a solution
            # Set free variables to 0
            x = [Fr(0)] * cols
            # Identify pivot columns
            pivot_cols = []
            for r in range(rows):
                for c in range(cols):
                    if M[r][c] != 0:
                        pivot_cols.append(c)
                        break
            
            # Back substitute
            for r in range(rows - 1, -1, -1):
                # Find pivot column for this row
                pc = None
                for c in range(cols):
                    if M[r][c] != 0:
                        pc = c
                        break
                if pc is None:
                    continue
                # x[pc] = M[r][cols] - sum_{c > pc} M[r][c] * x[c]
                val = M[r][cols]
                for c in range(pc + 1, cols):
                    val -= M[r][c] * x[c]
                x[pc] = val / M[r][pc]
            
            # x is a solution for W_vec
            # Reshape to W
            W = [[Fr(0)] * rank for _ in range(p*n)]
            for idx, val in enumerate(x):
                w_idx = idx // rank
                r_idx = idx % rank
                W[w_idx][r_idx] = val
            
            # Verify
            if verify([list(map(int, row)) if all(v.denominator == 1 for v in row) else row for row in U],
                      [list(map(int, row)) if all(v.denominator == 1 for v in row) else row for row in V],
                      W, rank):
                # We found a solution!
                # Convert to plain lists of numbers (int or Fraction)
                # The problem accepts integer/half-integer entries.
                # Our solution is rational. We can return Fractions or convert to floats?
                # "integer/half-integer entries" suggests we might need to scale?
                # No, it says "entries that exactly decompose". Fractions are exact.
                # But the interface expects lists. Let's return the lists of Fractions or numbers.
                # The evaluator likely converts to float or uses exact arithmetic.
                # Let's return the lists as they are (Fractions are hashable/comparable).
                # However, the seed returns lists of ints.
                # Let's try to return lists of Fractions.
                
                U_out = [list(row) for row in U_fr]
                V_out = [list(row) for row in V_fr]
                W_out = [list(row) for row in W]
                
                return {"U": U_out, "V": V_out, "W": W_out}
        
        return None

    # Try ranks from 8 to 15
    for rank in range(8, 16):
        result = try_rank(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
