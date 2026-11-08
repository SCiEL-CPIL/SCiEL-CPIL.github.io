"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition using structured search
    # For 2x3*3x7, the matrix multiplication tensor has dimensions 6x21x14
    # We look for rank r where r < 42
    
    def verify_decomposition(U, V, W, r):
        """Verify that U, V, W of rank r correctly decompose matrix multiplication."""
        # The tensor T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # But wait, the convention is:
        # U: (n*m, r), indexed by (i*m+j)
        # V: (m*p, r), indexed by (j*p+k)
        # W: (p*n, r), indexed by (k*n+i)
        # The product is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal 1 if we're looking at the (i,j,k) entry of the multiplication tensor
        # which is 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
        # Actually the tensor is T[i,j,k] = delta_{i,j,k} which is always 1 for valid indices
        # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # because (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition using structured approach."""
        # For small ranks, we can use a greedy or structured approach
        # The key insight: we need to find U, V, W such that the tensor factorizes
        
        # Try a simple structured form first
        # Based on known results, 2x3*3x7 might have rank less than 42
        
        # Use a randomized approach with verification
        import random
        random.seed(42)
        
        max_attempts = 100
        deadline = time.time() + 120
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Try random integer/half-integer matrices
            # For simplicity, try small integer values first
            U = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(n*m)]
            V = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(m*p)]
            W = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(p*n)]
            
            if verify_decomposition(U, V, W, r):
                return U, V, W
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound for 2x3*3x7 is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Actually the lower bound is more subtle, but let's try from 21 up
    
    for r in range(21, schoolbook_rank):
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
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
