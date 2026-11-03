"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # Build the multiplication tensor T[i,j,k] = delta for (i,j,k) where C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all valid (i,j,k)
    
    # Schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # And for all other (i,j,k) combinations, the sum should be 0
        # Actually, the tensor T has T[i,j,k]=1 for all i,j,k in valid ranges
        # The decomposition must satisfy: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Check all (i,j,k) tuples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(len(U[0])):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    def try_merge_terms(U, V, W, c1, c2):
        """Try to replace terms c1 and c2 with a single term or keep as is.
        Returns None if no improvement, or new (U, V, W, new_rank) if successful."""
        
        # Check if terms c1 and c2 are identical
        # If they are, we can just remove one
        u1 = [U[i][c1] for i in range(len(U))]
        u2 = [U[i][c2] for i in range(len(U))]
        v1 = [V[i][c1] for i in range(len(V))]
        v2 = [V[i][c2] for i in range(len(V))]
        w1 = [W[i][c1] for i in range(len(W))]
        w2 = [W[i][c2] for i in range(len(W))]
        
        if u1 == u2 and v1 == v2 and w1 == w2:
            # Remove duplicate term c2
            new_U = [[U[i][c] for c in range(len(U[0])) if c != c2] for i in range(len(U))]
            new_V = [[V[i][c] for c in range(len(V[0])) if c != c2] for i in range(len(V))]
            new_W = [[W[i][c] for c in range(len(W[0])) if c != c2] for i in range(len(W))]
            return (new_U, new_V, new_W, len(U[0]) - 1)
        
        # Try to find if a single term can replace both
        # This is more complex - skip for now in initial implementation
        
        return None
    
    # Start with schoolbook
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # Try to find and remove duplicate terms
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_rank = len(best_U[0])
        for c1 in range(current_rank):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, current_rank):
                result = try_merge_terms(best_U, best_V, best_W, c1, c2)
                if result is not None:
                    best_U, best_V, best_W, best_rank = result
                    changed = True
                    break
            if changed:
                break
    
    # Additional search: try to find linear dependencies among terms
    # This is a more advanced technique that may reduce rank further
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
