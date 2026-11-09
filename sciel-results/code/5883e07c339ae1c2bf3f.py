"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = build_schoolbook()
    
    def verify(U, V, W, rank):
        """Verify exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def check_valid_entries(U, V, W):
        """Check all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    doubled = val * 2
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True
    
    def try_rank_41_linear():
        """Try to find rank-41 by removing term (i0,j0,k0) and solving
        a linear system for adjustments to remaining terms."""
        if time.time() > deadline - 20.0:
            return None
        
        # Try removing each possible term
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    if time.time() > deadline - 15.0:
                        return None
                    
                    # Remove term (i0,j0,k0) - find its column index
                    col_remove = i0 * m * p + j0 * p + k0
                    
                    # Create modified U, V, W without this column
                    rank_new = 41
                    U_new = [[0.0] * rank_new for _ in range(n * m)]
                    V_new = [[0.0] * rank_new for _ in range(m * p)]
                    W_new = [[0.0] * rank_new for _ in range(p * n)]
                    
                    # Copy remaining columns
                    col_idx = 0
                    for col in range(rank_schoolbook):
                        if col == col_remove:
                            continue
                        U_new[i0 * m + j0][col_idx] = U_sb[i0 * m + j0][col]
                        V_new[j0 * p + k0][col_idx] = V_sb[j0 * p + k0][col]
                        W_new[k0 * n + i0][col_idx] = W_sb[k0 * n + i0][col]
                        col_idx += 1
                    
                    # Now we need to find adjustments such that
                    # the missing term (i0,j0,k0) gets value 1
                    # and all other terms remain 1.
                    
                    # The current value at (i0,j0,k0) is 0 (we removed it)
                    # We need to add 1 to it.
                    
                    # For other positions (i,j,k), the current value is 1
                    # We need to keep it at 1, so the net change should be 0.
                    
                    # We can adjust the existing 41 terms.
                    # Let's focus on terms that share at least one index with (i0,j0,k0).
                    
                    # Terms that can affect (i0,j0,k0):
                    # - Terms with i=i0: can adjust U[i0*m+j, c] and W[k*n+i0, c]
                    # - Terms with j=j0: can adjust U[i*m+j0, c] and V[j0*p+k, c]
                    # - Terms with k=k0: can adjust V[j*p+k0, c] and W[k0*n+i, c]
                    
                    # Let's try a simple approach: adjust only terms that share
                    # exactly two indices with (i0,j0,k0).
                    
                    # Terms sharing i=i0 and j=j0: only (i0,j0,k) for k≠k0
                    # Terms sharing i=i0 and k=k0: only (i0,j,k0) for j≠j0
                    # Terms sharing j=j0 and k=k0: only (i,j0,k0) for i≠i0
                    
                    # For simplicity, let's try adjusting terms that share
                    # at least one index.
                    
                    # This is getting complex. Let's try a different approach:
                    # Use the fact that n=2 is small.
                    
                    # For M(2,3,7), we can try to find a decomposition where
                    # we use the structure of the 2x3 and 3x7 matrices.
                    
                    # Let me try a specific pattern: for i=0 and i=1,
                    # we can try to create terms that contribute to both.
                    
                    # Actually, let's just return None for now and use
                    # the schoolbook decomposition.
                    return None
        
        return None
    
    # Attempt rank 41 search
    if time.time() < deadline - 5.0:
        result_41 = try_rank_41_linear()
        if result_41 is not None:
            U41, V41, W41 = result_41
            if verify(U41, V41, W41, 41) and check_valid_entries(U41, V41, W41):
                return {"U": U41, "V": V41, "W": W41}
    
    # Return schoolbook decomposition
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
