"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x8 matrix.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    and the decomposition satisfies:
      for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    start_time = time.time()
    deadline = 145.0  # 5 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W, rank):
        """Verify the tensor decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to construct a lower-rank decomposition using Strassen-like sharing
    # For (2,2,p), we can use 6 shared terms + p specific terms = p+6 rank
    
    def try_strassen_construction(rank):
        """
        Try to construct a rank-r decomposition for (2,2,p) matmul.
        Uses the structure that (2,2,2) has rank 7 (Strassen).
        For (2,2,p), we can potentially share across columns.
        """
        if rank < 7:  # Minimum possible is at least 7 for (2,2,2) subblock
            return None
        
        # Strategy: Use 6 "shared" Strassen terms that work across all columns,
        # plus p "specific" terms for each column
        # Total rank = 6 + p = 14 for p=8
        
        if rank != 6 + p:
            return None
            
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen's 7 terms for (2,2,2):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # For (2,2,p), we can share the 6 "interaction" terms across columns,
        # and use p terms for the diagonal-like parts
        
        # Let me try a different approach: use the fact that the tensor is
        # a "strip" and see if we can get rank p + 6
        
        # Actually, let me try a direct construction and verify
        # The 6 shared terms correspond to the off-diagonal Strassen terms
        # The p specific terms handle the diagonal contributions
        
        # This is complex, so let me try a simpler approach:
        # Use rank = p + 6 and see if it works with a specific pattern
        
        # For now, return None to indicate this construction needs more work
        return None
    
    # Try the Strassen-based construction
    strassen_result = try_strassen_construction(p + 6)
    
    if strassen_result is not None:
        U, V, W = strassen_result
        if verify(U, V, W, p + 6):
            return {"U": U, "V": V, "W": W}
    
    # Check time
    if time.time() > deadline - 10:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try other ranks between p+6 and schoolbook
    for rank in range(p + 7, rank_schoolbook):
        if time.time() > deadline - 5:
            break
        # This is where we could try more sophisticated constructions
        # For now, just return schoolbook
        pass
    
    # Return schoolbook as fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
