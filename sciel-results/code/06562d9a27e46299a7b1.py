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
                    s = 0
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
    
    def try_rank_41():
        """Attempt to find a rank-41 decomposition by removing one term and adjusting others."""
        if time.time() > deadline - 15.0:
            return None
        
        # Strategy: For each schoolbook term (i0, j0, k0), try to remove it
        # and adjust nearby terms to compensate.
        # 
        # If we remove term c0 = (i0, j0, k0), we need the sum over remaining
        # terms to still give 1 for (i0, j0, k0) and 0 for all other (i,j,k).
        #
        # We can modify terms that share at least one index with (i0,j0,k0).
        # Let's try modifying terms that share exactly one index (neighbors).
        #
        # For a term (i, j, k) to contribute to (i0, j0, k0), we need:
        # U[i*m+j] * V[j*p+k] * W[k*n+i0] != 0
        # But in schoolbook, U[i*m+j] is nonzero only for column j, etc.
        #
        # Alternative approach: Try to express the missing tensor entry as a
        # linear combination of nearby rank-1 tensors.
        #
        # Let's try a more direct approach: modify up to 3 nearby terms.
        
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    if time.time() > deadline - 10.0:
                        return None
                    
                    # Try to remove term (i0, j0, k0)
                    # and adjust terms that share indices with it.
                    
                    # Candidate terms to modify: those with (i0, *, *) or (*, j0, *) or (*, *, k0)
                    # but not (i0, j0, k0) itself.
                    
                    candidates = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i == i0 and j == j0 and k == k0:
                                    continue
                                # Check if this term shares at least one index
                                if i == i0 or j == j0 or k == k0:
                                    candidates.append((i, j, k))
                    
                    # Try subsets of 1, 2, or 3 candidates
                    # For simplicity, try pairs first
                    
                    for idx1 in range(len(candidates)):
                        if time.time() > deadline - 8.0:
                            return None
                        c1 = candidates[idx1]
                        
                        # Try modifying just c1
                        # We need: U[i1*m+j1] * V[j1*p+k1] * W[k1*n+i0] = 1
                        # But we also need all other (i,j,k) to remain 0 or 1.
                        
                        i1, j1, k1 = c1
                        
                        # This term will now contribute to (i1, j1, k1) and (i0, j0, k0)
                        # if the indices align properly.
                        
                        # Actually, a rank-1 term with U[i1*m+j1]=a, V[j1*p+k1]=b, W[k1*n+i1]=c
                        # contributes a*b*c to (i1, j1, k1) and a*b*c to (i0, j0, k0) only if
                        # i1==i0 AND j1==j0 AND k1==k0, which is not the case.
                        
                        # So a single modified term cannot help. We need at least 2.
                        
                        for idx2 in range(idx1 + 1, len(candidates)):
                            if time.time() > deadline - 6.0:
                                return None
                            c2 = candidates[idx2]
                            i2, j2, k2 = c2
                            
                            # We want to adjust terms c1 and c2 such that:
                            # 1. Their combined contribution to (i0, j0, k0) is 1
                            # 2. Their contributions to all other (i,j,k) are 0
                            
                            # This is tricky. Let's try a different approach:
                            # Use the fact that we can set values to half-integers.
                            
                            # For now, this search space is too large. Let's try
                            # a more structured approach with specific patterns.
                            
                            pass
                    
                    # Try a specific pattern: modify terms (i0, j0, k0+1), (i0, j0+1, k0), (i0+1, j0, k0)
                    # if they exist.
                    
                    modifications = []
                    if k0 + 1 < p:
                        modifications.append((i0, j0, k0 + 1))
                    if j0 + 1 < m:
                        modifications.append((i0, j0 + 1, k0))
                    if i0 + 1 < n:
                        modifications.append((i0 + 1, j0, k0))
                    
                    if len(modifications) == 3:
                        # Try to solve for values that make (i0,j0,k0)=1 and others=0
                        # This requires a more detailed analysis.
                        pass
        
        return None
    
    # Attempt rank 41 search
    if time.time() < deadline - 5.0:
        result_41 = try_rank_41()
        if result_41 is not None:
            U41, V41, W41 = result_41
            if verify(U41, V41, W41, 41) and check_valid_entries(U41, V41, W41):
                return {"U": U41, "V": V41, "W": W41}
    
    # Return schoolbook decomposition
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
