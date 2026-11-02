import numpy as np
import random
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Tensor entries: T[i,j,k] = 1 if i*m+j, j*p+k, k*n+i form a valid triple
    # U has 6 rows (i*m+j for i in 0..1, j in 0..2)
    # V has 9 rows (j*p+k for j in 0..2, k in 0..2)
    # W has 6 rows (k*n+i for k in 0..2, i in 0..1)
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly matches the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_rank_with_support_pattern(U_support, V_support, W_support, rank):
        """
        Given support patterns (which entries are non-zero), try to find coefficients.
        U_support: list of (row, col) pairs that are non-zero in U
        V_support: list of (row, col) pairs that are non-zero in V  
        W_support: list of (row, col) pairs that are non-zero in W
        
        We set up a linear system where each tensor entry T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        For fixed supports, this becomes a system of equations in the non-zero entries.
        
        Strategy: Fix U and V supports, then for each column c, the contribution to T[i,j,k]
        is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
        
        Actually, let's use a different approach: enumerate all possible rank-r decompositions
        by trying all combinations of which tensor entries are "covered" by each rank-1 term.
        """
        pass
    
    # Approach: Use the fact that for small ranks, we can enumerate support patterns
    # and solve the resulting system.
    
    # For rank r, we need r rank-1 terms. Each term c contributes to T[i,j,k] as:
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # The key insight: for a fixed set of non-zero positions in U, V, W, 
    # the equations T[i,j,k] = sum_c ... become polynomial equations.
    
    # Better approach: Use the structure. For 2x3x3, the tensor has 18 non-zero entries.
    # Lower bound on rank: max(6, 9, 6) = 9 (from dimension arguments)
    # But actually the lower bound is higher. Let's try ranks from 9 to 18.
    
    # Strategy: For each rank r, try to find a decomposition by:
    # 1. Enumerate possible support patterns (which (row,col) entries are non-zero)
    # 2. For each pattern, set up and solve the system
    
    # More practical: Use a structured search
    # For rank r, each column c of (U,V,W) defines a rank-1 tensor.
    # We need sum of r rank-1 tensors = the matrix multiplication tensor.
    
    # Let's use a different strategy: enumerate all possible "types" of rank-1 terms
    # A rank-1 term is determined by which (i,j), (j,k), (k,i) combinations it covers.
    
    # For 2x3x3:
    # U has 6 entries: (i,j) for i in {0,1}, j in {0,1,2}
    # V has 9 entries: (j,k) for j in {0,1,2}, k in {0,1,2}
    # W has 6 entries: (k,i) for k in {0,1,2}, i in {0,1}
    
    # A rank-1 term with non-zero U[i0,j0], V[j1,k1], W[k2,i2] contributes to
    # T[i,j,k] only if i=i0 or j=j0, etc. Actually no - it contributes to ALL T[i,j,k]
    # with value U[i0,j0]*V[j1,k1]*W[k2,i2] * delta(i,i0)*delta(j,j0)*delta(j,j1)*delta(k,k1)*delta(k,k2)*delta(i,i2)
    
    # Wait, that's not right either. Let me reconsider.
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So for a single column c, the contribution to T[i,j,k] is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is non-zero only if U[i*m+j,c] != 0 AND V[j*p+k,c] != 0 AND W[k*n+i,c] != 0.
    
    # For a given column c, let:
    # S_U = {(i,j) : U[i*m+j,c] != 0}
    # S_V = {(j,k) : V[j*p+k,c] != 0}
    # S_W = {(k,i) : W[k*n+i,c] != 0}
    
    # Then the column contributes to T[i,j,k] iff (i,j) in S_U, (j,k) in S_V, (k,i) in S_W.
    
    # The value contributed is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
    
    # For the sum over all columns to equal 1 for each (i,j,k), we need:
    # sum_{c: (i,j) in S_U(c), (j,k) in S_V(c), (k,i) in S_W(c)} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of 18 equations in the non-zero entries.
    
    # Let's try a structured enumeration approach for small ranks.
    
    # For rank r, we have r columns. Each column has a "type" determined by its support.
    # The number of possible support patterns is large, but for small r we can enumerate.
    
    # Actually, let's use a more efficient approach:
    # 1. For each rank r from 9 to 18, try to find a decomposition
    # 2. Use a combinatorial search over support patterns
    # 3. For each pattern, solve the resulting system (which may be nonlinear)
    
    # Given the complexity, let's use a hybrid approach:
    # - For ranks 9-12, use exhaustive support pattern enumeration
    # - For ranks 13-18, use randomized search with verification
    
    def solve_for_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        # For each rank, try different strategies
        if r <= 12:
            # Try structured enumeration
            result = enumerate_support_patterns(r)
            if result is not None:
                return result
        
        # Fallback: randomized search
        result = randomized_search(r)
        if result is not None:
            return result
        
        return None
    
    def enumerate_support_patterns(r):
        """Enumerate support patterns for small ranks."""
        # This is complex; for now, use a simpler approach
        # Try to find decompositions where each rank-1 term covers a "block"
        
        # For 2x3x3, one natural decomposition is to use terms that cover
        # specific (i,j,k) triples.
        
        # Actually, let's just use randomized search for all ranks
        # The enumeration space is too large for exhaustive search
        return None
    
    def randomized_search(r, max_attempts=50000):
        """Randomized search for a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        # Use small integer/half-integer entries
        # Try different entry sets
        entry_sets = [
            [-1, 0, 0, 0, 1],
            [-1, 0, 0, 1],
            [-2, -1, 0, 1, 2],
            [-1, 0, 1, 2],
            [0, 1],
            [-1, 1],
        ]
        
        for entries in entry_sets:
            if time.time() > deadline - 5:
                break
            
            for _ in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                U = np.random.choice(entries, size=(6, r))
                V = np.random.choice(entries, size=(9, r))
                W = np.random.choice(entries, size=(6, r))
                
                if check_decomposition(U, V, W, r):
                    return (U, V, W, r)
        
        return None
    
    # Try ranks from 9 to 18
    for r in range(9, 19):
        if time.time() > deadline - 5:
            break
        
        result = solve_for_rank(r)
        if result is not None:
            U, V, W, rank = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 18
    rank = 18
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(6)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
