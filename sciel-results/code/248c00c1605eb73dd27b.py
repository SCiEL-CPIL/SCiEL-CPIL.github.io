"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = len(U[0])
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                    # Tensor: T[i*m+j, j*p+k, k*n+i] = 1
                    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0.0
                    for r in range(r):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(r):
                        pass
        # For non-matching indices, the tensor entry should be 0
        # The tensor T[a,b,c] = 1 iff a = i*m+j, b = j'*p+k, c = k'*n+i' 
        # where i=i', j=j', k=k'
        # So for any (a,b,c) not of that form, T=0
        # But we only need to verify the 1s are correct and the decomposition works
        # Actually, we need to verify that for ALL (a,b,c), 
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # where T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
        # and the same i,j,k work for all three
        
        # More carefully: T[i*m+j, j*p+k, k*n+i] = 1
        # For other (a,b,c), T=0
        # Let's check a sample of zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    # Determine if this is a valid triple
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    s = 0.0
                    for r in range(r):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    def snap_to_half(x):
        """Snap to nearest half-integer."""
        return round(2*x) / 2.0
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0.0]*rank_school for _ in range(n*m)]
    V_school = [[0.0]*rank_school for _ in range(m*p)]
    W_school = [[0.0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1.0
                V_school[j*p+k][col] = 1.0
                W_school[k*n+i][col] = 1.0
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: Try various ranks from low to high
    # For each rank, try to find U, V, W with half-integer entries
    # 
    # Key insight: For n=2, we can exploit structure
    # Let's try rank = n*m*p - something
    # 
    # Approach: Fix U and V to have simple patterns, solve for W
    # 
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = 0
    #
    # This is a system of equations in the W entries (and U,V entries)
    #
    # Simpler approach: Try specific structured decompositions
    #
    # For 2x5x6, known bounds suggest rank can be lower than 60
    # Let's try to find a decomposition with rank < 60
    
    # Try rank values from 40 to 59
    for rank in range(40, 60):
        if time.time() > deadline - 30:
            break
        
        # Try to construct a valid decomposition with this rank
        # We'll use a randomized approach with local search
        
        # Initialize U, V, W with random half-integers
        best_found = None
        
        for attempt in range(100):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = [[snap_to_half(np.random.uniform(-1, 1)) for _ in range(rank)] for _ in range(n*m)]
            V = [[snap_to_half(np.random.uniform(-1, 1)) for _ in range(rank)] for _ in range(m*p)]
            W = [[snap_to_half(np.random.uniform(-1, 1)) for _ in range(rank)] for _ in range(p*n)]
            
            # Check if valid
            if verify(U, V, W):
                best_found = (U, V, W)
                break
        
        if best_found:
            best_result = {"U": best_found[0], "V": best_found[1], "W": best_found[2]}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
