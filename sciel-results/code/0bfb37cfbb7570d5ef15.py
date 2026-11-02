"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    # Build target tensor T[a,b,c,d] = delta_{a,i} * delta_{b,c} * delta_{d,k}
    # Actually T is indexed by (a, b, c, d) where:
    # a = row of A (0..n-1), b = col of A / row of B (0..m-1)
    # c = row of B / col of B (0..m-1), d = col of B (0..p-1)
    # T[a,b,c,d] = 1 if b==c, else 0
    # And the decomposition is:
    # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = T_target[i, b, c, d] ... 
    # Wait, I need to be more careful.
    
    # The condition is:
    # For all (a,b,c,d,i,k):
    # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = delta_{a,i} * delta_{b,c} * delta_{d,k}
    
    # Let me define the target tensor more carefully.
    # The output C[i,k] = sum_j A[i,j] * B[j,k]
    # So the coefficient of A[a,b] * B[c,d] in C[i,k] is:
    # 1 if a==i, b==c, d==k
    # 0 otherwise
    
    # So we need:
    # sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] = 1 if (a==i, b==c, d==k) else 0
    
    # Number of (a,b,c,d,i,k) tuples: n*m*m*p*n*p = 2*4*4*7*2*7 = 3136
    # But many are 0, so the actual constraints are 224 (for the 1's) and the rest must be 0.
    
    # Let me build the full constraint matrix.
    # Variables: U (8*r), V (28*r), W (14*r) = 50*r total
    # Constraints: for each (i,k,a,b,c,d), the equation above
    
    # This is a system of polynomial equations. Let me try a different approach.
    
    # Approach: Use a randomized search with local optimization.
    # For a given rank r, try to find U, V, W that satisfy the equations.
    
    # Let me first try the schoolbook solution as a fallback.
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, r_idx] = 1
                    V[v_idx, r_idx] = 1
                    W[w_idx, r_idx] = 1
                    r_idx += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Reconstruct the tensor
        # For each (i,k), C[i,k] = sum_r W[k*n+i, r] * (sum_{a,b} U[a*m+b, r]*A[a,b]) * (sum_{c,d} V[c*p+d, r]*B[c,d])
        # The coefficient of A[a,b]*B[c,d] in C[i,k] should be delta_{a,i}*delta_{b,c}*delta_{d,k}
        
        # Check for all (a,b,c,d,i,k)
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                lhs = 0.0
                                for r in range(rank):
                                    lhs += W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r]
                                rhs = 1.0 if (a == i and b == c and d == k) else 0.0
                                if abs(lhs - rhs) > 1e-9:
                                    return False
        return True
    
    # Try to find a lower rank solution
    # Start with schoolbook and try to reduce rank
    
    # For now, let's try ranks from 50 down to 40 and see if we can find solutions
    # using a simple randomized approach
    
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]
    
    # Try a few ranks below schoolbook
    # This is a placeholder - the actual search will be more sophisticated
    # Given time constraints, let's just return schoolbook for now
    # but try a quick search for rank 55
    
    if time.time() < deadline - 10:
        # Try rank 55 with a simple approach
        # This is a stub - in a real implementation, we'd do more
        pass
    
    # Convert to lists for the return format
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_U.tolist()]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_V.tolist()]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
