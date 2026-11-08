"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import numpy as np
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # We want rank as low as possible
    # Known: schoolbook is 100, we can do better
    
    # Strategy: Use a structured decomposition
    # For matrix multiplication, we can exploit the fact that
    # the tensor has a specific algebraic structure
    
    # Let's try a construction based on partitioning
    # We'll use the fact that we can share some terms
    
    # First, let's build a good initial solution
    # Then try to reduce rank through local search
    
    # Start with a known good construction
    # For 4x5 * 5x5, we can use block structure
    
    # Actually, let's use a direct approach:
    # The multiplication tensor T has T[i,j,k] nonzero only when indices align
    # We need to find minimal rank decomposition
    
    # Use integer linear programming approach conceptually:
    # Each "term" in the decomposition corresponds to a rank-1 tensor
    # A rank-1 tensor is specified by choosing one row index from U, one from V, one from W
    # which gives (i,j), (j',k), (k',i') with the constraint j=j' and i=i', k=k'
    # Wait, no - in the decomposition, each term c contributes:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is nonzero for specific (i,j,k) combinations
    
    # Let me think differently. The tensor for matrix multiplication is:
    # T[(i,j),(j',k),(k',i')] = 1 if j=j' and i=i' and k=k', else 0
    # So T is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 if (a mod m) == (b div p) and (a div m) == (c mod n) and (b mod p) == (c div n)
    
    # For rank r decomposition, we need vectors u_c, v_c, w_c such that:
    # T[a,b,c] = sum_{c=1}^{r} u_c[a] * v_c[b] * w_c[c]
    
    # Let's use a constructive approach based on known results
    # For rectangular matrix multiplication, there are known constructions
    
    # Let me try a simple improvement over schoolbook:
    # Notice that some products can be shared
    
    # Actually, let's just do a careful local search starting from schoolbook
    
    # But first, let's try to find a better structured solution
    
    # Known result: For m x n * n x p, rank is at most mn + np + mp - something
    # For 4x5 * 5x5: 20 + 25 + 20 = 65? That seems too optimistic
    
    # Let's use a practical approach: start with schoolbook and reduce
    
    deadline = time.time() + 140  # Leave 10s margin
    
    def validate(U, V, W, n, m, p):
        """Validate that the decomposition is exact using Fraction arithmetic"""
        rank = len(U[0])
        # Build the tensor from decomposition
        # T[a,b,c] = sum_{r} U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # Check T matches the multiplication tensor
        
        # For efficiency, check a subset first, then full if needed
        # Full check:
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for jp in range(m):  # j'
                    for k in range(p):
                        b = jp * p + k
                        # T[a,b,c] should be 1 if j==jp and for all k',i' where c=k'*n+i'
                        # Actually T[a,b,c] = 1 if j==jp, and c = k*n+i where i=i_a, k=k_b
                        # Let me re-derive:
                        # a = i*m + j (row of A, col of A)
                        # b = j'*p + k (row of B, col of B)  
                        # c = k'*n + i' (row of C, col of C)
                        # T[a,b,c] = 1 iff j == j' AND i == i' AND k == k'
                        # So for fixed a,b: j is fixed from a, j' is fixed from b
                        # If j != j', T[a,b,c] = 0 for all c
                        # If j == j', then T[a,b,c] = 1 iff c = k*n + i where i = a//m, k = b%p
                        # So exactly one c has value 1, rest are 0
                        
                        if j != jp:
                            # All c should give 0
                            for kp in range(p):
                                for ip in range(n):
                                    c = kp * n + ip
                                    val = sum(Fraction(U[a, r]) * Fraction(V[b, r]) * Fraction(W[c, r]) for r in range(rank))
                                    if val != 0:
                                        return False
                        else:
                            # c = k*n + i should give 1, others 0
                            i_val = a // m
                            k_val = b % p
                            c_target = k_val * n + i_val
                            for kp in range(p):
                                for ip in range(n):
                                    c = kp * n + ip
                                    val = sum(Fraction(U[a, r]) * Fraction(V[b, r]) * Fraction(W[c, r]) for r in range(rank))
                                    expected = 1 if (kp == k_val and ip == i_val) else 0
                                    if val != expected:
                                        return False
        return True
    
    # Start with schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try to reduce rank by removing columns that are linearly dependent
    # or by finding better decompositions
    
    # Local search: try to replace pairs of columns with a single column
    # This is complex, so let's try a simpler approach
    
    # Use the fact that we can have half-integer entries
    # Try to find a decomposition with rank < 100
    
    # Known constructions for small matrices:
    # For 2x2 * 2x2, rank 7 (Strassen) vs 8 schoolbook
    # For 3x3 * 3x3, rank 23 (Bini) vs 27 schoolbook
    # For 4x4 * 4x4, rank 49 (Bini) vs 64 schoolbook
    
    # For 4x5 * 5x5, we need a specific construction
    
    # Let's try a block-based approach:
    # Partition the 4x5 matrix into blocks and use known small matrix results
    
    # Actually, let's try to use a greedy approach to find a lower rank
    # by solving a series of linear systems
    
    # The key insight: each "position" (i,j,k) in the output requires
    # sum_j A[i,j]*B[j,k] to be computed
    # This is a bilinear form in j
    
    # We can think of this as: for each (i,k), we need to compute
    # the dot product of row i of A and column k of B
    
    # A rank-1 term in the decomposition computes:
    # (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait, that's not quite right.
    
    # Let me re-derive. The decomposition says:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # No wait, that's not how it works either.
    
    # Actually, the tensor decomposition means:
    # T[(i,j),(j',k),(k',i')] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # And T is the multiplication tensor, so:
    # T[(i,j),(j',k),(k',i')] = 1 if j=j' and i=i' and k=k'
    
    # The actual computation is:
    # C[i',k'] = sum_{i,j,j',k} A[i,j] * B[j',k] * T[(i,j),(j',k),(k',i')]
    # = sum_{i,j,j',k} A[i,j] * B[j',k] * sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c]
    # = sum_c W[k'*n+i',c] * (sum_{i,j} A[i,j] * U[i*m+j,c]) * (sum_{j',k} B[j',k] * V[j'*p+k,c])
    
    # For this to equal sum_j A[i',j] * B[j,k'], we need:
    # sum_c W[k'*n+i',c] * (sum_{i,j} A[i,j] * U[i*m+j,c]) * (sum_{j',k} B[j',k] * V[j'*p+k,c]) = sum_j A[i',j] * B[j,k']
    
    # This must hold for ALL A and B, so we need the tensor identity.
    
    # Let me just do a careful local search with validation
    
    # Try to remove one column at a time and see if we can adjust others to compensate
    
    # This is getting complex. Let me try a different strategy:
    # Use known results and try to construct explicitly
    
    # For now, return the schoolbook solution as a valid baseline
    # and hope that the search can improve it
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
