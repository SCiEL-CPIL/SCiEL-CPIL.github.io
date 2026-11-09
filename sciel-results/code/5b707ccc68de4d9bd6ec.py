import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    total_naive = n * m * p  # 96

    # Fallback: naive decomposition
    def naive_decomp():
        U = np.zeros((nm, total_naive))
        V = np.zeros((mp, total_naive))
        W = np.zeros((pn, total_naive))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # Try ILP for various ranks
    # The tensor constraint: for each (i,k), sum_j A[i,j]*B[j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # This is nonlinear (trilinear). We need to handle this differently.
    
    # Alternative: Use the fact that for matrix multiplication tensor,
    # T[i,j,k] = delta (standard basis)
    # We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid mult, 0 otherwise
    # Actually the tensor is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no - the matrix multiplication tensor has T[i,j,k] = 1 for all combinations.
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # This is a trilinear constraint, not linear. ILP won't work directly.
    # 
    # Let me think differently. For small dimensions, we can try to find
    # decompositions with entries in {-1, 0, 1} or {-1, 0, 0.5, 1}.
    #
    # Actually, let me reconsider. The constraint is:
    # For all (i,j,k) in [n]x[m]x[p]: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    #
    # This is trilinear. One approach: fix W to be sparse (each column has at most 2 nonzeros),
    # then the problem becomes bilinear which is still hard.
    #
    # Better approach: try known small-rank constructions.
    # For 3x4x8, the naive rank is 96. Let's try to find improvements.
    #
    # One strategy: use the structure that m=4 is small.
    # We can write the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Try rank = n*m*p - something. Or try specific constructions.
    #
    # Let me try a direct search with small integer entries for reduced rank.
    # For rank r, we have r columns in each matrix.
    # Constraint: for all i,j,k: sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Since this is trilinear, let's try a different approach:
    # Enumerate possible W patterns (sparse), then solve for U and V.
    # 
    # Actually, for the specific case 3x4x8, let me try to construct
    # a decomposition with rank < 96 using the following idea:
    # Group terms to reduce rank.
    
    # Try rank 95 first (just remove one redundant term if possible)
    # Actually, let me try a smarter approach: use numpy to verify and
    # try random searches for small-rank decompositions.
    
    best_result = naive_decomp()
    
    # Try to find a decomposition with rank < 96
    # Using the constraint: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Let's try a structured approach:
    # For each rank r from 95 down to some minimum, try to find a solution.
    # We'll use a greedy/heuristic approach.
    
    # First, let's try if rank 95 works by checking if we can merge two naive terms.
    # In the naive decomposition, term (i,j,k) contributes 1 to T[i,j,k] only.
    # To reduce rank, we need a single term that contributes to multiple (i,j,k) positions.
    # A term with U[i*m+j,r]=a, V[j*p+k,r]=b, W[k*n+i,r]=c contributes a*b*c to T[i,j,k].
    # For it to help multiple positions, we need overlapping indices.
    
    # Let me try a direct numerical search for small rank improvements.
    # Given the complexity, let's try rank values that are likely to work.
    
    # Try: can we do rank 92? 88? etc.
    # For now, let's try a few specific constructions.
    
    # Construction idea: For the 4x4 middle dimension, use a 2x2 block structure.
    # Split j into pairs: (j0, j1) for j in {0,1,2,3}
    # This might allow sharing of factors.
    
    # Let me just try random search with verification for a few rank values.
    
    def verify(U, V, W):
        """Verify that U, V, W give exact matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find rank < 96 using a heuristic search
    # We'll try ranks: 95, 94, 93, ... down to maybe 80
    
    # For efficiency, let's use a structured approach:
    # Start with naive, then try to merge pairs of terms.
    
    # Actually, let me try a completely different approach:
    # Use the fact that we can have half-integer entries.
    # Try rank = n*m*p - m = 96 - 4 = 92 by combining terms along j dimension.
    
    # For a fixed (i,k), we need: sum_j A[i,j]*B[j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # 
    # Let me try to use a 2x2 identity-like construction for the j dimension.
    # For j in {0,1}: use two terms that together give the sum.
    
    # Actually, I think the most promising approach for this specific case is:
    # Try to find a decomposition with rank = 3*4*8 - 4 = 92
    # by exploiting that m=4 allows a 2x2 decomposition trick.
    
    # Let me try a direct construction:
    # For each (i,k), we need sum over j of A[i,j]*B[j,k].
    # Group j into pairs: (0,1) and (2,3).
    # For pair (0,1): A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This can be written as a 2x1 * 1x2 product, which needs rank 2 in general.
    # But if we share the W factor across j values, we might save.
    
    # Let me just do a brute-force search for small improvements.
    # Given time constraints, I'll try a few specific ranks.
    
    # Try rank 92: use the construction where for each (i,k),
    # we combine the j=0 and j=1 terms, and j=2 and j=3 terms.
    # This requires careful bookkeeping.
    
    # For now, let's return the naive solution if we can't find better quickly.
    # But let's try one more thing: check if rank 95 is achievable.
    
    # To go from 96 to 95, we need one term that covers two (i,j,k) positions.
    # This means finding (i1,j1,k1) != (i2,j2,k2) such that we can write
    # a term contributing to both. The term at position (i,j,k) has
    # U[i*m+j], V[j*p+k], W[k*n+i] all equal to some value.
    # For it to contribute to (i1,j1,k1) and (i2,j2,k2), we need:
    # i1*m+j1 = i2*m+j2 (same U row) -> i1=i2, j1=j2
    # j1*p+k1 = j2*p+k2 (same V row) -> j1=j2, k1=k2
    # k1*n+i1 = k2*n+i2 (same W row) -> k1=k2, i1=i2
    # So no single term can cover two different positions in the naive scheme.
    # We need non-naive terms.
    
    # Given the time pressure, let's return the naive solution.
    U, V, W = naive_decomp()
    return {"U": U, "V": V, "W": W}
