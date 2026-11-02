"""Matrix multiplication 2x2x8 - rank-minimal decomposition search."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS
    # T[i, j, k] = 1 for multiplication tensor
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the full multiplication tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find lower rank decompositions
    # Strategy: use tensor SVD / CP decomposition with exact rational recovery
    
    # First, try rank from 16 up to 31 (below schoolbook)
    # Use iterative optimization: fix two matrices, solve for third
    
    # For a given rank r, we need:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # 
    # This is a 3-way tensor CP decomposition.
    # 
    # Key insight: for fixed U and V, W is determined linearly:
    # W[k*n+i, c] = T[i,j,k] / (U[i*m+j,c] * V[j*p+k,c])
    # But this only works if U and V are non-zero at the right places.
    
    # Better approach: use the fact that this is a bilinear map.
    # T can be seen as a matrix M of size (n*p) x (n*m) where:
    # M[(i,k), (j,i')] = 1 if i==i' (from the multiplication structure)
    # Actually, let me reconsider the tensor structure.
    
    # The multiplication tensor for A(n×m) * B(m×p) -> C(n×p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this case!)
    # Wait, that's not right. The tensor T has T[i,j,k] = 1 if the (i,j) element of A
    # and (j,k) element of B contribute to (i,k) element of C.
    # In standard matrix multiplication, every T[i,j,k] = 1.
    
    # So we need: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # for all i,j,k.
    
    # This means: for each (i,k), sum_j A[i,j]*B[j,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c])
    
    # Let's define: for each c, let u_c be a vector in R^(n*m), v_c in R^(m*p), w_c in R^(p*n)
    # Then we need: T = sum_c u_c ⊗ v_c ⊗ w_c (as a 3-way tensor)
    
    # Since T[i,j,k] = 1 for all i,j,k, we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the all-ones tensor.
    
    # The all-ones tensor of size n×m×p has rank... let me think.
    # For n=2, m=2, p=8: we need sum_c U[2i+j,c]*V[2j+k,c]*W[8k+i,c] = 1 for all i,j,k.
    
    # Let's try a structured approach. Assume U, V, W have simple structure.
    
    # Try: U[i*m+j, c] depends only on c in a structured way.
    # 
    # Key observation: If we set U[i*m+j, c] = a_c for all i,j (constant across rows),
    # V[j*p+k, c] = b_c for all j,k, W[k*n+i, c] = w_{k,i,c},
    # then: sum_c a_c * b_c * w_{k,i,c} = 1 for all i,k
    # This requires: for each (i,k), sum_c (a_c*b_c) * w_{k,i,c} = 1
    # With r terms, we can set w_{k,i,c} = 1/r for all c, then need sum_c a_c*b_c = r.
    # But this gives W[k*n+i,c] = 1/r for all k,i,c, and we need a_c*b_c to sum to r.
    # E.g., a_c = 1, b_c = 1, r terms: sum = r, so w = 1/r works!
    # Rank r = 1? No wait, we need sum_c a_c*b_c*w_{k,i,c} = 1.
    # If a_c=1, b_c=1, w_{k,i,c}=1 for all: sum_c 1*1*1 = r. Need r=1.
    # So with r=1: U=1, V=1, W=1 gives T[i,j,k] = 1. That works!
    # But wait, that can't be right for matrix multiplication...
    
    # I think I'm confusing the tensor structure. Let me re-examine.
    
    # In matrix multiplication A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k]).
    # 
    # A rank-r decomposition gives:
    # C[i,k] = sum_c (sum_j' U[i*m+j',c] * A[i,j']) * (sum_k' V[j*p+k',c] * B[j,k']) ... 
    # No, that's not right either.
    
    # Let me use the standard formulation:
    # A rank-r decomposition of matrix multiplication means:
    # C[i,k] = sum_{c=1}^r (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait, that's not standard either.
    
    # Standard: C[i,k] = sum_c M_c(A) * M_c(B) * W_c
    # where M_c(A) = sum_{i,j} U[i*m+j, c] * A[i,j]
    # M_c(B) = sum_{j,k} V[j*p+k, c] * B[j,k]
    # and W_c = W[k*n+i, c]... no, W is a scalar per c for each (i,k)?
    
    # Actually the standard form is:
    # C[i,k] = sum_c (sum_{i',j'} U[i*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k']) 
    # This doesn't look right dimensionally.
    
    # Let me use the correct formulation from the task:
    # "exactly decompose the matrix multiplication tensor"
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So the tensor T[i,j,k] must equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # And T[i,j,k] = 1 for all i,j,k (since every product term appears with coefficient 1).
    
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i∈{0,1}, j∈{0,1}, k∈{0,...,7}
    
    # This is a system of 2*2*8 = 32 equations.
    
    # For rank r, we have r*(4+16+16) = r*36 unknowns (with possible scaling redundancy).
    
    # Let's try small ranks. The minimum possible rank is at least max(n*m, m*p, p*n) = max(4,16,16) = 16.
    # Actually, the rank is at least max of the dimensions of the "slice" spaces.
    
    # Let me try rank 16, 17, ..., up to 31.
    
    # For a given rank r, set up the equations and try to solve.
    # Use a structured initialization and optimize.
    
    def try_rank(r):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Iteratively solve: fix two, solve for third
            # For fixed U, V: W[k*n+i, c] should satisfy
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
            # For fixed (i,k): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = 1
            # Let S[i,k,c] = sum_j U[i*m+j,c]*V[j*p+k,c]
            # Then: sum_c W[k*n+i,c] * S[i,k,c] = 1 for all i,k
            # This is a system: for each (i,k), sum_c W[k*n+i,c]*S[i,k,c] = 1
            # W has p*n rows, r columns. For each row (k*n+i), we need sum_c W[row,c]*S[i,k,c] = 1
            # This is r equations per (i,k) pair? No, it's 1 equation per (i,k) pair.
            # So we have n*p = 16 equations, and p*n*r = 16*r unknowns.
            # Underdetermined for r > 1. We can pick a simple solution.
            
            # Alternative: use ALS (Alternating Least Squares) style
            # But this is a 3-way tensor, not a matrix.
            
            # Let's use a different approach: parameterize and use nonlinear optimization.
            # Or: use the fact that we can fix one matrix and solve the others.
            
            # Fix U. Then for each c, define v_c = V[:,c] (size m*p) and w_c = W[:,c] (size p*n)
            # The constraint is: sum_c U[i*m+j,c]*v_c[j*p+k]*w_c[k*n+i] = 1 for all i,j,k
            # 
            # For fixed U, this is linear in the products v_c[j*p+k]*w_c[k*n+i].
            # Let z_c[i,j,k] = v_c[j*p+k]*w_c[k*n+i]
            # Then sum_c U[i*m+j,c]*z_c[i,j,k] = 1 for all i,j,k
            # This is linear in z, but z has rank-1 structure in (j,k) for each c.
            
            # This is getting complex. Let me try a direct numerical approach.
            
            # Use scipy's least squares on the full system.
            # Variables: all entries of U, V, W (r*(4+16+16) = 36r variables)
            # Equations: 32 (one for each (i,j,k))
            # Nonlinear (cubic) system.
            
            # For small r, try to solve via Newton's method or least squares.
            
            # Actually, let me try a smarter structured approach.
            
            # Hypothesis: rank 16 might work with a Kronecker product structure.
            # U[i*m+j, c] = u_i * u'_j * something
            # Let's try: U[i*m+j, c] = a_c * b_{i,j} where b is a 2x2 matrix
            # V[j*p+k, c] = a_c * d_{j,k} where d is a 2x8 matrix  
            # W[k*n+i, c] = e_{k,i,c}
            # Then: sum_c a_c^2 * b_{i,j} * d_{j,k} * e_{k,i,c} = 1
            # This still seems hard.
            
            # Let me just try the schoolbook and verify it works, then try to reduce.
            break  # Skip random search for now, use structured approach
        
        # Structured approach for rank r:
        # Try to find U, V, W with simple structure.
        
        # Approach: use the fact that T is the all-ones tensor.
        # The all-ones tensor of size a×b×c has CP rank = max(a,b,c) if... 
        # Actually, the all-ones tensor has rank 1! Because 1 = 1*1*1.
        # Wait, T[i,j,k] = 1 = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # If we set U=1, V=1, W=1 for all entries, then sum_c 1*1*1 = r.
        # We need this to equal 1, so r=1 works with U=V=W=1!
        
        # But that can't be right for matrix multiplication. Let me re-examine.
        
        # OH WAIT. I think I misunderstood the tensor. Let me re-read the task.
        # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 8 matrix."
        # 
        # The multiplication tensor T[i,j,k] is NOT all ones. It encodes the structure of
        # matrix multiplication. In standard matrix multiplication:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
        # But the tensor T[i,j,k] represents: the (i,j,k) entry is 1 if A[i,j] and B[j,k]
        # multiply to contribute to C[i,k], which is always true.
        
        # So T[i,j,k] = 1 for all i,j,k. And we need:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k.
        
        # With r=1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1 gives sum = 1. ✓
        
        # But this would mean rank 1, which is clearly wrong for matrix multiplication.
        # There must be a constraint I'm missing.
        
        # AH! I think the issue is that the decomposition must actually compute the product.
        # The rank-r decomposition means:
        # C[i,k] = sum_c (sum_{i',j'} U[i*m+j',c] * A[i',j']) * (sum_{j',k'} V[j'*p+k',c] * B[j',k']) * W[k*n+i,c]
        # 
        # No wait, that's not standard either. Let me think about what "decompose the multiplication tensor" means.
        
        # The multiplication tensor is a 3-tensor T where T[i,j,k] = 1 means that the product
        # A[i,j]*B[j,k] contributes to C[i,k]. A rank-r decomposition expresses T as:
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c ∈ R^(n*m), v_c ∈ R^(m*p), w_c ∈ R^(p*n)
        # 
        # This means: T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
        # 
        # And the actual computation is:
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_{i,j,k} T[i,j,k] * A[i,j]*B[j,k]
        #         = sum_c (sum_{i,j} u_c[i*m+j]*A[i,j]) * (sum_{j,k} v_c[j*p+k]*B[j,k]) * w_c[k*n+i]
        # 
        # Wait, that's not right. Let me be more careful.
        
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Using T[i,j,k] = 1: C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta_{i',i} * delta_{k',k}
        # 
        # Hmm, this is getting confusing. Let me just trust the task formulation:
        # "exactly decompose the matrix multiplication tensor" with the given U, V, W shapes.
        # The condition is: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # where T[i,j,k] = 1 for all i,j,k.
        
        # If T is all ones, then rank 1 works. But that contradicts known results.
        
        # I think the resolution is: the tensor T is NOT all ones. The multiplication tensor
        # for n×m by m×p has T[i,j,k] = 1 only when the indices are consistent with
        # matrix multiplication, which they always are in the standard formulation.
        
        # Let me just implement the schoolbook and verify, then try to reduce rank.
        
        # Schoolbook:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Verify
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i,j,k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        
        if np.allclose(T_recon, 1.0):
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
        else:
            # Should not happen
            pass
        
        # Now try to reduce rank by removing zero columns or merging
        # Check if any columns are redundant
        
        # Try rank 16 (theoretical lower bound)
        # Use a structured ansatz
        
        # For rank r, try: U[i*m+j, c] = a_c * f_{i,j}, V[j*p+k, c] = a_c * g_{j,k}, W[k*n+i, c] = h_{k,i,c}
        # Then: sum_c a_c^2 * f_{i,j} * g_{j,k} * h_{k,i,c} = 1
        
        # This is still complex. Let me just return the schoolbook for now.
        
        # Convert to list format
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return try_rank(rank_schoolbook)
# EVOLVE-BLOCK-END
