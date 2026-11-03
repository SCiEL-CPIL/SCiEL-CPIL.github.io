"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 140  # 10s safety margin
    
    # Build the matrix multiplication tensor T[i,j,k] = 1
    # We need: for all i,j,k: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Try to find a lower-rank decomposition by exploiting structure.
    # For 3x3x7, try ranks from a lower bound upward.
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    
    # Try a constructive approach using linear algebra.
    # Fix a candidate rank r, and try to find U, V, W.
    
    # Approach: Use the fact that we can write the multiplication as
    # C = A @ B where A is 3x3, B is 3x7, C is 3x7.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Try to find a decomposition with rank < 63.
    # One approach: for each pair (i,j), the contribution to all k is A[i,j] * B[j,:].
    # We can potentially share the "A[i,j]" factor across k values.
    
    # Let's try a specific construction:
    # Use rank = n*m + m*p + p*n - (n-1)*(p-1) or similar.
    # Actually, let's just try to solve for a low-rank decomposition numerically.
    
    # Alternative: Use the known structure. For 3x3x3, minimum is 21.
    # For 3x3x7, let's try to find a good decomposition.
    
    # Try rank 50, 45, 40, etc. by solving a linear system.
    
    # Method: Fix U and V randomly (or with structure), then solve for W.
    # The equation is: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is linear in W if U,V are fixed. Let's set up the system.
    
    # For each output position (k,i), the equation involves W[k*n+i, r] for all r.
    # But the equation couples different (i,j,k) through shared r.
    
    # Let's use a different approach: alternating least squares.
    
    def try_rank(r, max_iter=100, tol=1e-10):
        """Try to find a decomposition of rank r using ALS."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to help convergence
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update W: for each (k,i), solve linear system
            # For each (k,i): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is: sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            # Let M_W[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then: M_W[k*n+i, :] @ W[k*n+i, :] = 1 ... wait, that's not right.
            
            # Actually, the constraint is: for ALL (i,j,k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a single scalar equation per (i,j,k), but W entries are shared
            # across different (i,j,k) tuples.
            
            # Let me reformulate. The full tensor equation is:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is a 3-tensor equation. It's not straightforward to solve linearly.
            
            # Let's use a different strategy: build the full linear system.
            # Unknowns: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for all i,j,k,r
            # Equations: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear (products of unknowns).
            
            # Use alternating minimization: fix two, solve for the third.
            # Fix U and V, solve for W:
            # For each (k,i): sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A_W[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then: sum_r A_W[k*n+i, r] * W[k*n+i, r] = 1
            # This is a system: for each (k,i), one equation with r unknowns (W[k*n+i, 0..r-1])
            # But each W[k*n+i, r] appears in exactly one equation (for that k,i).
            # So we have p*n independent equations, each with r unknowns.
            # This is underdetermined if r > 1. We need more constraints.
            
            # The issue is that the tensor equation gives n*m*p = 63 equations,
            # but we have (n*m + m*p + p*n) * r = (9 + 21 + 21) * r = 51*r unknowns.
            # For r=63: 51*63 = 3213 unknowns, 63 equations. Very underdetermined.
            
            # So we can't just solve it linearly. We need a different approach.
            
            break  # ALS approach is complex, try something else
        
        return None
    
    # Let's try a more direct constructive approach.
    # For 3x3x7, use the following idea:
    # Split the 7 columns into groups and use block structure.
    
    # Actually, let's just use the schoolbook decomposition but check if we can
    # find any linear dependency that allows rank reduction.
    
    # Build the tensor as a 3D array
    T = np.ones((n, m, p))
    
    # The schoolbook decomposition has rank 63 with:
    # U[i*m+j, i*m*p + j*p + k] = 1
    # V[j*p+k, i*m*p + j*p + k] = 1  
    # W[k*n+i, i*m*p + j*p + k] = 1
    
    # Check if there are linear dependencies among these 63 "basis" tensors.
    # Each basis tensor is T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    # where U_r, V_r, W_r are the r-th column of U, V, W.
    
    # For the schoolbook, each basis tensor is 1 at exactly one (i,j,k) and 0 elsewhere.
    # So they are linearly independent! Rank is exactly 63.
    
    # To get rank < 63, we need a different decomposition where the basis tensors
    # are not "one-hot" but can overlap and cancel.
    
    # Let's try to construct such a decomposition.
    # Idea: use the fact that 1 = a + b - (a-1) etc.
    
    # For a specific (i,j,k), we have 1 = U[i,j,r1]*V[j,k,r1]*W[k,i,r1] + ...
    # We can try to write 1 as a sum of products where the same U,V,W entries
    # are reused across different (i,j,k) positions.
    
    # Let's try a small example: can we do 1x1x2 with rank < 2?
    # C[0,0] = A[0,0]*B[0,0], C[0,1] = A[0,0]*B[0,1]
    # Schoolbook: rank 2. Can we do rank 1?
    # Need: U[0,0]*V[0,0]*W[0,0] = 1 and U[0,0]*V[0,1]*W[0,0] = 1
    # From first: U[0,0]*V[0,0]*W[0,0] = 1
    # From second: U[0,0]*V[0,1]*W[0,0] = 1
    # Divide: V[0,0]/V[0,1] = 1, so V[0,0] = V[0,1]
    # Then both equations are the same: U[0,0]*V[0,0]*W[0,0] = 1
    # So rank 1 works! Set U[0,0]=1, V[0,0]=V[0,1]=1, W[0,0]=1.
    # Wait, but V has shape (m*p, rank) = (1*2, 1) = (2,1)
    # V[0,0] is for (j=0,k=0), V[1,0] is for (j=0,k=1)
    # So V[0,0]=1, V[1,0]=1, U[0,0]=1, W[0,0]=1
    # Check: C[0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # C[0,1] = U[0,0]*V[1,0]*W[0,0] = 1*1*1 = 1 ✓
    # So 1x1x2 has rank 1 < 2!
    
    # This suggests that for our problem, we can also reduce rank.
    # The key is that V[j*p+k, r] can be the same for different k values (same j).
    
    # For 3x3x7: For fixed i and j, the terms for different k all share U[i*m+j, r].
    # If we set U[i*m+j, r] = 1 for some r, and V[j*p+k, r] = 1 for all k,
    # and W[k*n+i, r] = 1 for all k, then we get 1 for all (i,j,k) with that r.
    # But we need to handle all (i,j,k) pairs.
    
    # Let's try: use r=1 with U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then for all (i,j,k): sum = 1*1*1 = 1. 
    # But wait, this gives C[i,k] = A[i,j]*B[j,k] for a single j, not the sum over j!
    
    # I made an error. The tensor equation is:
    # T[i,j,k] = 1 means that the (i,j,k)-th entry of the multiplication tensor is 1.
    # The actual multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 indicates that A[i,j] and B[j,k] contribute to C[i,k].
    
    # The decomposition says: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k']) ... 
    # No, that's not right either.
    
    # Let me re-read the problem. The tensor M has entries M[a,b,c] where:
    # a = i*m + j (index into A), b = j*p + k (index into B), c = k*n + i (index into C)
    # M[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for the same i,j,k.
    
    # The decomposition is: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # And the actual matrix multiplication is:
    # C = A @ B, so C[i,k] = sum_j A[i,j] B[j,k]
    
    # The tensor M encodes which products A[i,j]*B[j,k] contribute to which C[i,k].
    # Specifically, M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    
    # So the decomposition M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] means:
    # For each (i,j,k): 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is what we need to satisfy. It's a constraint on U, V, W.
    
    # The actual computation is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    # Wait, that's not how it works. The tensor decomposition is for the structure, not the computation.
    
    # I think I'm overcomplicating this. Let me just use the schoolbook decomposition
    # and accept rank 63, since all inspirations also have rank 63.
    
    # But wait - the 1x1x2 example shows rank can be reduced. Let me revisit.
    
    # For 1x1x2: M[0,0,0] = 1 (i=0,j=0,k=0) and M[0,1,1] = 1 (i=0,j=0,k=1)
    # Decomposition: M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # M[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1
    # M[0,1,1] = U[0,0]*V[1,0]*W[1,0] = 1
    # With r=1: U[0,0]=1, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=1
    # Check: M[0,0,0] = 1*1*1 = 1 ✓, M[0,1,1] = 1*1*1 = 1 ✓
    # So rank 1 works for 1x1x2!
    
    # For 3x3x7, can we do better than 63? Let's try to find a construction.
    
    # Key insight: if we can find U, V, W such that for all (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # One approach: set U[i*m+j, r] = 1 for all i,j and some r, V[j*p+k, r] = 1 for all j,k,
    # and W[k*n+i, r] = 1 for all k,i. Then sum_r 1*1*1 = r. We need r=1.
    # But then all entries are 1, which works! 
    # Wait, but U has shape (n*m, r) = (9, 1), V has shape (m*p, r) = (21, 1), W has shape (p*n, r) = (21, 1).
    # Set all entries to 1. Then for all (i,j,k): 1*1*1 = 1. 
    # So rank 1 works?? That can't be right.
    
    # Let me double-check. The tensor M[a,b,c] = 1 for all valid (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i.
    # If U[a,0] = 1 for all a, V[b,0] = 1 for all b, W[c,0] = 1 for all c,
    # then M[a,b,c] = 1*1*1 = 1 for all a,b,c. But M is only defined for specific (a,b,c) tuples!
    # For other (a,b,c) tuples, M[a,b,c] = 0, but our decomposition gives 1.
    
    # So the decomposition must match M exactly, including the zeros.
    # M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and M[a,b,c] = 0 otherwise.
    
    # With U=V=W=1 (all ones), we get M[a,b,c] = 1 for ALL a,b,c, not just the valid ones.
    # So rank 1 doesn't work.
    
    # The schoolbook decomposition has M[a,b,c] = 1 only for the specific (a,b,c) tuples.
    # To get rank < 63, we need the sum of products to be 1 for valid tuples and 0 for others.
    
    # This is a much harder constraint. Let me just use the schoolbook decomposition.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
