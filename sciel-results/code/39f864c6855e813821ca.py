"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 36 (from Strassen-based construction)
    target_rank = 36
    
    # The multiplication tensor T is a 3-way tensor of shape (n, m, p) = (2, 4, 5)
    # T[i, j, k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B 
    # contribute to the (i,k) entry of C. In matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor)
    # Actually, for matrix multiplication, the tensor is:
    # T_{i,j,k} = 1 if we're multiplying A[i,j] by B[j,k] to get C[i,k]
    # The tensor is simply all ones in a certain sense, but let me think more carefully.
    
    # The matrix multiplication tensor for multiplying an (n x m) matrix by an (m x p) matrix:
    # T is a tensor in R^{n x m x p} where T[i, j, k] = 1 for all valid i, j, k
    # Wait, that's not right either. Let me reconsider.
    
    # For matrix multiplication: C = A * B where A is n x m, B is m x p
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    # The multiplication tensor T has entries T[i, j, k] = 1 for all i in range(n), j in range(m), k in range(p)
    # No wait - the tensor T is defined such that:
    # For basis vectors e_i^A, e_j^B, e_k^C:
    # T(e_i^A, e_j^B, e_k^C) = 1 if j is the shared index, i.e., T[i,j,k] = delta_{shared}
    # Actually the standard multiplication tensor is:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * [the coefficient]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k is NOT correct.
    
    # Let me be precise. The matrix multiplication tensor T ∈ R^{n×m×p} is defined by:
    # T[i, j, k] = 1 if and only if the product A[i,j] * B[j,k] contributes to C[i,k]
    # Since every pair (i,j) in A and (j,k) in B with the same j contributes to C[i,k],
    # we have T[i, j, k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    
    # Hmm, but that would make the tensor all-ones, which has rank 1, which is wrong.
    
    # Let me reconsider. The correct formulation:
    # The multiplication tensor T is such that for matrices A, B:
    # C = T(A, B) where C[i,k] = sum_j A[i,j] B[j,k]
    # As a 3-tensor: T ∈ (R^n)^* ⊗ (R^m)^* ⊗ (R^p)^* or similar
    # The entries are: T[i, j, k] = 1 for all i, j, k
    # No, that's still all ones.
    
    # OK here's the correct view. The tensor T is in R^{n*m*p} (flattened) or as a 3-way tensor.
    # For the bilinear map (A, B) -> C where C = AB:
    # The tensor T satisfies: C_{ik} = T_{ikj} A_{ij} B_{jk}
    # So T_{ikj} = 1 for all i, k, j. 
    # Wait, that means T is the all-ones tensor of shape (n, p, m) or (n, m, p)?
    
    # Let me use the standard convention from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # Where T is the multiplication tensor.
    
    # The multiplication tensor for n×m times m×p matrix multiplication:
    # T is a 3-way tensor with T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # And T = 0 elsewhere.
    
    # So T[a, b, c] = 1 if and only if there exist i,j,k such that:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # i.e., a mod m = b div p and b mod p = c div n and c mod n = a div m
    
    # Let's verify: n=2, m=4, p=5
    # a ranges over 0..7 (n*m = 8)
    # b ranges over 0..19 (m*p = 20)
    # c ranges over 0..9 (p*n = 10)
    
    # T[a, b, c] = 1 iff (a div m)*... let me index:
    # i = a // m, j = a % m
    # j' = b // p, k = b % p
    # k' = c // n, i' = c % n
    # T[a,b,c] = 1 iff i == i' and j == j' and k == k'
    
    # So the nonzero entries are: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    # There are n*m*p = 2*4*5 = 40 such entries.
    
    # Now we want to find rank-minimal decomposition:
    # T = sum_{c=0}^{rank-1} u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    
    # The schoolbook solution has rank = n*m*p = 40.
    # We want to find if rank 36 works.
    
    # Strategy: Use the known Strassen-based construction for 2x4x5.
    # The key insight is that we can use Strassen's algorithm on blocks.
    
    # Let me try a numerical approach: initialize with a rank-40 decomposition,
    # then try to reduce rank by merging terms.
    
    # First, create the rank-40 schoolbook decomposition:
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank40 = schoolbook_decomposition()
    
    # Now try to find a rank-36 decomposition using optimization.
    # We'll use alternating least squares or gradient descent.
    
    # Initialize with random perturbations of the rank-40 solution,
    # keeping only 36 terms.
    
    best_U, best_V, best_W, best_rank = U, V, W, rank40
    best_error = float('inf')
    
    deadline = time.time() + 100  # Leave some safety margin
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        rank = target_rank  # Try rank 36
        
        # Initialize: take random combinations of the 40 schoolbook terms
        # and try to fit them into 36 terms.
        
        # Strategy: Start with 36 random terms from the 40, then optimize.
        indices = np.random.choice(rank40, size=rank, replace=False)
        
        U_curr = np.zeros((n*m, rank))
        V_curr = np.zeros((m*p, rank))
        W_curr = np.zeros((p*n, rank))
        
        for c, idx in enumerate(indices):
            U_curr[:, c] = U[:, idx]
            V_curr[:, c] = V[:, idx]
            W_curr[:, c] = W[:, idx]
        
        # Add small random perturbation
        U_curr += np.random.randn(n*m, rank) * 0.01
        V_curr += np.random.randn(m*p, rank) * 0.01
        W_curr += np.random.randn(p*n, rank) * 0.01
        
        # Optimize using alternating least squares
        # For fixed V, W: minimize ||T - sum_c u_c ⊗ v_c ⊗ w_c||
        # This is a linear system for u_c.
        
        # Actually, let's use a simpler approach: gradient-based optimization
        # or use the fact that we can solve for one mode at a time.
        
        # For fixed V and W, the problem is:
        # T[a, b, c] = sum_{r} U[a, r] * V[b, r] * W[c, r]
        # For fixed V, W, this is linear in U.
        # Specifically, for each a:
        # sum_r U[a,r] * (V[:,r] * W[:,r]) = T[a, :, :] (flattened)
        
        # Let's build the linear system for U:
        # For each a, we have: sum_r U[a,r] * V[:,r] * W[:,r] = T_flat[a, :]
        # where T_flat is T reshaped to (n*m, m*p*p*n)
        
        # This is getting complex. Let me use a different approach.
        
        # Use numpy to build the tensor and try to do CP decomposition.
        
        # Build T as a 3D array
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Try CP decomposition with rank 36
        # Use a simple ALS approach
        
        U_opt = U_curr.copy()
        V_opt = V_curr.copy()
        W_opt = W_curr.copy()
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each column r, minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||
            # Fix V, W. For each r, U[:,r] should satisfy:
            # sum_r U[:,r] * V[:,r] * W[:,r] = T (tensor equation)
            # This is linear in U. Let's build the system.
            
            # Reshape T to (n*m, m*p, p*n)
            # For fixed V, W, define: for each pair (b,c), the "slice" T[:,b,c]
            # sum_r U[:,r] * V[b,r] * W[c,r] = T[:,b,c]
            
            # Let M_bc = sum_r V[b,r] * W[c,r] * U[:,r] ... no
            # T[:,b,c] = sum_r U[:,r] * (V[b,r] * W[c,r])
            
            # So for each (b,c): T[:,b,c] = U @ (V[b,:]*W[c,:])
            # This is: T[:,b,c] = U * diag(V[b,:]*W[c,:]) ... no
            
            # Let me think again. U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # For fixed V, W, to find U:
            # For each a: T[a,b,c] for all b,c = sum_r U[a,r] * V[b,r] * W[c,r]
            # Let's define for each r: M_r = V[:,r] ⊗ W[:,r] (a (m*p)*(p*n) matrix)
            # Then T[a] (flattened to (m*p*p*n,)) = sum_r U[a,r] * M_r.flatten()
            
            # So we have: T_flat[a] = U[a,:] @ M_flat where M_flat is (rank, m*p*p*n)
            # M_flat[r, b*p*n + c] = V[b,r] * W[c,r]
            
            M_flat = np.zeros((rank, m*p*p*n))
            for r in range(rank):
                M_flat[r, :] = (V[:,r] * W[:,r]).reshape(m*p, p*n).flatten()
            
            T_flat = T.reshape(n*m, m*p*p*n)
            
            # Solve: T_flat = U @ M_flat
            # This is a linear system: U = T_flat @ M_flat^+ (pseudo-inverse)
            # But M_flat is (rank, m*p*p*n) which is (36, 200)
            # We want U of shape (n*m, rank) = (8, 36)
            # T_flat is (8, 200)
            
            # U @ M_flat = T_flat
            # U = T_flat @ pinv(M_flat)
            
            try:
                U_new = T_flat @ np.linalg.pinv(M_flat)
            except:
                U_new = U_opt
            
            # Update V similarly
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed U, W:
            # T[a,b,c] = V[b,:] @ (U[a,:]*W[c,:])
            # Let N_flat[s, a*p*n + c] = U[a,s] * W[c,s]
            # T_flat[b] = N_flat @ V[b,:]^T ... let me be more careful
            
            # T[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
            # For fixed U, W, define for each r: P_r[a, c] = U[a,r] * W[c,r]
            # T[a,b,c] = sum_r V[b,r] * P_r[a,c]
            # Flattened: T_flat2[b] (of shape (n*m*p*n,)) = sum_r V[b,r] * P_r.flatten()
            
            P_flat = np.zeros((rank, n*m*p*n))
            for r in range(rank):
                P_flat[r, :] = (U[:,r] * W[:,r]).reshape(n*m, p*n).flatten()
            
            T_flat2 = T.reshape(m*p, n*m*p*n)
            
            try:
                V_new = T_flat2 @ np.linalg.pinv(P_flat)
            except:
                V_new = V_opt
            
            # Update W
            # T[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
            Q_flat = np.zeros((rank, n*m*m*p))
            for r in range(rank):
                Q_flat[r, :] = (U[:,r] * V[:,r]).reshape(n*m, m*p).flatten()
            
            T_flat3 = T.reshape(p*n, n*m*m*p)
            
            try:
                W_new = T_flat3 @ np.linalg.pinv(Q_flat)
            except:
                W_new = W_opt
            
            U_opt = U_new
            V_opt = V_new
            W_opt = W_new
            
            # Check convergence
            T_reconstructed = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
            error = np.linalg.norm(T_reconstructed - T)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W, best_rank = U_opt.copy(), V_opt.copy(), W_opt.copy(), rank
            
            if error < 1e-8:
                break
        
        if time.time() > deadline:
            break
    
    # Check if we found a good rank-36 solution
    if best_error < 1e-6:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If not, return the schoolbook rank-40 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

