"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    DEADLINE = 135  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a decomposition with rank less than schoolbook
    # For matrix multiplication, the tensor is T_{i,j,k} = delta_{i,k} * delta_{j,j} (standard)
    # Actually, the multiplication tensor for A (n x m) * B (m x p) -> C (n x p) is:
    # T_{i,j,k} = 1 if the index corresponds to C[i,k] += A[i,j] * B[j,k]
    # In the standard tensor decomposition, we want:
    # T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]  (or similar)
    
    # Let me be more precise about the tensor structure.
    # The matrix multiplication tensor M_{i,j,k} where:
    # i indexes rows of A (and C): 0..n-1
    # j indexes cols of A (and rows of B): 0..m-1  
    # k indexes cols of B (and C): 0..p-1
    # M_{i,j,k} = 1 for all valid (i,j,k), representing C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A rank-r decomposition is:
    # M_{i,j,k} = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # Wait, the interface says:
    # U: (n*m, rank) - so U[i*m+j, r]
    # V: (m*p, rank) - so V[j*p+k, r]
    # W: (p*n, rank) - so W[k*n+i, r]
    
    # And the decomposition is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # And for "wrong" combinations, they should be 0. But wait, in matrix multiplication,
    # every triple (i,j,k) contributes to the product, so actually all entries of the 
    # multiplication tensor are 1.
    
    # Hmm, let me reconsider. The matrix multiplication tensor has entries:
    # T[i,j,k] = 1 for all i,j,k in valid ranges
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need a rank-r decomposition where:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a very specific structure. Let me think about what rank is achievable.
    
    # For small cases, known results:
    # 2x2 matrix mult: rank 7 (Strassen)
    # 3x3 matrix mult: rank 23 (known lower bound 19, upper bound 23)
    # 3x3 x 3x8: not well studied
    
    # Let me try a greedy/randomized approach to find a low-rank decomposition.
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        rank = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Verify using numpy for speed."""
        rank = U.shape[1]
        # Build the tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this as: for each r, outer product
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # u[i,j] * v[j,k] * w[k,i] summed over j,k... no wait
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is element-wise: for each (i,j,k), u[i,j]*v[j,k]*w[k,i]
            # We need to broadcast properly
            # u[i,j], v[j,k], w[k,i]
            # Create a 3D array
            # u has shape (n,m), v has shape (m,p), w has shape (p,n)
            # We want for each (i,j,k): u[i,j] * v[j,k] * w[k,i]
            # Reshape: u -> (n, m, 1), v -> (1, m, p), w -> (p, 1, n)
            # But w[k,i] means index k then i, so w is (p,n), w[k,i]
            # So: u[i,j,0] * v[0,j,k] * w[k,0,i]... this is tricky
            # Let me just do it directly with einsum or explicit
            # Actually, let's use:
            # tmp[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            # = u[i,j,0] * v[0,j,k] * w[k,0,i]
            # Reshape u to (n,m,1), v to (1,m,p), w to (p,1,n)
            # Then u[:,:,0] * v[:, :, :] won't work directly because of the index cycling
            # Let me just loop or use a different approach
            # For small sizes, just do:
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        T[i2, j2, k2] += u[i2, j2] * v[j2, k2] * w[k2, i2]
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    def verify_fast2(U, V, W):
        """Faster verification using numpy broadcasting."""
        rank = U.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Use einsum: 'ij,jk,ki->ijk'
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)
    
    # Try to find a decomposition with rank < schoolbook
    # Start with a reasonable guess and try to reduce
    
    # For 3x3x3, Strassen-like decompositions exist. For 3x3x8, we can try
    # to use known low-rank patterns.
    
    # Let's try a randomized greedy approach
    best_rank = schoolbook_rank
    best_decomp = None
    
    # First, set up the schoolbook as baseline
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    if verify_fast2(U_school, V_school, W_school):
        best_rank = schoolbook_rank
        best_decomp = (U_school, V_school, W_school)
    
    # Try to find better decompositions using optimization
    # We'll try ranks from a lower bound up
    
    # Lower bound for matrix multiplication rank: at least n*m + m*p + p*n - (n-1)*(m-1)*(p-1)... 
    # Actually, simple lower bound: rank >= n*m*p / (something)
    # For our case, let's try ranks from 20 up to schoolbook
    
    # Use a simple iterative refinement / random search
    import random
    random.seed(42)
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(20, schoolbook_rank + 1):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Try multiple random initializations
        max_attempts = 5 if target_rank < 40 else 3
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random values
            U[:] = np.random.randn(n*m, target_rank) * 0.1
            V[:] = np.random.randn(m*p, target_rank) * 0.1
            W[:] = np.random.randn(p*n, target_rank) * 0.1
            
            # Greedy: try to satisfy the constraints one at a time
            # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a system of n*m*p equations in rank*3*... variables
            # But it's nonlinear (product of three variables)
            
            # Alternative: use the fact that for a fixed r, the term is
            # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # This is like a rank-1 tensor in the (i,j,k) space for each r,
            # but with the specific index structure.
            
            # Let me think differently. The constraint is:
            # For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # This is equivalent to: the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # equals the all-ones tensor.
            
            # For a fixed r, define:
            # A_r[i,j] = U[i*m+j, r]
            # B_r[j,k] = V[j*p+k, r]  
            # C_r[k,i] = W[k*n+i, r]
            # Then T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
            
            # Note the cyclic index structure: A[i,j], B[j,k], C[k,i]
            
            # This is a specific type of tensor decomposition.
            
            # Let's try a simple approach: for each r, set A_r, B_r, C_r to be
            # rank-1 matrices in their respective spaces, i.e.,
            # A_r[i,j] = a_r[i] * b_r[j]
            # B_r[j,k] = b_r[j] * c_r[k]  (sharing b_r)
            # C_r[k,i] = c_r[k] * a_r[i]  (sharing c_r and a_r)
            # Then A_r[i,j]*B_r[j,k]*C_r[k,i] = a_r[i]*b_r[j]*b_r[j]*c_r[k]*c_r[k]*a_r[i]
            # = a_r[i]^2 * b_r[j]^2 * c_r[k]^2
            # So T[i,j,k] = sum_r (a_r[i]*b_r[j]*c_r[k])^2
            # We need this to equal 1 for all i,j,k.
            
            # This is a sum of squares equaling 1. This can be done if we have
            # enough terms. For example, if we set for each (i,j,k) a dedicated term,
            # we get rank = n*m*p (schoolbook).
            
            # But we can do better. If we choose a_r, b_r, c_r such that
            # sum_r a_r[i]^2 * b_r[j]^2 * c_r[k]^2 = 1 for all i,j,k,
            # this is a separable approximation of the all-ones matrix.
            
            # The all-ones tensor can be written as:
            # 1 = sum_r a_r[i] * b_r[j] * c_r[k]  (standard tensor decomposition)
            # But we have squares, so it's different.
            
            # Actually, let me reconsider. The cyclic structure means:
            # T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
            # where A_r is n x m, B_r is m x p, C_r is p x n.
            
            # This is NOT the standard matrix multiplication tensor structure.
            # The standard one would be C[i,k] = sum_j A[i,j] * B[j,k], which gives
            # T[i,j,k] = A[i,j] * B[j,k] * delta_{something}... 
            
            # I think I'm overcomplicating this. Let me just use the verification
            # and try to optimize.
            
            # Simple gradient-free approach: randomly perturb and keep if improvement
            # Define objective: sum of squared errors
            def objective(U, V, W):
                T = np.zeros((n, m, p))
                for r in range(target_rank):
                    u = U[:, r].reshape(n, m)
                    v = V[:, r].reshape(m, p)
                    w = W[:, r].reshape(p, n)
                    T += np.einsum('ij,jk,ki->ijk', u, v, w)
                err = T - 1.0
                return np.sum(err**2)
            
            # Try a few iterations of random perturbation
            best_obj = objective(U, V, W)
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(50):
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Random perturbation
                U_trial = U.copy() + np.random.randn(n*m, target_rank) * 0.05
                V_trial = V.copy() + np.random.randn(m*p, target_rank) * 0.05
                W_trial = W.copy() + np.random.randn(p*n, target_rank) * 0.05
                
                obj_trial = objective(U_trial, V_trial, W_trial)
                if obj_trial < best_obj:
                    best_obj = obj_trial
                    U, V, W = U_trial, V_trial, W_trial
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_obj < 1e-8:
                        break
            
            if best_obj < 1e-8:
                if verify_fast2(best_U, best_V, best_W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_decomp = (best_U, best_V, best_W)
                    break  # Found a valid decomposition for this rank
    
    # Convert to list format
    if best_decomp is not None:
        U, V, W = best_decomp
    else:
        U, V, W = U_school, V_school, W_school
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
