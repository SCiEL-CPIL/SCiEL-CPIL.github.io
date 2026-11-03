"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    # The tensor T[a,b,c] = 1 for all a,b,c in the schoolbook convention
    # We need sum_r U[a,r] * V[b,r] * W[c,r] = 1 for all a,b,c
    
    # Strategy: 
    # 1. Try structured low-rank decompositions (Kronecker products)
    # 2. Use alternating least squares with structured initialization
    # 3. Fall back to schoolbook rank 54 if needed
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Precompute the structure
    # For matrix multiplication 2x3 * 3x9:
    # A[i,j] (i in 0..1, j in 0..2) -> index a = i*3 + j
    # B[j,k] (j in 0..2, k in 0..8) -> index b = j*9 + k
    # C[i,k] (i in 0..1, k in 0..8) -> index c = k*2 + i (trace convention W[k*n+i,r])
    
    # The tensor is T[a,b,c] = 1 for all a,b,c
    # This is because in the schoolbook, each term contributes 1 to exactly one (a,b,c)
    # and the sum over all terms gives 1 for each (a,b,c)
    
    # Wait, let me reconsider. In the schoolbook:
    # For each (i,j,k), we have U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So for a given (a,b,c), the sum is 1 only if there exists (i,j,k) such that:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # But a, b, c are independent indices in the tensor.
    # Actually, the tensor being decomposed is the "identity" tensor where
    # T[a,b,c] = 1 if the indices are "compatible" and 0 otherwise?
    
    # No, looking at the schoolbook more carefully:
    # The product sum_r U[a,r]V[b,r]W[c,r] should equal 1 for ALL (a,b,c)
    # because each (a,b,c) triple corresponds to exactly one (i,j,k) in the sum
    # and that term contributes 1.
    
    # Wait, that's not right either. Let me trace through:
    # a = i0*m + j0, b = j1*p + k0, c = k1*n + i1
    # The term for (i,j,k) contributes to U[a,r]V[b,r]W[c,r] only if:
    # U[i*m+j, r] is non-zero -> i*m+j = a -> i=i0, j=j0
    # V[j*p+k, r] is non-zero -> j*p+k = b -> j=j1, k=k0
    # W[k*n+i, r] is non-zero -> k*n+i = c -> k=k1, i=i1
    # So we need i0=i1, j0=j1, k0=k1
    # This means the sum is 1 only if a=i0*m+j0, b=j0*p+k0, c=k0*n+i0 for some i0,j0,k0
    # Otherwise it's 0.
    
    # So the tensor T is NOT all ones. It's 1 only on the "diagonal" where the indices match.
    # T[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 0 otherwise
    
    # Let's verify: for n=2, m=3, p=9:
    # a in 0..5, b in 0..26, c in 0..17
    # a = i*3+j: i in {0,1}, j in {0,1,2}
    # b = j*9+k: j in {0,1,2}, k in {0..8}
    # c = k*2+i: k in {0..8}, i in {0,1}
    
    # For a given (a,b,c), we can recover:
    # i = a // 3, j = a % 3
    # j' = b // 9, k = b % 9
    # k' = c // 2, i' = c % 2
    # T[a,b,c] = 1 if i==i' and j==j' and k==k'
    
    # So the tensor has 2*3*9 = 54 ones and the rest are zeros.
    # Total size: 6*27*18 = 2916 entries
    
    # Let's build the target tensor
    target = np.zeros((num_a, num_b, num_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Now we need to find U (num_a x R), V (num_b x R), W (num_c x R)
    # such that sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
    
    # This is equivalent to finding a rank-R decomposition of the 3-tensor target.
    
    # Approach 1: Alternating Least Squares
    # For fixed V, W: U[a,:] = argmin sum_{b,c} (sum_r U[a,r]V[b,r]W[c,r] - target[a,b,c])^2
    # This is a linear system: for each a, solve for U[a,:]
    # sum_r U[a,r] * (sum_{b,c} V[b,r]W[c,r] * delta_{target}) ... this is complex
    
    # Better: use the fact that for fixed V, W, the equation for U[a,:] is:
    # sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c] for all b,c
    # Let M[a,b,c] = V[b,r] * W[c,r] summed over r? No.
    
    # Actually, for fixed V, W, define for each a:
    # We need U[a,:] such that for all b,c: sum_r U[a,r] * (V[b,r] * W[c,r]) = target[a,b,c]
    # Let X[a,b,c] = target[a,b,c]
    # Let Y[b,c,r] = V[b,r] * W[c,r]
    # Then for each a: sum_r U[a,r] * Y[:, :, r] = X[a, :, :]
    # This is a linear system: U[a,:] @ Y_vec = X[a,:].flatten()
    # where Y_vec is the vectorization of Y over (b,c,r)
    
    # But Y depends on r, so it's:
    # For each a, b, c: sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c]
    
    # Let's define A_a as a matrix where A_a[b*c + (b2*c2), r] ... this is getting complicated.
    
    # Simpler approach: use the tensor unfolding.
    # Unfold the tensor along the first mode:
    # T_(1) is a matrix of size (num_a) x (num_b * num_c)
    # T_(1)[a, b*num_c + c] = target[a, b, c]
    # Then we need U * (W ⊗ V)^T = T_(1) approximately
    # where W ⊗ V is the Kronecker product
    
    # Actually, the CP decomposition says:
    # T_(1) = U * diag(W_vec) * V^T ... no, that's not quite right.
    
    # Let me use a different approach: alternating minimization with direct solving.
    
    def compute_residual(U, V, W):
        """Compute the residual sum_r U[a,r]V[b,r]W[c,r] - target[a,b,c]"""
        # Efficient computation using einsum
        # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r (U[a,r] * V[b,r]) * W[c,r]
        # Let's compute in steps
        # temp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = sum_r U[a,r] * (V[b,r] * W[c,r])
        
        # Use einsum
        # temp = np.einsum('ar,br,cr->abc', U, V, W)
        # But this might be slow. Let's try a different way.
        
        # Compute UV = U @ V.T -> shape (num_a, num_b, R)? No.
        # UV[a,b] = sum_r U[a,r] * V[b,r] -> shape (num_a, num_b)
        # Then result[a,b,c] = sum_r UV[a,b,r] * W[c,r]? No, UV doesn't have r.
        
        # Let's do: 
        # temp1[a,b,c] = U[a,r] * V[b,r] for each r, then sum over r? No.
        
        # Actually, the simplest is:
        # result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.einsum('ar,br,cr->abc', U, V, W) - target
    
    def compute_error(U, V, W):
        res = compute_residual(U, V, W)
        return np.sum(res**2)
    
    def solve_for_U(V, W, target):
        """Solve for U given V, W"""
        # For each a: sum_r U[a,r] * V[b,r] * W[c,r] = target[a,b,c] for all b,c
        # Let's form the matrix M where M[bc, r] = V[b,r] * W[c,r]
        # Then U[a,:] @ M = target[a,:].flatten()
        
        # M has shape (num_b * num_c, R)
        M = np.zeros((num_b * num_c, R))
        for b in range(num_b):
            for c in range(num_c):
                M[b * num_c + c, :] = V[b, :] * W[c, :]
        
        U = np.zeros((num_a, R))
        for a in range(num_a):
            rhs = target[a, :, :].flatten()
            # Solve M @ U[a,:] = rhs
            # This is underdetermined if R < num_b*num_c, which it will be
            # Use least squares
            U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return U
    
    def solve_for_V(U, W, target):
        """Solve for V given U, W"""
        # For each b: sum_r V[b,r] * U[a,r] * W[c,r] = target[a,b,c] for all a,c
        M = np.zeros((num_a * num_c, R))
        for a in range(num_a):
            for c in range(num_c):
                M[a * num_c + c, :] = U[a, :] * W[c, :]
        
        V = np.zeros((num_b, R))
        for b in range(num_b):
            rhs = target[:, b, :].transpose(1, 0).flatten()  # target[a,b,c] -> for fixed b, shape (num_a, num_c)
            V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return V
    
    def solve_for_W(U, V, target):
        """Solve for W given U, V"""
        # For each c: sum_r W[c,r] * U[a,r] * V[b,r] = target[a,b,c] for all a,b
        M = np.zeros((num_a * num_b, R))
        for a in range(num_a):
            for b in range(num_b):
                M[a * num_b + b, :] = U[a, :] * V[b, :]
        
        W = np.zeros((num_c, R))
        for c in range(num_c):
            rhs = target[:, :, c].flatten()  # shape (num_a, num_b)
            W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return W
    
    def alternating_minimization(R, n_iters=200, n_restarts=3):
        """Run alternating minimization for rank R"""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            U = np.random.randn(num_a, R) * 0.1
            V = np.random.randn(num_b, R) * 0.1
            W = np.random.randn(num_c, R) * 0.1
            
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                # Solve for U
                U = solve_for_U(V, W, target)
                # Solve for V
                V = solve_for_V(U, W, target)
                # Solve for W
                W = solve_for_W(U, V, target)
                
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-12:
                    break
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-10:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err
    
    # Try ranks from low to high
    # The schoolbook rank is 54
    # Let's try to find a lower rank
    
    # First, try some structured initializations
    # Idea: use Kronecker product structure
    
    # For 2x3x9, we can think of it as:
    # C = A * B where A is 2x3, B is 3x9
    # We can decompose B into 3 blocks of 3x3
    # B = [B1, B2, B3] where each Bi is 3x3
    # C = A * B1 + A * B2 + A * B3 (block-wise)
    # Each A*Bi is 2x3 multiplication
    # The rank of 2x3x3 matrix multiplication is known to be 6
    # So total rank would be 3 * 6 = 18
    
    # Let's try to construct a rank-18 decomposition using this structure
    
    def construct_kronecker_decomposition():
        """
        Construct a rank-18 decomposition using Kronecker product structure.
        2x3x9 = 3 parallel 2x3x3 multiplications
        Each 2x3x3 has rank 6
        """
        # For 2x3x3 matrix multiplication:
        # A is 2x3, B is 3x3, C is 2x3
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[a,b,c] where a=(i,j), b=(j',k'), c=(k'',i'')
        # T[a,b,c] = 1 if i=i'', j=j', k=k''
        
        # Rank-6 decomposition for 2x3x3:
        # We need U6 (6x6), V6 (9x6), W6 (6x6)
        
        # One known decomposition:
        # For r in 0..5, let i_r = r % 2, j_r = r // 2
        # But this only gives 2*3=6 terms if we use i in {0,1}, j in {0,1,2}
        
        # Actually, let's use the following:
        # The 2x3x3 tensor can be decomposed as:
        # sum_{i=0}^{1} sum_{j=0}^{2} e_i ⊗ e_j ⊗ e_j ⊗ e_i
        # Wait, that's not a rank-1 decomposition.
        
        # Let me think differently. The tensor for 2x3x3 is:
        # T[i,j; j',k; k',i'] = delta(i,i') delta(j,j') delta(k,k')
        # This is the identity tensor.
        
        # A rank-1 decomposition would be:
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is a vector in R^6, v_r in R^9, w_r in R^6
        
        # For the identity tensor, we can use:
        # For each (i,j,k), define a rank-1 tensor
        # u_{ijk} = e_{i*3+j}, v_{ijk} = e_{j*3+k}, w_{ijk} = e_{k*2+i}
        # But this gives rank 18, not 6.
        
        # The minimal rank of 2x3x3 is actually 6, but finding it is non-trivial.
        # Let's just use the schoolbook for 2x3x3 which has rank 18,
        # and then the total would be 3*18 = 54, which is the same as the original schoolbook.
        
        # So this approach doesn't help. Let's stick with alternating minimization.
        
        return None
    
    # Try alternating minimization for various ranks
    for R in [18, 24, 30, 36, 42, 48, 54]:
        if time.time() > deadline - 10:
            break
        
        U, V, W, err = alternating_minimization(R, n_iters=100, n_restarts=5)
        
        if U is not None and err < 1e-10:
            # Verify the solution
            final_err = compute_error(U, V, W)
            if final_err < 1e-10:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(num_a)]
    V = [[0] * rank for _ in range(num_b)]
    W = [[0] * rank for _ in range(num_c)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
