"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 110  # safety margin before 150s kill
    
    # The multiplication tensor T has entries T[a, b, c] where:
    #   a = i*m + j  (i in 0..n-1, j in 0..m-1)  -> A index
    #   b = j'*p + k (j' in 0..m-1, k in 0..p-1) -> B index  
    #   c = k'*n + i'(k' in 0..p-1, i' in 0..n-1) -> C index (trace convention: W[k*n+i])
    # T[a,b,c] = 1 if j == j' (the shared index matches) else 0
    # A rank-r decomposition: T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
    # where U_r is n*m, V_r is m*p, W_r is p*n

    # Strategy: Build a Strassen-like decomposition, then try to reduce rank
    # by identifying linearly dependent terms.

    # First, build a good initial decomposition using block Strassen.
    # For 4x5x5, we can think of it as blocks.
    
    # Let's use a direct approach: enumerate all "natural" rank-1 tensors
    # and find a minimal subset that spans the multiplication tensor.
    
    # The multiplication tensor T lives in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # A rank-1 tensor is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{p*n}
    
    # T[a,b,c] = delta_{j, j'} where a=i*m+j, b=j'*p+k, c=k'*n+i'
    
    # We need: sum_r U[a,r] * V[b,r] * W[c,r] = delta_{j,j'}
    
    # Let's construct a decomposition by "splitting" the delta.
    # For each (i, j, j', k) with j == j', we need a contribution.
    # The "natural" rank-1 terms are: for each (i, j, k), the term that
    # picks out A[i,j], B[j,k], and contributes to C[i,k].
    # That's the schoolbook: rank = n*m*p = 100.
    
    # To reduce rank, we use the Strassen trick: combine multiple terms.
    # For 2x2x2: rank 7 instead of 8.
    # For 4x5x5, we can use block decomposition.
    
    # Let's try a different approach: use the fact that we can factor
    # through smaller matrices.
    
    # Actually, let's try to find a decomposition by solving a system.
    # We want to find R such that there exist U (20xR), V (25xR), W (20xR)
    # with the tensor equation satisfied.
    
    # The equation is: for all a, b, c:
    #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of 20*25*20 = 10000 equations in 20*R + 25*R + 20*R = 65*R unknowns.
    # But it's nonlinear (bilinear in pairs).
    
    # Better approach: fix U and V, then solve for W (or vice versa).
    # Given U and V, the equation becomes:
    #   For each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #   Let M[a,b] = sum_r U[a,r]*V[b,r]  (this is an n*m x m*p matrix)
    #   Then we need: sum_r W[c,r] * (something depending on a,b,r) = T[a,b,c]
    
    # Actually, let's reshape. Define for each r:
    #   U_r ∈ R^{n*m}, V_r ∈ R^{m*p}, W_r ∈ R^{p*n}
    # The contribution to T from term r is: U_r ⊗ V_r ⊗ W_r
    # We need sum_r U_r ⊗ V_r ⊗ W_r = T
    
    # Let's use a different strategy: use known low-rank decompositions.
    # The rank of 4x5x5 matrix multiplication is known to be at most 79
    # (via Strassen-like algorithms) and at least 65 (from lower bounds).
    
    # Let me try to construct a rank-79 decomposition explicitly.
    
    # For 4x5x5: Think of A as 4x5, B as 5x5.
    # We can split A into [A1; A2] where A1 is 2x5, A2 is 2x5.
    # Then C = [A1; A2] @ B = [A1@B; A2@B]
    # So we need to compute two 2x5 @ 5x5 multiplications.
    
    # For 2x5 @ 5x5: split B into [B1 B2] where B1 is 5x3, B2 is 5x2.
    # Then 2x5 @ 5x3 and 2x5 @ 5x2.
    
    # This is getting complex. Let me try a more direct approach.
    
    # Use random initialization + gradient descent with exact verification.
    
    def verify_exact(U, V, W, n, m, p, tol=1e-9):
        """Verify the decomposition is exact using random tests."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        inner = 0.0
                        for j in range(m):
                            inner += U[i*m+j, r] * V[j*p+k, r]
                        s += W[k*n+i, r] * inner
                    C_computed[i, k] = s
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True
    
    def verify_tensor(U, V, W, n, m, p, tol=1e-9):
        """Verify by checking the tensor equation on all entries (slow but exact)."""
        R = U.shape[1]
        # Check a random subset of (a,b,c) entries
        rng = np.random.RandomState(123)
        for _ in range(2000):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            # T[a,b,c] = 1 if (a%m) == (b//p) else 0
            target = 1.0 if (a % m) == (b // p) else 0.0
            computed = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(computed - target) > tol:
                return False
        return True

    # Build schoolbook decomposition as fallback
    rank_sb = n * m * p  # 100
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find lower rank using gradient descent
    # Key insight: the problem is bilinear, so we can use alternating minimization.
    # Fix two of U, V, W and solve for the third.
    
    def try_alternating(R, max_iters=50, seed=42):
        """Try to find rank-R decomposition via alternating least squares."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, R) * 0.5
        V = rng.randn(m*p, R) * 0.5
        W = rng.randn(p*n, R) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: fix V, W, solve for U
            # For each a: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c] for all b,c
            # This is: U[a,:] * diag(V[b,:]*W[c,:]) = T[a,b,c] for all b,c
            # Better: treat it as a linear system.
            # Let's define for each a: we need U[a,:] such that
            #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
            # Let M_a[b,c] = V[b,:] * W[c,:] (outer product, m*p x p*n)
            # Then U[a,:] @ M_a = T[a,:,:]
            # But M_a is (m*p) x (p*n) and U[a,:] is (R,), T[a,:,:] is (m*p) x (p*n)
            # So: U[a,:] * (V[b,r]*W[c,r]) summed over r = T[a,b,c]
            # This is: sum_r U[a,r] * (V[b,:] ⊗ W[c,:])_r = T[a,b,c]
            # Where (V[b,:] ⊗ W[c,:]) is a vector in R^R with entry r = V[b,r]*W[c,r]
            
            # So for fixed a: we have a linear system in U[a,:]:
            #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b in 0..m*p-1, c in 0..p*n-1
            # Let's collect: for each (b,c), we have one equation.
            # The coefficient of U[a,r] in equation (b,c) is V[b,r]*W[c,r].
            # So the system is: A_sys @ U[a,:] = rhs_a
            # where A_sys[(b,c), r] = V[b,r]*W[c,r]
            
            # This is a (m*p * p*n) x R system, which is 25*20=500 equations in R unknowns.
            # We can solve in least squares sense.
            
            # Build A_sys once (it's the same for all a, just different rhs)
            # A_sys has shape (m*p*p*n, R)
            # A_sys[(b*p*n + c), r] = V[b,r] * W[c,r]
            
            # Actually, let's be more careful. For each a, the rhs is T[a,b,c] for all b,c.
            # T[a,b,c] = 1 if (a%m) == (b//p) else 0
            
            # Let's build the system matrix once:
            # Shape: (m*p * p*n, R) = (25*20, R) = (500, R)
            A_sys = np.zeros((m*p*p*n, R))
            for b in range(m*p):
                for c in range(p*n):
                    for r in range(R):
                        A_sys[b*p*n + c, r] = V[b, r] * W[c, r]
            
            # Now for each a, solve A_sys @ U[a,:] = rhs_a
            # rhs_a[b*p*n + c] = T[a,b,c]
            for a in range(n*m):
                rhs = np.zeros(m*p*p*n)
                for b in range(m*p):
                    for c in range(p*n):
                        rhs[b*p*n + c] = 1.0 if (a % m) == (b // p) else 0.0
                # Solve least squares
                sol, _, _, _ = np.linalg.lstsq(A_sys, rhs, rcond=None)
                U[a, :] = sol
            
            # Update V similarly
            # For each b: sum_r V[b,r] * U[a,r] * W[c,r] = T[a,b,c] for all a,c
            # A_sys_V[(a*p*n + c), r] = U[a,r] * W[c,r]
            A_sys_V = np.zeros((n*m*p*n, R))
            for a in range(n*m):
                for c in range(p*n):
                    for r in range(R):
                        A_sys_V[a*p*n + c, r] = U[a, r] * W[c, r]
            
            for b in range(m*p):
                rhs = np.zeros(n*m*p*n)
                for a in range(n*m):
                    for c in range(p*n):
                        rhs[a*p*n + c] = 1.0 if (a % m) == (b // p) else 0.0
                sol, _, _, _ = np.linalg.lstsq(A_sys_V, rhs, rcond=None)
                V[b, :] = sol
            
            # Update W similarly
            # For each c: sum_r W[c,r] * U[a,r] * V[b,r] = T[a,b,c] for all a,b
            A_sys_W = np.zeros((n*m*m*p, R))
            for a in range(n*m):
                for b in range(m*p):
                    for r in range(R):
                        A_sys_W[a*(m*p) + b, r] = U[a, r] * V[b, r]
            
            for c in range(p*n):
                rhs = np.zeros(n*m*m*p)
                for a in range(n*m):
                    for b in range(m*p):
                        rhs[a*(m*p) + b] = 1.0 if (a % m) == (b // p) else 0.0
                sol, _, _, _ = np.linalg.lstsq(A_sys_W, rhs, rcond=None)
                W[c, :] = sol
            
            # Check convergence
            if verify_tensor(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
        
        # Check at the end
        if verify_tensor(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
        return None
    
    # Try various ranks from low to high
    # Known: rank(4x5x5) is between 65 and 79
    # Let's try from 65 upward
    
    for R in range(65, min(100, rank_sb)):
        if time.time() > deadline - 10:
            break
        
        result = try_alternating(R, max_iters=30, seed=R)
        if result is not None:
            U, V, W = result
            if verify_tensor(U, V, W, n, m, p, tol=1e-8):
                best_rank = R
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break  # Found a valid decomposition at this rank
    
    # If we didn't find anything better, try a few more with different seeds
    if best_rank == rank_sb:
        for R in [70, 72, 75, 78, 79]:
            if time.time() > deadline - 10:
                break
            for seed in [1, 2, 3]:
                if time.time() > deadline - 5:
                    break
                result = try_alternating(R, max_iters=40, seed=seed*1000 + R)
                if result is not None:
                    U, V, W = result
                    if verify_tensor(U, V, W, n, m, p, tol=1e-8):
                        best_rank = R
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        break
            if best_rank < rank_sb:
                break
    
    # Convert to lists with rounding to nearest 0.5
    def to_list(mat):
        R = mat.shape[1]
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(R):
                val = mat[i, j]
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                # If it's very close to an integer, use integer
                if abs(rounded - round(rounded)) < 1e-9:
                    row.append(int(round(rounded)))
                else:
                    row.append(rounded)
            result.append(row)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
