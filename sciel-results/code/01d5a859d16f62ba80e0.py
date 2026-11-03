"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def verify(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal: for all i,j,j2,k,k2,i3 where not (j==j2 and k==k2 and i==i3)
        # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i3,c] == 0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i3][c]
                                if abs(s) > 1e-9:
                                    return False
        return True

    def schoolbook_decomp(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
        return U, V, W, rank

    def try_rank(r):
        """
        Try to find a rank-r decomposition using structured parameterization.
        For n=2, we exploit that the two rows of the output matrix share the same
        input columns, so we can use a symmetric structure.
        
        We parameterize as:
        - For each (j, k) in [0..m) x [0..p), we have 2 components (one per output row)
        - Actually, let's try a different approach: use the fact that n=2 is small.
        
        Strategy: For n=2, the tensor has a special structure. The multiplication
        C = A @ B where A is 2x5 and B is 5x5.
        C[0,:] = A[0,:] @ B, C[1,:] = A[1,:] @ B.
        
        We can try to find rank r such that r < 50.
        A known technique for small n: use the fact that we can share components.
        
        Let's try a parameterization where for each j (column of A, row of B),
        we have p components that handle both output rows.
        
        Actually, for n=2, a simple improvement: since the two output rows
        multiply the same B matrix, we can potentially save factors.
        
        Let's try a direct optimization for small ranks using random init.
        """
        if time.time() > deadline - 10:
            return None

        # Try to find a decomposition with rank r using random initialization
        # and coordinate descent / gradient steps.
        
        # Initialize with random values
        np.random.seed(42)
        U_init = np.random.randn(n * m, r) * 0.5
        V_init = np.random.randn(m * p, r) * 0.5
        W_init = np.random.randn(p * n, r) * 0.5
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            r_val = r
            U_x = x[:n * m * r_val].reshape(n * m, r_val)
            V_x = x[n * m * r_val:(n * m + m * p) * r_val].reshape(m * p, r_val)
            W_x = x[(n * m + m * p) * r_val:].reshape(p * n, r_val)
            
            # Compute the tensor decomposition error
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We want T[i,j,k] = delta(i,j,k) for the multiplication tensor
            # Actually, the multiplication tensor has:
            # T[i, j, k] = 1 if this corresponds to a valid multiplication, else 0
            
            # For matrix multiplication C = A @ B:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
            # Wait, that's not right. The multiplication tensor M_{n,m,p} has:
            # M[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
            # and the decomposition is: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
            # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
            
            # So we want: for all i, j, k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # But that's only for the "diagonal" entries. The off-diagonal entries
            # (where the indices don't match the multiplication pattern) should be 0.
            
            # Actually, the tensor is defined as:
            # T[i, j, k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
            # No wait. The matrix multiplication tensor is:
            # T in R^{n x m x p} where T[i,j,k] = 1 if the entry (i,k) of the product
            # depends on A[i,j] * B[j,k]. So T[i,j,k] = 1 for all i,j,k in their ranges.
            
            # The decomposition satisfies:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # for ALL i, j, k. But T is only 1 for valid indices, and the tensor
            # is defined on the full grid [0,n) x [0,m) x [0,p).
            
            # Wait, I need to re-read the task. The tensor for matrix multiplication
            # M_{n,m,p} has entries:
            # M[i, j, k] = 1 for all 0 <= i < n, 0 <= j < m, 0 <= k < p
            # and the decomposition is:
            # M[i, j, k] = sum_{c=1}^{r} U[i, j, c] * V[j, k, c] * W[k, i, c]
            
            # But that would mean the tensor is all ones, which doesn't make sense
            # for matrix multiplication. Let me reconsider.
            
            # The correct formulation: The multiplication tensor M_{n,m,p} is in
            # R^{n*m} x R^{m*p} x R^{p*n} and has entries:
            # M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
            # and 0 otherwise.
            
            # So the decomposition is:
            # For all (a, b, c) in [0,n*m) x [0,m*p) x [0,p*n):
            # M[a,b,c] = sum_{t=1}^{r} U[a,t] * V[b,t] * W[c,t]
            # where M[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k,
            # and 0 otherwise.
            
            # So we need to compute the error as:
            # error = sum over all (a,b,c) of (M[a,b,c] - sum_t U[a,t]*V[b,t]*W[c,t])^2
            
            # But that's expensive for large dimensions. Let's compute it efficiently.
            
            # Build the target tensor implicitly.
            # For each (i,j,k), the target is 1 at position (i*m+j, j*p+k, k*n+i).
            
            # Compute the decomposed tensor and compare.
            # This is O(r * n * m * p) which for r=40, n=2, m=5, p=5 is 40*50=2000, fine.
            
            # Actually, let's compute the error only on the support of the tensor
            # (where M is 1) and check that off-support entries are close to 0.
            
            # For the support: for each (i,j,k), compute sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
            # and check it's close to 1.
            # For off-support: this is harder. But if we minimize the error on the support
            # and add a penalty for the off-support, we can do it.
            
            # Let's just compute the full error on the support for now, and verify separately.
            
            # Compute sum on support
            s = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_x[a] * V_x[b] * W_x[c])
                        s += (val - 1.0) ** 2
            
            # We also need to penalize off-support entries. But that's expensive.
            # For now, just use the support error and verify at the end.
            return s

        # Use random restarts
        best_x = None
        best_obj = float('inf')
        
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + restart)
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                      'adaptive': True})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        
        if best_x is None:
            return None
        
        U_sol = best_x[:n * m * r].reshape(n * m, r)
        V_sol = best_x[n * m * r:(n * m + m * p) * r].reshape(m * p, r)
        W_sol = best_x[(n * m + m * p) * r:].reshape(p * n, r)
        
        # Convert to lists of lists
        U_list = U_sol.tolist()
        V_list = V_sol.tolist()
        W_list = W_sol.tolist()
        
        # Verify
        if verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list, r
        
        # Try a polish step
        # Snap near-integer values
        U_sol2 = np.round(U_sol * 2) / 2
        V_sol2 = np.round(V_sol * 2) / 2
        W_sol2 = np.round(W_sol * 2) / 2
        
        U_list2 = U_sol2.tolist()
        V_list2 = V_sol2.tolist()
        W_list2 = W_sol2.tolist()
        
        if verify(U_list2, V_list2, W_list2, n, m, p):
            return U_list2, V_list2, W_list2, r
        
        return None

    # Try ranks from 25 up to 49 (below schoolbook)
    for r in range(25, 50):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook
    U, V, W, rank = schoolbook_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
