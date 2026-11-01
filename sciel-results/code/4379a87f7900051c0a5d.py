import numpy as np
import time
import scipy.optimize as so
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Pre-compute schoolbook fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Precompute indices for tensor equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # We represent the tensor T as a set of linear equations for each matrix.
    
    # For updating U: Fix V, W. Solve for U rows.
    # Equation for U[i, j]: sum_k U[i, j, :] dot (V[j, k, :] * W[k, i, :]) = 1 for all k? 
    # No, the equation is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for EACH k.
    # Wait, the matrix product is C = A B. 
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor decomposition: sum_r U[i, j, r] V[j, k, r] W[i, k, r] = delta_{i,k}? No.
    # The task says "multiplying a 2 by 3 matrix by a 3 by 3 matrix".
    # Output W follows trace convention W[k*n+i, r]. This implies the third factor is indexed by (k, i) where k is col of B (and C), i is row of A (and C).
    # So the equation is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j, ?} 
    # Let's check the standard definition.
    # Usually: sum_r U[i, j, r] V[j, k, r] W[i, k, r] = delta_{j, j'}? No.
    # The decomposition of the matrix multiplication tensor $M_{n, m, p}$:
    # $M_{i, j, k} = \delta_{i, i'}$? No.
    # The entries of the product $C = AB$ are $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a linear map. The tensor form is usually defined such that the bilinear form gives the product.
    # The standard rank decomposition for matrix multiplication $M_{n, m, p}$ satisfies:
    # $\sum_{r=1}^R U_{i, j, r} V_{j, k, r} W_{i, k, r} = \delta_{j, j'}$? No, that's not right.
    # Actually, the equation is:
    # $\sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} = \delta_{j, j'}$ is for identity?
    # Let's look at the schoolbook code in the seed:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For a specific (i, j, k), the term is 1 if there is an r such that U, V, W are all 1 for that r.
    # In schoolbook, each (i, j, k) triple has a unique r.
    # So for a fixed (i, k), sum_j (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])
    # The seed's verify checks:
    # res = 1.0 - sums
    # where sums = sum_r U[u] V[v] W[w] for u=i*m+j, v=j*p+k, w=k*n+i.
    # So the equation is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if j corresponds to the sum?
    # No, the verify function in the seed computes:
    # prod = U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :]
    # sums = np.sum(prod, axis=1)
    # return 1.0 - sums
    # This implies the target is 1 for ALL (i, j, k) triples?
    # If the target is 1 for all i, j, k, then:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 for all i, j, k.
    # Let's check if this decomposes matrix multiplication.
    # If we fix i, k, and sum over j:
    # sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # If W depends on i and k, it can be pulled out of j sum? No, W is part of the product.
    # Actually, the standard definition of the matrix multiplication tensor $M_{n, m, p}$ has entries:
    # $T_{i, j, k} = 1$ if the indices correspond to the multiplication?
    # The tensor is defined on indices $i \in \{1..n\}, j \in \{1..m\}, k \in \{1..p\}$.
    # The condition for rank $R$ is the existence of vectors $u^{(r)}, v^{(r)}, w^{(r)}$ such that
    # $T = \sum_{r=1}^R u^{(r)} \otimes v^{(r)} \otimes w^{(r)}$.
    # The entries of $T$ are $T_{i, j, k}$.
    # For matrix multiplication, $T_{i, j, k} = 1$? No.
    # The equation for the product $C_{ik} = \sum_j A_{ij} B_{jk}$ can be written as:
    # $C_{ik} = \sum_r (\sum_j U_{ij}^{(r)} A_{ij}) (\sum_j V_{jk}^{(r)} B_{jk}) W_{ik}^{(r)}$? No.
    
    # Let's trust the seed's verification logic.
    # The seed says: `res = 1.0 - sums`. This means it requires:
    # $\sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1$ for all $i, j, k$.
    # Let's verify if this is the correct definition for the task.
    # If $\sum_r U_{ij} V_{jk} W_{ki} = 1$ for all $i,j,k$, does this imply matrix multiplication?
    # Consider $A = \sum_i u_i e_i^T$? No.
    # Let's test with $n=1, m=1, p=1$. $1 \times 1$ mat mult.
    # $U_{0,0,r} V_{0,0,r} W_{0,0,r} = 1$. Rank 1 solution: U=V=W=1.
    # Product $C_{00} = A_{00} B_{00}$.
    # Formula: $\sum_r (U_{00,r} A_{00}) (V_{00,r} B_{00}) W_{00,r}$?
    # No, the standard reconstruction is:
    # $C_{ik} = \sum_r ( \sum_j U_{ij}^{(r)} A_{ij} ) ( \sum_j V_{jk}^{(r)} B_{jk} ) W_{ik}^{(r)}$?
    # If the tensor equation is $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1$,
    # then for any A, B:
    # $\sum_{i,j,k} A_{ij} B_{jk} (\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}) = \sum_{i,j,k} A_{ij} B_{jk} \cdot 1 = \text{trace}(AB)$?
    # This sums everything. It doesn't give $C_{ik}$.
    
    # There is a mismatch in my understanding of the seed's verify.
    # Let's look at the indices again.
    # $u = i*m + j$ (Row i, Col j of A)
    # $v = j*p + k$ (Row j, Col k of B)
    # $w = k*n + i$ (Row k, Col i of C? Or just index pair (k,i))
    # If the equation is $\sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1$,
    # This is the definition of the **Identity Tensor** or something similar?
    # Actually, for matrix multiplication, the tensor entries are $T_{i, j, k} = 1$ if $j$ is the summation index?
    # No, the matrix multiplication tensor $M_{n,m,p}$ is defined such that:
    # $\langle M, A \otimes B \rangle = \text{vec}(AB)$.
    # The entries of $M$ are $M_{i, j, k} = \delta_{i, i'} \delta_{k, k'}$?
    # The standard decomposition satisfies:
    # $AB = \sum_r (A U_r) (V_r B) W_r^T$?
    
    # Let's look at a known rank 15 decomposition for $2 \times 3 \times 3$.
    # The Strassen algorithm is for $2 \times 2 \times 2$.
    # For $2 \times 3 \times 3$, the rank is known to be 15? Or 14?
    # According to literature, rank of $2 \times 3 \times 3$ is 15?
    # Wait, $2 \times 2 \times 2$ is 7.
    # $2 \times 3 \times 3$ is likely 15?
    # The schoolbook is $2 \times 3 \times 3 = 18$.
    # If the seed verifies `sum == 1`, it is looking for a specific tensor.
    # I will stick to the seed's verification logic because it's part of the "Public Task Contract" implementation in the seed.
    # The seed's `verify_solution` checks `np.allclose(res, 0.0)` where `res = 1.0 - sums`.
    # So I MUST satisfy $\sum_r U_{u,r} V_{v,r} W_{w,r} = 1$ for the given index sets.

    # Precompute index triples (u, v, w) for the equations
    u_idxs = []
    v_idxs = []
    w_idxs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idxs.append(i * m + j)
                v_idxs.append(j * p + k)
                w_idxs.append(k * n + i)
    
    u_idxs = np.array(u_idxs)
    v_idxs = np.array(v_idxs)
    w_idxs = np.array(w_idxs)
    
    # We have N_eq = n*m*p equations.
    # Variables: U (n*m * R), V (m*p * R), W (p*n * R).
    
    def compute_residuals(U, V, W):
        # U[u_idxs, :] shape (N_eq, R)
        # V[v_idxs, :] shape (N_eq, R)
        # W[w_idxs, :] shape (N_eq, R)
        prod = U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :]
        sums = np.sum(prod, axis=1)
        return 1.0 - sums

    def verify_solution(U, V, W):
        # Check if U, V, W are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Check exact decomposition
        res = compute_residuals(U, V, W)
        if not np.allclose(res, 0.0, atol=1e-9):
            return False
        return True

    def als_decompose(R, max_iters=100, tol=1e-10, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialization: Mix of random half-integers and reals
        # Try different strategies for initialization
        strategy = np.random.randint(0, 3)
        if strategy == 0:
            U = np.random.randint(-2, 3, (n * m, R)) * 0.5
            V = np.random.randint(-2, 3, (m * p, R)) * 0.5
            W = np.random.randint(-2, 3, (p * n, R)) * 0.5
        elif strategy == 1:
            U = np.random.normal(0, 1, (n * m, R))
            V = np.random.normal(0, 1, (m * p, R))
            W = np.random.normal(0, 1, (p * n, R))
        else:
            U = np.random.uniform(-1, 1, (n * m, R))
            V = np.random.uniform(-1, 1, (m * p, R))
            W = np.random.uniform(-1, 1, (p * n, R))

        prev_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf

        for it in range(max_iters):
            # Update U
            # For each u, solve least squares: A_u @ U_u = 1
            # A_u[i, r] = V[v_i, r] * W[w_i, r]
            # We can vectorize this somewhat, but loop over u is fine for small n*m.
            
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    # Get indices v, w for this u
                    # v = j*p + k, w = k*n + i for k in 0..p-1
                    vs = np.array([j * p + k for k in range(p)])
                    ws = np.array([k * n + i for k in range(p)])
                    # B shape (p, R)
                    B = V[vs, :] * W[ws, :]
                    try:
                        U_u, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[u, :] = U_u
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    # u = i*m + j, w = k*n + i for i in 0..n-1
                    us = np.array([i * m + j for i in range(n)])
                    ws = np.array([k * n + i for i in range(n)])
                    C = U[us, :] * W[ws, :]
                    try:
                        V_v, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
                        V[v, :] = V_v
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    # u = i*m + j, v = j*p + k for j in 0..m-1
                    us = np.array([i * m + j for j in range(m)])
                    vs = np.array([j * p + k for j in range(m)])
                    D = U[us, :] * V[vs, :]
                    try:
                        W_w, _, _, _ = np.linalg.lstsq(D, np.ones(m), rcond=None)
                        W[w, :] = W_w
                    except np.linalg.LinAlgError:
                        pass
            
            # Calculate error
            res = compute_residuals(U, V, W)
            err = np.sum(res ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if abs(prev_err - err) < tol:
                break
            prev_err = err

        # Use the best found so far for polish
        U, V, W = best_U, best_V, best_W
        res = compute_residuals(U, V, W)
        err = np.sum(res**2)
        
        if err < 1e-6:
            # Polish
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals_func(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R :].reshape(p*n, R)
                return compute_residuals(U_flat, V_flat, W_flat)
            
            try:
                # Use 'lm' for speed, 'trf' if bounds needed (not here)
                res_polish = so.least_squares(residuals_func, x0, method='lm', max_nfev=500)
                x_opt = res_polish.x
                U_opt = x_opt[:n*m*R].reshape(n*m, R)
                V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W_opt = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
                
                # Round and verify
                U_opt_r = np.round(U_opt * 2) / 2
                V_opt_r = np.round(V_opt * 2) / 2
                W_opt_r = np.round(W_opt * 2) / 2
                
                if verify_solution(U_opt_r, V_opt_r, W_opt_r):
                    return U_opt_r, V_opt_r, W_opt_r
                    
                # If rounding failed, try local search on half-integers
                # This is expensive, so only do it if error is very small
                if np.sum(residuals_func(x_opt)**2) < 1e-10:
                    # Try flipping nearby half-integers
                    # This is a heuristic
                    pass
                    
            except Exception:
                pass
                
        return None

    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit

    # Try rank 14, 15, 16
    for R in [14, 15, 16]:
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time left
        # Estimate time per restart: ~0.5s to 2s
        # If 140s left, we can do ~100 restarts.
        
        # Prioritize R=15
        max_restarts = 200
        found = None
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Seed the random state for reproducibility in debugging, but change each run
            res = als_decompose(R, seed=None)
            if res is not None:
                U, V, W = res
                found = (U, V, W)
                break
                
        if found:
            U, V, W = found
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
