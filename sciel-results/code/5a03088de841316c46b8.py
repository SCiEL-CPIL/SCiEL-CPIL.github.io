"""Construct a tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import nnls, minimize, differential_evolution
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    n, m, p = 3, 3, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j, j',k, k',i'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, let's re-derive.
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # We want sum_r U[i,j,r] V[j',k,r] W[k',i',r] ... no.
    # The task says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The reconstruction is:
    # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i'', r]
    # This sum should equal the tensor T.
    # For matrix multiplication, T[i,j, j',k, k',i'] is non-zero only when:
    # i == i', j == j', k == k' ? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # So the "output" index is (i,k). The "input" indices are (i',j') for A and (j'',k'') for B.
    # The tensor T connects (i,k) with (i',j',j'',k'').
    # T[i,k, i',j',j'',k''] = 1 if i==i' and k==k'' and j==j'' else 0.
    # Wait, the factorization is sum_r U[i,j,r] V[j'',k'',r] W[k,i',r]?
    # Let's check the schoolbook in the prompt:
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # Sum over c=(i0,j0,k0):
    # U[i,j,r] is 1 if i==i0, j==j0.
    # V[j',k',r] is 1 if j'==j0, k'==k0.
    # W[k'',i'',r] is 1 if k''==k0, i''==i0.
    # Product is 1 if i==i0, j==j0, j'==j0, k'==k0, k''==k0, i''==i0.
    # This implies i==i'', j==j', k==k''.
    # So the tensor is T[i,k, i'',j',j'',k''] ... wait.
    # The sum is over r. The result is a 6-index tensor?
    # No, the result of the contraction is just the value for specific indices.
    # Let's look at the shape: U (n*m), V (m*p), W (p*n).
    # The "outer product" U[:,r] V[:,r] W[:,r] creates a tensor of shape (n*m, m*p, p*n).
    # This tensor T[a, b, c] where a=i*m+j, b=j'*p+k', c=k''*n+i''.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # We want this T to be the matrix multiplication tensor.
    # The matrix multiplication tensor T_mm[a,b,c] is 1 if:
    # a = i*m+j, b = j'*p+k', c = k''*n+i''
    # AND i == i'' AND j == j' AND k == k''.
    # Wait, C[i,k] depends on A[i,j] and B[j,k].
    # So A index is (i,j) -> a. B index is (j,k) -> b. Output index is (i,k) -> c?
    # No, W is (p*n). Index c = k''*n + i''.
    # So c corresponds to (k'', i'').
    # The condition for non-zero is:
    # i (from a) == i'' (from c)
    # j (from a) == j' (from b)
    # k' (from b) == k'' (from c)
    # So T_mm[a,b,c] = 1 if i==i'', j==j', k'==k''.
    
    # Let's verify schoolbook:
    # r = (i0, j0, k0).
    # U[a,r] = 1 if a = i0*m+j0.
    # V[b,r] = 1 if b = j0*p+k0.
    # W[c,r] = 1 if c = k0*n+i0.
    # Product is 1 if a=i0*m+j0, b=j0*p+k0, c=k0*n+i0.
    # This implies i=i0, j=j0, j'=j0, k'=k0, k''=k0, i''=i0.
    # So i=i'', j=j', k'=k''.
    # Yes, this matches.
    
    # So we need to construct U, V, W such that for all a, b, c:
    # sum_r U[a,r] V[b,r] W[c,r] = T_mm[a,b,c]
    
    # We can flatten T_mm and solve for the decomposition.
    # But rank minimization is hard.
    # We start with a valid high-rank decomposition (schoolbook rank 36) and try to reduce rank.
    # Or we can use the known rank-21 for 3x3x3 and extend.
    
    # Strategy:
    # 1. Start with schoolbook rank 36.
    # 2. Try to find a lower rank decomposition using optimization.
    # 3. Specifically, try rank 22, 23, etc.
    
    # Since the prompt asks for rank-minimal, and we know 3x3x3 is 21, 
    # 3x3x4 is likely 21 + 3 = 24? Or maybe less?
    # Actually, the rank of matrix multiplication tensor M_{m,n,p} satisfies:
    # R(M_{m,n,p}) <= R(M_{m,n,p-1}) + R(M_{m,n,1})
    # R(M_{m,n,1}) = m*n (since C[i,0] = sum_j A[i,j]B[j,0], which is just a dot product for each i,
    # but wait, B is m x 1. So B[j,0] is a vector.
    # C[i,0] = sum_j A[i,j] B[j,0].
    # This is n independent dot products of length m.
    # The tensor for this is sum_i (e_i in output) (A row i) (B column 0).
    # Rank is n * m? No.
    # Let's check rank of M_{m,n,1}.
    # T[i,0, i',j',j'',0'] = 1 if i==i', j==j'', 0==0'.
    # Since p=1, k and k' are always 0.
    # T[i, i',j',j''] = 1 if i==i', j==j''.
    # This is just the identity tensor for the i index and j index.
    # It can be decomposed as sum_{i,j} e_i (out) e_{i,j} (A) e_{j,0} (B)?
    # No, the factors are U (n*1), V (m*1), W (1*n).
    # U[i, r], V[j, r], W[0, i', r].
    # We need sum_r U[i,r] V[j,r] W[0,i',r] = delta_{i,i'} delta_{j,j'}.
    # This is the identity matrix of size n*m? No.
    # It's a tensor that is 1 when i=i' and j=j'.
    # This is exactly the identity tensor I_{n} \otimes I_{m} ?
    # The rank of the identity tensor I_d is d^2? No, rank of I_d is d.
    # Wait, I_d[x,y] = delta_{x,y}. Rank is d.
    # Here we have T[i, i', j, j'] = delta_{i,i'} delta_{j,j'}.
    # This is I_n \otimes I_m.
    # The rank of Kronecker product of tensors is product of ranks.
    # Rank(I_n) = n. Rank(I_m) = m.
    # So Rank(M_{m,n,1}) = n*m.
    # So R(M_{3,3,4}) <= R(M_{3,3,3}) + R(M_{3,3,1}) = 21 + 9 = 30.
    # We can do better?
    # R(M_{m,n,p}) <= R(M_{m,n,p-1}) + n*m.
    # So 3x3x4 <= 21 + 9 = 30.
    # But we can also view it as 3x3x3 + 3x3x1.
    # Is there a better bound?
    # R(M_{m,n,p}) <= R(M_{m,n,1}) * p? No.
    # Actually, for 3x3x4, the known rank is likely 24?
    # Let's check literature.
    # Rank of 3x3x3 is 21.
    # Rank of 3x3x4?
    # If we use the 21 terms for the first 3 columns, and 3 more for the 4th?
    # The 4th column C[:,3] = A B[:,3].
    # This is a 3x1 matrix.
    # To compute C[:,3], we need 9 terms (3x3).
    # So total 21 + 9 = 30.
    # Can we share terms?
    # Probably not easily.
    # However, the prompt's current program has rank 36 (schoolbook).
    # The inspiration 2 tries rank 24.
    # Let's try to construct rank 24.
    # If we can find a valid rank 24, that's an improvement over 36.
    # The score is 1.0 / rank.
    # 1/36 = 0.0277.
    # 1/24 = 0.0416.
    # This is an improvement.
    
    # How to get rank 24?
    # Use rank 21 for 3x3x3.
    # Add 3 terms for the 4th column?
    # If we add 3 terms, we need to compute C[i,3] = sum_j A[i,j] B[j,3] using 3 terms.
    # This requires rank 3 for the 3x1 multiplication.
    # But rank of 3x1 multiplication is 3*3=9.
    # So we can't do it in 3 terms.
    # So rank 24 is likely invalid unless we mix the terms.
    
    # Let's try to find the optimal rank using a solver.
    # We will try ranks from 21 upwards.
    # For each rank, we initialize U, V, W with random values (or schoolbook truncated)
    # and optimize to minimize the error.
    
    # Given time constraints, we should try a few specific ranks.
    # The known lower bound for 3x3x3 is 21.
    # For 3x3x4, the lower bound is likely higher.
    # Let's try rank 24, 25, 26, 27, 28, 29, 30.
    # If we find a valid one, we return it.
    
    # To speed up, we can use the structure of the 3x3x3 part.
    # But for now, let's just run a generic optimizer for a few ranks.
    
    # Deadline guard
    start_time = time.time()
    deadline = 140  # 10s margin
    
    # Target ranks to try
    target_ranks = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    # Generate the target tensor T_mm
    # T_mm[a,b,c]
    n, m, p = 3, 3, 4
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a,b,c] = 1.0
    
    # Flatten T_target
    T_flat = T_target.flatten()
    
    best_rank = 36
    best_U = None
    best_V = None
    best_W = None
    
    # Helper to verify
    def check_rank(rank, U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Reconstruct
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a 3-mode tensor product sum.
        # We can use einsum or loops.
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=1e-8)
    
    # Try schoolbook first to have a fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    if check_rank(n*m*p, U_school, V_school, W_school):
        best_rank = n*m*p
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    # Try lower ranks
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # If rank is >= best_rank, we can skip if we want strictly better, 
        # but we want minimal, so we go up from low.
        # Actually, we iterate from low to high.
        # If we find a valid one, we update best and continue to see if lower?
        # No, we iterate in increasing order. First valid is best.
        
        # Initialize with random small values
        # Or initialize with truncated schoolbook?
        # Random is better for optimization.
        
        found = False
        # Try a few restarts for each rank
        for attempt in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + rank + attempt)
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Objective function
            def objective(params):
                # Unpack
                U_flat = params[:n*m*rank]
                V_flat = params[n*m*rank : n*m*rank + m*p*rank]
                W_flat = params[n*m*rank + m*p*rank :]
                
                U_r = U_flat.reshape(n*m, rank)
                V_r = V_flat.reshape(m*p, rank)
                W_r = W_flat.reshape(p*n, rank)
                
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                error = np.sum((T_recon - T_target)**2)
                return error
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B or similar
            # Since the problem is non-convex, we might get stuck.
            # We'll run for a limited number of iterations.
            try:
                res = minimize(objective, params0, method='L-BFGS-B', 
                               options={'maxiter': 50, 'gtol': 1e-6})
                if res.fun < 1e-6:
                    # Verify
                    U_r = res.x[:n*m*rank].reshape(n*m, rank)
                    V_r = res.x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_r = res.x[n*m*rank + m*p*rank :].reshape(p*n, rank)
                    
                    if check_rank(rank, U_r, V_r, W_r):
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # Since we are iterating in increasing order, the first found is the minimum
            break
    
    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    return _verify_decomposition(0, 0, 0) # Dummy to trigger import? No, just call the logic.
    # Actually, let's just put the logic in construct directly or call a helper.
    # The prompt requires construct() to return the dict.
    pass

# Let's rewrite construct properly to call the search logic.

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit
    
    n, m, p = 3, 3, 4
    
    # Generate the target tensor T_mm
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a,b,c] = 1.0
    
    # Helper to verify
    def check_rank(rank, U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=1e-8)
    
    # Fallback: Schoolbook rank 36
    best_rank = n * m * p
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try lower ranks
    # We know 3x3x3 is 21. 3x3x4 is likely around 24-30.
    # Let's try 22 to 35.
    target_ranks = list(range(22, 36))
    
    # Precompute dimensions for objective
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
            
        found = False
        
        # Define objective for DE (returns scalar error)
        def de_objective(params):
            U_flat = params[:dim_U*rank]
            V_flat = params[dim_U*rank : dim_U*rank + dim_V*rank]
            W_flat = params[dim_U*rank + dim_V*rank :]
            
            U_r = U_flat.reshape(dim_U, rank)
            V_r = V_flat.reshape(dim_V, rank)
            W_r = W_flat.reshape(dim_W, rank)
            
            T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            error = np.sum((T_recon - T_target)**2)
            return error
        
        # Use Differential Evolution for global search
        # Bounds: [-10, 10] is usually sufficient for normalized tensors
        bounds = [(-10, 10)] * (dim_U + dim_V + dim_W) * rank
        
        try:
            # DE is expensive, so limit iterations and population
            # maxiter=100, popsize=15 (default is 15)
            # tol=1e-6
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=50, 
                popsize=10,
                tol=1e-6,
                seed=rank,
                polish=False # We will polish manually
            )
            
            if res_de.fun < 1e-4: # Good candidate found by global search
                # Polish with L-BFGS-B
                params0 = res_de.x
                def polish_objective(params):
                    U_flat = params[:dim_U*rank]
                    V_flat = params[dim_U*rank : dim_U*rank + dim_V*rank]
                    W_flat = params[dim_U*rank + dim_V*rank :]
                    
                    U_r = U_flat.reshape(dim_U, rank)
                    V_r = V_flat.reshape(dim_V, rank)
                    W_r = W_flat.reshape(dim_W, rank)
                    
                    T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    error = np.sum((T_recon - T_target)**2)
                    return error
                
                res_polish = minimize(polish_objective, params0, method='L-BFGS-B', 
                                       options={'maxiter': 200, 'gtol': 1e-12})
                
                if res_polish.fun < 1e-8:
                    U_r = res_polish.x[:dim_U*rank].reshape(dim_U, rank)
                    V_r = res_polish.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_r = res_polish.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                    
                    if check_rank(rank, U_r, V_r, W_r):
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
        except Exception:
            pass
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
