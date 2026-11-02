import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if it corresponds to (i, j, k) in multiplication
    # T is of shape (n, m, p)
    # Actually, the tensor T for matrix multiplication C = A B is defined such that
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has indices (i, j, k) and T[i, j, k] = 1.
    # The decomposition is T = sum_c U[:, c] \otimes V[:, c] \otimes W[:, c]
    # where U is (n*m, R), V is (m*p, R), W is (p*n, R) ?
    # Wait, the interface says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The trace convention is W[k*n + i, r].
    # Let's check the indices.
    # Standard tensor for matmul: T_{i, j, k}.
    # U indices: i, j -> index i*m + j.
    # V indices: j, k -> index j*p + k.
    # W indices: k, i -> index k*n + i.
    # So T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    # Create the target tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Flatten T to match the product of U, V, W columns
    # We want to solve for U, V, W such that
    # sum_c U[:, c] \otimes V[:, c] \otimes W[:, c] = T
    # This is equivalent to:
    # For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We will search for rank R
    # Start with a low rank and increase if not found
    # Theoretical lower bound is max(n*m, m*p, n*p) = 20
    # Schoolbook is 60
    # We try ranks from 20 to 60
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Pre-compute indices
    idx_U = np.arange(n*m)
    idx_V = np.arange(m*p)
    idx_W = np.arange(p*n)
    
    # Map (i, j, k) to indices in U, V, W
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    # We define the cost function for least squares
    def cost_func(params, R):
        # params is a flat array of length R * (n*m + m*p + p*n)
        # Split into U, V, W
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = size_U + size_V + size_W
        
        U = params[:R*size_U].reshape(R, size_U).T # (size_U, R)
        V = params[R*size_U:(R+1)*size_U].reshape(R, size_V).T # (size_V, R)
        W = params[(R+1)*size_U:(R+2)*size_U].reshape(R, size_W).T # (size_W, R)
        
        # Reconstruct T
        # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can vectorize this
        # Create a tensor of shape (n, m, p, R)
        # U_part[i, j, c] = U[i*m+j, c]
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        
        # Sum over c
        # T_recon[i, j, k] = sum_c U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
        # This is a bit tricky to vectorize directly without loops over i, j, k
        # But n, m, p are small (3, 4, 5)
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = np.dot(U_t[i, j, :], V_t[j, k, :]) * 0 # No, it's element-wise product then sum
                    # T_recon[i, j, k] = sum_c U_t[i, j, c] * V_t[j, k, c] * W_t[k, i, c]
                    # This is np.einsum('c,c,c->', U_t[i,j,:], V_t[j,k,:], W_t[k,i,:])
                    T_recon[i, j, k] = np.einsum('c,c,c->', U_t[i, j, :], V_t[j, k, :], W_t[k, i, :])
        
        # Residual
        residual = T_recon - T
        return residual.flatten()

    # Try ranks
    # We prioritize lower ranks
    ranks_to_try = list(range(20, 61))
    # Shuffle slightly to mix? No, lower is better.
    # But maybe 30 is easier to find than 20.
    # We will try a few random restarts for each rank.
    
    # Optimization: Use a simpler cost if possible?
    # The cost function is expensive due to the loop.
    # We can pre-compute the indices for the einsum.
    
    # Let's refine the cost function for speed
    # Pre-compute the triplets (u_idx, v_idx, w_idx) for all (i, j, k)
    triplets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triplets.append((i*m+j, j*p+k, k*n+i))
    triplets = np.array(triplets)
    u_indices = triplets[:, 0]
    v_indices = triplets[:, 1]
    w_indices = triplets[:, 2]
    
    def fast_cost_func(params, R):
        size_U = n * m
        size_V = m * p
        size_W = p * p # Wait, W is p*n
        size_W = p * n
        total_size = size_U + size_V + size_W
        
        U = params[:R*size_U].reshape(R, size_U).T
        V = params[R*size_U:(R+1)*size_U].reshape(R, size_V).T
        W = params[(R+1)*size_U:(R+2)*size_U].reshape(R, size_W).T
        
        # Extract rows
        U_rows = U[u_indices, :] # (n*m*p, R)
        V_rows = V[v_indices, :] # (n*m*p, R)
        W_rows = W[w_indices, :] # (n*m*p, R)
        
        # Element-wise product and sum over R
        # T_recon_flat = sum_c U_rows[:, c] * V_rows[:, c] * W_rows[:, c]
        T_recon_flat = np.einsum('ic,ic,ic->i', U_rows, V_rows, W_rows)
        
        # T_flat is all 1s
        residual = T_recon_flat - 1.0
        return residual

    # We will use scipy.optimize.least_squares
    from scipy.optimize import least_squares
    
    found = False
    # Try ranks from low to high
    # But lower ranks might be harder to find.
    # We will try a few ranks in parallel or sequentially with time checks.
    
    # Strategy:
    # 1. Try rank 30 with 5 restarts.
    # 2. If fail, try rank 40 with 5 restarts.
    # 3. If fail, try rank 50 with 5 restarts.
    # 4. If fail, return schoolbook (rank 60).
    
    # Actually, let's just iterate ranks and use remaining time.
    # We'll try ranks: 25, 30, 35, 40, 45, 50, 55, 60
    # And for each, a few restarts.
    
    rank_list = [25, 30, 35, 40, 45, 50, 55, 60]
    
    for R in rank_list:
        if time.time() > deadline - 5:
            break
            
        if R >= best_rank:
            # If we already found a better one, we can skip higher ones?
            # No, we want the *best* (lowest) rank.
            # So we should try lower ranks first.
            # The list is sorted ascending.
            pass
            
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_size = size_U + size_V + size_W
        
        # Try a few random initializations
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale: values should be around 1/sqrt(R) or similar?
            # For rank 1, values are 1. For rank R, maybe 1?
            # Let's try normal distribution with std 1.0
            x0 = np.random.randn(R * total_size)
            
            # Solve
            try:
                res = least_squares(fast_cost_func, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception as e:
                continue
                
            if res.success or res.cost < 1e-8:
                # Check if we can round to integers/half-integers
                params = res.x
                U = params[:R*size_U].reshape(R, size_U).T
                V = params[R*size_U:(R+1)*size_U].reshape(R, size_V).T
                W = params[(R+1)*size_U:(R+2)*size_U].reshape(R, size_W).T
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                # Recalculate with rounded values
                U_rows_r = U_rounded[u_indices, :]
                V_rows_r = V_rounded[v_indices, :]
                W_rows_r = W_rounded[w_indices, :]
                T_recon_r = np.einsum('ic,ic,ic->i', U_rows_r, V_rows_r, W_rows_r)
                
                if np.allclose(T_recon_r, 1.0, atol=1e-9):
                    # Found a valid exact decomposition
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    # Ensure integer/half-integer representation
                    # The task asks for integer/half-integer entries.
                    # Floats like 1.5 are fine.
                    
                    # We found a rank R decomposition.
                    # Since we are iterating from low to high, this is the best so far.
                    # We can break and return this.
                    # But maybe we can find an even lower one?
                    # We are iterating 25, 30... so if we find 25, it's likely the best.
                    # Let's return it immediately to save time?
                    # Or continue to see if we can find lower?
                    # We started at 25. If we find 25, we can't go lower in this list.
                    # But maybe rank 20 is possible?
                    # We didn't try 20.
                    # Let's add 20 to the list.
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # If not exact, but close, we might store it as a candidate?
            # But the task requires exact.
            # So we discard it.
            
    # If no low rank found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Pre-compute indices for the full tensor
    triplets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triplets.append((i*m+j, j*p+k, k*n+i))
    triplets = np.array(triplets)
    u_indices = triplets[:, 0]
    v_indices = triplets[:, 1]
    w_indices = triplets[:, 2]
    
    size_U = n * m
    size_V = m * p
    size_W = p * n
    total_size = size_U + size_V + size_W
    
    def fast_cost_func(params, R):
        U = params[:R*size_U].reshape(R, size_U).T
        V = params[R*size_U:(R+1)*size_U].reshape(R, size_V).T
        W = params[(R+1)*size_U:(R+2)*size_U].reshape(R, size_W).T
        
        U_rows = U[u_indices, :]
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        
        T_recon_flat = np.einsum('ic,ic,ic->i', U_rows, V_rows, W_rows)
        
        residual = T_recon_flat - 1.0
        return residual

    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        U_rows = U[u_indices, :]
        V_rows = V[v_indices, :]
        W_rows = W[w_indices, :]
        T_recon = np.einsum('ic,ic,ic->i', U_rows, V_rows, W_rows)
        return np.allclose(T_recon, 1.0, atol=1e-9)
    
    def round_to_half_integer(mat):
        """Round matrix entries to nearest half-integer."""
        return np.round(mat * 2) / 2
    
    # Strategy 1: Divide and conquer by splitting the middle dimension
    # Split m=4 into m1=2, m2=2
    # This gives us two sub-problems: (n, m1, p) and (n, m2, p)
    # But we need to combine them carefully
    
    # Actually, let's try a different approach:
    # Use the fact that for small matrices, we can try known low-rank constructions
    
    # First, try to find a low-rank solution using optimization
    from scipy.optimize import least_squares
    
    # Try ranks from low to high
    rank_list = [20, 22, 24, 25, 28, 30, 32, 35, 40, 45, 50, 55, 60]
    
    for R in rank_list:
        if time.time() > deadline - 15:
            break
            
        # Multiple restarts for this rank
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            # Different initialization strategies
            if restart % 3 == 0:
                # Random normal
                x0 = np.random.randn(R * total_size) * 0.5
            elif restart % 3 == 1:
                # Random uniform
                x0 = np.random.uniform(-1, 1, R * total_size)
            else:
                # Sparse initialization
                x0 = np.zeros(R * total_size)
                non_zero_idx = np.random.choice(R * total_size, size=R * total_size // 3, replace=False)
                x0[non_zero_idx] = np.random.randn(len(non_zero_idx))
            
            try:
                res = least_squares(fast_cost_func, x0, args=(R,), method='lm', max_nfev=3000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            except Exception as e:
                continue
                
            if res.cost < 1e-8:
                params = res.x
                U = params[:R*size_U].reshape(R, size_U).T
                V = params[R*size_U:(R+1)*size_U].reshape(R, size_V).T
                W = params[(R+1)*size_U:(R+2)*size_U].reshape(R, size_W).T
                
                # Try rounding to half-integers
                U_rounded = round_to_half_integer(U)
                V_rounded = round_to_half_integer(V)
                W_rounded = round_to_half_integer(W)
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                
                # If rounding doesn't work, try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if verify_decomposition(U_int, V_int, W_int):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # If neither works, check if the unrounded solution is close enough
                # and try a local search around it
                if verify_decomposition(U, V, W):
                    # Try to find a nearby half-integer solution
                    for attempt in range(10):
                        if time.time() > deadline - 15:
                            break
                        # Perturb and round
                        U_pert = U + np.random.randn(*U.shape) * 0.1
                        V_pert = V + np.random.randn(*V.shape) * 0.1
                        W_pert = W + np.random.randn(*W.shape) * 0.1
                        
                        U_pert_rounded = round_to_half_integer(U_pert)
                        V_pert_rounded = round_to_half_integer(V_pert)
                        W_pert_rounded = round_to_half_integer(W_pert)
                        
                        if verify_decomposition(U_pert_rounded, V_pert_rounded, W_pert_rounded):
                            return {"U": U_pert_rounded.tolist(), "V": V_pert_rounded.tolist(), "W": W_pert_rounded.tolist()}
            
    # Strategy 2: Constructive approach using tensor rank properties
    # For 3x4x5 matmul, try to build a decomposition from smaller blocks
    
    # Try a structured initialization based on Kronecker products
    # This might find lower rank solutions
    
    for R in [24, 28, 30, 32]:
        if time.time() > deadline - 20:
            break
            
        # Create structured initial guess
        # Use block structure: split dimensions into smaller parts
        # n=3: [1,2], m=4: [2,2], p=5: [2,3]
        
        # Try a rank-R initialization with some structure
        U_struct = np.zeros((size_U, R))
        V_struct = np.zeros((size_V, R))
        W_struct = np.zeros((size_W, R))
        
        # Assign random values but with some pattern
        for c in range(R):
            # Each column gets a "pattern"
            pattern = c % 4
            if pattern == 0:
                U_struct[::4, c] = 1.0
                V_struct[::4, c] = 1.0
                W_struct[::4, c] = 1.0
            elif pattern == 1:
                U_struct[1::4, c] = 1.0
                V_struct[1::4, c] = 1.0
                W_struct[1::4, c] = 1.0
            elif pattern == 2:
                U_struct[2::4, c] = 1.0
                V_struct[2::4, c] = 1.0
                W_struct[2::4, c] = 1.0
            else:
                U_struct[3::4, c] = 1.0
                V_struct[3::4, c] = 1.0
                W_struct[3::4, c] = 1.0
        
        # Add random perturbation
        U_struct += np.random.randn(*U_struct.shape) * 0.3
        V_struct += np.random.randn(*V_struct.shape) * 0.3
        W_struct += np.random.randn(*W_struct.shape) * 0.3
        
        params = np.concatenate([U_struct.flatten(), V_struct.flatten(), W_struct.flatten()])
        
        try:
            res = least_squares(fast_cost_func, params, args=(R,), method='lm', max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        except Exception as e:
            continue
        
        if res.cost < 1e-8:
            params = res.x
            U = params[:R*size_U].reshape(R, size_U).T
            V = params[R*size_U:(R+1)*size_U].reshape(R, size_V).T
            W = params[(R+1)*size_U:(R+2)*size_U].reshape(R, size_W).T
            
            U_rounded = round_to_half_integer(U)
            V_rounded = round_to_half_integer(V)
            W_rounded = round_to_half_integer(W)
            
            if verify_decomposition(U_rounded, V_rounded, W_rounded):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
