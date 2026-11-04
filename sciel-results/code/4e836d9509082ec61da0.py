"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Precompute valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                valid_triples.append((a, b, c))
    
    # Precompute all triples and mark which are valid
    all_triples = []
    valid_set = set(valid_triples)
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                all_triples.append((a, b, c))
    
    def verify_decomposition_full(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check all non-zero entries
        for a, b, c in valid_triples:
            val = np.sum(U[a,:] * V[b,:] * W[c,:])
            if abs(val - 1) > tol:
                return False
        # Check all zero entries
        for a, b, c in all_triples:
            if (a, b, c) not in valid_set:
                val = np.sum(U[a,:] * V[b,:] * W[c,:])
                if abs(val) > tol:
                    return False
        return True
    
    def compute_loss(U, V, W):
        """Compute the total squared error over all tensor entries."""
        r = U.shape[1]
        loss = 0.0
        # Non-zero entries
        for a, b, c in valid_triples:
            val = np.sum(U[a,:] * V[b,:] * W[c,:])
            loss += (val - 1)**2
        # Zero entries
        for a, b, c in all_triples:
            if (a, b, c) not in valid_set:
                val = np.sum(U[a,:] * V[b,:] * W[c,:])
                loss += val**2
        return loss
    
    def solve_for_U(V, W, rank):
        """Solve for U given V and W using least squares."""
        # Build linear system: for each valid (a,b,c), sum_d U[a,d]*V[b,d]*W[c,d] = 1
        # For zero entries, sum_d U[a,d]*V[b,d]*W[c,d] = 0
        # This is linear in U
        # Unknowns: U[a,d] for a in range(dim_u), d in range(rank)
        # Total unknowns: dim_u * rank
        
        # Build A and b for the linear system
        # We'll use all equations (both valid and invalid)
        num_eqs = len(all_triples)
        A = np.zeros((num_eqs, dim_u * rank))
        b = np.zeros(num_eqs)
        
        for idx, (a, b_idx, c_idx) in enumerate(all_triples):
            is_valid = (a, b_idx, c_idx) in valid_set
            b[idx] = 1.0 if is_valid else 0.0
            for d in range(rank):
                A[idx, a*rank + d] = V[b_idx, d] * W[c_idx, d]
        
        # Solve least squares
        try:
            U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U = U_flat.reshape(dim_u, rank)
        except:
            return None
        
        return U
    
    def solve_for_V(U, W, rank):
        """Solve for V given U and W using least squares."""
        num_eqs = len(all_triples)
        A = np.zeros((num_eqs, dim_v * rank))
        b = np.zeros(num_eqs)
        
        for idx, (a_idx, b, c_idx) in enumerate(all_triples):
            is_valid = (a_idx, b, c_idx) in valid_set
            b[idx] = 1.0 if is_valid else 0.0
            for d in range(rank):
                A[idx, b*rank + d] = U[a_idx, d] * W[c_idx, d]
        
        try:
            V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(dim_v, rank)
        except:
            return None
        
        return V
    
    def solve_for_W(U, V, rank):
        """Solve for W given U and V using least squares."""
        num_eqs = len(all_triples)
        A = np.zeros((num_eqs, dim_w * rank))
        b = np.zeros(num_eqs)
        
        for idx, (a_idx, b_idx, c) in enumerate(all_triples):
            is_valid = (a_idx, b_idx, c) in valid_set
            b[idx] = 1.0 if is_valid else 0.0
            for d in range(rank):
                A[idx, c*rank + d] = U[a_idx, d] * V[b_idx, d]
        
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape(dim_w, rank)
        except:
            return None
        
        return W
    
    def alternating_minimization(U, V, W, rank, max_iters=100, tol=1e-12):
        """Perform alternating least squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Solve for U
            U_new = solve_for_U(V, W, rank)
            if U_new is not None:
                U = U_new
            
            # Solve for V
            V_new = solve_for_V(U, W, rank)
            if V_new is not None:
                V = V_new
            
            # Solve for W
            W_new = solve_for_W(U, V, rank)
            if W_new is not None:
                W = W_new
            
            # Compute loss
            loss = compute_loss(U, V, W)
            
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        return U, V, W, prev_loss
    
    # Try different ranks from low to high
    # Start with a reasonable guess
    min_rank_to_try = 30
    for rank in range(min_rank_to_try, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            rng = np.random.RandomState(trial * 1000 + rank)
            U = rng.randn(dim_u, rank) * 0.1
            V = rng.randn(dim_v, rank) * 0.1
            W = rng.randn(dim_w, rank) * 0.1
            
            # Run alternating minimization
            U, V, W, loss = alternating_minimization(U, V, W, rank, max_iters=50, tol=1e-14)
            
            # Check if the decomposition is valid
            if loss < 1e-10:
                # Verify fully
                if verify_decomposition_full(U, V, W, tol=1e-10):
                    # Round to half-integers if possible
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify_decomposition_full(U_r, V_r, W_r, tol=1e-10):
                        best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = rank
                        found = True
                    else:
                        # Use the unrounded version
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = rank
                        found = True
                    break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
