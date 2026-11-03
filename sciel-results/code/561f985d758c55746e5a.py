"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Shapes
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    # Build the target tensor structure
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    triple_set = set(triples)
    
    # Schoolbook solution (rank = n*m*p = 144)
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_decomposition_exact(U, V, W, tol=1e-6):
        """Check if the decomposition is exact for all entries using vectorized ops where possible."""
        r = U.shape[1]
        # We need to check T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to checking that the reconstructed tensor matches T.
        # Reconstructing the full tensor is 16*36*36 = 20736 elements.
        # We can do this efficiently.
        
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Let's compute this in chunks to avoid huge memory if R is large, but R <= 144 is small.
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        try:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        except:
            return False
        
        # Create target tensor
        T_target = np.zeros((U_rows, V_rows, W_rows))
        for (a, b, c) in triples:
            T_target[a, b, c] = 1.0
        
        # Check difference
        diff = np.abs(T_recon - T_target)
        if np.max(diff) > tol:
            return False
        return True

    def check_decomposition_fast(U, V, W, tol=1e-4):
        """Faster check using sampling for intermediate steps."""
        r = U.shape[1]
        # Check all triples
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Sample non-triples
        rng = np.random.RandomState(42)
        num_checks = 2000
        for _ in range(num_checks):
            a = rng.randint(0, U_rows)
            b = rng.randint(0, V_rows)
            c = rng.randint(0, W_rows)
            if (a, b, c) in triple_set:
                continue
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val) > tol:
                return False
        return True

    # Vectorized ALS
    def als_decompose(target_rank, max_iters=200, tol=1e-8, seed=42):
        """Alternating Least Squares for tensor decomposition with vectorized updates."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        # Initialize
        # Strategy: Random initialization is often bad. 
        # Let's try initializing with random small values.
        U = rng.randn(U_rows, target_rank) * 0.01
        V = rng.randn(V_rows, target_rank) * 0.01
        W = rng.randn(W_rows, target_rank) * 0.01
        
        # Precompute indices for target tensor construction if needed, 
        # but we can use the triple set.
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            # We want to solve for U[a, :] such that sum_r U[a, r] V[b, r] W[c, r] approx T[a, b, c]
            # For each a, the design matrix B_a has rows r, columns (b, c)
            # B_a[r, b, c] = V[b, r] W[c, r]
            # We can form B for all a at once? No, B depends on V, W but not U.
            # So B is the same for all a in terms of structure, but the target vector changes.
            # Let's construct B once per iteration.
            # B is shape (target_rank, V_rows * W_rows)
            # B[r, b*W_rows + c] = V[b, r] * W[c, r]
            
            # Efficient construction of B
            # V is (V_rows, R), W is (W_rows, R)
            # We need outer product of V[:, r] and W[:, r] for each r.
            # This is a tensor of shape (R, V_rows, W_rows).
            # B_tensor = V[:, :, None] * W[None, :, :] -> (V_rows, R, W_rows)
            # We want B[r, b, c] = V[b, r] * W[c, r]
            # So B_tensor = np.einsum('br,cr->bcr', V, W) -> (V_rows, W_rows, R)
            # Then transpose to (R, V_rows, W_rows) and flatten last two dims.
            
            B_tensor = np.einsum('br,cr->bcr', V, W) # (V_rows, W_rows, R)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows) # (R, V_rows * W_rows)
            
            # Target vectors for U
            # T_target is (U_rows, V_rows, W_rows)
            # We can precompute T_target once outside the loop if we do it.
            # Let's do it here for simplicity, it's small.
            T_target = np.zeros((U_rows, V_rows, W_rows))
            for (a, b, c) in triples:
                T_target[a, b, c] = 1.0
            T_target_flat = T_target.reshape(U_rows, V_rows * W_rows)
            
            # Solve U[a, :] @ B.T = T_target_flat[a, :]
            # U[a, :] = T_target_flat[a, :] @ pinv(B.T)
            # Or use lstsq.
            # Since B is (R, M) where M = V_rows * W_rows = 1296, and R <= 144.
            # We are solving for x (size R) in B.T x = y.
            # This is an underdetermined system (R < M typically).
            # We want the minimum norm solution.
            # x = pinv(B.T) @ y = (B B.T)^-1 B y
            
            # Let's compute G = B B.T (R x R)
            G = B @ B.T
            # Add small regularization for stability
            G += 1e-8 * np.eye(target_rank)
            
            # For each a, U[a, :] = T_target_flat[a, :] @ G^-1
            # We can compute G_inv once.
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)
            
            # U = T_target_flat @ G_inv
            # T_target_flat is (U_rows, M), G_inv is (R, R)
            # We need U (U_rows, R)
            # U[a, r] = sum_k T_target_flat[a, k] * (G_inv^T)[k, r]? No.
            # Equation: B.T @ U[a, :] = T_target_flat[a, :]
            # (B.T)^T B.T U = B.T T -> B B.T U = B T
            # G U = B T_target_flat.T
            # U = G_inv @ (B @ T_target_flat.T)
            
            BT = B @ T_target_flat.T # (R, U_rows)
            U = G_inv @ BT # (R, U_rows)
            U = U.T # (U_rows, R)
            
            # Update V
            # Similarly for V[b, :]
            # C[r, a, c] = U[a, r] W[c, r]
            C_tensor = np.einsum('ar,cr->acr', U, W) # (U_rows, W_rows, R)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            
            # Target for V: T_target[b, a, c]
            # T_target is (U_rows, V_rows, W_rows)
            # We need T_target_V[b, a, c] = T_target[a, b, c]
            T_target_V = np.transpose(T_target, (1, 0, 2)) # (V_rows, U_rows, W_rows)
            T_target_V_flat = T_target_V.reshape(V_rows, U_rows * W_rows)
            
            Gv = C @ C.T
            Gv += 1e-8 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)
            
            CT = C @ T_target_V_flat.T
            V = (Gv_inv @ CT).T
            
            # Update W
            # D[r, a, b] = U[a, r] V[b, r]
            D_tensor = np.einsum('ar,br->abr', U, V) # (U_rows, V_rows, R)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            
            # Target for W: T_target[c, a, b]
            # T_target is (U_rows, V_rows, W_rows)
            # We need T_target_W[c, a, b] = T_target[a, b, c]
            T_target_W = np.transpose(T_target, (2, 0, 1)) # (W_rows, U_rows, V_rows)
            T_target_W_flat = T_target_W.reshape(W_rows, U_rows * V_rows)
            
            Gw = D @ D.T
            Gw += 1e-8 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)
            
            DT = D @ T_target_W_flat.T
            W = (Gw_inv @ DT).T
            
            # Check convergence
            if it % 10 == 0:
                # Compute error on a subset
                err = 0.0
                # Check triples
                for (a, b, c) in triples:
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += (val - 1.0) ** 2
                # Check some non-triples
                rng_check = np.random.RandomState(it)
                for _ in range(100):
                    a = rng_check.randint(0, U_rows)
                    b = rng_check.randint(0, V_rows)
                    c = rng_check.randint(0, W_rows)
                    if (a, b, c) in triple_set:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += val ** 2
                
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
        
        return U, V, W

    # Try to find lower rank decompositions
    # Start from a reasonable lower bound
    # Theoretical lower bound is max(U_rows, V_rows, W_rows) = 36
    # But for matrix multiplication, it's often higher.
    # Let's try ranks from 100 down to 36.
    
    ranks_to_try = list(range(100, 35, -2))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            result = als_decompose(target_rank, max_iters=50, seed=seed)
            if result is None:
                continue
            
            U, V, W = result
            
            # Check if this is a valid decomposition (fast check first)
            if check_decomposition_fast(U, V, W, tol=1e-4):
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_decomposition_exact(U_rounded, V_rounded, W_rounded, tol=1e-9):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # If rounding fails, check if unrounded is exact enough (unlikely to be accepted if not half-int)
                    # The problem requires integer/half-integer entries.
                    # So we MUST round. If rounding breaks it, we discard this candidate.
                    # But maybe we can try a different rounding or refinement?
                    # For now, discard if rounded version is not exact.
                    pass
        
        if found:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
