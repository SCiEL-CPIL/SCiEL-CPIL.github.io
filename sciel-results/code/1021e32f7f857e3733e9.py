"""Matrix multiplication 4x4x9 rank-minimal decomposition via Strassen-style block recursion."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36
    
    # Build the target tensor: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    triple_set = set(triples)
    
    # Schoolbook solution (rank = 144)
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
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact for all entries."""
        r = U.shape[1]
        # Check all triple entries = 1
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check all non-triple entries = 0
        for a in range(U_rows):
            for b in range(V_rows):
                for c in range(W_rows):
                    if (a, b, c) in triple_set:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, tol=1e-9):
        """Faster check using matrix operations and sampling."""
        r = U.shape[1]
        # Check all triples exactly
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # For non-triples, use a deterministic subset for speed
        # Check a structured subset of non-triple entries
        rng = np.random.RandomState(42)
        num_checks = 10000
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
    
    # Strassen-style block decomposition
    # Split A (4x4) into 2x2 blocks of 2x2
    # Split B (4x9) into 2x2 blocks of 2x(9/2)... but 9 is odd
    # Alternative: use a 7-multiplication scheme for 2x2 matrices
    # For 4x4 * 4x9, we can use block Strassen
    
    # Strategy: Use the fact that 4x4 matrix multiplication can be done with 7^2=49 multiplications
    # if we use Strassen recursively. But B is 4x9, not 4x4.
    
    # Let's try a different approach: direct algebraic decomposition
    # The tensor T has 144 ones. We want to find rank-r decomposition.
    
    # Lower bound: max(U_rows, V_rows, W_rows) = 36
    # For 4x4x4, the rank is known to be 49 (Strassen) or possibly lower.
    # For 4x4x9, the rank should be at most 144 (schoolbook).
    
    # Let's try to use the structure of the problem more directly.
    # The key insight: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    
    # Try to find a decomposition with rank < 144 using ALS or other methods
    
    # First, let's try a greedy approach: start with schoolbook and try to merge terms
    # Or use a random initialization and ALS
    
    def als_decompose(target_rank, max_iters=200, tol=1e-10, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        if time.time() > deadline - 15:
            return None
        
        rng = np.random.RandomState(seed)
        # Initialize with small random values
        U = rng.randn(U_rows, target_rank) * 0.01
        V = rng.randn(V_rows, target_rank) * 0.01
        W = rng.randn(W_rows, target_rank) * 0.01
        
        # Precompute the target tensor as a 3D array
        T = np.zeros((U_rows, V_rows, W_rows))
        for (a, b, c) in triples:
            T[a, b, c] = 1.0
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
            # Build design matrix B where B[r, (b,c)] = V[b,r] * W[c,r]
            B = np.zeros((target_rank, V_rows * W_rows))
            for r in range(target_rank):
                B[r, :] = np.outer(V[:, r], W[:, r]).flatten()
            
            for a in range(U_rows):
                target_vec = T[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(B.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Update V
            # For each b: T[:,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Design matrix C where C[r, (a,c)] = U[a,r] * W[c,r]
            C = np.zeros((target_rank, U_rows * W_rows))
            for r in range(target_rank):
                C[r, :] = np.outer(U[:, r], W[:, r]).flatten()
            
            for b in range(V_rows):
                target_vec = T[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(C.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Update W
            # For each c: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Design matrix D where D[r, (a,b)] = U[a,r] * V[b,r]
            D = np.zeros((target_rank, U_rows * V_rows))
            for r in range(target_rank):
                D[r, :] = np.outer(U[:, r], V[:, r]).flatten()
            
            for c in range(W_rows):
                target_vec = T[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(D.T, target_vec, rcond=None)[0]
                except:
                    pass
            
            # Check convergence
            if it % 20 == 0:
                err = np.sum((T - np.einsum('ar,br,cr->abc', U, V, W))**2)
                if err < tol:
                    break
        
        return U, V, W
    
    # Try ranks from 40 to 100
    for target_rank in range(40, 101):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            result = als_decompose(target_rank, max_iters=100, seed=seed)
            if result is None:
                continue
            
            U, V, W = result
            
            # Check if this is a valid decomposition
            if check_decomposition_fast(U, V, W, tol=1e-8):
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Use the unrounded version if it's valid
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_school:
        # Ensure entries are half-integers if possible
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        
        if check_decomposition(U_out, V_out, W_out, tol=1e-9):
            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        else:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
