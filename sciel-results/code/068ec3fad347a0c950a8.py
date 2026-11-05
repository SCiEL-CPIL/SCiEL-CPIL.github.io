"""Matrix multiplication 2,3,3 rank search."""
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 3, 3
    # Row indices: A has n*m = 6 rows, B has m*p = 9 rows, C has p*n = 6 rows
    
    # Known rank-17 decomposition (Strassen-like block):
    # We'll first verify and return rank 17 if valid.
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition."""
        rank = len(U[0])
        # Check: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] == delta_{k,n+i} ... 
        # Actually the tensor condition is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But for matrix multiplication, T[i,j,k] = 1 if k is the "output index" matching (i,j,k)
        # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i, j, k] = 1 iff we need A[i,j]*B[j,k] to contribute to C[i,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) in range.
        # The decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Rank 18 schoolbook - always valid
    U18 = [[0]*18 for _ in range(6)]
    V18 = [[0]*18 for _ in range(9)]
    W18 = [[0]*18 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U18[i*3+j][col] = 1
                V18[j*3+k][col] = 1
                W18[k*2+i][col] = 1
                col += 1
    
    # Try to find rank 17 by structured search
    # Use a known approach: the rank of M_{2,3,3} is at most 17.
    # Let's try a randomized/structured search for rank 16 and 17.
    
    # Strategy: enumerate sparse support patterns
    # For rank r, each matrix U (6xr), V (9xr), W (6xr) has r columns.
    # The key insight: we can fix U and V to be simple (e.g., identity-like) and solve for W,
    # or use a more systematic approach.
    
    # Actually, let's try a direct approach:
    # The tensor T is 6x9x6 (flattened indices).
    # We need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (using flattened indices)
    
    # Let's try rank 17 with a known construction pattern.
    # One approach: start from rank 18 and try to merge two rank-1 terms.
    
    # For now, let's try a randomized search for rank 16 and 17
    # with small integer/half-integer entries.
    
    def try_rank_search(rank, max_iters=500):
        """Try to find a valid decomposition of given rank."""
        import random
        random.seed(42)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            scale = [1, -1, 0.5, -0.5, 2, -2]
            U = [[random.choice(scale) for _ in range(rank)] for _ in range(6)]
            V = [[random.choice(scale) for _ in range(rank)] for _ in range(9)]
            
            # For each (i,j,k), we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # This is a system of linear equations in W entries (and products).
            # Actually it's nonlinear. Let's use a different approach.
            
            # Fix U and V, then for each pair (k,i), W[k,i,c] appears in equations
            # for all j. Specifically, for fixed (k,i), the equation over j is:
            # sum_c V[j,k,c] * W[k,i,c] = 1/U[i,j,c] ... no, that's not right.
            
            # Let's use: for each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
            # Group by (k,i): sum_j' sum_c U[i,j',c] * V[j',k,c] * W[k,i,c] = sum_j' 1 = m = 3
            # Wait, that's not right either. Each (i,j,k) is a separate equation.
            
            # Total equations: 2*3*3 = 18
            # Unknowns: 6*rank (W entries)
            
            # For fixed U,V, the system in W is:
            # For each (i,j,k): sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1
            # This is linear in W! Let's set up the linear system.
            
            # W has 6*rank unknowns. We have 18 equations.
            # If rank >= 3, we have 6*rank >= 18 unknowns, so underdetermined.
            
            # Build the system: A_W * W_vec = b
            # Where A_W has 18 rows and 6*rank columns.
            
            num_eq = n * m * p  # 18
            num_unk = p * n * rank  # 6*rank
            
            A_w = [[0.0] * num_unk for _ in range(num_eq)]
            b = [1.0] * num_eq
            
            for eq_idx, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
                for c in range(rank):
                    col_idx = k * n + i * rank + c  # W[k*n+i, c] -> index
                    A_w[eq_idx][col_idx] = U[i*m+j][c] * V[j*p+k][c]
            
            # Solve: we want any solution. Use least squares or null space.
            # Since it's underdetermined (num_unk > num_eq typically), find a particular solution.
            
            # Use numpy for the linear solve
            import numpy as np
            A_np = np.array(A_w)
            b_np = np.array(b)
            
            try:
                # Find minimum-norm solution
                W_vec, residuals, rank_sv, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
            except:
                continue
            
            # Check if it's an exact solution
            if not np.allclose(A_np @ W_vec, b_np, atol=1e-9):
                continue
            
            # Reshape W_vec back to W matrix
            W = [[0]*rank for _ in range(p*n)]
            for idx, val in enumerate(W_vec):
                k_idx = idx // (n * rank)
                rem = idx % (n * rank)
                i_idx = rem // rank
                c_idx = rem % rank
                W[k_idx * n + i_idx][c_idx] = val
            
            # Verify
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 16 first (best case)
    if time.time() < deadline - 30:
        result = try_rank_search(16, max_iters=200)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try rank 17
    if time.time() < deadline - 20:
        result = try_rank_search(17, max_iters=300)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 18 schoolbook
    return {"U": U18, "V": V18, "W": W18}

# EVOLVE-BLOCK-END
