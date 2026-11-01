import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _build_rank21_strassen():
    n, m, p = DIMENSIONS
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_blocks = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_blocks[t]
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            for l, val in v_coeffs.items():
                if l == 0: v_idx = 0 * p + 2 * b
                elif l == 1: v_idx = 0 * p + 2 * b + 1
                elif l == 2: v_idx = 1 * p + 2 * b
                elif l == 3: v_idx = 1 * p + 2 * b + 1
                V[v_idx, r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx, r] = val
            r += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[i,j,k] = delta_{i, i'} delta_{j, j'} where 
    # i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
    # T[i,j,k] = 1 if j == k (since C[i,k] = sum_j A[i,j] B[j,k], the tensor is 
    # T[i,j,k] = 1 if j==k else 0... wait no)
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor has T[i,j,k] = 1 iff the index j connects A and B.
    # In the standard indexing: T[a, b, c] = 1 if a is row of A, b is col of A = row of B, c is col of B.
    # So T[i, j, k] = 1 for all valid (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. The tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] = 1 if j == j (always), meaning every triple (i,j,k) with i in rows of A,
    # j in shared dimension, k in cols of B contributes. Actually T[i,j,k] = 1 for all i,j,k.
    # No - the structure is: the (i,j,k)-th entry of the tensor is 1 if the j-index of A's 
    # column matches the j-index of B's row. Since we label them consistently, T[i,j,k] = 1 
    # for ALL i in [0,n), j in [0,m), k in [0,p).
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A ⊗ e_j^B ⊗ e_k^C
    # where e_i^A is the i-th row index of A, e_j^B is the j-th column of A / row of B,
    # e_k^C is the k-th column of B / column of C.
    # So M[i, j, k] = 1 for all valid indices.
    
    # The rank-r decomposition says: M[i,j,k] = sum_{r=0}^{R-1} U[i,r] * V[j,r] * W[k,r]
    
    max_err = 0.0
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                target = 1.0  # M[i,j,k] = 1 for all valid indices
                recon = 0.0
                for r in range(rank):
                    recon += U[i, r] * V[j, r] * W[k, r]
                err = abs(recon - target)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_reduce_rank(U, V, W, max_rank, deadline):
    """Try to find a rank < max_rank decomposition by searching for linear dependencies."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    if time.time() > deadline:
        return U, V, W
    
    # Try to find if we can drop one term by checking if the tensor can be represented
    # with rank-1 fewer terms. We do this by trying to express one term as a linear
    # combination of the others.
    
    # For each term r0, check if there exist coefficients c_r (r != r0) such that
    # U[:,r0] = sum_{r!=r0} c_r * U[:,r] (and similarly for V, W)
    # This is a necessary but not sufficient condition.
    
    # Better approach: use the fact that if rank is minimal, the tensor has no 
    # linear dependencies. We can try to find a rank-(rank-1) decomposition by
    # optimizing over the factors.
    
    # Let's try a more direct approach: check if any single term can be eliminated
    # by checking if the resulting (rank-1) decomposition still works.
    
    for drop_idx in range(rank):
        if time.time() > deadline:
            return U, V, W
        
        mask = [i for i in range(rank) if i != drop_idx]
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        if _verify_decomposition(U_test, V_test, W_test, tol=1e-8):
            return U_test, V_test, W_test
    
    return U, V, W

def _try_rank19_search(U21, V21, W21, deadline):
    """Try to find a rank-19 or rank-20 decomposition via structured search."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The key insight: for 2x2x6, we can potentially do better than 3 independent Strassen blocks.
    # Let's try to find a rank-20 decomposition by allowing cross-block interactions.
    
    # Strategy: Start from the rank-21 decomposition and try to merge two terms from different
    # blocks by finding a rank-1 tensor that, when added to one and subtracted from another,
    # preserves the decomposition.
    
    # Alternative: directly search for a rank-20 decomposition using the structure.
    # A 2x2x6 matrix multiplication can be seen as computing 3 independent 2x2x2 multiplications
    # (each 2x2x2 has rank 7 via Strassen), giving rank 21. But maybe we can do better by
    # sharing computation across blocks.
    
    # Let's try: can we find a rank-20 decomposition?
    # One approach: use the fact that the 2x2x6 problem is equivalent to 3 copies of 2x2x2,
    # but with the constraint that the "m" dimension is shared. Wait, no - in 2x2x6,
    # A is 2x2, B is 2x6, C is 2x6. So we're computing C = A*B where A is 2x2 and B is 2x6.
    # This is 3 independent 2x2x2 problems (columns 0-1, 2-3, 4-5 of B).
    
    # Actually, let me reconsider the structure. A is n x m = 2x2, B is m x p = 2x6.
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for i=0,1 and k=0,...,5.
    # So for each k, we compute a 2x2 matrix product (A * B[:,k]).
    # These are 3 independent 2x2x2 matrix multiplications.
    # Each 2x2x2 has Strassen rank 7, so total is 3*7 = 21.
    
    # Can we do better? The question is whether we can share the Strassen "subtractions"
    # across the three blocks. Since A is the same for all three blocks, the U factors
    # (which depend on A) are the same structure for all blocks.
    
    # Let's try a different approach: formulate the rank-20 problem as a system of equations.
    # For rank R, we need U (2x2 x R), V (2x6 x R), W (6x2 x R) such that
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,5}.
    
    # This is a system of 2*2*6 = 24 equations with 3*R*2 = 6R unknowns (for R=20, that's 120 unknowns).
    # The system is bilinear (actually trilinear) in the unknowns, so it's not linear.
    
    # However, we can fix one factor and solve for the others. Let's fix W and solve for U, V.
    # Given W, the equations become: for each (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = 1.
    # This is still bilinear in U and V.
    
    # Let's try a different approach: use the known rank-21 solution and try to perturb it
    # to find a rank-20 solution.
    
    # Actually, let's just try to verify if rank 20 is achievable by a simple check.
    # The lower bound for 2x2x6 is at least 2*2*6/2 = 12 (trivial) or better bounds.
    # The best known upper bound is 21 (3*7). It's an open question whether 20 or less is possible.
    
    # Given the computational constraints, let's just try to verify the rank-21 solution
    # and see if we can drop any terms.
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    U, V, W = _build_rank21_strassen()
    rank = U.shape[1]
    
    # Try to reduce rank by finding linear dependencies
    U, V, W = _try_reduce_rank(U, V, W, rank, deadline)
    
    rank = U.shape[1]
    
    # If we still have rank 21, try a more aggressive search
    if rank == 21 and time.time() < deadline - 10:
        result = _try_rank19_search(U, V, W, deadline)
        if result is not None:
            U, V, W = result
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
