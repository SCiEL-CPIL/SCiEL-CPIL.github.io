"""Combined program: structural rank-32 base with local search for rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 2, 9

    # Build the rank-32 base decomposition using Strassen blocks
    # Block 1: cols 0-3 -> two Strassen-7 blocks (rank 14)
    # Block 2: cols 4-8 -> Strassen(4,5) + schoolbook(6) + Strassen(7,8) (rank 7+4+7=18)
    # Total: 32

    rank_base = 32
    U = [[0.0] * rank_base for _ in range(n * m)]   # 4 x 32
    V = [[0.0] * rank_base for _ in range(m * p)]   # 18 x 32
    W = [[0.0] * rank_base for _ in range(p * n)]   # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b00+b11) -> C00, C11
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M2: (a10+a11)b00 -> C10, -C11
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        # M3: a00(b01-b11) -> C01, C11
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        # M4: a11(b10-b00) -> C00, C10
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        # M5: (a00+a01)b11 -> -C00, C01
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        # M6: (a10-a00)(b00+b01) -> C11
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        # M7: (a01-a11)(b10+b11) -> C00
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
        U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
        U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_trivial(6)
    add_strassen(7, 8)

    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        """Check tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_arr.shape[1]):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr, tol=1e-8):
        """Vectorized verification."""
        r = U_arr.shape[1]
        # Build tensor via einsum
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        # Check against ones tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1.0) > tol:
                        return False
        return True

    # Try to reduce rank by 1: for each candidate term to remove,
    # check if the remaining rank-31 terms can be adjusted to still decompose T.
    # This is a non-linear problem, but we can use a local approach:
    # Fix one matrix (say W), and solve for U and V adjustments.
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_base

    # Strategy: try removing each term and see if we can find a valid rank-31 decomposition
    # by solving a least-squares problem on the residual.
    
    def try_rank_reduction(target_rank, U0, V0, W0):
        """Try to find a decomposition of rank target_rank starting from rank_base.
        We'll use a simple approach: for each term to drop, solve for corrections."""
        if time.time() > deadline - 5:
            return None
        
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Total number of tensor entries
        total = n * m * p
        
        # For each candidate term to remove, try to solve
        for drop_idx in range(rank_base):
            if time.time() > deadline - 5:
                return None
            
            # Keep all terms except drop_idx
            keep = [i for i in range(rank_base) if i != drop_idx]
            rank_new = len(keep)
            
            if rank_new < target_rank:
                continue
            
            # Extract sub-matrices
            U_sub = U0[:, keep]
            V_sub = V0[:, keep]
            W_sub = W0[:, keep]
            
            # The residual we need to compensate: T - sum_{r in keep} u_r v_r w_r
            # But we can adjust U, V, W. This is non-linear.
            # Simpler: fix W_sub, solve for U and V such that sum_r U[:,r] V[:,r] W_sub[:,r] = T
            # This is still non-linear. 
            # Alternative: use the fact that for fixed W, the problem becomes:
            # For each output index (i,k), we need: sum_j sum_r U[i*m+j,r] V[j*p+k,r] W_sub[k*n+i,r] = 1
            # Let's try a different approach: perturb one matrix and solve for others.
            
            # Approach: Fix U_sub and W_sub, solve for V_sub' such that the tensor matches.
            # For each (j,k): sum_i sum_r U[i*m+j,r] * V'[j*p+k,r] * W_sub[k*n+i,r] = 1
            # This gives: for each (j,k), a linear system in V'[j*p+k, :]
            # But it's coupled across r through the sum over i.
            
            # Let's use a simpler heuristic: check if the term is "redundant" by 
            # checking if its contribution can be matched by a linear combination of others.
            # Build the 3D tensor for each term
            # term_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # Check if term_drop_idx is in the span of the other terms
            # This requires solving a linear system in the tensor product space.
            # The space has dimension n*m*p = 36.
            # Each term is a vector in this 36-dim space.
            
            # Build matrix A where column r is the flattened term tensor
            A = np.zeros((total, rank_base))
            for r in range(rank_base):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A[i*m*p + j*p + k, r] = U0[i*m+j, r] * V0[j*p+k, r] * W0[k*n+i, r]
            
            # Target tensor T
            T_vec = np.ones(total)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_vec[i*m*p + j*p + k] = 1.0
            
            # Check if we can express T_vec as linear combination of columns of A
            # excluding column drop_idx
            A_keep = A[:, keep]
            
            # Solve least squares: A_keep @ x ≈ T_vec
            try:
                x, residuals, rank_a, sv = np.linalg.lstsq(A_keep, T_vec, rcond=None)
            except:
                continue
            
            # Check if the solution is exact
            residual = np.linalg.norm(A_keep @ x - T_vec)
            if residual < 1e-10:
                # We found coefficients! But these coefficients don't directly give us
                # a valid tensor decomposition because the terms are not independent.
                # The linear combination of tensors doesn't correspond to a rank decomposition.
                # This approach is flawed. Let's use a different strategy.
                pass
            
            # Better approach: use local optimization to adjust the remaining terms
            # to compensate for the dropped term.
            # We'll use a simple gradient-free approach: try small perturbations.
            
            # Actually, let's try a more principled approach:
            # The tensor decomposition constraint is:
            # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
            # If we drop term r0, we need to find U', V', W' (of size rank-1) such that
            # sum_{r != r0} U'[:,r] ⊗ V'[:,r] ⊗ W'[:,r] = T
            # 
            # This is equivalent to finding a rank-(rank-1) decomposition of T.
            # We can use iterative methods: fix two matrices, solve for the third.
            
            # Fix W_sub, solve for U' and V'
            # For each k*n+i: sum_j sum_r U'[i*m+j,r] V'[j*p+k,r] W_sub[k*n+i,r] = 1
            # Let's define for each r: w_r = W_sub[:, r]
            # Then: sum_j sum_r U'[i*m+j,r] V'[j*p+k,r] w_r[k*n+i] = 1
            # 
            # This is still non-linear. Let's use a different strategy:
            # Start from the current solution and try to "absorb" the dropped term
            # into the others via a change of basis.
            
            # Change of basis: if we have rank R decomposition, we can apply
            # invertible transformations: U' = U * P, V' = V * P^{-T}, W' = W * P^{-T}
            # This preserves the tensor. We can use this to set one entry to zero
            # and then drop that term if it becomes zero.
            
            # Let's try: for each pair of terms (r1, r2), see if we can combine them
            # into one term by finding a linear combination that simplifies.
            
            if time.time() > deadline - 5:
                return None
        
        return None

    # The above approach is too complex. Let's use a simpler but effective method:
    # Try all pairs of terms and see if their sum can be expressed as a single term
    # in the tensor product space. This is equivalent to checking if the sum of two
    # rank-1 tensors is rank 1.
    
    def is_rank1_3tensor(U_col, V_col, W_col):
        """Check if the 3-tensor u⊗v⊗w has rank 1 (always true by construction)."""
        return True
    
    def can_merge_two_terms(U0, V0, W0, r1, r2):
        """Check if terms r1 and r2 can be merged into a single term.
        This requires that u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w for some u,v,w.
        This is a non-trivial condition. A necessary condition is that
        the 2-tensor u1⊗v1 + u2⊗v2 has rank ≤ 2 and similar for other contractions."""
        u1, u2 = U0[:, r1], U0[:, r2]
        v1, v2 = V0[:, r1], V0[:, r2]
        w1, w2 = W0[:, r1], W0[:, r2]
        
        # Check if u1 and u2 are linearly dependent
        if np.linalg.matrix_rank(np.column_stack([u1, u2])) <= 1:
            # u1 = a*u2, then we need v1*w1 + a*v2*w2 = v*w for some v,w
            # This is possible if v1*w1 + a*v2*w2 is a rank-1 matrix
            M = np.outer(v1, w1) + np.dot(u1, u2)/np.dot(u2, u2) * np.outer(v2, w2) if np.dot(u2,u2) != 0 else np.outer(v1, w1)
            if np.linalg.matrix_rank(M) <= 1:
                return True
        
        # Check if v1 and v2 are linearly dependent
        if np.linalg.matrix_rank(np.column_stack([v1, v2])) <= 1:
            return True
        
        # Check if w1 and w2 are linearly dependent
        if np.linalg.matrix_rank(np.column_stack([w1, w2])) <= 1:
            return True
        
        return False

    # Try merging pairs
    merged = []
    for r1 in range(rank_base):
        if r1 in merged:
            continue
        for r2 in range(r1 + 1, rank_base):
            if r2 in merged:
                continue
            if time.time() > deadline - 10:
                break
            if can_merge_two_terms(U_np, V_np, W_np, r1, r2):
                merged.append(r1)
                merged.append(r2)
                break
        if time.time() > deadline - 10:
            break

    if merged:
        keep = [r for r in range(rank_base) if r not in merged]
        U_new = [[U[i][r] for r in keep] for i in range(n*m)]
        V_new = [[V[i][r] for r in keep] for i in range(m*p)]
        W_new = [[W[i][r] for r in keep] for i in range(p*n)]
        
        U_np_new = np.array(U_new, dtype=float)
        V_np_new = np.array(V_new, dtype=float)
        W_np_new = np.array(W_new, dtype=float)
        
        if verify(U_np_new, V_np_new, W_np_new):
            return {"U": U_new, "V": V_new, "W": W_new}

    # Return the base decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
