"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    def time_left():
        return deadline - (time.time() - start_time)

    # Build a rank-54 decomposition using 2x2x2 Strassen blocks + 2x1x2 schoolbook blocks
    # This is the base solution.
    
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (12, rank)

    # We'll build the decomposition by blocks.
    # j-groups: {0,1}, {2,3}, {4}
    # k-groups: {0,1}, {2,3}, {4,5}
    
    rank = 54
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_step(u_terms, v_terms, w_terms):
        nonlocal col
        if col >= rank:
            return
        if time_left() < 0.1:
            return
        for idx, val in u_terms:
            U[idx][col] += val
        for idx, val in v_terms:
            V[idx][col] += val
        for idx, val in w_terms:
            W[idx][col] += val
        col += 1

    def strassen_block(j1, j2, k1, k2):
        """Rank-7 Strassen for A[{0,1}, {j1,j2}] * B[{j1,j2}, {k1,k2}]"""
        if time_left() < 0.1:
            return
        # m1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
        add_step(
            [(u_idx(0,j1),1),(u_idx(1,j2),1)],
            [(v_idx(j1,k1),1),(v_idx(j2,k2),1)],
            [(w_idx(k1,0),1),(w_idx(k2,1),1)]
        )
        # m2 = (a10+a11)*b00 -> C10 += 1, C11 -= 1
        add_step(
            [(u_idx(1,j1),1),(u_idx(1,j2),1)],
            [(v_idx(j1,k1),1)],
            [(w_idx(k1,1),1),(w_idx(k2,1),-1)]
        )
        # m3 = a00*(b01-b11) -> C01 += 1, C11 += 1
        add_step(
            [(u_idx(0,j1),1)],
            [(v_idx(j1,k2),1),(v_idx(j2,k2),-1)],
            [(w_idx(k2,0),1),(w_idx(k2,1),1)]
        )
        # m4 = a11*(b10-b00) -> C00 += 1, C10 += 1
        add_step(
            [(u_idx(1,j2),1)],
            [(v_idx(j2,k1),1),(v_idx(j1,k1),-1)],
            [(w_idx(k1,0),1),(w_idx(k1,1),1)]
        )
        # m5 = (a00+a01)*b11 -> C00 -= 1, C01 += 1
        add_step(
            [(u_idx(0,j1),1),(u_idx(0,j2),1)],
            [(v_idx(j2,k2),1)],
            [(w_idx(k1,0),-1),(w_idx(k2,0),1)]
        )
        # m6 = (a10-a00)*(b00+b01) -> C11 += 1
        add_step(
            [(u_idx(1,j1),1),(u_idx(0,j1),-1)],
            [(v_idx(j1,k1),1),(v_idx(j1,k2),1)],
            [(w_idx(k2,1),1)]
        )
        # m7 = (a01-a11)*(b10+b11) -> C00 += 1
        add_step(
            [(u_idx(0,j2),1),(u_idx(1,j2),-1)],
            [(v_idx(j2,k1),1),(v_idx(j2,k2),1)],
            [(w_idx(k1,0),1)]
        )

    def schoolbook_2x1x2(j, k1, k2):
        """Rank-4 schoolbook for A[:,j] * B[j,{k1,k2}]"""
        if time_left() < 0.1:
            return
        add_step([(u_idx(0,j),1)], [(v_idx(j,k1),1)], [(w_idx(k1,0),1)])
        add_step([(u_idx(0,j),1)], [(v_idx(j,k2),1)], [(w_idx(k2,0),1)])
        add_step([(u_idx(1,j),1)], [(v_idx(j,k1),1)], [(w_idx(k1,1),1)])
        add_step([(u_idx(1,j),1)], [(v_idx(j,k2),1)], [(w_idx(k2,1),1)])

    # Apply blocks
    for j1, j2 in [(0,1), (2,3)]:
        for k1, k2 in [(0,1), (2,3), (4,5)]:
            strassen_block(j1, j2, k1, k2)
    
    for k1, k2 in [(0,1), (2,3), (4,5)]:
        schoolbook_2x1x2(4, k1, k2)

    base_U = [row[:] for row in U]
    base_V = [row[:] for row in V]
    base_W = [row[:] for row in W]
    
    # Now attempt to find a lower-rank decomposition by searching for linear dependencies
    # or by trying alternative block structures.
    
    # Alternative: try to use a 3x3 Strassen-like approach or other algebraic tricks
    # For 2x5x6, we can try different partitions.
    
    # Let's try to verify the rank-54 solution and see if we can find rank 53 or lower
    # by checking if any column is a linear combination of others.
    
    import numpy as np
    
    U_np = np.array(base_U, dtype=float)
    V_np = np.array(base_V, dtype=float)
    W_np = np.array(base_W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to verify this equals the standard multiplication tensor.
    
    # Let's check rank of the 3-mode tensor
    # Reshape into matrices for rank computation
    # T as a 3D tensor: shape (n*m, m*p, p*n)
    # T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # But the indices are: i = row of A, j = col of A / row of B, k = col of B
    # Actually the tensor is: T[a_idx, b_idx, c_idx] where a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    # And T[a_idx, b_idx, c_idx] = 1 if the indices are consistent, else 0
    # More precisely: for A (n x m), B (m x p), C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has entries T[i,j,k] = delta (the identity in the summation)
    # In the decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's construct the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Check our decomposition
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
    
    # Verify
    max_err = np.max(np.abs(T_decomp - T_target))
    
    if max_err > 1e-10:
        # Decomposition is wrong, fall back to schoolbook
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank by column elimination
    # If a column in (U,V,W) is redundant, we can remove it
    # Check linear dependence of the "rank-1" tensors
    # The rank-1 tensors are u_r ⊗ v_r ⊗ w_r
    # If sum_{r in S} c_r * (u_r ⊗ v_r ⊗ w_r) = 0, we can reduce rank
    
    # Flatten each rank-1 tensor into a vector of length n*m*m*p*p*n
    # This is too large (10*30*12 = 3600), but we have 54 vectors, so it's feasible
    
    # Actually, let's use a different approach: try to find if any column can be expressed
    # as a linear combination of others by solving a linear system.
    
    # For each column r, check if there exist coefficients c_s (s != r) such that:
    # u_r ⊗ v_r ⊗ w_r = sum_{s != r} c_s * (u_s ⊗ v_s ⊗ w_s)
    
    # This is equivalent to checking if the matrix formed by all columns (as vectors in R^{3600})
    # has rank < 54.
    
    # Let's compute the rank of the 54 x 3600 matrix
    cols = []
    for r in range(rank):
        vec = np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r]).flatten()
        cols.append(vec)
    
    M = np.column_stack(cols)
    rank_found = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_found < rank and time_left() > 30:
        # We found redundancy! Try to find a basis
        # Use SVD to find the rank-deficient directions
        try:
            U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
            # The right singular vectors corresponding to near-zero singular values give null space
            # But we want to find which columns are linearly dependent
            
            # Find the null space of M
            tol = 1e-8
            null_mask = S < tol
            if np.any(null_mask):
                # The null space is spanned by Vt[null_mask]
                # But this doesn't directly tell us which columns to remove.
                # Let's use a different approach: Gaussian elimination on the columns
                pass
        except:
            pass
    
    # Alternative approach: try to find a rank-53 decomposition by modifying the structure
    # Try different block partitions
    
    # Try: j-groups {0,1,2}, {3,4} and k-groups {0,1,2}, {3,4,5}
    # 3x2 blocks and 2x3 blocks - these don't have known low-rank decompositions easily
    
    # Try: keep the same partition but see if we can merge some terms
    
    # Actually, let's try a completely different partition:
    # j-groups: {0,1,2,3}, {4}  -> 4x2 and 2x1
    # k-groups: {0,1,2}, {3,4,5} -> 3x3 blocks
    
    # For 4x2x3: schoolbook is 24, but maybe we can do better?
    # For 2x1x3: schoolbook is 6
    
    # This gives 24 + 6 = 30, which is worse than 54. So the 2x2 blocks are better.
    
    # Let's try to see if rank 53 is achievable by a small modification
    # Try to perturb the decomposition slightly and see if we can maintain exactness
    # while using one fewer term
    
    # This is a hard problem. For now, let's try to use the rank-54 solution
    # and see if we can find any column that is exactly zero or a simple combination.
    
    # Check for zero columns
    for r in range(rank):
        if np.all(U_np[:,r] == 0) and np.all(V_np[:,r] == 0) and np.all(W_np[:,r] == 0):
            # This column is unused
            # Remove it
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank -= 1
            break
    
    if rank < 54 and time_left() > 10:
        # Verify again
        T_decomp2 = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp2 += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
        if np.max(np.abs(T_decomp2 - T_target)) < 1e-10:
            U = U_np.astype(int).tolist()
            V = V_np.astype(int).tolist()
            W = W_np.astype(int).tolist()
            return {"U": U, "V": V, "W": W}
    
    # Return the rank-54 solution
    U_out = [row[:] for row in base_U]
    V_out = [row[:] for row in base_V]
    W_out = [row[:] for row in base_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
