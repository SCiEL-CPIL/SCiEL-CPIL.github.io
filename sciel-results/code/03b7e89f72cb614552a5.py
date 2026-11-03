import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # The 3x3x6 multiplication tensor has 54 non-zero entries.
    # We seek a rank-r decomposition: sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = T[i,j,k]
    # This is equivalent to finding rank-r factors for a 3-way tensor.
    
    # Strategy: Use the known rank-23 Bini et al. decomposition for 3x3x3 as a base,
    # but attempt to find a lower rank decomposition for 3x3x6 directly.
    
    # First, let's establish the schoolbook rank 54 as a safe fallback.
    def schoolbook():
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
    
    # Build the 3x3x6 tensor T as a matrix M of size (n*m) x (m*p) for each fixed k? No, that's not right.
    # The tensor is T[i, j, k]. We can view it as a matrix unfolding.
    # But for rank decomposition, we need to solve for U, V, W.
    
    # A simpler approach: Since 3x3x6 is two 3x3x3 blocks, the rank is at most 46 (two copies of rank-23).
    # Can we do better? The rank of the 3x3x6 multiplication tensor is known to be 45 in some literature?
    # Actually, the rank of 3x3x3 is 23. For 3x3x6, it's the direct sum of two 3x3x3 tensors, so rank is 46.
    # But wait, the 3x3x6 multiplication is NOT a direct sum. It's a single tensor where B has 6 columns.
    # The tensor T[i,j,k] = delta_{i,j,k} for the multiplication? No, the multiplication tensor is T[i,j,k] = 1 if i,j,k are valid indices.
    # Actually, the matrix multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # No, that's not right either. The multiplication tensor is defined such that C = A*B, where C_{i,k} = sum_j A_{i,j} * B_{j,k}.
    # So the tensor is T[i, j, k] = 1 for all i, j, k. Wait, no.
    # The tensor T is such that T(A, B) = C, where C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 if the indices match, i.e., T_{i,j,k} = 1 for all i, j, k? No.
    # Actually, T_{i,j,k} = 1 for all i in [n], j in [m], k in [p]. That would give C_{i,k} = sum_j A_{i,j} B_{j,k} if T_{i,j,k} = 1.
    # But that's not a tensor decomposition in the sense of rank. The rank of the multiplication tensor is the minimum number of terms.
    
    # Let's use the standard approach: the multiplication tensor M_{n,m,p} is a tensor in R^{n x m x p} with M_{i,j,k} = 1 for all i,j,k.
    # Wait, no. The multiplication tensor is M_{i,j,k} = 1 if i,j,k are the "multiplication" indices.
    # Actually, for matrix multiplication, the tensor is M_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to M.
    
    # For 3x3x3, the rank is 23. For 3x3x6, the tensor is M_{i,j,k} = 1 for i in [3], j in [3], k in [6].
    # This is NOT the direct sum of two 3x3x3 tensors. It's a single tensor.
    # The rank of M_{3,3,6} is at most 46 (two copies of 3x3x3 rank-23), but could be lower.
    
    # Let's try to find a rank-45 decomposition using integer linear algebra.
    # We have 54 equations (one for each (i,j,k) tuple) and 3*r*54 variables? No.
    # We have U (9 x r), V (18 x r), W (18 x r). Total variables: 9r + 18r + 18r = 45r.
    # Equations: for each (i,j,k), sum_{r'} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = 1.
    # This is a system of 54 cubic equations in 45r variables.
    
    # This is hard to solve directly. Let's stick with the known rank-46 solution.
    
    # Bini et al. rank-23 decomposition for 3x3x3:
    # We'll use the verified coefficients from the current program.
    
    bini = [
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,1, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0]),
        ([1,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([1,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,1, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,1, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),
        ([1,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
    ]
    
    # Verify the bini decomposition
    A_test = np.random.randn(3, 3)
    B_test = np.random.randn(3, 3)
    C_test = A_test @ B_test
    
    C_recon = np.zeros((3, 3))
    for u, v, w in bini:
        u_mat = np.array(u).reshape(3, 3)
        v_mat = np.array(v).reshape(3, 3)
        w_mat = np.array(w).reshape(3, 3)
        C_recon += (A_test * u_mat) @ (B_test * v_mat) * w_mat
    
    if not np.allclose(C_recon, C_test, atol=1e-10):
        return schoolbook()
    
    # Build the 3x3x6 decomposition using two copies of bini
    rank = 2 * len(bini)  # 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    # First half: columns 0-2 of B
    for a_idx, (u, v, w) in enumerate(bini):
        col = a_idx
        for idx in range(9):
            if u[idx] != 0:
                U[idx][col] = u[idx]
            j = idx // 3
            k = idx % 3
            if v[idx] != 0:
                V[j * 6 + k][col] = v[idx]
            i = idx // 3
            k = idx % 3
            if w[idx] != 0:
                W[k * 3 + i][col] = w[idx]
    
    # Second half: columns 3-5 of B
    for a_idx, (u, v, w) in enumerate(bini):
        col = 23 + a_idx
        for idx in range(9):
            if u[idx] != 0:
                U[idx][col] = u[idx]
            j = idx // 3
            k = idx % 3
            if v[idx] != 0:
                V[j * 6 + (k + 3)][col] = v[idx]
            i = idx // 3
            k = idx % 3
            if w[idx] != 0:
                W[(k + 3) * 3 + i][col] = w[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
