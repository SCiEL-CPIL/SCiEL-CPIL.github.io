"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    rank_trivial = n * m * p  # 105

    # Build the trivial decomposition as fallback
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, rank):
        """Verify tensor decomposition T_{i,j,k} = sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r}"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    # Try to find a lower rank decomposition
    # Strategy: Use the structure that T_{i,j,k} = 1 for all (i,j,k)
    # We can write this as a sum of rank-1 tensors
    
    # Approach: Group by j. For each j, the sub-tensor over (i,k) is all ones (n×p).
    # An n×p all-ones matrix has rank 1, so it can be written as u * v^T where u is n×1, v is p×1.
    # So T = sum_j (e_j ⊗ u_j ⊗ v_j) where e_j is the j-th basis vector in R^m
    
    # But this still gives m * (rank of all-ones n×p) = m * 1 = m terms if we could use rank-1 in (i,k)
    # However, in the 3D tensor sense, each term e_j ⊗ u_j ⊗ v_j is rank-1 in the 3D sense.
    # So we need m terms? No, because u_j and v_j can be the same for all j.
    
    # Actually: T_{i,j,k} = 1 = (1) * (1) * (1) for all i,j,k
    # This is already rank 1! But wait, that's not right for the tensor structure.
    
    # Let me reconsider. The tensor T has T_{i,j,k} = 1 for all valid indices.
    # A rank-1 tensor would be T_{i,j,k} = a_i * b_j * c_k.
    # If a_i = 1, b_j = 1, c_k = 1 for all i,j,k, then T_{i,j,k} = 1.
    # So the all-ones tensor IS rank 1!
    
    # But wait, that can't be right for matrix multiplication. Let me re-read the problem.
    
    # The matrix multiplication tensor: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The bilinear map is (A, B) -> C, and the tensor T satisfies:
    # C_{i,k} = sum_{j} sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r} A_{i,j} B_{j,k}
    # So T_{i,j,k} should be such that the decomposition reproduces matrix multiplication.
    
    # Actually, the tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor entries T_{i,j,k} represent the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # For standard matrix multiplication, C_{i,k} = sum_j A_{i,j} B_{j,k}, so
    # the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1 if the indices match, else 0.
    # But in the tensor decomposition, T_{i,j,k} is defined such that:
    # sum_j A_{i,j} B_{j,k} = sum_r (sum_j U_{i*m+j,r} A_{i,j}) (sum_k V_{j*p+k,r} B_{j,k}) W_{k*n+i,r}
    # Wait, that's not quite right either.
    
    # Let me use the standard definition: The matrix multiplication tensor M_{n,m,p} is a 3-tensor
    # with entries M_{i,j,k} = 1 if the indices are consistent with matrix multiplication, else 0.
    # Specifically, for C = AB, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is indexed by (i, j, k) where i is row of A/C, j is column of A/row of B, k is column of B/C.
    # The entry M_{i,j,k} = 1 for all valid (i,j,k) because each term A_{i,j} B_{j,k} contributes to C_{i,k}.
    
    # So actually, the tensor IS the all-ones tensor for valid indices!
    # And the all-ones tensor of size n×m×p has rank 1 (since it's 1*1*1).
    
    # But that would mean rank 1 decomposition exists, which is clearly wrong.
    # The issue is that in the tensor decomposition for bilinear maps, the structure is different.
    
    # Let me look at this more carefully. The bilinear map is:
    # (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This can be written as: C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # In tensor form: C = T(A, B) where T is the tensor.
    # The rank of T is the minimum number of rank-1 tensors needed to represent T.
    
    # For matrix multiplication, it's known that the rank is at least max(nm, mp, pn) and at most nmp.
    # For n=3, m=5, p=7: lower bound = 35, upper bound = 105.
    
    # The all-ones tensor interpretation is wrong because the tensor structure for bilinear maps
    # has a specific indexing that prevents the trivial rank-1 interpretation.
    
    # Let me just use the trivial decomposition and try to improve it with a search.
    
    # Try rank r from 35 to 104, and for each, try to find a solution
    # using a simple iterative method
    
    import numpy as np
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_trivial
    
    # Try a few candidate ranks
    for r in range(35, min(60, rank_trivial)):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random values
        np.random.seed(42 + r)
        U_r = np.random.randn(n * m, r) * 0.1
        V_r = np.random.randn(m * p, r) * 0.1
        W_r = np.random.randn(p * n, r) * 0.1
        
        # Iterative refinement: for each (i,j,k), adjust to satisfy the constraint
        # This is a non-convex problem, so use a simple alternating minimization
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Fix U and V, solve for W, etc.
            
            # Update W: for each (k,i), W[k*n+i,:] should satisfy
            # sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for each i,k
            # This is: sum_j (U[i*m+j,:] * V[j*p+k,:]) dot W[k*n+i,:] = 1
            # Let A_ki = sum_j U[i*m+j,:] * V[j*p+k,:] (element-wise product, summed over j)
            # Then A_ki dot W[k*n+i,:] = 1
            # This is a linear system for each (k,i)
            
            for i in range(n):
                for k in range(p):
                    # Compute the vector A = sum_j U[i*m+j,:] * V[j*p+k,:]
                    A = np.zeros(r)
                    for j in range(m):
                        A += U_r[i*m+j, :] * V_r[j*p+k, :]
                    # Solve A dot W[k*n+i,:] = 1
                    if np.linalg.norm(A) < 1e-10:
                        continue
                    # Least squares solution
                    W_r[k*n+i, :] = np.linalg.lstsq(A.reshape(1, -1), np.array([1.0]), rcond=None)[0]
            
            # Update U: for each (i,j), U[i*m+j,:] should satisfy
            # sum_k sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for each i,j
            # Let B_ij = sum_k V[j*p+k,:] * W[k*n+i,:]
            # Then B_ij dot U[i*m+j,:] = 1
            for i in range(n):
                for j in range(m):
                    B = np.zeros(r)
                    for k in range(p):
                        B += V_r[j*p+k, :] * W_r[k*n+i, :]
                    if np.linalg.norm(B) < 1e-10:
                        continue
                    U_r[i*m+j, :] = np.linalg.lstsq(B.reshape(1, -1), np.array([1.0]), rcond=None)[0]
            
            # Update V: for each (j,k), V[j*p+k,:] should satisfy
            # sum_i sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for each j,k
            # Let C_jk = sum_i U[i*m+j,:] * W[k*n+i,:]
            # Then C_jk dot V[j*p+k,:] = 1
            for j in range(m):
                for k in range(p):
                    C = np.zeros(r)
                    for i in range(n):
                        C += U_r[i*m+j, :] * W_r[k*n+i, :]
                    if np.linalg.norm(C) < 1e-10:
                        continue
                    V_r[j*p+k, :] = np.linalg.lstsq(C.reshape(1, -1), np.array([1.0]), rcond=None)[0]
        
        # Check if this decomposition is valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                    if abs(val - 1) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to lists
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            if verify(U_list, V_list, W_list, r):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = r
                break  # Found a valid decomposition, stop
    
    if best_rank < rank_trivial:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
