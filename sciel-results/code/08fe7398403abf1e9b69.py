import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 4
    rank_schoolbook = n * m * p
    deadline = time.time() + 100.0
    
    # 1. Schoolbook Decomposition (Rank 24)
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # 2. Heuristic Search for Rank 23
    # This is a structural addition: attempting to find a lower rank.
    # Given the difficulty, this is likely to fail, but it's a new direction.
    target_rank = rank_schoolbook - 1
    search_time_limit = 5.0 # Seconds
    start_search = time.time()
    
    # We will try a few random sparse initializations and check for exactness.
    # This is extremely unlikely to work for rank 23 of M_234, but it's a valid search strategy.
    while time.time() - start_search < search_time_limit:
        # Generate random sparse matrices
        # Density 0.1
        U_rand = (np.random.rand(n * m, target_rank) < 0.1).astype(float)
        V_rand = (np.random.rand(m * p, target_rank) < 0.1).astype(float)
        W_rand = (np.random.rand(p * n, target_rank) < 0.1).astype(float)
        
        # Check if U * V * W reconstructs the tensor?
        # The tensor T is defined by T[i,j,k] = 1 if (i,j,k) is a valid multiplication term.
        # Actually, the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
        # We need to check if the sum of outer products matches the target tensor.
        
        # Construct target tensor (flattened)
        # T[i, j, k] = 1 if the term exists.
        # In the schoolbook, each (i,j,k) is a separate term.
        # So the target tensor has 24 ones at specific positions.
        
        # Let's check the reconstruction of a few random slices or the full tensor.
        # Full tensor check is 2*3*4*23 multiplications. Very fast.
        
        # Target tensor T (2, 3, 4)
        # T[i,j,k] is 1 if this specific multiplication is part of the sum?
        # No, the tensor for matrix multiplication is such that 
        # sum_{j} A[i,j] B[j,k] = C[i,k].
        # The tensor M satisfies <M, A \otimes B> = tr(C^T ...)?
        # The standard tensor M_{n,m,p} has entries M_{i,j,k} = 1 if the "triplet" (i,j,k) is valid?
        # No, the tensor is the identity of the multiplication map.
        # The map is A, B -> C.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T has indices (i, j, k).
        # T_{i,j,k} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
        # Since every A_{ij}B_{jk} contributes to C_{ik}, T_{i,j,k} = 1 for all i,j,k?
        # No, that would be a rank 1 tensor? No.
        
        # Let's look at the schoolbook decomposition provided in the seed.
        # U[i*m+j, col] = 1
        # V[j*p+k, col] = 1
        # W[k*n+i, col] = 1
        # This corresponds to a rank-1 tensor with non-zero entry at (i*m+j, j*p+k, k*n+i).
        # The sum of these rank-1 tensors forms the full tensor.
        # The full tensor T has T[a,b,c] = 1 if there exists a col such that U[a,col]=1, V[b,col]=1, W[c,col]=1.
        # In schoolbook, each col is unique.
        # So T is a tensor with 1s at positions (i*m+j, j*p+k, k*n+i).
        
        # To verify a candidate (U,V,W) of rank R:
        # Compute S = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
        # Check if S == T_target.
        
        # T_target is sparse.
        # We can check S by computing the tensor product sum.
        
        # Efficient check:
        # For each r, the contribution is sparse.
        # But summing R sparse tensors is easy.
        
        # Let's just compute the full tensor S (2*3*4 = 24 elements) and compare to T_target.
        
        # Initialize S as zeros
        S = np.zeros((n, m, p))
        
        # Add contributions from each rank-1 term
        # This is slow if done in Python loops.
        # Use einsum?
        # S = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand)
        
        S_candidate = np.einsum('ir,jr,kr->ijk', U_rand, V_rand, W_rand)
        
        # Check against target
        # Target T has 1s at (i,j,k) for all i,j,k?
        # No, in the schoolbook, the indices are mapped.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # So T_target[u, v, w] = 1 if there exists i,j,k such that u=i*m+j, v=j*p+k, w=k*n+i.
        # Note that for a given i,j,k, the indices are fixed.
        # Are there collisions?
        # i in 0..1, j in 0..2, k in 0..3.
        # u in 0..5, v in 0..11, w in 0..7.
        # The mapping (i,j,k) -> (u,v,w) is injective?
        # u determines i, j.
        # v determines j, k.
        # w determines k, i.
        # So yes, it's a bijection between the 24 triples and the non-zero entries.
        
        # So T_target has 1s at these 24 positions.
        
        # Create T_target
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1
        
        if np.allclose(S_candidate, T_target):
            # Found exact solution!
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            best_rank = target_rank
            break

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
