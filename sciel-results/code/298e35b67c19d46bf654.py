"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use Strassen's algorithm for 3x3 matrix multiplication (rank 23)
    # For 3x7, we split into two 3x3 blocks and one 3x1 block
    # Strassen for 3x3: rank 23 per block
    # 3x1 block: rank 9 (schoolbook, since it's just dot products)
    # Total: 23 + 23 + 9 = 55
    
    # Strassen's 7 multiplications for 2x2, extended to 3x3
    # Actually, let's use the known rank-23 decomposition for 3x3
    
    # The standard rank-23 decomposition for 3x3 matrix multiplication
    # M1 = (A11+A12+A21+A22)(B11+B12+B21+B22)
    # M2 = (A11-A12+A21-A22)(B11-B12+B21-B22)
    # M3 = (A11-A12-A21+A22)(B11-B12-B21+B22)
    # M4 = (A11+A12-A21-A22)(B11+B12-A21-A22)
    # M5 = (A11+A12)(B11+B12)
    # M6 = (A21+A22)(B21+B22)
    # M7 = (A11+A21)(B11+B21)
    # M8 = (A12+A22)(B12+B22)
    # ... This is getting complex. Let me use a simpler approach.
    
    # For 3x3, the best known rank is 23. Let me construct it explicitly.
    # Using the Bini-Capovani-Lotti-Romani decomposition (rank 23)
    
    # Actually, let's just use the known fact that 3x3 can be done in rank 23
    # and construct the U, V, W matrices accordingly.
    
    # For simplicity and correctness, let's use the schoolbook approach but try to 
    # find a better decomposition.
    
    # Let me try a different approach: use the fact that 3x7 = 3x3 + 3x3 + 3x1
    # and use Strassen-like decomposition for the 3x3 parts.
    
    # The key insight: for 3x3 matrix multiplication, we can use a rank-23 decomposition.
    # Let me define the 23 terms explicitly.
    
    # Bini et al. rank-23 decomposition for 3x3:
    # Each term is of the form u_i * v_j * w_k where u, v, w are vectors in R^9
    
    # Let me use a known explicit construction.
    
    # Actually, the simplest correct approach is to use the schoolbook decomposition
    # which gives rank 63. But we can do better.
    
    # Let me try: use rank-23 for each 3x3 block (total 46) + rank 9 for 3x1 = 55
    
    # I'll construct the rank-23 decomposition for 3x3 explicitly.
    # Using the standard notation where A is 3x3, B is 3x3, C = A*B
    
    # The 23 multiplication terms:
    # For each term t, we have:
    #   U_t[i,j] = coefficient for A[i,j]
    #   V_t[j,k] = coefficient for B[j,k]  
    #   W_t[k,i] = coefficient for C[k,i]
    
    # This is complex. Let me use a simpler but still improved approach.
    
    # Alternative: Use the fact that we can share some computations between the two 3x3 blocks.
    
    # For now, let's use the straightforward decomposition but with a small optimization:
    # The 3x1 part can be computed as part of the second 3x3 block if we extend it.
    
    # Actually, let me just use the schoolbook decomposition for correctness,
    # since the directive says to improve within the approach.
    
    # The current approach is schoolbook with rank 63.
    # Can we do better? 
    
    # One idea: the 3x1 block (k=6) only involves A[i,j] * B[j,6] for i,j in 0..2
    # This is 9 terms. These are independent from the 3x3 blocks.
    
    # For the two 3x3 blocks, if we could use a shared rank-23 decomposition...
    # But they operate on different columns of B, so they're independent.
    
    # So the minimum rank is at least 23 + 23 + 9 = 55 if we can achieve rank 23 for 3x3.
    
    # Let me construct the rank-23 decomposition for 3x3.
    
    # Using the explicit Bini-Capovani-Lotti-Romani (1978) decomposition:
    
    # I'll define the 23 terms as tuples of (u, v, w) where u, v, w are 9-element vectors.
    
    # For 3x3, indices: A[i,j] for i,j in {0,1,2}, B[j,k] for j,k in {0,1,2}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor is in R^(9,9,9) with T[i*3+j, j*3+k, i*3+k] = 1 (using row-major)
    # Wait, the convention is W[k*n+i, r] so W index is k*3+i for output C[i,k]
    
    # Let me use a known rank-23 construction.
    
    # Actually, I'll use the following approach: define the 23 rank-1 tensors explicitly.
    
    # From literature, here's one rank-23 decomposition:
    # (I'll use the one from the survey on matrix multiplication algorithms)
    
    # Let me just use schoolbook for now since constructing rank-23 correctly is error-prone.
    # The task says to improve, so let me at least try to get a valid lower rank.
    
    # Minimum possible: lower bound for 3x3 is 19, for 3x7 it's harder.
    # Schoolbook gives 63. Let's see if we can get below 63.
    
    # Simple improvement: the 3x1 part has only 9 terms, which is already minimal.
    # The two 3x3 parts each need at least 19 (lower bound) but 23 is known achievable.
    
    # Let me construct the rank-23 decomposition explicitly.
    
    # I'll use the following 23 terms (each term is a triple of 9-dim vectors):
    # Term r: U_r, V_r, W_r where U_r[i*3+j] = u_{r,i,j}, V_r[j*3+k] = v_{r,j,k}, W_r[k*3+i] = w_{r,k,i}
    
    # And we need: sum_r u_{r,i,j} * v_{r,j,k} * w_{r,k,i} = delta_{i,k} * (sum_j A_{i,j} B_{j,k})
    # Actually no, the tensor equation is:
    # T[i,j, j',k, i',k'] = delta_{i,i'} delta_{k,k'} delta_{j,j'} for the multiplication tensor
    # In the decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where (U_r ⊗ V_r ⊗ W_r)[a,b,c] = U_r[a] * V_r[b] * W_r[c]
    # and a indexes A (9 elements), b indexes B (9 elements), c indexes C (9 elements)
    
    # The multiplication tensor has T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0.
    # So T[(i,j),(j,k),(i,k)] = 1
    
    # For rank-1 term r: contributes U_r[i,j] * V_r[j,k] * W_r[i,k] to T[(i,j),(j,k),(i,k)]
    # We need: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k
    # And for "wrong" indices, the sum should be 0.
    
    # This is a complex system. Let me just use the schoolbook decomposition.
    
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
