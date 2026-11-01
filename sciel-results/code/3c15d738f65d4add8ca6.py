"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]
DEADLINE = 115.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
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
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using the block structure.
    # The 2x5x5 tensor can be viewed as a direct sum of two 1x5x5 tensors.
    # However, a 1x5x5 tensor is just a matrix-vector product (or vector-matrix),
    # which is a 5x5 matrix multiplication where one factor is 1x5 and 5x1.
    # The rank of a 1xN x Nx1 multiplication is N (you need N multiplications).
    # So two such blocks would have rank 5+5=10? 
    # Wait, let's re-evaluate.
    # T_2x5x5[i, j, k] = 1 if the indices match the multiplication pattern.
    # Actually, the tensor for 2x5 * 5x5 -> 2x5 is:
    # T[a, b, c] where a in {0..9} (2*5), b in {0..24} (5*5), c in {0..9} (5*2? No, output is 2x5 so c indexes (i, k) where i is row of A, k is col of B. 
    # Let's check the index convention: W[k*n+i, r].
    # n=2, m=5, p=5.
    # A is 2x5 (n x m). B is 5x5 (m x p). C is 2x5 (n x p).
    # C[i, k] = sum_j A[i, j] * B[j, k].
    # Index for U: i*m + j. (0..9)
    # Index for V: j*p + k. (0..24)
    # Index for W: k*n + i. (0..9) -- Wait, the task says W[k*n+i, r].
    # So W index is k*2 + i. Since k is 0..4 and i is 0..1, this covers 0..9.
    # This is a permutation of the standard C[i, k] index (i*p + k).
    # Standard C[i, k] index is i*5 + k.
    # Here W index is k*2 + i.
    # This just permutes the rows of W. It doesn't change the rank.
    
    # The tensor is the direct sum of two 1x5x5 tensors?
    # Let's decompose by i (row of A and C).
    # For a fixed i, we have A[i, :] (1x5) * B (5x5) = C[i, :] (1x5).
    # This is a 1x5 x 5x5 -> 1x5 multiplication.
    # The tensor for this sub-problem has rank 5 (since it's just 5 dot products of length 5? No, 1x5 * 5x5 is 5 scalar multiplications per output, total 5*5=25? No.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # For fixed i, we have 5 outputs (k=0..4), each is a sum of 5 products.
    # This is a matrix multiplication of size 1x5 * 5x5.
    # The rank of 1xN * NxP matrix multiplication is N*P? No, it's N.
    # Wait. The tensor T_{i, j, k} for fixed i is T_{j, k} = A[i, j] B[j, k].
    # This is a bilinear form in A[i, :] and B.
    # Actually, the tensor is T_{(i,j), (j,k), (k,i)}.
    # If we fix i, the indices are (j, j, k).
    # The sub-tensor is T_jk = A_i,j * B_j,k.
    # This is the outer product of vector A_i and matrix B? No.
    # It is the tensor for multiplying a 1x5 matrix by a 5x5 matrix.
    # The rank of 1x5 * 5x5 is 5.
    # Why? Because C_i = A_i * B. This is a linear map from B (25 dims) to C_i (5 dims), linear in A_i (5 dims).
    # Actually, it's bilinear.
    # The tensor is sum_j e_j (in A space) tensor e_j (in B space, but B is 5x5 so it's more complex).
    # Let's use the known rank.
    # Rank of 1xN * NxP is N.
    # Proof: C_k = sum_j A_j B_jk.
    # We can write this as sum_j (A_j) * (B_j .) where B_j . is the j-th row of B.
    # So we need N terms. Each term multiplies a scalar A_j by the vector B_j.
    # But the output is a vector.
    # The decomposition is:
    # U_{(i,j), r} = delta_{j, r}
    # V_{(j,k), r} = delta_{j, r} * B_{j,k}? No, V must be constant.
    # V_{(j,k), r} = delta_{j, r}.
    # W_{(k,i), r} = delta_{j, r}? No, W depends on k and i.
    # Let's check: sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
    # = sum_r delta_{j,r} delta_{j,r} W_{(k,i), r}
    # = delta_{j,j} W_{(k,i), j} = W_{(k,i), j}.
    # We want this to be 1 if the indices match the multiplication, i.e., if we are computing C[i,k] from A[i,j] and B[j,k].
    # The term in the sum is A[i,j] B[j,k].
    # So we need the tensor coefficient to be 1 for the triple (i,j) in U, (j,k) in V, (k,i) in W.
    # Wait, the tensor is T_{(i,j), (j,k), (k,i)}.
    # If we set U_{(i,j), r} = 1 if r=j, V_{(j,k), r} = 1 if r=j, W_{(k,i), r} = 1 if r=j?
    # Then sum_r U V W = sum_r delta_{r,j} delta_{r,j} delta_{r,j} = 1.
    # This works!
    # So for a fixed i, the rank is 5 (j=0..4).
    # Since there are two i's (0 and 1), and they are independent (the indices in U and W are distinct for different i),
    # the total rank is 5 + 5 = 10.
    
    # Let's construct this rank-10 decomposition.
    rank_new = 10
    U_new = [[0] * rank_new for _ in range(n * m)]
    V_new = [[0] * rank_new for _ in range(m * p)]
    W_new = [[0] * rank_new for _ in range(p * n)]
    
    # For i=0, use ranks 0..4 corresponding to j=0..4
    # For i=1, use ranks 5..9 corresponding to j=0..4
    for i in range(n):
        for j in range(m):
            r = i * m + j  # r ranges from 0 to 9
            
            # U index: i*m + j
            U_new[i * m + j][r] = 1
            
            # V index: j*p + k. We need V to be 1 for all k?
            # No, V_{(j,k), r} should be 1 if the "j" part of V matches r's j.
            # In our derivation: V_{(j,k), r} = delta_{j, r_mod_m}?
            # Let's re-verify the sum:
            # sum_r U_{(i,j), r} V_{(j',k), r} W_{(k',i'), r}
            # We want this to be 1 iff j=j', i=i', and k=k'? No.
            # The tensor is T_{(i,j), (j,k), (k,i)}.
            # The indices must match: j from U must equal j from V.
            # k from V must equal k from W? No, k is in V index and W index.
            # i from W must equal i from U.
            
            # My proposed decomposition:
            # r = i*m + j.
            # U_{(i,j), r} = 1 if r == i*m+j.
            # V_{(j,k), r} = 1 if r % m == j. (And implicitly r // m == i? No, V doesn't know i).
            # This is the problem. V does not depend on i.
            # If V_{(j,k), r} = 1 if r % m == j, then for a given j, it is 1 for r=j and r=m+j.
            # Then sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
            # = sum_{r in {j, m+j}} U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}
            # If i=0, U_{(0,j), j}=1, U_{(0,j), m+j}=0.
            # So only r=j contributes.
            # Term: 1 * 1 * W_{(k,0), j}.
            # We need W_{(k,0), j} to be 1.
            # But W_{(k,i), r} index is k*n+i.
            # For i=0, W index is k*2+0 = 2k.
            # We need W_{2k, j} = 1.
            # But W is shared.
            # If we set W_{2k, j} = 1 for all k, then for i=1, we might have interference.
            
            # Let's look at i=1.
            # U_{(1,j), m+j} = 1.
            # V_{(j,k), m+j} = 1 (since (m+j) % m = j).
            # Term: 1 * 1 * W_{(k,1), m+j}.
            # W index for i=1 is k*2+1 = 2k+1.
            # We need W_{2k+1, m+j} = 1.
            
            # So if we define:
            # U_{i*m+j, i*m+j} = 1
            # V_{j*p+k, r} = 1 if r % m == j
            # W_{k*n+i, r} = 1 if r == i*m + (r % m)? No.
            # We need W_{k*n+i, i*m+j} = 1.
            # This means for a fixed i and j, W_{k*n+i, i*m+j} must be 1 for all k.
            # Is this possible?
            # W has dimensions (p*n) x rank.
            # Rows are indexed by k*n+i.
            # Columns by r.
            # We want W_{k*n+i, i*m+j} = 1.
            # This sets specific entries to 1.
            # Does this cause conflicts?
            # For i=0, j=0, we set W_{2k, 0} = 1 for all k.
            # For i=0, j=1, we set W_{2k, 1} = 1 for all k.
            # ...
            # For i=1, j=0, we set W_{2k+1, 5} = 1 for all k.
            # These are all distinct entries in W.
            # So this works!
            
            pass # Just logic check
    
    # Construction:
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U_new[i * m + j][r] = 1
            
            for k in range(p):
                V_new[j * p + k][r] = 1
                
            for k in range(p):
                W_new[k * n + i][r] = 1
    
    # Verify
    U_np = np.array(U_new, dtype=float)
    V_np = np.array(V_new, dtype=float)
    W_np = np.array(W_new, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U_np[a], V_np[b]), W_np[c])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    # Check zeros
    if valid:
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # If (a,b,c) is not a valid multiplication triple, it should be 0.
                    # Valid if a%m == b//p and b%p == c//n and c%n == a//m
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        val = np.dot(np.dot(U_np[a], V_np[b]), W_np[c])
                        if abs(val) > 1e-9:
                            valid = False
                            break
                if not valid: break
            if not valid: break
    
    if valid:
        return {"U": U_new, "V": V_new, "W": W_new}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
