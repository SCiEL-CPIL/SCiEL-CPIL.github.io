import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130  # 20s safety margin before the 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # --- Schoolbook Baseline ---
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    # --- Strassen Hybrid Approach ---
    # We will attempt to construct a decomposition using Strassen for the 4x4x4 part
    # and Schoolbook for the 4x4x3 part.
    
    # Inner Strassen terms for 2x2x2 multiplication
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are lists of (index, value)
    
    inner_terms = [
        # M1 = (x11+x22)(y11+y22)
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # M2 = (x21+x22)y11
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # M3 = x11(y12-y22)
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # M4 = x22(y21-y11)
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # M5 = (x11+x12)y22
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # M6 = (x21-x11)(y11+y12)
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # M7 = (x12-x22)(y21+y22)
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
    ]
    
    # Outer Strassen terms for 4x4x4 (composed of 2x2 blocks)
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22 (and similarly for B, C)
    outer_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11:1, C22:1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # M2 = (A21+A22)B11 -> C21:1, C22:-1
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # M3 = A11(B12-B22) -> C12:1, C22:1
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # M4 = A22(B21-B11) -> C11:1, C21:1
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # M5 = (A11+A12)B22 -> C11:-1, C12:1
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # M6 = (A21-A11)(B11+B12) -> C22:1
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # M7 = (A12-A22)(B21+B22) -> C11:1
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
    ]
    
    # We will build the Strassen part for the first 4 columns of B (k=0..3)
    # and Schoolbook for the last 3 columns (k=4..6)
    
    # Helper to map global indices for the 4x4x4 part
    # A is 4x4. Index (i, j) where i in 0..3, j in 0..3.
    # Block row br = i//2, block col bc = j//2.
    # Local row ir = i%2, local col ic = j%2.
    # Global U index for A(i,j) is i*m + j = 4*i + j.
    
    # B is 4x4 (for the Strassen part). Index (j, k) where j in 0..3, k in 0..3.
    # Block row br = j//2, block col bc = k//2.
    # Local row jr = j%2, local col kc = k%2.
    # Global V index for B(j,k) is j*p + k. 
    # Wait, V index depends on the full p=7. 
    # For the Strassen part, we are only using k=0..3.
    # V index is j*7 + k.
    
    # C is 4x4 (for the Strassen part). Index (k, i) where k in 0..3, i in 0..3.
    # Block row br = k//2, block col bc = i//2.
    # Local row kr = k%2, local col ic = i%2.
    # Global W index for C(k,i) is k*n + i = 4*k + i.
    
    # Let's construct the tensors for the Strassen part.
    # Rank contribution from Strassen part: 7 (outer) * 7 (inner) = 49.
    
    U_strassen = [[0] * 49 for _ in range(n * m)] # 16 x 49
    V_strassen = [[0] * 49 for _ in range(m * p)] # 28 x 49
    W_strassen = [[0] * 49 for _ in range(p * n)] # 28 x 49
    
    for outer_idx in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[outer_idx]
        for inner_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = inner_terms[inner_idx]
            
            term_idx = outer_idx * 7 + inner_idx
            
            # Process U (A side)
            for block_idx, coeff in a_blocks:
                # block_idx 0=A11, 1=A12, 2=A21, 3=A22
                # br = block_idx // 2, bc = block_idx % 2
                br = block_idx // 2
                bc = block_idx % 2
                for local_idx, local_coeff in u_coeffs:
                    # local_idx 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    ir = local_idx // 2
                    ic = local_idx % 2
                    # Global A index: i = br*2 + ir, j = bc*2 + ic
                    i = br * 2 + ir
                    j = bc * 2 + ic
                    U_strassen[i * m + j][term_idx] += coeff * local_coeff
            
            # Process V (B side)
            for block_idx, coeff in b_blocks:
                br = block_idx // 2
                bc = block_idx % 2
                for local_idx, local_coeff in v_coeffs:
                    jr = local_idx // 2
                    kc = local_idx % 2
                    # Global B index: j = br*2 + jr, k = bc*2 + kc
                    j = br * 2 + jr
                    k = bc * 2 + kc
                    V_strassen[j * p + k][term_idx] += coeff * local_coeff
            
            # Process W (C side)
            for block_idx, coeff in c_blocks:
                br = block_idx // 2
                bc = block_idx % 2
                for local_idx, local_coeff in w_coeffs:
                    kr = local_idx // 2
                    ic = local_idx % 2
                    # Global C index: k = br*2 + kr, i = bc*2 + ic
                    k = br * 2 + kr
                    i = bc * 2 + ic
                    W_strassen[k * n + i][term_idx] += coeff * local_coeff
    
    # Schoolbook part for k=4..6
    rank_school_part = n * m * 3 # 4*4*3 = 48
    U_school_part = [[0] * rank_school_part for _ in range(n * m)]
    V_school_part = [[0] * rank_school_part for _ in range(m * p)]
    W_school_part = [[0] * rank_school_part for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U_school_part[i * m + j][col] = 1
                V_school_part[j * p + k][col] = 1
                W_school_part[k * n + i][col] = 1
                col += 1
    
    # Combine
    total_rank = 49 + rank_school_part
    U_hybrid = [[0] * total_rank for _ in range(n * m)]
    V_hybrid = [[0] * total_rank for _ in range(m * p)]
    W_hybrid = [[0] * total_rank for _ in range(p * n)]
    
    for r in range(49):
        for i in range(n * m):
            U_hybrid[i][r] = U_strassen[i][r]
        for j in range(m * p):
            V_hybrid[j][r] = V_strassen[j][r]
        for k in range(p * n):
            W_hybrid[k][r] = W_strassen[k][r]
            
    offset = 49
    for r in range(rank_school_part):
        for i in range(n * m):
            U_hybrid[i][offset + r] = U_school_part[i][r]
        for j in range(m * p):
            V_hybrid[j][offset + r] = V_school_part[j][r]
        for k in range(p * n):
            W_hybrid[k][offset + r] = W_school_part[k][r]
            
    # Verify Hybrid
    # Check if the hybrid decomposition is correct
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Our decomposition: C[i,k] = sum_r U[i,j]_r V[j,k]_r W[k,i]_r ??
    # Wait, the tensor decomposition is T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # where T_{ijk} is 1 if i,j,k form a valid multiplication term?
    # No, the tensor for matrix multiplication is T_{ijk} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is T_{i,j,k} = delta_{i,k}? No.
    # The tensor is defined such that the contraction is sum_{j} A_{ij} B_{jk} = C_{ik}.
    # In tensor notation, T_{i,j,k} is 1 if the indices correspond to the multiplication.
    # Specifically, the rank-1 decomposition is U_{ij,r} V_{jk,r} W_{ki,r}.
    # Summing over r gives T_{i,j,k}.
    # For matrix multiplication, T_{i,j,k} is 1 if the "middle" indices match?
    # Actually, the standard definition is:
    # M_{i,j,k} = 1 if j is the shared index? No.
    # The tensor M is defined by M_{i,j,k} = 1 if the indices i,j,k are such that
    # they participate in the product.
    # Let's check the seed program's logic.
    # U[i*m+j][col] = 1
    # V[j*p+k][col] = 1
    # W[k*n+i][col] = 1
    # This implies the tensor entry T_{i,j,k} is reconstructed as sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # For the schoolbook term (i,j,k), U=1, V=1, W=1.
    # So T_{i,j,k} = 1.
    # But matrix multiplication only requires T_{i,j,k} to be non-zero when?
    # Actually, the tensor for matrix multiplication is T_{i,j,k} = 1 for all i,j,k?
    # No, that would be the identity tensor.
    # The matrix multiplication tensor is defined such that:
    # sum_{i,j} U_{ij,r} A_{ij} * V_{jk,r} B_{jk} ... no.
    
    # Let's verify the correctness of the hybrid decomposition by checking if it matches the schoolbook tensor.
    # The schoolbook tensor has T_{i,j,k} = 1 for all i,j,k.
    # Wait, is T_{i,j,k} = 1 for all i,j,k?
    # If A is n x m and B is m x p, C is n x p.
    # C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    # The tensor T is of size n x m x p.
    # T_{i,j,k} = 1.
    # Yes, the matrix multiplication tensor is the all-ones tensor in this context?
    # No, that's not right. The tensor is fixed.
    # T_{i,j,k} = 1 if the indices are valid.
    # In the decomposition U, V, W, we are decomposing the tensor T.
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # If T_{i,j,k} = 1 for all i,j,k, then we are decomposing the all-ones tensor.
    # But matrix multiplication is not just the all-ones tensor.
    # The tensor is defined by the structure of the multiplication.
    # Actually, the tensor IS the all-ones tensor if we consider the indices correctly?
    # Let's check:
    # If we set U_{i,j,r} = delta_{i, i_r} delta_{j, j_r}, etc.
    # The standard "schoolbook" decomposition uses rank n*m*p.
    # For each (i,j,k), we have a term.
    # U_{i,j,r} = 1 if r corresponds to (i,j).
    # V_{j,k,r} = 1 if r corresponds to (j,k).
    # W_{k,i,r} = 1 if r corresponds to (k,i).
    # Sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # This sum is 1 only if there is an r such that r matches (i,j), (j,k), and (k,i).
    # In the schoolbook case, r is the tuple (i,j,k).
    # So U_{i,j,r} = 1 if r=(i,j,k).
    # V_{j,k,r} = 1 if r=(i,j,k).
    # W_{k,i,r} = 1 if r=(i,j,k).
    # So the sum is 1.
    # Thus, the tensor T_{i,j,k} = 1 for all i,j,k.
    
    # So the task is to decompose the all-ones tensor of size n x m x p.
    # Wait, is that true?
    # If T_{i,j,k} = 1 for all i,j,k, then the rank is 1?
    # No, because the indices in U, V, W are coupled.
    # U is indexed by (i,j). V by (j,k). W by (k,i).
    # The tensor is T_{i,j,k}.
    # If T_{i,j,k} = 1, then we can write T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # Can we do this with rank 1?
    # U_{ij,1} = 1, V_{jk,1} = 1, W_{ki,1} = 1.
    # Sum = 1.
    # So the rank would be 1?
    # That can't be right. Matrix multiplication has rank n*m*p in the naive case, and lower with Strassen.
    # Where is the mistake?
    
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor is defined such that the contraction yields the product.
    # Let's look at the definition again.
    # The tensor T is a 3-way array.
    # T_{i,j,k} is the coefficient of the term A_{i,j} B_{j,k} in the expression for C_{?,?}.
    # Actually, the tensor is defined by T_{i,j,k} = 1 if the indices form a valid "triangle" in the multiplication graph?
    # No.
    # Let's look at the seed code again.
    # U[i*m+j][col] = 1
    # V[j*p+k][col] = 1
    # W[k*n+i][col] = 1
    # This constructs a tensor where T_{i,j,k} = 1.
    # If the seed code is correct, then the tensor IS the all-ones tensor.
    # But then why is the rank not 1?
    # Because the indices of U, V, W are not independent.
    # U is a matrix of size (n*m) x rank.
    # V is (m*p) x rank.
    # W is (p*n) x rank.
    # The tensor T is of size n x m x p.
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # If we set U_{i*m+j, 1} = 1, V_{j*p+k, 1} = 1, W_{k*n+i, 1} = 1 for all i,j,k,
    # then T_{i,j,k} = 1.
    # So the rank is 1?
    
    # This implies my understanding of the tensor is wrong.
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # It is a sparse tensor?
    # No, the tensor is fixed.
    # Let's check the literature.
    # The matrix multiplication tensor M_{n,m,p} is defined by M_{i,j,k} = 1 if i,j,k are such that...
    # Actually, the tensor is defined by the indices.
    # M_{i,j,k} = 1.
    # But the rank is not 1.
    # Why?
    # Because the decomposition must be valid for ALL matrices A and B.
    # The tensor T is the "structure tensor" of the multiplication.
    # T_{i,j,k} = 1.
    # If T is the all-ones tensor, its rank is 1.
    # But matrix multiplication rank is > 1.
    
    # Ah! The tensor is NOT the all-ones tensor.
    # The tensor is defined by T_{i,j,k} = 1 if the indices are distinct? No.
    # Let's look at the indices in the seed code.
    # U index: i*m + j.
    # V index: j*p + k.
    # W index: k*n + i.
    # The tensor entry is T_{i,j,k}.
    # If we use rank 1 with all 1s, we get T_{i,j,k} = 1.
    # So the tensor IS the all-ones tensor.
    # But then the rank is 1.
    
    # There must be a constraint I am missing.
    # Maybe the tensor is not all-ones.
    # Let's check the definition of the matrix multiplication tensor.
    # The tensor is M_{n,m,p} in V^{n*m} * V^{m*p} * V^{p*n}.
    # The components are M_{i,j,k} = 1.
    # Wait, if M_{i,j,k} = 1, then the rank is 1.
    # This is a well-known fact: the tensor of matrix multiplication is NOT the all-ones tensor.
    # The all-ones tensor has rank 1.
    # The matrix multiplication tensor has rank at least n*m*p? No, Strassen shows it's lower.
    # But it's definitely not 1.
    
    # Let's re-read the tensor definition.
    # The tensor is defined by the coefficients of the monomials in the product.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The term A_{i,j} B_{j,k} appears in C_{i,k}.
    # So the tensor component T_{i,j,k} corresponds to the coefficient of A_{i,j} B_{j,k} in C_{?,?}.
    # But C is indexed by i,k.
    # So for a fixed (i,j,k), the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So the tensor should be T_{i,j,k} = 1 if it contributes to C_{i,k}.
    # But the tensor is 3-indexed: i, j, k.
    # And the output is 2-indexed: i, k.
    # So the tensor T is a map from (i,j,k) to something?
    # No, the tensor is a 3-way array T_{i,j,k}.
    # And the decomposition is T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # If T_{i,j,k} = 1 for all i,j,k, then rank 1 works.
    
    # I must be misunderstanding the problem statement or the tensor definition.
    # Let's look at the inspiration 2 code.
    # It constructs U, V, W with specific coefficients.
    # It does NOT use all 1s.
    # This implies T_{i,j,k} is NOT 1 for all i,j,k.
    # Or, my rank-1 argument is flawed.
    # Why is rank 1 flawed?
    # U_{ij,1} = 1, V_{jk,1} = 1, W_{ki,1} = 1.
    # Sum = 1.
    # It seems correct.
    
    # Is it possible that the tensor is NOT the all-ones tensor?
    # Let's check the indices again.
    # U is (n*m) x rank.
    # V is (m*p) x rank.
    # W is (p*n) x rank.
    # The tensor is T_{i,j,k}.
    # i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # If T_{i,j,k} = 1, then rank 1 works.
    
    # Maybe the tensor is T_{i,j,k} = 1 if i,j,k are such that...
    # Actually, I recall now. The tensor for matrix multiplication is NOT the all-ones tensor.
    # The all-ones tensor is the tensor for the "sum of all products" which is not matrix multiplication.
    # Matrix multiplication is a specific bilinear map.
    # The tensor is defined by the basis of the space of bilinear forms.
    # The space of bilinear forms from V^n x V^m to V^p? No.
    # From V^{n*m} x V^{m*p} to V^{p*n}?
    # The tensor is an element of V^{n*m} * V^{m*p} * V^{p*n}.
    # The component T_{a,b,c} where a is index in V^{n*m}, b in V^{m*p}, c in V^{p*n}.
    # a = i*m + j.
    # b = j*p + k.
    # c = k*n + i.
    # T_{a,b,c} = 1 if a,b,c are consistent?
    # i.e., if the j in a matches the j in b, and the k in b matches the k in c, and the i in c matches the i in a.
    # If so, T_{a,b,c} = 1 only for specific triples (a,b,c).
    # Specifically, for each (i,j,k), there is exactly one triple (a,b,c) = (i*m+j, j*p+k, k*n+i) such that T_{a,b,c} = 1.
    # And T_{a,b,c} = 0 otherwise.
    # So the tensor is NOT the all-ones tensor. It is a sparse tensor!
    # It has n*m*p non-zero entries, each being 1.
    # And these entries are "diagonal" in a sense.
    
    # So the tensor T is defined by:
    # T_{i*m+j, j*p+k, k*n+i} = 1 for all i,j,k.
    # T_{a,b,c} = 0 otherwise.
    
    # Now, can we decompose this with rank 1?
    # U_{a,1} = 1, V_{b,1} = 1, W_{c,1} = 1.
    # Sum_r U_{a,r} V_{b,r} W_{c,r} = 1 for all a,b,c.
    # But T_{a,b,c} is 0 for most (a,b,c).
    # So rank 1 does NOT work.
    # We need U_{a,r} V_{b,r} W_{c,r} to be 0 for most (a,b,c).
    # This requires the vectors to be orthogonal or something.
    
    # Okay, so the tensor is the "permutation" tensor or similar.
    # The schoolbook decomposition uses rank n*m*p.
    # For each (i,j,k), we have a term r=(i,j,k).
    # U_{i*m+j, r} = 1 if r=(i,j,k), else 0.
    # V_{j*p+k, r} = 1 if r=(i,j,k), else 0.
    # W_{k*n+i, r} = 1 if r=(i,j,k), else 0.
    # Sum_r U_{a,r} V_{b,r} W_{c,r}.
    # If a=i*m+j, b=j*p+k, c=k*n+i, then only r=(i,j,k) contributes, giving 1.
    # If a,b,c do not correspond to a valid (i,j,k), then no r contributes, giving 0.
    # So the schoolbook decomposition works.
    
    # Now, does the Strassen hybrid work?
    # The Strassen part decomposes the sub-tensor for k=0..3.
    # The sub-tensor T' has entries T'_{i*m+j, j*p+k, k*n+i} = 1 for k=0..3.
    # Strassen decomposes the 4x4x4 multiplication tensor.
    # The 4x4x4 tensor is exactly T' (with p=4).
    # So the Strassen decomposition is valid for T'.
    # The schoolbook part decomposes the sub-tensor for k=4..6.
    # So the hybrid should be valid.
    
    # Let's verify the hybrid decomposition.
    # We need to check if the sum of the Strassen part and the schoolbook part equals the full tensor.
    # The Strassen part handles k=0..3.
    # The schoolbook part handles k=4..6.
    # They are disjoint in the k dimension.
    # So they should sum to the full tensor.
    
    # However, we must ensure that the Strassen decomposition is correct for the 4x4x4 case.
    # The Strassen algorithm is known to be correct for 2x2x2 and 4x4x4.
    # So the hybrid should be correct.
    
    # Let's verify the rank.
    # Strassen part: 49.
    # Schoolbook part: 48.
    # Total: 97.
    # Schoolbook total: 112.
    # So 97 < 112.
    # This is an improvement.
    
    # Let's check if the hybrid is valid.
    # We can do a quick check with random matrices.
    
    # Generate random A (4x4) and B (4x7).
    # Compute C = A @ B.
    # Compute C_hybrid using the decomposition.
    # C_hybrid[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # Wait, the decomposition is T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # The product is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Using the decomposition:
    # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j A_{i,j} B_{j,k} * 1
    # = sum_j A_{i,j} B_{j,k} * T_{i,j,k}  (since T_{i,j,k}=1 for valid indices)
    # = sum_j A_{i,j} B_{j,k} * sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # = sum_r W_{ki,r} * (sum_j A_{i,j} U_{ij,r}) * (sum_j B_{j,k} V_{jk,r})
    # Let U_r = sum_j A_{i,j} U_{ij,r} (depends on i,r)
    # Let V_r = sum_j B_{j,k} V_{jk,r} (depends on k,r)
    # Then C_{i,k} = sum_r W_{ki,r} U_r V_r.
    
    # Let's test this.
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    # Compute C_hybrid
    C_hybrid = np.zeros((n, p))
    for r in range(total_rank):
        # U_r is a vector of size n*m? No, U is (n*m) x rank.
        # U[:, r] is a vector of size n*m.
        # We need to compute sum_j A[i,j] U[i*m+j, r].
        # This is like a tensor contraction.
        # Let's do it simply.
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    val += A[i,j] * U_hybrid[i*m+j, r]
                # Now we have U_r for this i.
                # But V_r depends on k.
                # This is inefficient.
                pass
    
    # Better way:
    # C_hybrid[i,k] = sum_r W_hybrid[k*n+i, r] * (sum_j A[i,j] U_hybrid[i*m+j, r]) * (sum_j B[j,k] V_hybrid[j*p+k, r])
    
    # Let's precompute the contractions.
    # AU[i, r] = sum_j A[i,j] U_hybrid[i*m+j, r]
    AU = np.zeros((n, total_rank))
    for i in range(n):
        for r in range(total_rank):
            s = 0
            for j in range(m):
                s += A[i,j] * U_hybrid[i*m+j, r]
            AU[i, r] = s
            
    # BV[k, r] = sum_j B[j,k] V_hybrid[j*p+k, r]
    BV = np.zeros((p, total_rank))
    for k in range(p):
        for r in range(total_rank):
            s = 0
            for j in range(m):
                s += B[j,k] * V_hybrid[j*p+k, r]
            BV[k, r] = s
            
    # C_hybrid[i,k] = sum_r W_hybrid[k*n+i, r] * AU[i, r] * BV[k, r]
    for i in range(n):
        for k in range(p):
            s = 0
            for r in range(total_rank):
                s += W_hybrid[k*n+i, r] * AU[i, r] * BV[k, r]
            C_hybrid[i,k] = s
            
    # Check if C_hybrid == C_true
    if np.allclose(C_hybrid, C_true):
        # The hybrid is valid!
        if total_rank < best["rank"]:
            best = {"U": U_hybrid, "V": V_hybrid, "W": W_hybrid, "rank": total_rank}
    else:
        # The hybrid is invalid. Stick with schoolbook.
        pass
        
    # Check time
    if time.time() > deadline:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
        
    # Try other optimizations if time permits?
    # For now, return the best found.
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
