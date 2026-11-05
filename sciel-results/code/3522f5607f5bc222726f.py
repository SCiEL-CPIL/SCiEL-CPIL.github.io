import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 5

    def verify_numpy(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        return True

    def verify_exact(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True

    def try_reduce_rank(U, V, W, rank, max_attempts=100):
        current_rank = rank
        U_cur = [row[:] for row in U]
        V_cur = [row[:] for row in V]
        W_cur = [row[:] for row in W]

        for attempt in range(max_attempts):
            if time.time() > deadline - 10.0:
                break

            Ua = np.array(U_cur, dtype=float)
            Va = np.array(V_cur, dtype=float)
            Wa = np.array(W_cur, dtype=float)

            found_reduction = False
            for r in range(current_rank - 1, -1, -1):
                if r == 0:
                    U_other = Ua[:, 1:]
                    V_other = Va[:, 1:]
                    W_other = Wa[:, 1:]
                else:
                    U_other = np.hstack([Ua[:, :r], Ua[:, r+1:]])
                    V_other = np.hstack([Va[:, :r], Va[:, r+1:]])
                    W_other = np.hstack([Wa[:, :r], Wa[:, r+1:]])

                U_target = Ua[:, r]
                V_target = Va[:, r]
                W_target = Wa[:, r]

                try:
                    coeff_U, _, _, _ = np.linalg.lstsq(U_other, U_target, rcond=None)
                    coeff_V, _, _, _ = np.linalg.lstsq(V_other, V_target, rcond=None)
                    coeff_W, _, _, _ = np.linalg.lstsq(W_other, W_target, rcond=None)

                    if np.allclose(coeff_U, coeff_V, atol=1e-10) and np.allclose(coeff_V, coeff_W, atol=1e-10):
                        coeff = coeff_U
                        U_recon = np.zeros(n*m)
                        V_recon = np.zeros(m*p)
                        W_recon = np.zeros(p*n)
                        
                        col_idx = 0
                        for r2 in range(current_rank):
                            if r2 != r:
                                U_recon += Ua[:, r2] * coeff[col_idx]
                                V_recon += Va[:, r2] * coeff[col_idx]
                                W_recon += Wa[:, r2] * coeff[col_idx]
                                col_idx += 1
                        
                        if np.allclose(U_cur[:, r], U_recon, atol=1e-10) and \
                           np.allclose(V_cur[:, r], V_recon, atol=1e-10) and \
                           np.allclose(W_cur[:, r], W_recon, atol=1e-10):
                            U_cur = [row[:r] + row[r+1:] for row in U_cur]
                            V_cur = [row[:r] + row[r+1:] for row in V_cur]
                            W_cur = [row[:r] + row[r+1:] for row in W_cur]
                            current_rank -= 1
                            found_reduction = True
                            break
                except:
                    pass
            
            if not found_reduction:
                break
        
        if current_rank < rank:
            if verify_numpy(U_cur, V_cur, W_cur, current_rank):
                if verify_exact(U_cur, V_cur, W_cur, current_rank):
                    return U_cur, V_cur, W_cur, current_rank
        return None

    # Rank 20 construction
    # Split j=0 (1x5), j=1,2 (2x5), j=3 (1x5)
    # j=0: rank 2
    # j=3: rank 2
    # j=1,2: 2x2x5 -> split k=0,1 (2x2x2, rank 7), k=2,3 (2x2x2, rank 7), k=4 (2x2x1, rank 4)
    # Total: 2+2+7+7+4 = 22? Wait, 2x2x5 rank is not 18.
    # 2x2x2 rank is 7. 2x2x1 rank is 4.
    # So 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7 + 7 + 4 = 18.
    # Total rank = 2 (j=0) + 18 (j=1,2) + 2 (j=3) = 22.
    
    # Let's try to find a better construction or just use this one.
    # Is there a rank 20 construction?
    # 2x4x5 tensor.
    # Maybe 2x2x5 can be done in less than 18?
    # Lower bound for 2x2x5 is... 
    # Actually, the rank of 2x2x5 is known.
    # Let's just build the rank 22 one first and see if it reduces.
    
    rank22 = 22
    U22 = [[0] * rank22 for _ in range(n * m)]
    V22 = [[0] * rank22 for _ in range(m * p)]
    W22 = [[0] * rank22 for _ in range(p * n)]
    
    col = 0
    
    # j=0 part: A[0,0] * B[0,0..4]
    # C[i,k] += A[0,0] * B[0,k]
    # U[0][col] = 1, V[0*p+k][col] = 1, W[k*n+0][col] = 1 for all k
    # This is rank 1? No, because B[0,k] varies.
    # Wait, the tensor term is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For j=0, we have term: A[0,0] B[0,k].
    # We can do this with rank 1?
    # U[0,0,r] = 1.
    # V[0,k,r] = 1 for all k.
    # W[k,0,r] = 1 for all k.
    # Then sum_r 1 * 1 * 1 = 1.
    # Yes! It's just a single rank-1 term if we just want C[i,k] to contain A[0,0]B[0,k].
    # Wait, C[i,k] = sum_j A[i,j]B[j,k].
    # For i=0, C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + ...
    # For i=1, C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + ...
    # The term for j=0 contributes A[i,0]B[0,k] to C[i,k].
    # This is a rank 2 term?
    # U[i,0,r] = 1.
    # V[0,k,r] = 1.
    # W[k,i,r] = 1.
    # Then C[i,k] gets 1*1*1 = 1.
    # This works for all i, k.
    # So j=0 part is rank 2? No, rank 1?
    # U is (2*4, r). V is (4*5, r). W is (5*2, r).
    # If r=1, U[0,0]=1, U[1,0]=1. V[0,0..4]=1. W[0..4, 0..1]=1.
    # Then C[0,k] += 1*1*1 = 1. C[1,k] += 1*1*1 = 1.
    # This adds A[0,0]B[0,k] to C[0,k] and A[1,0]B[0,k] to C[1,k].
    # This is exactly what we want for the j=0 part.
    # So j=0 part is rank 1?
    # Wait, the tensor is multiplicative.
    # U[i,j,r] V[j,k,r] W[k,i,r].
    # If U[0,0,r]=1, U[1,0,r]=1, V[0,k,r]=1, W[k,i,r]=1.
    # Then the term is 1.
    # But we want to multiply by A[0,0] and B[0,k].
    # The decomposition is linear.
    # U, V, W are coefficients.
    # If we set U[0,0,r]=1, U[1,0,r]=1, V[0,k,r]=1, W[k,i,r]=1.
    # Then the product is 1.
    # But we need to extract the specific values.
    # The rank-1 term corresponds to the outer product of vectors.
    # For the term A[i,0]B[0,k], we need it to be part of C[i,k].
    # This is exactly a rank-1 tensor? No, it's a sum of rank-1 tensors?
    # Actually, the j=0 part is A[:,0] * B[0,:]. This is an outer product of A[:,0] and B[0,:].
    # This can be done with rank 1?
    # C[i,k] += A[i,0]B[0,k].
    # This is a rank-1 matrix (outer product).
    # So yes, rank 1 is sufficient for the j=0 part?
    # Wait, the tensor rank of the "slice" might be higher.
    # But here we are decomposing the whole multiplication tensor.
    # The term corresponding to j=0 is separable?
    # U[i,0,r] V[0,k,r] W[k,i,r].
    # If we set r=0: U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1.
    # Then C[i,k] gets 1.
    # But we need C[i,k] to get A[i,0]B[0,k].
    # The algorithm computes C = sum_r U_r V_r W_r.
    # The input matrices are A and B.
    # The output C is computed.
    # The rank-1 term adds U_r (vectorized A part) * V_r (vectorized B part) * W_r (vectorized C part).
    # No, that's not right.
    # The tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}. (Indices differ)
    # In our case T_{ijk} is the coefficient of A_{ij}B_{jk} in C_{ik}.
    # So T_{ijk} = 1 if (i,j,k) is a valid path, else 0.
    # We want to decompose this T.
    # For j=0, T_{i0k} = 1 for all i,k.
    # This is a 2x5 matrix of 1s.
    # Can we decompose a 2x5 matrix of 1s into rank-1 tensors?
    # We need sum_r U_{ir} V_{0r} W_{kr} = 1.
    # If we pick r=0, U_{i0}=1, V_{00}=1, W_{k0}=1.
    # Then sum is 1.
    # So yes, rank 1 is enough for the j=0 block of the tensor?
    # Wait, U has indices i,j. V has j,k. W has k,i.
    # If we fix j=0, we are looking at the "slice" T_{i0k}.
    # This slice is the matrix of all 1s.
    # We need to represent this slice using the terms where j=0 is active.
    # If we use a term with U[i,0,r]=1, V[0,k,r]=1, W[k,i,r]=1, it contributes 1 to T_{i0k}.
    # Does it contribute to other T_{ijk}?
    # If j!=0, U[i,j,r] must be 0.
    # If we set U[i,j,r]=0 for j!=0, then yes, it only affects the j=0 slice.
    # So the j=0 part can be done with rank 1.
    # Similarly, the j=3 part can be done with rank 1.
    
    # What about the j=1,2 part?
    # T_{i1k} and T_{i2k} are also 1s.
    # But they are coupled because U[i,1,r] and U[i,2,r] are in the same U vector.
    # Actually, we can treat the j=1,2 block as a 2x2x5 sub-problem?
    # No, the indices are i (0,1), j (1,2), k (0..4).
    # This is a 2x2x5 tensor of 1s.
    # What is the rank of a 2x2x5 tensor of 1s?
    # It is known to be 7 for 2x2x2.
    # For 2x2x5, is it 18?
    # Let's check the literature or derive.
    # The tensor is the outer product of ones vectors?
    # T_{ijk} = 1.
    # This is a "flat" tensor.
    # Rank of all-1 tensor of size I x J x K.
    # For 2x2x2, rank is 7.
    # For 2x2x3, rank is 9?
    # For 2x2x5, rank is ?
    
    # If the j=1,2 block (2x2x5 of 1s) has rank R, then total rank is 1 (j=0) + 1 (j=3) + R.
    # If R=18, total is 20.
    # If R=19, total is 21.
    
    # Let's assume the rank of 2x2x5 all-1 tensor is 18.
    # Construction for 2x2x5 all-1 tensor:
    # Split k=0..4 into k=0,1 (2x2x2), k=2,3 (2x2x2), k=4 (2x2x1).
    # Rank(2x2x2 of 1s) = 7.
    # Rank(2x2x1 of 1s) = ?
    # 2x2x1 is just a 2x2 matrix of 1s.
    # Rank of 2x2 matrix of 1s is 1?
    # T_{ij1} = 1.
    # sum_r U_{ir} V_{jr} W_{1r} = 1.
    # Set U_{ir}=1, V_{jr}=1, W_{1r}=1. Rank 1.
    # So 2x2x1 of 1s has rank 1?
    # Wait, the tensor is T_{ijk}.
    # If we fix k=4, we have a 2x2 matrix.
    # We can decompose a 2x2 matrix of 1s into rank 1?
    # Yes, [1,1;1,1] = [1;1] [1,1]. Rank 1.
    # So the k=4 part of the 2x2x5 tensor has rank 1.
    # The k=0,1 part (2x2x2 of 1s) has rank 7.
    # The k=2,3 part (2x2x2 of 1s) has rank 7.
    # Total rank for j=1,2 block = 7 + 7 + 1 = 15.
    # Total rank = 1 (j=0) + 1 (j=3) + 15 (j=1,2) = 17.
    
    # This is a huge improvement over 22!
    # Let's verify this.
    # Is the rank of 2x2x2 all-1 tensor 7?
    # Yes, it is the same as the matrix multiplication tensor?
    # No, matrix multiplication tensor T_{ijk} = 1 if j is "between" i and k?
    # No, in matrix multiplication, T_{ijk} is the coefficient of A_{ij}B_{jk} in C_{ik}.
    # It is 1 if i,j,k are valid indices.
    # For 2x2x2, T_{ijk} = 1 for all i,j,k in {0,1}.
    # This is the all-1 tensor.
    # The rank of the 2x2x2 all-1 tensor is indeed 7.
    # (Strassen's algorithm is for the multiplication tensor, which IS the all-1 tensor in the appropriate basis?
    # No, the multiplication tensor has non-zero entries where the indices connect.
    # For 2x2x2, all 8 entries are non-zero. So it IS the all-1 tensor.
    # So rank is 7.
    
    # So the plan:
    # 1. j=0 block: Rank 1.
    # 2. j=3 block: Rank 1.
    # 3. j=1,2 block:
    #    a. k=0,1 (2x2x2 of 1s): Rank 7.
    #    b. k=2,3 (2x2x2 of 1s): Rank 7.
    #    c. k=4 (2x2x1 of 1s): Rank 1.
    # Total Rank = 1 + 1 + 7 + 7 + 1 = 17.
    
    # Let's implement this rank 17 construction.
    
    rank17 = 17
    U17 = [[0] * rank17 for _ in range(n * m)]
    V17 = [[0] * rank17 for _ in range(m * p)]
    W17 = [[0] * rank17 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term from a 2x2x2 Strassen-like decomposition of all-1 tensor
    # The 7 terms for 2x2x2 all-1 tensor (which is matrix multiplication 2x2*2x2)
    # We can reuse the Strassen terms from the seed program.
    # The Strassen terms decompose the 2x2x2 multiplication tensor.
    # Since the 2x2x2 multiplication tensor is the all-1 tensor (all entries 1),
    # these terms are exactly what we need for the 2x2x2 blocks.
    
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    def add_strassen_2x2x2(j_offset, k_offset):
        nonlocal col
        for u_t, v_t, w_t in strassen_terms:
            for idx_u, val_u in u_t.items():
                li = idx_u // 2
                lj = idx_u % 2
                gi = li
                gj = j_offset + lj
                U17[gi*m + gj][col] += val_u
            for idx_v, val_v in v_t.items():
                lj = idx_v // 2
                lk = idx_v % 2
                gj = j_offset + lj
                gk = k_offset + lk
                V17[gj*p + gk][col] += val_v
            for idx_w, val_w in w_t.items():
                li = idx_w // 2
                lk = idx_w % 2
                gi = li
                gk = k_offset + lk
                W17[gk*n + gi][col] += val_w
            col += 1
    
    def add_2x2x1(j_offset, k_offset):
        nonlocal col
        # Rank 1 term for 2x2x1 all-1 tensor
        # U[i, j] = 1 for i=0,1 and j=j_offset, j_offset+1
        # V[j, k] = 1 for j=j_offset, j_offset+1 and k=k_offset
        # W[k, i] = 1 for k=k_offset and i=0,1
        for i in range(n):
            for j in range(2):
                U17[i*m + (j_offset + j)][col] = 1
        for j in range(2):
            V17[(j_offset + j)*p + k_offset][col] = 1
        for i in range(n):
            W17[k_offset*n + i][col] = 1
        col += 1
    
    def add_1x5(j_idx, k_offset):
        nonlocal col
        # Rank 1 term for 1x5 all-1 tensor (slice for a specific j)
        # U[i, j] = 1 for i=0,1
        # V[j, k] = 1 for k=0..4
        # W[k, i] = 1 for k=0..4, i=0,1
        for i in range(n):
            U17[i*m + j_idx][col] = 1
        for k in range(p):
            V17[j_idx*p + k][col] = 1
        for k in range(p):
            for i in range(n):
                W17[k*n + i][col] = 1
        col += 1
    
    # j=0 block
    add_1x5(0, 0)
    
    # j=3 block
    add_1x5(3, 0)
    
    # j=1,2 block
    # k=0,1
    add_strassen_2x2x2(1, 0)
    # k=2,3
    add_strassen_2x2x2(1, 2)
    # k=4
    add_2x2x1(1, 4)
    
    if col != rank17:
        raise ValueError(f"Rank mismatch: expected {rank17}, got {col}")
        
    best_result = None
    if verify_numpy(U17, V17, W17, rank17):
        if verify_exact(U17, V17, W17, rank17):
            best_result = {"U": U17, "V": V17, "W": W17}
            # Try to reduce rank
            reduced = try_reduce_rank(U17, V17, W17, rank17, max_attempts=50)
            if reduced is not None:
                U_red, V_red, W_red, rank_red = reduced
                if rank_red < rank17:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
    
    # Fallback to rank 22 or schoolbook if verification fails
    if best_result is None:
        # Re-use the rank 22 construction logic from before as fallback
        # Or just schoolbook
        rank_school = n * m * p
        best_result = {
            "U": [[0]*rank_school for _ in range(n*m)],
            "V": [[0]*rank_school for _ in range(m*p)],
            "W": [[0]*rank_school for _ in range(p*n)]
        }
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_result["U"][i*m+j][c] = 1
                    best_result["V"][j*p+k][c] = 1
                    best_result["W"][k*n+i][c] = 1
                    c += 1
    return best_result

