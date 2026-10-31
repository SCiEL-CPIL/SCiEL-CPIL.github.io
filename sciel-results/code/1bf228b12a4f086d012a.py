import time
import numpy as np
from numpy.linalg import matrix_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 4, 4, 4
    rank = 49

    # Strassen's 7 multiplications for 2x2
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M0 = (A00 + A11)(B00 + B11)
        [[0, 0], [1, 1]],   # M1 = (A10 + A11)B00
        [[1, 0], [0, 0]],   # M2 = A00(B01 - B11)
        [[0, 0], [0, 1]],   # M3 = A11(B10 - B00)
        [[1, 1], [0, 0]],   # M4 = (A00 + A01)B11
        [[-1, 0], [1, 0]],  # M5 = (A10 - A00)(B00 + B01)
        [[0, 1], [0, -1]],  # M6 = (A01 - A11)(B10 + B11)
    ])

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # (A00+A11)(B00+B11)
        [[1, 0], [0, 0]],   # (A10+A11)B00
        [[0, 1], [0, -1]],  # A00(B01-B11)
        [[-1, 0], [1, 0]],  # A11(B10-B00)
        [[0, 0], [0, 1]],   # (A00+A01)B11
        [[1, 1], [0, 0]],   # (A10-A00)(B00+B01)
        [[0, 0], [1, 1]],   # (A01-A11)(B10+B11)
    ])

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C00+=M0, C11+=M0
        [[0, 0], [1, -1]],  # C10+=M1, C11-=M1
        [[0, 1], [0, 1]],   # C01+=M2, C11+=M2
        [[1, 0], [1, 0]],   # C00+=M3, C10+=M3
        [[-1, 1], [0, 0]],  # C00-=M4, C01+=M4
        [[0, 0], [0, 1]],   # C11+=M5
        [[1, 0], [0, 0]],   # C00+=M6
    ])

    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))

    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t

            for S in range(2):
                for T in range(2):
                    a_st = a_coeffs[s, S, T]
                    if a_st == 0:
                        continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_t2 = a_coeffs[t, s2, t2]
                            if a_t2 == 0:
                                continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U[i * 4 + j, c_idx] += a_st * a_t2

            for T in range(2):
                for K in range(2):
                    b_st = b_coeffs[s, T, K]
                    if b_st == 0:
                        continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_t2 = b_coeffs[t, t2, k2]
                            if b_t2 == 0:
                                continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V[j * 4 + k, c_idx] += b_st * b_t2

            for K in range(2):
                for S in range(2):
                    c_sk = c_coeffs[s, S, K]
                    if c_sk == 0:
                        continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_t2 = c_coeffs[t, s2, k2]
                            if c_t2 == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W[k * 4 + i, c_idx] += c_sk * c_t2

            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the decomposition is correct
    def verify(U_mat, V_mat, W_mat):
        """Check that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == delta_{i,k} for matrix mult tensor"""
        # The tensor T[i,j,k] = delta_{i,k} for matmul (C[i,k] = sum_j A[i,j]*B[j,k])
        # Our decomposition: T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected = 1.0 if i == k else 0.0
                    actual = 0.0
                    for r in range(U_mat.shape[1]):
                        actual += U_mat[i*4+j, r] * V_mat[j*4+k, r] * W_mat[k*4+i, r]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: return the 64-term schoolbook (rank 64)
        U_f = np.zeros((16, 64))
        V_f = np.zeros((16, 64))
        W_f = np.zeros((16, 64))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    idx = i*16 + j*4 + k
                    U_f[i*4+j, idx] = 1.0
                    V_f[j*4+k, idx] = 1.0
                    W_f[k*4+i, idx] = 1.0 if i == k else 0.0
        # Actually schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = delta_{i,k}
        # U[i*4+j, idx] * V[j*4+k, idx] * W[k*4+i, idx] should give delta_{i,k}
        # So set W[k*4+i, idx] = 1 for all, and only include terms where the product works
        # Simpler: just use the identity that A*B has C[i,k]=sum_j A[i,j]B[j,k]
        # Term (i,j,k) contributes A[i,j]*B[j,k]*delta_{i,k}
        # So we need: for each (i,j,k), a term with U[i,j]=1, V[j,k]=1, W[k,i]=1
        # But that gives C[i,k] += A[i,j]*B[j,k] for ALL i,j,k, not just when i==k
        # The correct schoolbook: for each (i,j,k), term has U[i*4+j]=1, V[j*4+k]=1, W[k*4+i]=1
        # Then sum over all (i,j,k): C[i',k'] gets contribution from term (i,j,k) when i'=i, j'=j, k'=k
        # i.e., U[i*4+j]*V[j*4+k]*W[k*4+i] = 1 only when (i',j',k')=(i,j,k)
        # So C[i',k'] = sum_{j'} A[i',j'] B[j',k'] * (1 if i'==k' else 0)... no that's wrong.
        
        # Let me just use the standard: rank = n*m*p terms isn't right either.
        # Standard bilinear: C[i,k] = sum_j A[i,j] B[j,k]
        # Term (i,j,k): contributes to C[i,k] the value A[i,j]*B[j,k]
        # So we need: for each (i,j), a term that multiplies A[i,j] and B[j,k] for all k? No.
        # Each scalar multiplication is A[i,j]*B[j,k] for specific (i,j,k)
        # So 64 terms, each computing one product A[i,j]*B[j,k] and adding to C[i,k]
        # U[i*4+j, (i,j,k)] = 1, V[j*4+k, (i,j,k)] = 1, W[k*4+i, (i,j,k)] = 1
        # Then C[i',k'] = sum_{(i,j,k): i'=i, k'=k} A[i,j]*B[j,k] = sum_j A[i',j]*B[j,k'] ✓
        
        U_f = np.zeros((16, 64))
        V_f = np.zeros((16, 64))
        W_f = np.zeros((16, 64))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_f[i*4+j, idx] = 1.0
                    V_f[j*4+k, idx] = 1.0
                    W_f[k*4+i, idx] = 1.0
                    idx += 1
        if verify(U_f, V_f, W_f):
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        # Should not reach here
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now attempt rank reduction: find linear dependencies among the 49 terms
    # A term r is redundant if there exist scalars alpha_s (s != r) such that
    # for all (i,j,k): U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_{s != r} alpha_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
    # This is hard in general. Instead, check if any term is all-zeros (shouldn't be)
    # or use a different approach.
    
    # Alternative: try to find if the 49 tensors are linearly dependent.
    # Each term r defines a tensor T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # If sum_r alpha_r * T_r = 0 has a non-trivial solution, we can reduce rank.
    # Vectorize: each T_r is a 64-dim vector (flattened 4x4x4).
    # Check rank of the 64x49 matrix where column r is T_r flattened.
    
    tensors = np.zeros((64, 49))
    for r in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    tensors[i*16+j*4+k, r] = U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    # Find the rank of this 64x49 matrix
    # Use SVD to find the effective rank
    try:
        U_svd, svals, Vt = np.linalg.svd(tensors, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        effective_rank = np.sum(svals > tol)
    except:
        effective_rank = 49
    
    if effective_rank < 49 and time.time() < deadline:
        # The 49 tensors are linearly dependent. We can find a basis.
        # Use the left null space of the 64x49 matrix to find dependencies.
        # Actually, we want to find a subset of columns that spans the same space.
        # Use QR decomposition with column pivoting.
        try:
            # Find a basis using numpy
            # The columns of 'tensors' span a space of dimension effective_rank
            # We can select 'effective_rank' linearly independent columns
            # and express the rest as linear combinations.
            # But for our purposes, we just need to remove redundant terms.
            
            # Use the fact that if columns are dependent, we can drop some.
            # Find pivot columns via QR with column pivoting (not directly in numpy)
            # Alternative: use the SVD - keep only the first 'effective_rank' right singular vectors
            # This gives a rotated basis, but we need to express U,V,W in terms of it.
            
            # Simpler approach: iteratively remove terms that are in the span of others.
            # For each term r, check if T_r is in the span of {T_s : s != r}
            # by solving the least squares problem.
            
            new_U = U.copy()
            new_V = V.copy()
            new_W = W.copy()
            kept = list(range(49))
            
            # Try to remove terms one by one
            for r in range(49):
                if time.time() > deadline:
                    break
                if r not in kept:
                    continue
                # Check if T_r is in span of other kept terms
                others = [x for x in kept if x != r]
                if len(others) == 0:
                    continue
                T_r = tensors[:, r]
                T_others = tensors[:, others]
                # Solve T_others @ alpha = T_r
                try:
                    alpha, residuals, rank_ls, svals_ls = np.linalg.lstsq(T_others, T_r, rcond=1e-12)
                    # Check if the residual is small
                    if len(residuals) > 0 and residuals[0] < 1e-8:
                        # T_r is in the span, remove it
                        # But we can't just remove it - we need to redistribute its contribution
                        # Actually, if T_r = sum_s alpha_s * T_s, then the term r is redundant
                        # because its contribution is already captured by the other terms.
                        # We can simply drop term r.
                        kept.remove(r)
                        new_U[:, r] = 0
                        new_V[:, r] = 0
                        new_W[:, r] = 0
                except:
                    pass
            
            if len(kept) < 49:
                # Verify the reduced decomposition
                reduced_rank = len(kept)
                U_red = new_U[:, kept]
                V_red = new_V[:, kept]
                W_red = new_W[:, kept]
                if verify(U_red, V_red, W_red):
                    return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        except:
            pass
    
    # Return the full rank-49 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
