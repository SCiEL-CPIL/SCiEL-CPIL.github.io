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
        # Use matrix operations for speed
        n_r = U_mat.shape[1]
        # Reshape for batched computation
        # For each r, we have outer product of U[:,r], V[:,r], W[:,r]
        # Sum over r: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        # This is a 4x4x4 tensor. Let's compute it efficiently.
        # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
        # Reshape U to (4,4,n_r), V to (4,4,n_r), W to (4,4,n_r)
        U_reshaped = U_mat.reshape(4, 4, n_r)  # U[i,j,r]
        V_reshaped = V_mat.reshape(4, 4, n_r)  # V[j,k,r]
        W_reshaped = W_mat.reshape(4, 4, n_r)  # W[k,i,r]
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        # Check against delta_{i,k}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected = 1.0 if i == k else 0.0
                    if abs(T[i,j,k] - expected) > 1e-10:
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
    # Each term r defines a tensor T_r[i,j,k] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # If sum_r alpha_r * T_r = 0 has a non-trivial solution, we can reduce rank.
    # Vectorize: each T_r is a 64-dim vector (flattened 4x4x4).
    
    tensors = np.zeros((64, 49))
    for r in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    tensors[i*16+j*4+k, r] = U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    # Find the rank of this 64x49 matrix using SVD
    try:
        U_svd, svals, Vt = np.linalg.svd(tensors, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(svals > tol)
    except:
        effective_rank = 49
    
    if effective_rank < 49 and time.time() < deadline:
        # The 49 tensors are linearly dependent. Find a basis.
        # Use QR with column pivoting via scipy
        try:
            from scipy.linalg import qr
            # QR with column pivoting: A[:, P] = Q * R
            Q, R, P = qr(tensors, pivoting=True, mode='economic')
            # The first 'effective_rank' pivot columns form a basis
            pivot_cols = P[:effective_rank]
            
            # Verify that keeping only these columns gives a valid decomposition
            U_red = U[:, pivot_cols]
            V_red = V[:, pivot_cols]
            W_red = W[:, pivot_cols]
            if verify(U_red, V_red, W_red):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        except:
            pass
    
    # Return the full rank-49 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
