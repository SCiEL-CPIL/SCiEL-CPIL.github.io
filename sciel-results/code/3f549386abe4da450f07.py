import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = 2, 4, 8
    
    # Build the rank-56 Strassen decomposition (8 blocks of 2x2x2)
    # This is a known valid decomposition.
    # We then attempt to reduce the rank via a structural change:
    # We treat the problem as finding a low-rank decomposition of the tensor.
    # We start with the rank-56 factors and try to eliminate linear dependencies
    # among the rank-1 terms.
    
    # 1. Construct the rank-56 decomposition
    rank_init = 56
    U = np.zeros((n*m, rank_init), dtype=np.int32)
    V = np.zeros((m*p, rank_init), dtype=np.int32)
    W = np.zeros((p*n, rank_init), dtype=np.int32)
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen 7 multiplications for 2x2 matrix mult
            # Block A: rows 0,1; cols j0, j0+1
            # Block B: rows j0, j0+1; cols k0, k0+1
            # Block C: rows 0,1; cols k0, k0+1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[0*m + j0, col] += 1; U[1*m + j0+1, col] += 1
            V[j0*p + k0, col] += 1; V[j0+1*p + k0+1, col] += 1
            W[k0*n + 0, col] += 1; W[k0+1*n + 1, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[1*m + j0, col] += 1; U[1*m + j0+1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1; W[k0+1*n + 1, col] -= 1
            col += 1
            
            # M3 = A00(B01 - B11)
            U[0*m + j0, col] += 1
            V[j0*p + k0+1, col] += 1; V[j0+1*p + k0+1, col] -= 1
            W[k0+1*n + 0, col] += 1; W[k0+1*n + 1, col] += 1
            col += 1
            
            # M4 = A11(B10 - B00)
            U[1*m + j0+1, col] += 1
            V[j0+1*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1; W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[0*m + j0, col] += 1; U[0*m + j0+1, col] += 1
            V[j0+1*p + k0+1, col] += 1
            W[k0*n + 0, col] -= 1; W[k0+1*n + 0, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[1*m + j0, col] += 1; U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + k0+1, col] += 1
            W[k0+1*n + 1, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[0*m + j0+1, col] += 1; U[1*m + j0+1, col] -= 1
            V[j0+1*p + k0, col] += 1; V[j0+1*p + k0+1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    # 2. Attempt Rank Reduction via Linear Algebra
    # The tensor T is defined by T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0.
    # Actually, T[i,j,k] is the coefficient in the bilinear form.
    # The decomposition is sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] = T.
    # We want to find a smaller set of rank-1 tensors that sum to T.
    # This is equivalent to finding the tensor rank.
    # 
    # We can try to see if the current 56 vectors (u_r, v_r, w_r) are linearly independent
    # in a way that allows reduction.
    # A simple check: if we can express one term as a linear combination of others, we can remove it.
    # However, tensor rank is not simply the matrix rank of the flattened tensor.
    #
    # A more robust approach: 
    # We know the lower bound is max(n*m, m*p, p*n) = 32? No, that's not right.
    # The lower bound for matrix multiplication (n, m, p) is at least max(n*m, m*p, p*n) / something?
    # Actually, a standard lower bound is that the rank must be at least the dimension of the space spanned by the slices.
    #
    # Let's try a numerical optimization approach to see if we can push below 56.
    # We will fix the structure of the first R terms and try to solve for the coefficients.
    # But we don't know the structure.
    #
    # Instead, let's try to merge terms that are "close" or linearly dependent.
    # We can compute the matrix product for random A, B and see if we can fit a lower rank model.
    # This is essentially doing a rank-R SVD on the tensor, but SVD doesn't minimize tensor rank.
    #
    # Given the time limit and the difficulty of tensor rank minimization, we will stick with 56
    # unless we can find a very specific algebraic cancellation.
    #
    # Let's check if there are any duplicate or zero columns in U, V, W.
    # (There shouldn't be, but good to check).
    
    # Check for zero columns
    mask_u = np.all(U == 0, axis=0)
    mask_v = np.all(V == 0, axis=0)
    mask_w = np.all(W == 0, axis=0)
    
    if np.any(mask_u) or np.any(mask_v) or np.any(mask_w):
        # Remove zero columns
        valid = ~(mask_u | mask_v | mask_w)
        U = U[:, valid]
        V = V[:, valid]
        W = W[:, valid]
    
    # Check for duplicate columns (which can be merged)
    # This is O(R^2) and R=56, so it's fast.
    def remove_duplicates(U, V, W):
        # We can't just merge duplicates because the coefficients must match.
        # If U[:, i] == U[:, j], V[:, i] == V[:, j], and W[:, i] == W[:, j], we can merge.
        # If only some match, we can't simply merge.
        # We can check if (u_i, v_i, w_i) is a linear combination of others? No, that's hard.
        # We just check for exact duplicates of the triple.
        # Since entries are integers, we can hash the tuples.
        seen = {}
        new_cols = []
        for r in range(U.shape[1]):
            key = (tuple(U[:, r]), tuple(V[:, r]), tuple(W[:, r]))
            if key in seen:
                # Merge: add the coefficients? No, if they are identical, we just have a redundant term.
                # In a decomposition sum u v w, if two terms are identical, we can combine them: 2 u v w.
                # But the rank counts the number of terms. If we have 2 identical terms, we can replace them with 1 term with coeff 2.
                # So we can reduce rank.
                # However, our current representation assumes coeff 1 for each term in the sum.
                # The output format is just the matrices U, V, W. The sum is implicit.
                # If we have two identical columns, we can just drop one? No, that changes the sum.
                # We need to scale.
                # But the problem asks for "integer/half-integer entries".
                # If we merge, we double the entries. That's fine.
                # But we only reduce rank if we can represent the sum with fewer terms.
                # If we have term X and term X, sum is 2X. We can represent 2X as one term with 2X coefficients.
                # So yes, we can reduce rank.
                idx = seen[key]
                # Double the coefficients in the existing column
                U[:, idx] = U[:, idx] + U[:, r]
                V[:, idx] = V[:, idx] + V[:, r]
                W[:, idx] = W[:, idx] + W[:, r]
            else:
                seen[key] = len(new_cols)
                new_cols.append(r)
        
        U = U[:, new_cols]
        V = V[:, new_cols]
        W = W[:, new_cols]
        return U, V, W

    U, V, W = remove_duplicates(U, V, W)
    
    # Check for linear dependencies among the columns of U, V, W jointly?
    # This is the tensor rank problem.
    # We can try to see if the rank is actually lower by checking the "slice" ranks.
    # But let's assume 56 is the best we can get with this method.
    
    # One more check: can we reduce rank by 1?
    # If we can find a vector a, b, c such that a \otimes b \otimes c is in the span of the existing terms,
    # and we can adjust the existing terms to cancel it out... this is complex.
    
    # Let's verify the current decomposition is valid.
    def verify(U, V, W, n, m, p):
        # Use a few random tests
        for _ in range(10):
            A = np.random.randint(-2, 2, size=(n, m))
            B = np.random.randint(-2, 2, size=(m, p))
            C_expected = A @ B
            C_actual = np.zeros((n, p))
            for r in range(U.shape[1]):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                # C[i,k] += sum_j u[i,j] v[j,k] w[k,i]
                # This is (u @ v) * w.T element-wise? No.
                # (u @ v)[i, k] = sum_j u[i,j] v[j,k]
                # We need to multiply by w[k,i] for each k,i.
                # So C += (u @ v) * w.T (where w.T[i,k] = w[k,i])
                C_actual += (u @ v) * w.T
            if not np.allclose(C_actual, C_expected):
                return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = 64
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
