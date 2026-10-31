import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS

    # Build the 2x2x2 Strassen decomposition (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # t=0 (M1 inner)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # t=1 (M2 inner)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # t=2 (M3 inner)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # t=3 (M4 inner)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # t=4 (M5 inner)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # t=5 (M6 inner)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # t=6 (M7 inner)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 using 7 products of 2x2x2 (rank 49)
    rank_full = 49
    U = np.zeros((16, rank_full))
    V = np.zeros((16, rank_full))
    W = np.zeros((16, rank_full))

    # Strassen coefficients for 4x4 -> 2x2 blocks
    beta = np.zeros((7, 2, 2))
    gamma = np.zeros((7, 2, 2))
    delta = np.zeros((7, 2, 2))

    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    for s in range(7):
        for t in range(7):
            c = 7 * s + t

            for i in range(4):
                for j in range(4):
                    I = i // 2
                    i_prime = i % 2
                    J = j // 2
                    j_prime = j % 2
                    row_u = i * 4 + j
                    coeff = beta[s, I, J] * U2[i_prime * 2 + j_prime, t]
                    if coeff != 0:
                        U[row_u, c] = coeff

            for j in range(4):
                for k in range(4):
                    J = j // 2
                    j_prime = j % 2
                    K = k // 2
                    k_prime = k % 2
                    row_v = j * 4 + k
                    coeff = gamma[s, J, K] * V2[j_prime * 2 + k_prime, t]
                    if coeff != 0:
                        V[row_v, c] = coeff

            for k in range(4):
                for i in range(4):
                    K = k // 2
                    k_prime = k % 2
                    I = i // 2
                    i_prime = i % 2
                    row_w = k * 4 + i
                    coeff = delta[s, I, K] * W2[k_prime * 2 + i_prime, t]
                    if coeff != 0:
                        W[row_w, c] = coeff

    if time.time() > deadline - 20.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((16, 16, 16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i*4+j, j*4+k, k*4+i] = 1.0

    def verify(U_mat, V_mat, W_mat):
        """Check if U,V,W exactly decompose T."""
        # Reconstruct: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[i, r] * V_mat[j, r] * W_mat[k, r]
                    if abs(val - T[i, j, k]) > 1e-10:
                        return False
        return True

    # Check if the full 49-column decomposition is valid
    if not verify(U, V, W):
        # Something went wrong; return as-is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find linearly dependent columns and remove them
    # Strategy: for each column, check if it's in the span of the others
    # by checking if the tensor contribution is zero when we set that column to zero
    # and compensate... but that's complex. Instead, use a simpler approach:
    # Check if any column is exactly zero (unlikely) or if two columns are identical.
    
    # Better approach: use the fact that the tensor rank is the minimum number of
    # rank-1 tensors. We can try greedy removal:
    # For each column r, temporarily zero it out and check if the remaining
    # columns can still span T. This is expensive but doable for 49 columns.
    
    # Actually, a more efficient check: the columns define a point in a
    # 16*16*16 dimensional space (flattened tensor). If one column's
    # contribution is a linear combination of others' contributions, we can remove it.
    
    # Let's flatten the tensor contributions
    # Column r contributes: outer(U[:,r], V[:,r], W[:,r])
    # Flatten to vector of length 16*16*16 = 4096
    
    # This is expensive. Let's try a simpler heuristic first:
    # Check for duplicate or zero columns.
    
    # Check for zero columns
    nonzero_cols = []
    for r in range(49):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            nonzero_cols.append(r)
    
    if len(nonzero_cols) < 49:
        U = U[:, nonzero_cols]
        V = V[:, nonzero_cols]
        W = W[:, nonzero_cols]
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find columns whose removal still allows exact decomposition
    # by checking if the tensor can be represented with fewer terms.
    # Use a simple greedy approach: try removing each column and checking
    # if the residual can be absorbed by adjusting others (not straightforward).
    
    # Alternative: use the structure. The 49 columns come from 7*7 combinations.
    # Check if any (s,t) pair produces a zero contribution.
    
    # For now, return the 49-column solution which we know is valid.
    # The rank-49 Strassen decomposition for 4x4x4 is known to be valid.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
