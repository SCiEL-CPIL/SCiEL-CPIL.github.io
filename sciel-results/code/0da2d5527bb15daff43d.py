import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = 2, 3, 6

    # Build the matrix multiplication tensor T[a*m+b, c*p+d, k*n+i] = delta_{a,i} * delta_{c,k} * delta_{b,d}
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor entry: sum_r U[a*m+b, r] * V[c*p+d, r] * W[k*n+i, r]
    # Target: 1 if a==i, c==k, b==d, else 0

    def build_tensor():
        """Build the target tensor T of shape (n*m, m*p, p*n)."""
        T = np.zeros((n * m, m * p, p * n))
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for k in range(p):
                            for i in range(n):
                                if a == i and c == k and b == d:
                                    T[a * m + b, c * p + d, k * n + i] = 1.0
        return T

    def verify_decomposition(U, V, W, n, m, p):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        T = build_tensor()
        # Compute the tensor from the decomposition
        # T_decomp[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Use einsum for efficiency
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_decomp, T, atol=1e-9)

    def verify_fast(U, V, W, n, m, p):
        """Faster verification using the structure."""
        r = U.shape[1]
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for k in range(p):
                            for i in range(n):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                s = 0.0
                                for r_idx in range(r):
                                    s += U[a*m+b, r_idx] * V[c*p+d, r_idx] * W[k*n+i, r_idx]
                                if abs(s - target) > 1e-9:
                                    return False
        return True

    def try_rank(r, time_budget=10.0):
        """Try to find a rank-r decomposition using randomized search."""
        if time.time() - start_time > time_budget:
            return None

        n, m, p = 2, 3, 6
        dim_u = n * m  # 6
        dim_v = m * p  # 18
        dim_w = p * n  # 12

        # Strategy: Start with a known higher-rank solution, then try to reduce
        # by finding linear dependencies among the rank-1 components.

        # First, get a baseline decomposition (rank 36 = schoolbook)
        # Schoolbook: U[i,r]=1 if i==r, V[j,r]=1 if j==r, W[k,r]=1 if k==r
        # where r indexes (a,b,c,d,k,i) but that's 36 terms for the identity
        # Actually schoolbook for 2x3*3x6:
        # C[a,i] = sum_b A[a,b]*B[b,i]
        # Term (a,b,i): U[a*3+b, r]=1, V[b*6+i, r]=1, W[i*2+a, r]=1
        # Wait, W index is k*n+i where k is the output column index.
        # For C[a, k] = sum_b A[a,b] * B[b,k]
        # W[k*n + a, r] should be 1 for the term corresponding to output (a,k)

        # Let me be more careful:
        # Output C is 2x6. C[a, k] = sum_{b=0}^{2} A[a,b] * B[b,k]
        # U index: a*m + b = a*3 + b
        # V index: b*p + k = b*6 + k
        # W index: k*n + a = k*2 + a

        # Schoolbook rank = n*m*p = 2*3*6 = 36
        U_school = np.zeros((dim_u, 36))
        V_school = np.zeros((dim_v, 36))
        W_school = np.zeros((dim_w, 36))

        idx = 0
        for a in range(n):
            for b in range(m):
                for k in range(p):
                    U_school[a*m + b, idx] = 1.0
                    V_school[b*p + k, idx] = 1.0
                    W_school[k*n + a, idx] = 1.0
                    idx += 1

        # Verify schoolbook
        if not verify_decomposition(U_school, V_school, W_school, n, m, p):
            # Fallback: return schoolbook
            return {'U': U_school.tolist(), 'V': V_school.tolist(), 'W': W_school.tolist()}

        # Now try to reduce rank by finding linear dependencies
        # Use a greedy approach: try removing each term and see if the remaining
        # terms can still span the tensor space.

        # Alternative: Use the fact that we can combine terms.
        # For 2x3x6, we can use Strassen-like tricks.

        # Let's try a direct approach: use the known rank-34 solution as baseline
        # and try to improve it.

        # Actually, let's try a completely different approach:
        # Use the tensor structure to find a minimal rank decomposition.

        # The tensor T has shape (6, 18, 12) with 36 non-zero entries.
        # We want to find the minimum rank r such that T = sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s

        # Use a randomized approach:
        # 1. Initialize with random rank-r factors
        # 2. Use ALS (Alternating Least Squares) to fit the tensor
        # 3. Check if the fit is exact

        def als_decomposition(T, r, n_iter=50, seed=42):
            """ALS for CP decomposition of a 3-tensor."""
            np.random.seed(seed)
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)

            for it in range(n_iter):
                if time.time() - start_time > DEADLINE - 5:
                    return None

                # Update U: for each i, U[i,:] solves the least squares
                # T[i,:,:] = sum_s U[i,s] * (v_s ⊗ w_s)
                # T[i,:,:].reshape(dim_v*dim_w) = U[i,:] @ (V ⊙ W)
                VW = np.kron(V.T, W.T)  # (r, dim_v*dim_w)
                Ti = T[i].reshape(-1) if False else T[i].reshape(dim_v * dim_w)
                # Actually T is (dim_u, dim_v, dim_w)
                # T[i, :, :] has shape (dim_v, dim_w)
                # We want: T[i,j,k] = sum_s U[i,s] * V[j,s] * W[k,s]
                # For fixed i: T[i,:,:] = sum_s U[i,s] * (V[:,s] ⊗ W[:,s])
                # Let M_i be (dim_v*dim_w, r) with M_i[(j,k), s] = V[j,s]*W[k,s]
                # Then T[i,:,:].flatten() = M_i.T @ U[i,:]
                # So U[i,:] = (M_i @ M_i.T)^{-1} @ M_i @ T[i,:,:].flatten()

                # More efficiently:
                # U[i,:] = (V ⊙ W)^T @ T[i,:,:].flatten() / ((V ⊙ W)^T @ (V ⊙ W))
                # where ⊙ is Khatri-Rao product

                # Let's do it properly with matrix operations
                # Khatri-Rao product of V and W: (dim_v*dim_w, r)
                # KR[:, s] = V[:, s] ⊗ W[:, s]

                # For each i:
                # T_i = T[i, :, :].reshape(-1)  # (dim_v*dim_w,)
                # U[i, :] = solve(KR.T @ KR, KR.T @ T_i)

                KR = np.zeros((dim_v * dim_w, r))
                for s in range(r):
                    KR[:, s] = np.kron(V[:, s], W[:, s])

                KRtKR = KR.T @ KR  # (r, r)
                KRtKR += 1e-10 * np.eye(r)  # regularization

                for i in range(dim_u):
                    T_i = T[i, :, :].reshape(-1)
                    KRtTi = KR.T @ T_i
                    U[i, :] = np.linalg.solve(KRtKR, KRtTi)

                # Similarly update V and W
                KW = np.zeros((dim_u * dim_w, r))
                for s in range(r):
                    KW[:, s] = np.kron(U[:, s], W[:, s])
                KWtKW = KW.T @ KW
                KWtKW += 1e-10 * np.eye(r)

                for j in range(dim_v):
                    T_j = T[:, j, :].reshape(-1)
                    KWtTj = KW.T @ T_j
                    V[j, :] = np.linalg.solve(KWtKW, KWtTj)

                KU = np.zeros((dim_u * dim_v, r))
                for s in range(r):
                    KU[:, s] = np.kron(U[:, s], V[:, s])
                KUtKU = KU.T @ KU
                KUtKU += 1e-10 * np.eye(r)

                for k in range(dim_w):
                    T_k = T[:, :, k].reshape(-1)
                    KUtTk = KU.T @ T_k
                    W[k, :] = np.linalg.solve(KUtKU, KUtTk)

            # Check if the decomposition is exact
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_recon, T, atol=1e-9):
                return U, V, W
            return None

        # Try ranks from 22 upward
        best_result = None

        # First, try to get a good baseline by using the rank-34 construction
        # and then try to reduce it.

        # Let's try ALS for various ranks
        for r in [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]:
            if time.time() - start_time > DEADLINE - 10:
                break

            # Try multiple random initializations
            for seed in range(5):
                if time.time() - start_time > DEADLINE - 5:
                    break

                result = als_decomposition(T, r, n_iter=30, seed=seed)
                if result is not None:
                    U, V, W = result
                    # Round to nearest half-integer for cleanliness
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2

                    if verify_decomposition(U_round, V_round, W_round, n, m, p):
                        best_result = {'U': U_round.tolist(), 'V': V_round.tolist(), 'W': W_round.tolist()}
                        return best_result
                    elif verify_decomposition(U, V, W, n, m, p):
                        best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                        return best_result

        # If ALS didn't find an exact solution, try the known rank-34 construction
        # and return that as a safe fallback.

        # Build the rank-34 solution (two copies of rank-17 for 2x3x3)
        U233 = np.zeros((6, 17))
        V233 = np.zeros((9, 17))
        W233 = np.zeros((6, 17))

        # Strassen for 2x2x2 part
        U233[0, 0] = 1; U233[4, 0] = 1
        V233[0, 0] = 1; V233[4, 0] = 1
        W233[0, 0] = 1; W233[3, 0] = 1

        U233[3, 1] = 1; U233[4, 1] = 1
        V233[0, 1] = 1
        W233[1, 1] = 1; W233[3, 1] = -1

        U233[0, 2] = 1
        V233[1, 2] = 1; V233[4, 2] = -1
        W233[2, 2] = 1; W233[3, 2] = 1

        U233[4, 3] = 1
        V233[3, 3] = 1; V233[0, 3] = -1
        W233[0, 3] = 1; W233[1, 3] = 1

        U233[0, 4] = 1; U233[1, 4] = 1
        V233[4, 4] = 1
        W233[0, 4] = -1; W233[2, 4] = 1

        U233[3, 5] = 1; U233[0, 5] = -1
        V233[0, 5] = 1; V233[1, 5] = 1
        W233[3, 5] = 1

        U233[1, 6] = 1; U233[4, 6] = -1
        V233[3, 6] = 1; V233[4, 6] = 1
        W233[0, 6] = 1

        # Outer product terms for the 3rd column/row
        U233[0, 7] = 1
        V233[2, 7] = 1
        W233[4, 7] = 1

        U233[1, 8] = 1
        V233[5, 8] = 1
        W233[4, 8] = 1

        U233[3, 9] = 1
        V233[2, 9] = 1
        W233[5, 9] = 1

        U233[4, 10] = 1
        V233[5, 10] = 1
        W233[5, 10] = 1

        # Additional terms for full 2x3x3
        U233[2, 11] = 1
        V233[6, 11] = 1
        W233[0, 11] = 1

        U233[2, 12] = 1
        V233[7, 12] = 1
        W233[2, 12] = 1

        U233[2, 13] = 1
        V233[8, 13] = 1
        W233[4, 13] = 1

        U233[5, 14] = 1
        V233[6, 14] = 1
        W233[1, 14] = 1

        U233[5, 15] = 1
        V233[7, 15] = 1
        W233[3, 15] = 1

        U233[5, 16] = 1
        V233[8, 16] = 1
        W233[5, 16] = 1

        # Combine two copies for 2x3x6
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))

        U_comb[:, :17] = U233
        U_comb[:, 17:] = U233

        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233[j * 3 + k]

        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233[k * 2 + i]

        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {'U': U_comb.tolist(), 'V': V_comb.tolist(), 'W': W_comb.tolist()}

        # Last resort: schoolbook rank 36
        U_s = np.zeros((6, 36))
        V_s = np.zeros((18, 36))
        W_s = np.zeros((12, 36))
        idx = 0
        for a in range(n):
            for b in range(m):
                for k in range(p):
                    U_s[a*m + b, idx] = 1.0
                    V_s[b*p + k, idx] = 1.0
                    W_s[k*n + a, idx] = 1.0
                    idx += 1

        return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}

    # Run the search
    result = try_rank(34, time_budget=130.0)
    if result is not None:
        return result

    # Fallback: return schoolbook
    U_s = np.zeros((6, 36))
    V_s = np.zeros((18, 36))
    W_s = np.zeros((12, 36))
    idx = 0
    for a in range(n):
        for b in range(m):
            for k in range(p):
                U_s[a*m + b, idx] = 1.0
                V_s[b*p + k, idx] = 1.0
                W_s[k*n + a, idx] = 1.0
                idx += 1
    return {'U': U_s.tolist(), 'V': V_s.tolist(), 'W': W_s.tolist()}

if __name__ == "__main__":
    construct()
