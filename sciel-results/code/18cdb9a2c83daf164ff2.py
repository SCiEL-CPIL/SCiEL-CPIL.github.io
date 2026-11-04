import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    # Get the rank-50 decomposition as a starting point
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Slice-rank approach:
    # Reshape T into a matrix M of shape (n, m*p*p*n) where M[i, j*p*(p*n) + j2*(p*n) + k] 
    # Actually, let's think of it as: for each output index (i,k), the "slice" is a 4x7 matrix
    # S_ik[j, k2] such that C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed (i,k), the slice over (j, k2) is: T[i*m+j, j*p+k2, k*n+i] which is 1 if k2==k else 0
    # Wait, that's not quite right. Let me re-examine.
    #
    # The tensor T has T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # So T[i*m+j, j*p+k, k*n+i] = 1.
    #
    # Slice-rank: fix the first index a = i*m+j. Then T[a, b, c] as a function of (b,c).
    # For a = i*m+j: T[i*m+j, j*p+k, k*n+i] = 1 for all k in 0..p-1.
    # So the slice at a=i*m+j is a matrix of size (m*p) x (p*n) with entries:
    #   S_a[b, c] = 1 iff b = j*p+k and c = k*n+i for some k, i.e., b=j*p+k, c=k*n+i.
    #
    # For the slice-rank method, we want to find rank-1 decompositions of these slices.
    # A rank-1 term U[:,r]*V[:,r]*W[:,r] contributes to slice a as:
    #   U[a,r] * (V[:,r] * W[:,r])^T
    # So the slice at a is: sum_r U[a,r] * outer(V[:,r], W[:,r])
    #
    # The slice-rank is the minimum number of rank-1 matrices needed to express all slices.
    # Since n=2, m=4, p=7: slices are at a=0..7 (8 slices), each of size 28x14.
    #
    # For slice a = i*m+j (i=0,1; j=0,1,2,3):
    #   S_a[b,c] = 1 iff b = j*p+k and c = k*n+i
    #   i.e., b is in {j*p, j*p+1, ..., j*p+6} and c = (b - j*p)*n + i
    #   So S_a has exactly 7 non-zero entries (one per k).
    #
    # The key insight: S_a for different a values share structure.
    # For i=0: S_{4j}[j*p+k, k] = 1 for k=0..6
    # For i=1: S_{4j+1}[j*p+k, 2k+1] = 1 for k=0..6
    #
    # Let's use ALS to find a low-rank decomposition directly.
    # The matrix form: T reshaped as (n*m) x (m*p * p*n) = 8 x 392
    # But that's just one unfolding. The CP decomposition requires all three unfoldings.
    
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(U_init, V_init, W_init, max_iters=200, tol=1e-12):
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U_new = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V_new = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W_new = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
            except:
                break
            change = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        return U, V, W

    rng = np.random.default_rng(42)

    for target_rank in range(49, 27, -1):
        if time.time() > deadline - 20:
            break
        found = False
        for restart in range(6):
            if time.time() > deadline - 15:
                break
            if restart == 0:
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            elif restart == 1:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            elif restart == 2:
                # Slice-rank inspired init: use the structure of the slices
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U_init[i*m+j, r] = 1.0
                    V_init[j*p+k, r] = 1.0
                    W_init[k*n+i, r] = 1.0
                U_init += rng.normal(0, 0.1, U_init.shape)
                V_init += rng.normal(0, 0.1, V_init.shape)
                W_init += rng.normal(0, 0.1, W_init.shape)
            elif restart == 3:
                # Use SVD of T1 for U init
                U_svd, s, _ = np.linalg.svd(T1, full_matrices=False)
                U_init = U_svd[:, :target_rank] * s[:target_rank]
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            elif restart == 4:
                # Strassen-inspired sparse init
                U_init = np.zeros((n*m, target_rank))
                V_init = np.zeros((m*p, target_rank))
                W_init = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    block = r // 7
                    sub = r % 7
                    j0 = (block % 2) * 2
                    k0 = (block // 2) * 2
                    if k0 >= p:
                        continue
                    # Strassen pattern
                    if sub == 0:
                        U_init[0*m+j0, r] = 1; U_init[1*m+(j0+1), r] = 1
                        V_init[j0*p+k0, r] = 1; V_init[(j0+1)*p+(k0+1), r] = 1
                        W_init[k0*n+0, r] = 1; W_init[(k0+1)*n+1, r] = 1
                    elif sub == 1:
                        U_init[1*m+j0, r] = 1; U_init[1*m+(j0+1), r] = 1
                        V_init[j0*p+k0, r] = 1
                        W_init[k0*n+1, r] = 1; W_init[(k0+1)*n+1, r] = -1
                    elif sub == 2:
                        U_init[0*m+j0, r] = 1
                        V_init[j0*p+(k0+1), r] = 1; V_init[(j0+1)*p+(k0+1), r] = -1
                        W_init[(k0+1)*n+0, r] = 1; W_init[(k0+1)*n+1, r] = 1
                    elif sub == 3:
                        U_init[1*m+(j0+1), r] = 1
                        V_init[(j0+1)*p+k0, r] = 1; V_init[j0*p+k0, r] = -1
                        W_init[k0*n+0, r] = 1; W_init[k0*n+1, r] = 1
                    elif sub == 4:
                        U_init[0*m+j0, r] = 1; U_init[0*m+(j0+1), r] = 1
                        V_init[(j0+1)*p+(k0+1), r] = 1
                        W_init[k0*n+0, r] = -1; W_init[(k0+1)*n+0, r] = 1
                    elif sub == 5:
                        U_init[1*m+j0, r] = 1; U_init[0*m+j0, r] = -1
                        V_init[j0*p+k0, r] = 1; V_init[j0*p+(k0+1), r] = 1
                        W_init[(k0+1)*n+1, r] = 1
                    elif sub == 6:
                        U_init[0*m+(j0+1), r] = 1; U_init[1*m+(j0+1), r] = -1
                        V_init[(j0+1)*p+k0, r] = 1; V_init[(j0+1)*p+(k0+1), r] = 1
                        W_init[k0*n+0, r] = 1
                U_init += rng.normal(0, 0.05, U_init.shape)
                V_init += rng.normal(0, 0.05, V_init.shape)
                W_init += rng.normal(0, 0.05, W_init.shape)
            else:
                U_init = rng.normal(0, 1, (n*m, target_rank))
                V_init = rng.normal(0, 1, (m*p, target_rank))
                W_init = rng.normal(0, 1, (p*n, target_rank))
            
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iters=100)
            
            recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
            residual = np.linalg.norm(recon - T)
            if residual < 1e-6:
                # Polish
                for _ in range(30):
                    if time.time() > deadline - 12:
                        break
                    KR_VW = khatri_rao(V_sol, W_sol)
                    U_sol = np.linalg.lstsq(KR_VW, T1.T, rcond=None)[0].T
                    KR_WU = khatri_rao(W_sol, U_sol)
                    V_sol = np.linalg.lstsq(KR_WU, T2.T, rcond=None)[0].T
                    KR_UV = khatri_rao(U_sol, V_sol)
                    W_sol = np.linalg.lstsq(KR_UV, T3.T, rcond=None)[0].T
                recon = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol)
                residual = np.linalg.norm(recon - T)
            
            if verify_decomp(U_sol, V_sol, W_sol):
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                found = True
                break
        
        if found:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
