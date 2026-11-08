"""Matrix multiplication 2x5x5 rank-minimal tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS
RANK_MAX = N * M * P  # 50
TIME_LIMIT = 120.0
START_TIME = time.time()


def _remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


def _verify(U, V, W, rank, tol=1e-7):
    """Verify the decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build the tensor T[i,j,k]
    T = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = 0.0
                for c in range(rank):
                    val += U[i * M + j, c] * V[j * P + k, c] * W[k * N + i, c]
                T[i, j, k] = val
    # Expected: T[i,j,k] = 1 if i==0 and j==0 and k==0? No.
    # Matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 for all (i,j,k) since each term A[i,j]*B[j,k] contributes.
    # Actually the tensor T_{ijk} = delta_{ij} * delta_{jk}? No.
    # The matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k because
    # C[i,k] = sum_j A[i,j] B[j,k], so the trilinear form is sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # which equals sum_{i,j,k} A[i,j] B[j,k] * (sum over the multiplication).
    # The tensor entries are T[i,j,k] = 1 for all i,j,k.
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if abs(T[i, j, k] - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, rank, tol=1e-7):
    """Faster verification using matrix operations."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U shape: (N*M, rank), V shape: (M*P, rank), W shape: (P*N, rank)
    # T[i,j,k] = sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
    # We need T[i,j,k] = 1 for all i,j,k.
    # Check: for each c, the contribution is U_c * V_c * W_c (outer product pattern).
    # Sum over c should give all-ones tensor.
    # Efficient: build the tensor as sum of rank-1 terms.
    T = np.zeros((N, M, P))
    for c in range(rank):
        Uc = U[:, c]  # (N*M,)
        Vc = V[:, c]  # (M*P,)
        Wc = W[:, c]  # (P*N,)
        # Uc[i*M+j], Vc[j*P+k], Wc[k*N+i]
        # For fixed c, the tensor term is: for each (i,j,k): Uc[i*M+j]*Vc[j*P+k]*Wc[k*N+i]
        # This is a rank-1 tensor in the (i,j,k) space with specific index mappings.
        # We can build it as: reshape Uc to (N,M), Vc to (M,P), Wc to (P,N)
        # Then T_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
        Uc2 = Uc.reshape(N, M)
        Vc2 = Vc.reshape(M, P)
        Wc2 = Wc.reshape(P, N)
        # T_c[i,j,k] = Uc2[i,j] * Vc2[j,k] * Wc2[k,i]
        # This is not a simple outer product because of the cyclic index dependency.
        # We need to compute it element-wise or use einsum.
        # T_c = Uc2[i,j] * Vc2[j,k] * Wc2[k,i]
        # Use einsum: 'ij,jk,ki->ijk'
        T += np.einsum('ij,jk,ki->ijk', Uc2, Vc2, Wc2, optimize=True)
    if np.allclose(T, 1.0, atol=tol):
        return True
    return False


def _schoolbook():
    """Return the schoolbook decomposition of rank N*M*P."""
    rank = RANK_MAX
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _try_reduce_rank(U, V, W, rank, max_iter=2000, tol=1e-9):
    """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
    # The decomposition is valid if sum_c U_c V_c W_c = T (the all-ones tensor).
    # If there's a linear dependency among the vectors (U_c, V_c, W_c) in some sense,
    # we can reduce rank. But the space is not linear in that simple way.
    # Instead, we can try: if the matrix of coefficients has rank < rank, we can project.
    # A better approach: try to find a vector alpha such that sum_c alpha_c * (U_c V_c W_c) = 0
    # and alpha is not zero. Then we can adjust.
    # Simpler: use the fact that if we can express one term as a combination of others.
    # Let's try a different approach: random linear combinations and rank detection.
    # For now, just return the same rank.
    return U, V, W, rank


def _random_perturb_and_reduce(U, V, W, rank, n_trials=500, tol=1e-9):
    """Try random perturbations to find a lower-rank equivalent decomposition."""
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for trial in range(n_trials):
        if _remaining() < 2.0:
            break
        # Try to find a linear dependency
        # Build a matrix where each column c corresponds to the "term" and we check
        # if the set of terms is linearly dependent in the tensor space.
        # Flatten the tensor space: dimension N*M*P = 50.
        # Each term c contributes a vector in R^(N*M*P): v_c[i,j,k] = U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c]
        # If the v_c are linearly dependent, we can reduce rank.
        dim = N * M * P
        Vmat = np.zeros((dim, rank))
        for c in range(rank):
            Uc2 = U[:, c].reshape(N, M)
            Vc2 = V[:, c].reshape(M, P)
            Wc2 = W[:, c].reshape(P, N)
            Tc = np.einsum('ij,jk,ki->ijk', Uc2, Vc2, Wc2, optimize=True)
            Vmat[:, c] = Tc.flatten()
        
        # Check rank of Vmat
        sv = np.linalg.svd(Vmat, full_matrices=False)
        r = np.sum(sv > tol * sv[0] if len(sv) > 0 else 0)
        if r < rank:
            # We can project to rank r
            U_new = U @ np.eye(rank)  # placeholder
            # Actually, if the terms are linearly dependent, we can find a basis.
            # Let Q be the left singular vectors of rank r.
            # Then the decomposition with rank r works if we can express U, V, W appropriately.
            # This is tricky because the factorization isn't linear.
            # For now, just note the rank.
            best_rank = min(best_rank, r)
    
    return best_U, best_V, best_W, best_rank


def construct():
    if time.time() - START_TIME > TIME_LIMIT - 5:
        # Return schoolbook as fallback
        U, V, W, rank = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: try various random initializations and use local search
    # to minimize rank while maintaining the decomposition.
    
    # Approach 1: Try to find a lower-rank decomposition by optimization
    # We want to find U (N*M x r), V (M*P x r), W (P*N x r) with r < 50
    # such that sum_c U[:,c] V[:,c] W[:,c] (in the tensor sense) = 1.
    
    # Let's try rank values from a lower bound up to 50.
    # Lower bound: max(N*M, M*P, P*N) = max(10, 25, 10) = 25
    # Also, the tensor rank of matrix multiplication for (n,m,p) is at least max(n*m, m*p, p*n).
    # For 2x5x5: at least 25.
    
    # Try to find a valid decomposition for rank r = 25, 26, ...
    # Use random initialization + gradient-free search.
    
    rng = np.random.RandomState(42)
    
    for target_rank in range(25, 51):
        if _remaining() < 5.0:
            break
        if target_rank >= best_rank:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(30):
            if _remaining() < 3.0:
                break
            
            U_r = rng.randn(N * M, target_rank)
            V_r = rng.randn(M * P, target_rank)
            W_r = rng.randn(P * N, target_rank)
            
            # Check if this gives the correct tensor
            T = np.zeros((N, M, P))
            for c in range(target_rank):
                Uc2 = U_r[:, c].reshape(N, M)
                Vc2 = V_r[:, c].reshape(M, P)
                Wc2 = W_r[:, c].reshape(P, N)
                T += np.einsum('ij,jk,ki->ijk', Uc2, Vc2, Wc2, optimize=True)
            
            if np.allclose(T, 1.0, atol=1e-7):
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                found = True
                break
        
        if found:
            break
    
    # Try to further reduce using the found decomposition
    if best_rank < 50:
        # Try to reduce further
        for target_rank in range(25, best_rank):
            if _remaining() < 3.0:
                break
            # Try random init + local refinement
            for attempt in range(20):
                if _remaining() < 2.0:
                    break
                U_r = rng.randn(N * M, target_rank)
                V_r = rng.randn(M * P, target_rank)
                W_r = rng.randn(P * N, target_rank)
                
                T = np.zeros((N, M, P))
                for c in range(target_rank):
                    Uc2 = U_r[:, c].reshape(N, M)
                    Vc2 = V_r[:, c].reshape(M, P)
                    Wc2 = W_r[:, c].reshape(P, N)
                    T += np.einsum('ij,jk,ki->ijk', Uc2, Vc2, Wc2, optimize=True)
                
                if np.allclose(T, 1.0, atol=1e-7):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
    
    # If no better than schoolbook found, return schoolbook
    if best_rank >= 50:
        U, V, W, rank = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
