"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import svd, lstsq
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct from decomposition
    T_recon = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        T_recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    
    return np.allclose(T_recon, T, atol=1e-10)

def _round_to_half_integers(M, tol=1e-6):
    """Check if matrix entries are close to half-integers, and round them."""
    M = np.array(M, dtype=float)
    M2 = M * 2
    M_rounded = np.round(M2)
    if np.allclose(M2, M_rounded, atol=tol):
        return (M_rounded / 2.0).tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    schoolbook_rank = n * m * p  # 72
    
    deadline = time.time() + 145  # Safety margin
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try various rank-minimization strategies
    best_result = None
    best_rank = schoolbook_rank
    
    # Strategy 1: Try known low-rank constructions
    # For 3x3x3, Winograd gives rank 16 (or 23 with certain conventions)
    # For 3x3x8, we can use tensor product / block structure
    
    # The matrix multiplication tensor for (n,m,p) can be decomposed using
    # the structure of the underlying matrix multiplication.
    
    # Key insight: 3x3x8 matrix multiplication can be seen as computing
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, j in 0..2, k in 0..7
    
    # We can try to exploit the fact that p=8 > m=3 to find structure.
    
    # Strategy: Use randomized tensor decomposition (CP decomposition)
    # with iterative refinement, then check for exact/half-integer solutions.
    
    def try_cp_decomposition(target_rank, max_iters=200, seed=42):
        """Try to find a CP decomposition of the tensor T with given rank."""
        rng = np.random.RandomState(seed)
        
        # Initialize randomly
        U = rng.randn(nm, target_rank)
        V = rng.randn(mp, target_rank)
        W = rng.randn(pn, target_rank)
        
        # Normalize
        for c in range(target_rank):
            norm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Alternating least squares
            # Update U: for each row a, solve (W * V) for U
            WV = np.einsum('ak,bk->ab', W, V)  # pn x mp
            for a in range(nm):
                # T[a, :, :] as vector of shape (mp*pn,)
                Ta = T[a].reshape(-1)
                # Solve U[a, :] such that sum_c U[a,c] * V[:,c] * W[:,c] approx Ta
                # This is: for each (b,c) pair in mp x pn, sum_k U[a,k] V[b,k] W[c,k] = T[a,b,c]
                # So: sum_k U[a,k] * (V[:,k] * W[:,k]) = Ta
                # Let M[a] be mp*pn x target_rank matrix where M[a][b*p+c, k] = V[b,k]*W[c,k]
                M_a = np.einsum('bk,ck->bc k', V, W).reshape(mp*pn, target_rank)
                U[a, :], residuals, _, _ = lstsq(M_a, Ta, rcond=None)
            
            # Update V
            for b in range(mp):
                M_b = np.einsum('ak,ck->ac k', U, W).reshape(nm*pn, target_rank)
                Tb = T[:, b, :].reshape(-1)
                V[b, :], residuals, _, _ = lstsq(M_b, Tb, rcond=None)
            
            # Update W
            for c in range(pn):
                M_c = np.einsum('ak,bk->ab k', U, V).reshape(nm*mp, target_rank)
                Tc = T[:, :, c].reshape(-1)
                W[c, :], residuals, _, _ = lstsq(M_c, Tc, rcond=None)
            
            # Compute residual
            T_recon = np.zeros((nm, mp, pn))
            for c in range(target_rank):
                T_recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
            residual = np.linalg.norm(T_recon - T, 'fro')
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if residual < 1e-8:
                break
            
            # Check convergence
            if it > 10 and residual > best_residual * 1.5:
                break
        
        # Check if we can round to half-integers
        U_rounded = _round_to_half_integers(best_U)
        V_rounded = _round_to_half_integers(best_V)
        W_rounded = _round_to_half_integers(best_W)
        
        if U_rounded is not None and V_rounded is not None and W_rounded is not None:
            # Verify exact decomposition
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Strategy 2: Try structured decomposition based on Kronecker products
    # The 3x3x8 multiplication tensor has a specific algebraic structure.
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k,l] = delta_{j,k} (where indices are A[i,j], B[j,k], C[i,k])
    # But here we have fixed dimensions 3x3x8.
    
    # Let's try a block approach:
    # Split the 8 columns of B into groups and use known 3x3x3 decompositions.
    
    # Actually, let's try a more systematic approach:
    # Use the fact that matrix multiplication is bilinear, and try to find
    # minimal rank via integer programming or exhaustive search for small ranks.
    
    # Strategy 3: Try specific known ranks
    # For 3x3x3, the minimal rank is 16 (Winograd).
    # For 3x3x8, we need to figure out the minimal rank.
    
    # One approach: the tensor for (n,m,p) can be related to (n,m,p') tensors.
    # If we have a decomposition of rank r for (3,3,3), can we extend it to (3,3,8)?
    
    # The (3,3,8) tensor T[i,j,k] = delta_{j,k} for i in 0..2, j in 0..2, k in 0..7
    # This is the same as the (3,3,3) tensor for the first 3 columns, plus 5 more columns.
    
    # Actually, T[i*m+j, j*p+k, k*n+i] = 1 if the indices match, 0 otherwise.
    # For p=8, k ranges 0..7, but j only ranges 0..2. So T[i*m+j, j*p+k, k*n+i] = 1
    # only when k = j (since j*p+k must have the "j" component matching).
    
    # Wait, let me re-read: T[i*m+j, j*p+k, k*n+i] = 1.
    # The indices are:
    # - First index: i*m + j, where i in 0..2, j in 0..2
    # - Second index: j*p + k, where j in 0..2, k in 0..7
    # - Third index: k*n + i, where k in 0..7, i in 0..2
    
    # So T[a, b, c] = 1 iff there exist i,j,k such that a = i*m+j, b = j*p+k, c = k*n+i.
    # This means: from b = j*p+k, we get j = b // p, k = b % p.
    # From a = i*m+j, we get i = a // m, j' = a % m. Need j' = j.
    # From c = k*n+i, we get k' = c // n, i' = c % n. Need k' = k and i' = i.
    
    # So T is the structure tensor of matrix multiplication.
    
    # Let's try CP decomposition with various ranks, starting from lower bounds.
    # Lower bound for rank of (n,m,p) matrix multiplication is at least max(n*m, m*p, p*n) / something,
    # but more precisely, the rank is at least the dimension of the space.
    
    # For 3x3x8: nm=9, mp=24, pn=24.
    # A simple lower bound: rank >= max(nm, mp, pn) / min(nm, mp, pn) * something...
    # Actually, a better lower bound comes from the fact that each "slice" must be rank-deficient.
    
    # Let's just try ranks from a reasonable range.
    # The schoolbook rank is 72. Let's try to beat that.
    
    ranks_to_try = [40, 45, 50, 55, 60, 65, 72]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        for seed in [42, 123, 456, 789, 1024, 2048]:
            if time.time() > deadline - 10:
                break
            
            result = try_cp_decomposition(r, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    # Strategy 4: Try a structured decomposition
    # Use the fact that 8 = 3 + 5, and try to combine a 3x3x3 decomposition with something.
    
    # Actually, let's try a different approach: use the "border rank" or "tensor rank"
    # via exhaustive search for small perturbations.
    
    # Strategy 5: Try the "Laderman" type decomposition adapted for rectangular case.
    # Laderman gives 7 multiplications for 2x2x2 (vs 8 schoolbook).
    # For 3x3x3, Winograd gives 16 (vs 27 schoolbook).
    
    # Let's try to construct a decomposition manually for 3x3x8.
    # The key is to find linear combinations of the entries that reduce the number
    # of required multiplications.
    
    # For now, if we didn't find a better result, fall back to schoolbook.
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
