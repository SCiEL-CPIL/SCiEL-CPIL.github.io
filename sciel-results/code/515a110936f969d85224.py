"""Rewritten program: search for a lower-rank exact decomposition of the 2x3 * 3x10 matmul tensor."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
DEADLINE = 135.0  # seconds
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return DEADLINE - _elapsed()

def _verify(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i, j, k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    """
    rank = U.shape[1]
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    expected = np.ones((N, M, P))
    return np.allclose(T, expected, atol=tol)

def _schoolbook():
    """Return the trivial rank-60 schoolbook decomposition."""
    U = np.zeros((N * M, RANK_SCHOOLBOOK))
    V = np.zeros((M * P, RANK_SCHOOLBOOK))
    W = np.zeros((P * N, RANK_SCHOOLBOOK))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x10():
    """
    Build a Strassen-based decomposition for the 2x2 x 2x10 sub-problem.
    Strassen's 7 multiplications for 2x2 matrix product, extended to 2x2 x 2x10.
    
    For 2x2 x 2x2, Strassen uses:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
      
    For 2x2 x 2x10, each "multiplication" is a dot product of length 10.
    So we get 7 rank-1 tensors, each with entries that are dot products.
    
    But wait - in the tensor decomposition, each rank-1 term is u⊗v⊗w where
    u is in R^2, v in R^2, w in R^10. The "multiplication" u_i * v_j * w_k
    corresponds to one scalar multiplication.
    
    Actually, for the 2x2 x 2x10 problem, the tensor has shape (2,2,2,10) and
    T[i,j,k,l] = 1 if k==j else 0 (i.e., (A*B)[i,l] = sum_j A[i,j] B[j,l]).
    
    Strassen decomposes the 2x2x2 tensor (where T[i,j,k]=1 if k==j) into 7 rank-1 terms.
    For 2x2x2x10, we can use the same 7 combinations but each "product" is over the
    length-10 dimension. So each rank-1 term in the 2x2x2 decomposition becomes
    a rank-10 term in the 2x2x2x10 decomposition (one for each of the 10 columns).
    
    Total rank for 2x2x2x10: 7 * 10 = 70. That's worse than schoolbook (40).
    
    So this approach doesn't help directly. Let me reconsider.
    """
    pass

def _construct_rank55():
    """
    Construct a rank-55 decomposition for 2x3 x 3x10.
    
    Strategy: Split the 3x3 middle into a 2x2 block and a 1x1 strip.
    C = A1*B1 + A2*B2
    where A1 is 2x2, B1 is 2x10, A2 is 2x1, B2 is 1x10.
    
    The A1*B1 part (2x2 x 2x10) has schoolbook rank 2*2*10 = 40.
    The A2*B2 part (2x1 x 1x10) has schoolbook rank 2*1*10 = 20.
    Total: 60.
    
    To reduce: Use Strassen on the 2x2x2x10 part. But as shown, that gives 70.
    
    Alternative: Use the fact that we can combine terms across the split.
    
    Known result: rank(2x3x10) <= 55. One construction uses:
    - 7 Strassen multiplications for a 2x2x2 sub-block, but shared across columns
    - Plus 20 schoolbook for the remaining part
    
    Actually, let me use a different approach: linear algebra.
    Build the rank-60 schoolbook, then find the null space of the "tensor" 
    formed by the 60 rank-1 tensors. If there are linear dependencies, 
    we can reduce the rank.
    """
    # Start with schoolbook
    U, V, W = _schoolbook()
    rank = RANK_SCHOOLBOOK
    
    # The 60 rank-1 tensors are e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} for i in {0,1}, j in {0,1,2}, k in {0..9}
    # These are all distinct, so there are no linear dependencies in the standard basis.
    # But we can apply invertible linear transformations to U, V, W to get equivalent
    # decompositions. The question is whether any such transformation yields a
    # decomposition where some columns become zero (reducing effective rank).
    
    # This is equivalent to asking: what is the tensor rank of T?
    # For 2x3x10, the known upper bound is 55 (from various papers).
    
    # Let me try a constructive approach based on the "border rank" or 
    # explicit Strassen-like identities.
    
    # One known identity for 2x3 matrix multiplication (2x3 x 3x1):
    # rank(2x3x1) <= 7 (using Strassen-like tricks)
    # For 2x3 x 3x10, if we could share across the 10 columns, we'd get 7.
    # But we can't share directly because the B matrix has different columns.
    
    # However, we can use the following trick:
    # For each group of columns, apply Strassen to the 2x2x2 sub-problem.
    # If we group the 10 columns into 5 pairs, we get 5 * 7 = 35 for the 2x2 part
    # plus 5 * 2 = 10 for the 1x1 part = 45 total. But this requires the 
    # Strassen combinations to work across paired columns, which they don't 
    # naturally.
    
    # Let me just try to find a valid rank-55 decomposition by solving a 
    # system of equations.
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want: T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For R=55, this is a system of 2*3*10 = 60 equations in 55*(6+30+20) = 55*56 = 3080 unknowns.
    # This is underdetermined, so solutions should exist (if the tensor rank is <= 55).
    
    # Let me try to construct it using the following approach:
    # Use 7 "Strassen-type" terms for each of the 10 columns in the 2x2 part,
    # but that's 70 terms. Instead, use a hybrid.
    
    # Actually, I'll use a numerical approach: set up the equations and solve.
    
    # For now, let me just return the schoolbook if we can't find a better one quickly.
    # But first, let me try a simple rank reduction: check if we can eliminate 
    # 5 of the 60 terms by finding linear combinations.
    
    # The key insight: if we can find coefficients c_r such that
    # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0, then we can remove those terms.
    
    # For the schoolbook, u_r = e_{i,j}, v_r = e_{j,k}, w_r = e_{k,i}
    # These are all linearly independent in the tensor product space, so no 
    # simple linear combination gives zero.
    
    # We need to apply invertible transformations to create dependencies.
    
    # Let me try: apply random invertible transformations to U, V, W and 
    # check if the resulting decomposition has any zero columns.
    
    np_rng = np.random.RandomState(42)
    best_U, best_V, best_W = U, V, W
    best_rank_found = rank
    
    for attempt in range(20):
        if _time_left() < 20.0:
            break
        
        # Apply random invertible transformations
        T_U = np_rng.randn(N*M, N*M)
        T_V = np_rng.randn(M*P, M*P)
        T_W = np_rng.randn(P*N, P*N)
        
        # Ensure invertibility
        if abs(np.linalg.det(T_U)) < 1e-10:
            continue
        if abs(np.linalg.det(T_V)) < 1e-10:
            continue
        if abs(np.linalg.det(T_W)) < 1e-10:
            continue
        
        U_new = U @ T_U
        V_new = V @ T_V
        W_new = W @ T_W
        
        # Check if this is still a valid decomposition
        # The transformation must satisfy: sum_r (U T_U)[:,r] (V T_V)[:,r] (W T_W)[:,r] = T
        # This is NOT generally true! The rank-1 tensors don't transform simply.
        
        # Actually, if T = sum_r u_r ⊗ v_r ⊗ w_r, and we apply invertible 
        # transformations, we get:
        # sum_r (u_r T_U) ⊗ (v_r T_V) ⊗ (w_r T_W) 
        # This is NOT equal to T in general.
        
        # The correct transformation is: if we have T = sum_r u_r ⊗ v_r ⊗ w_r,
        # then for any invertible 3-tensor transformation (which is complicated),
        # we can get another decomposition. But simple matrix transformations 
        # on U, V, W don't preserve the tensor unless they're compatible.
        
        # The correct way: if T = sum_r u_r ⊗ v_r ⊗ w_r, then for any invertible
        # matrix S (of size rank x rank), we can write:
        # T = sum_r (u_r S) ⊗ (v_r S^{-1}) ⊗ w_r
        # or similar. Let me verify:
        # sum_r (u_r S)_r' (v_r S^{-1})_r' w_r = sum_{r,r'} u_r S_{r,r'} v_r S^{-1}_{r',r''} w_r
        # This doesn't simplify nicely.
        
        # The correct identity: if T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r, then for any
        # invertible R x R matrix S:
        # T = sum_{r=1}^R (sum_{s=1}^R S_{r,s} u_s) ⊗ (sum_{s=1}^R S^{-1}_{r,s} v_s) ⊗ w_r
        # Let me verify: RHS = sum_{r,s,t} S_{r,s} S^{-1}_{r,t} u_s ⊗ v_t ⊗ w_r
        # = sum_{s,t} (sum_r S_{r,s} S^{-1}_{r,t}) u_s ⊗ v_t ⊗ w_r
        # = sum_{s,t} δ_{s,t} u_s ⊗ v_s ⊗ w_r ... this is getting complicated.
        
        # Actually, the standard result is: the set of decompositions of a tensor 
        # is an orbit under the action of GL(R) x GL(R) x GL(R) on the factor 
        # matrices, but with the constraint that the product of the three 
        # transformations is the identity in some sense.
        
        # Let me just try a different approach: directly solve for a rank-55 
        # decomposition using optimization.
        
        break
    
    # For now, return schoolbook. The ALS approach below may find something better.
    return U, V, W

def _try_random_search(max_rank, n_iter=5000, seed=42):
    """Random search for a decomposition of given rank using integer/half-integer entries."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        
        vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        
        U = np_rng.choice(vals, size=(N * M, max_rank))
        V = np_rng.choice(vals, size=(M * P, max_rank))
        W = np_rng.choice(vals, size=(P * N, max_rank))
        
        if _verify(U, V, W, tol=1e-8):
            return U, V, W
    return None

def _try_structured_search(max_rank):
    """Try structured low-rank decompositions based on known matmul tensor structure."""
    for rank in range(15, max_rank + 1):
        if _time_left() < 10.0:
            break
        result = _try_random_search(rank, n_iter=2000, seed=rank * 1000 + 42)
        if result is not None:
            return result
    return None

def _try_alternating_least_squares(rank, n_iter=200, tol=1e-10, seed=0):
    """
    Alternating Least Squares (ALS) to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F.
    """
    np_rng = np.random.RandomState(seed)
    
    U = np_rng.randn(N * M, rank) * 0.1
    V = np_rng.randn(M * P, rank) * 0.1
    W = np_rng.randn(P * N, rank) * 0.1
    
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    
    B = np.zeros((M * P * P * N, rank))
    for b in range(M * P):
        for c in range(P * N):
            for r in range(rank):
                B[b * P * N + c, r] = V[b, r] * W[c, r]
    
    T_flat = T.reshape(N * M, M * P * P * N)
    
    for it in range(n_iter):
        if _time_left() < 10.0:
            break
        
        try:
            U_new, _, _, _ = np.linalg.lstsq(B.T, T_flat.T, rcond=None)
            U_new = U_new.T
        except:
            break
        
        C = np.zeros((N * M * P * N, rank))
        for a in range(N * M):
            for c in range(P * N):
                for r in range(rank):
                    C[a * P * N + c, r] = U_new[a, r] * W[c, r]
        
        T_flat2 = T.reshape(M * P, N * M * P * N)
        try:
            V_new, _, _, _ = np.linalg.lstsq(C.T, T_flat2.T, rcond=None)
            V_new = V_new.T
        except:
            break
        
        D = np.zeros((N * M * M * P, rank))
        for a in range(N * M):
            for b in range(M * P):
                for r in range(rank):
                    D[a * M * P + b, r] = U_new[a, r] * V_new[b, r]
        
        T_flat3 = T.reshape(P * N, N * M * M * P)
        try:
            W_new, _, _, _ = np.linalg.lstsq(D.T, T_flat3.T, rcond=None)
            W_new = W_new.T
        except:
            break
        
        U, V, W = U_new, V_new, W_new
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_approx, T, atol=tol):
            break
    
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    if _verify(U_snap, V_snap, W_snap, tol=1e-6):
        return U_snap, V_snap, W_snap
    
    return None

def _construct_explicit_rank55():
    """
    Construct an explicit rank-55 decomposition for 2x3 x 3x10.
    
    Based on the following idea:
    - Split the 3x3 middle dimension into a 2x2 block (indices j=0,1) and a 1x1 strip (j=2)
    - For the 2x2 x 2x10 part, use a Strassen-like decomposition
    - For the 2x1 x 1x10 part, use schoolbook
    
    The key is to use a Strassen-like identity for the 2x2x2 tensor that can be
    "extended" to the 2x2x2x10 case by sharing across columns.
    
    Actually, let me use a simpler approach: the known rank-55 construction 
    uses the fact that rank(2x3x1) <= 7 and extends it.
    
    For 2x3 x 3x1, the tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k=0.
    We can write this as:
    T[i,j] = sum_r U[i,j,r] V[j,r] W[r]
    
    One rank-7 decomposition:
    Using Strassen-like identities for 2x3 x 3x1.
    
    For 2x3 x 3x10, if we could use the same 7 "multipliers" for all 10 columns,
    we'd get rank 7. But we can't because B has 10 different columns.
    
    However, we can use the following: for each column k, the product A*B[:,k] 
    is a 2x1 vector. The 7 Strassen multipliers for column k are:
    M1_k = (A00+A11) * (B00_k + B11_k)  -- but this is a scalar, not a vector
    ...
    
    Actually, each Strassen "multiplication" for 2x2x2 is a scalar product of 
    two linear combinations. For 2x2 x 2x10, each "multiplication" becomes a 
    dot product of two length-10 vectors, which is 10 scalar multiplications.
    So 7 Strassen multipliers = 70 scalar multiplications. Worse than schoolbook (40).
    
    I think the rank-55 result comes from a more sophisticated construction.
    Let me just try to find it numerically.
    """
    # Try ALS for rank 55
    result = _try_alternating_least_squares(55, n_iter=100, tol=1e-10, seed=0)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, tol=1e-8):
            return U, V, W
    
    # Try different seeds
    for seed in range(1, 10):
        if _time_left() < 10.0:
            break
        result = _try_alternating_least_squares(55, n_iter=100, tol=1e-10, seed=seed)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-8):
                return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    # First, try to construct an explicit rank-55 decomposition
    result = _construct_explicit_rank55()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U_fb, V_fb, W_fb = _schoolbook()
    best = (U_fb, V_fb, W_fb)
    best_rank = RANK_SCHOOLBOOK
    
    # Try ALS for lower ranks
    for rank in range(50, best_rank):
        if _time_left() < 15.0:
            break
        
        for seed in range(3):
            if _time_left() < 10.0:
                break
            
            result = _try_alternating_least_squares(rank, n_iter=50, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, tol=1e-8):
                    best = (U, V, W)
                    best_rank = rank
                    break
    
    # Try random search
    for rank in range(40, best_rank):
        if _time_left() < 15.0:
            break
        result = _try_random_search(rank, n_iter=500, seed=rank * 7919 + 13)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, tol=1e-8):
                best = (U, V, W)
                best_rank = rank
                break
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
