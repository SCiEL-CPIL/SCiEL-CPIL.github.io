"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

Strategy:
- The 4-dimensional inner product can be split into two 2-dimensional blocks.
- Each block is a 2x2 * 2x5 multiplication.
- For each 2x2 * 2x5, we can use a Strassen-like construction.
- The 2x2 * 2x2 part has rank 3 (Strassen).
- For 2x2 * 2x5, we can think of it as 2x2 * (2x2 + 2x3) but that doesn't help directly.
- Alternative: 2x2 * 2x5 can be done with rank at most 2*2*5 = 20 schoolbook,
  but we can do better.

Actually, let's think about this more carefully.

For 2x4 * 4x5:
- A is 2x4, B is 4x5, C is 2x5.
- C[i,k] = sum_j A[i,j] * B[j,k]

The tensor has 2*4*5 = 40 non-zero entries (all 1s) at positions (i*m+j, j*p+k, k*n+i).

Lower bound on rank: The tensor is a 3-tensor of shape (8, 20, 10).
A simple lower bound: rank >= max(8, 20, 10) = 20? No, that's not right.
The rank must be at least the maximum of the dimensions of the slices.

Actually, for matrix multiplication tensor of size n x m * m x p, the rank is at least
max(n*m, m*p, n*p) / max(n, m, p) ... this isn't a standard bound.

Let me think about known results:
- 2x2 * 2x2 has rank 3 (Strassen)
- 2x2 * 2x3: we can write this as 2x2 * (2x2 + 2x1) but that's not helpful.
  Actually, 2x2 * 2x3 can be done with rank 6? Let's see.
  
For 2x2 * 2x3:
C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k], for k=0,1,2.
This is like two independent 2x2 * 2x1 multiplications sharing the A matrix.
Each 2x2 * 2x1 has rank 2 (schoolbook: 2*2*1=4, but actually rank 2).
Wait, 2x2 * 2x1: C[i] = sum_j A[i,j] B[j]. This is just a linear combination.
The tensor for 2x2 * 2x1 has shape (4, 2, 2). Rank? 
T[a,b,c] where a=(i,j), b=j, c=i. T[i*m+j, j, i] = 1.
This is the "identity" tensor in some sense. Rank is 2? Let's check.
We need: sum_r U[i,j,r] * V[j,r] * W[i,r] = delta(i,j) ... no.
T[i*2+j, j, i] = 1.
For r=0: U[0,0,0]=1, V[0,0]=1, W[0,0]=1 gives T[0,0,0]=1.
U[0,1,0]=0, V[1,0]=0, W[1,0]=0.
U[1,0,0]=0, V[0,1]=0, W[0,1]=0.
U[1,1,0]=1, V[1,1]=1, W[1,1]=1 gives T[3,1,1]=1.
So rank 2 works for 2x2 * 2x1.

For 2x2 * 2x3, we have 3 columns in B. We can do:
C[:,0] = A * B[:,0] (rank 2)
C[:,1] = A * B[:,1] (rank 2)  
C[:,2] = A * B[:,2] (rank 2)
But these share the A factors. The total rank is at most 6.
Can we do better? Probably not much.

So for 2x4 * 4x5:
Split into two 2x2 * 2x5 blocks.
Each 2x2 * 2x5: think of B as 2x5 = 2x2 + 2x2 + 2x1 (three 2x1 blocks? No, 5 = 2+2+1).
Actually, 2x2 * 2x5:
C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k], for k=0..4.
This is 5 independent 2x2 * 2x1 multiplications, each of rank 2.
Total rank: at most 10. But we can share structure.

Actually, the rank of 2x2 * 2x5 is at most 2*2*5 = 20 schoolbook, but likely lower.
Known result: rank of 2x2 * 2xn is at most 2n for n>=2? Or is it 3n-1?

Let me just try to construct a rank-12 solution directly.

For 2x4 * 4x5:
Split A = [A1 | A2] where A1, A2 are 2x2.
Split B = [B1; B2] where B1, B2 are 2x5.
C = A1*B1 + A2*B2.

Each A_i * B_i is 2x2 * 2x5.
If we can do each with rank 6, total rank = 12.

For 2x2 * 2x5 with rank 6:
We need 6 rank-1 tensors that sum to the 2x2 * 2x5 tensor.
The 2x2 * 2x5 tensor has 2*2*5 = 20 non-zero entries.

Let me try to construct it. For 2x2 * 2x2, Strassen uses 3 multiplications.
For 2x2 * 2x5, we can think of it as 2x2 * (2x2 + 2x2 + 2x1).
If we use Strassen for each 2x2 * 2x2 block (3 each) and rank 2 for 2x2 * 2x1,
total: 3 + 3 + 2 = 8. That's more than 6.

Alternatively, maybe rank 6 is achievable for 2x2 * 2x5.

Let me just try to find a rank-12 decomposition computationally using a smarter approach.

I'll use a structured random search with the constraint that entries are half-integers,
and use the block structure to guide the search.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _build_target(n, m, p):
    """Build the target matmul tensor."""
    nm, mp, np_ = n * m, m * p, n * p
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _check(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    nm, mp, np_ = n * m, m * p, n * p
    r = U.shape[1]
    one_pos = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                one_pos.add((i * m + j, j * p + k, k * n + i))
    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                expected = 1.0 if (a, b, c) in one_pos else 0.0
                if abs(val - expected) > tol:
                    return False
    return True


def _snap_half(x):
    return np.round(np.asarray(x, dtype=float) * 2) / 2.0


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((n * p, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_rank_r(T, r, n, m, p, max_iters=500, time_budget=None, start_time=None):
    """Try to find rank-r decomposition using coordinate descent with half-integer snapping."""
    if start_time is None:
        start_time = time.time()
    nm, mp, np_ = T.shape
    
    best_err = np.inf
    best = None
    
    for restart in range(20):
        if time_budget is not None and time.time() > time_budget - 5:
            break
        
        # Initialize with small random half-integers
        U = _snap_half(np.random.randn(nm, r) * 0.5)
        V = _snap_half(np.random.randn(mp, r) * 0.5)
        W = _snap_half(np.random.randn(np_, r) * 0.5)
        
        prev_err = np.inf
        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget - 5:
                break
            
            # Coordinate descent: update one factor at a time
            for factor_idx in range(3):
                # Compute residual
                T_res = T.copy()
                for rr in range(r):
                    T_res -= np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
                
                # For each column, try to optimize
                for rr in range(r):
                    # Remove current contribution
                    T_res2 = T_res + np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
                    
                    # Optimize U[:, rr] given V[:, rr] and W[:, rr]
                    # T_res2[a, b, c] = sum_{rr' != rr} ... + U[a, rr] * V[b, rr] * W[c, rr]
                    # So U[a, rr] should minimize sum_{b,c} (T_res2[a,b,c] - U[a,rr]*V[b,rr]*W[c,rr])^2
                    # This is a linear system: for each a, U[a,rr] = sum_{b,c} T_res2[a,b,c]*V[b,rr]*W[c,rr] / sum_{b,c} V[b,rr]^2 * W[c,rr]^2
                    
                    vw = V[:, rr] * W[:, rr]  # shape (mp * np_,) ... no
                    # Actually: sum_{b,c} T_res2[a,b,c] * V[b,rr] * W[c,rr]
                    # and sum_{b,c} V[b,rr]^2 * W[c,rr]^2
                    
                    num = np.einsum('abc,b,c->a', T_res2, V[:, rr], W[:, rr])
                    den = np.sum(V[:, rr] ** 2) * np.sum(W[:, rr] ** 2)
                    if den > 1e-15:
                        U[:, rr] = num / den
                        U[:, rr] = _snap_half(U[:, rr])
                    
                    # Similarly for V
                    uw = U[:, rr] * W[:, rr]
                    num_v = np.einsum('abc,a,c->b', T_res2, U[:, rr], W[:, rr])
                    den_v = np.sum(U[:, rr] ** 2) * np.sum(W[:, rr] ** 2)
                    if den_v > 1e-15:
                        V[:, rr] = num_v / den_v
                        V[:, rr] = _snap_half(V[:, rr])
                    
                    # And W
                    uv = U[:, rr] * V[:, rr]
                    num_w = np.einsum('abc,a,b->c', T_res2, U[:, rr], V[:, rr])
                    den_w = np.sum(U[:, rr] ** 2) * np.sum(V[:, rr] ** 2)
                    if den_w > 1e-15:
                        W[:, rr] = num_w / den_w
                        W[:, rr] = _snap_half(W[:, rr])
            
            # Compute error
            T_recon = np.zeros_like(T)
            for rr in range(r):
                T_recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
            err = np.linalg.norm(T - T_recon)
            
            if err < 1e-9:
                best = (U.copy(), V.copy(), W.copy())
                return best
            
            if err > prev_err * (1 + 1e-6):
                break
            prev_err = err
        
        # Check final error
        T_recon = np.zeros_like(T)
        for rr in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
        err = np.linalg.norm(T - T_recon)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
    
    return best


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p
    
    T = _build_target(n, m, p)
    
    # Try ranks from 12 upward
    for r in range(12, 41):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_r(T, r, n, m, p, max_iters=300,
                             time_budget=deadline, start_time=start_time)
        if result is not None:
            U, V, W = result
            U_s, V_s, W_s = _snap_half(U), _snap_half(V), _snap_half(W)
            if _check(U_s, V_s, W_s, n, m, p):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = r
                break
            if _check(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
