"""Matrix multiplication 2x5 by 5x5: find a rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _schoolbook(n, m, p):
    """Return the standard (schoolbook) decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_z5_decomposition(n, m, p):
    """Construct a rank-25 decomposition using the cyclic group Z_5 structure.
    
    For M_{2,5,5}, the tensor T[a,b,c] is 1 iff a=i*5+j, b=j*5+k, c=k*2+i.
    Using the 5-point DFT, the tensor decomposes into 25 rank-1 terms over Q(zeta_5).
    Since the tensor is rational, the decomposition can be expressed in Q.
    
    We use the fact that the DFT matrix F over Z_5 satisfies F F^T = 5 I, and
    the cyclic convolution structure lets us write:
      T = sum_{ell, mu} (rank-1 term)
    where the entries involve powers of zeta_5 = exp(2*pi*i/5).
    
    Since zeta_5 satisfies 1 + zeta + zeta^2 + zeta^3 + zeta^4 = 0, and
    the tensor is rational, we can use the relation to express everything in Q.
    
    Concretely, the (ell, mu) term has:
      U[a, (ell,mu)] = zeta_5^{ell * j} * (something in i)
      V[b, (ell,mu)] = zeta_5^{-ell * j + mu * k} * ...
      W[c, (ell,mu)] = zeta_5^{-mu * k} * ...
    
    Let me derive this carefully.
    
    The tensor is: T[i,j,k] = 1 (for all i in {0,1}, j in {0..4}, k in {0..4}).
    In the (a,b,c) indexing: a = i*m+j, b = j*p+k, c = k*n+i.
    
    We want sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    
    Using the DFT: let F be the 5x5 DFT matrix with F[j, ell] = zeta^{j*ell}.
    Then sum_j x[j] y[j] zeta^{j*ell} = (x * F)[ell] ... 
    
    Actually, the standard Winograd/FFT approach for matrix multiplication:
    For C = A * B where A is n x m, B is m x p:
      C[i,k] = sum_j A[i,j] B[j,k]
    
    Using DFT on the j-index:
      Let A_hat[i, ell] = sum_j A[i,j] zeta^{-j*ell}
      Let B_hat[ell, k] = sum_j B[j,k] zeta^{j*ell}
      Then C[i,k] = (1/5) sum_ell A_hat[i, ell] B_hat[ell, k]
    
    But this requires complex arithmetic and the 1/5 factor. Moreover, this gives
    rank = n * 5 * p = 2 * 5 * 5 = 50, same as schoolbook! The DFT doesn't help
    for rectangular matrices in this naive way because we still need all (i, ell, k).
    
    The key insight for reduction: since n=2, we can share the B_hat computation
    between the two rows i=0 and i=1. But A_hat[i,ell] depends on i, so we can't
    share A_hat. However, B_hat[ell,k] is shared!
    
    So the decomposition is:
      For each (i, ell, k): term with U[i*5+j, (i,ell,k)] = zeta^{-j*ell} (for the A side)
                              V[j*5+k, (i,ell,k)] = zeta^{j*ell} (for the B side) -- but this depends on j!
    
    Wait, that's not right. Let me reconsider.
    
    The issue is that B_hat[ell,k] = sum_j B[j,k] zeta^{j*ell} involves a sum over j,
    but in the tensor decomposition, each rank-1 term is U[a] V[b] W[c] where
    U depends only on a, V only on b, W only on c. So we can't have V depend on
    both j and k in a way that sums over j.
    
    Let me think differently. The tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i.
    For fixed (i,k), T is nonzero only when j is consistent, i.e., a mod m = b div p = j.
    So T[i*m+j, j*p+k, k*n+i] = 1.
    
    This is a "permutation-like" tensor. The DFT approach works when the tensor has
    a convolution structure. Here, the structure is:
      T[i,j,k] = 1 for all (i,j,k).
    
    In the (a,b,c) space, the nonzero entries form a "grid" parameterized by (i,j,k).
    The DFT over j would give:
      T_hat[i, ell, k] = sum_j T[i,j,k] zeta^{-j*ell} = zeta^{-j*ell} ... but T[i,j,k]=1 for all j,
      so T_hat[i,ell,k] = sum_j zeta^{-j*ell} = 5 if ell=0, else 0.
    
    That's degenerate! The tensor is constant in j, so its DFT is only nonzero at ell=0.
    This means the DFT doesn't decompose it usefully.
    
    I think the rank-25 claim is incorrect for this specific tensor. The tensor
    T[i,j,k] = 1 (all ones) is actually the "all-ones" tensor, which has rank...
    
    Wait, the all-ones tensor in R^{2 x 5 x 5} has rank 1! Because T[i,j,k] = 1 = 1*1*1.
    But that's not our tensor. Our tensor is T[a,b,c] = 1 iff (a,b,c) = (i*5+j, j*5+k, k*2+i)
    for some (i,j,k). This is NOT the all-ones tensor; it's a sparse tensor with 50
    nonzero entries.
    
    Let me reconsider the structure. The nonzero entries are:
      (i,j,k) -> (5i+j, 5j+k, 2k+i)
    for i in {0,1}, j in {0..4}, k in {0..4}.
    
    This is a "cyclic" tensor. The DFT over the j-index (which is cyclic mod 5) might
    help because j appears in both a and b in a cyclic way.
    
    Let me try: define the DFT over j: F[j, ell] = zeta^{j*ell}.
    Then for fixed (i,k), the slice T[i*5+j, j*5+k, 2k+i] as a function of j is 1.
    Its DFT is 5 at ell=0.
    
    So the tensor, when DFT'd over j, is:
      T_hat[i, ell, k] = 5 * delta_{ell,0}
    
    This means the tensor is "diagonal" in the DFT basis, with only ell=0 contributing.
    So the DFT doesn't reduce the rank; it just re-indexes.
    
    I think the Winograd/FFT approach doesn't apply directly here because the tensor
    is not a convolution; it's a "selection" tensor.
    
    Let me try a different approach: use the fact that n=2 to find a rank-40 or
    rank-45 decomposition via a structured search.
    
    Actually, let me just try to find a rank-45 decomposition using the optimization
    approach from inspiration 2, with a better initialization.
    """
    return None  # Placeholder; will be replaced by actual search

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check whether the decomposition exactly reproduces the matmul tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(Ua[a], Va[b] * Wa[c]))
                if abs(val - 1.0) > tol:
                    return False
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = float(np.dot(Ua[a], Va[b] * Wa[c]))
                    if abs(val) > tol:
                        return False
    return True

def _try_rank_optimization(n, m, p, target_rank, time_budget=100.0):
    """Attempt to find a rank-target_rank decomposition via optimization."""
    start = time.time()
    deadline = start + time_budget
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2
    
    def validate(U, V, W, atol=1e-4):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None
    
    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_solution = None
    
    # Strategy: Start from schoolbook and try to merge terms
    # The schoolbook has 50 terms. We want to reduce to target_rank.
    # Initialize with random linear combinations of schoolbook terms.
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        R = target_rank
        rng = np.random.RandomState(42 + restart)
        
        U = np.zeros((U_dim, R))
        V = np.zeros((V_dim, R))
        W = np.zeros((W_dim, R))
        
        # For each rank position, randomly select 5-6 schoolbook terms to combine
        for r in range(R):
            num_terms = rng.choice([5, 6, 7])
            selected = rng.choice(range(n*m*p), size=num_terms, replace=False)
            
            for idx in selected:
                i = idx // (m*p)
                j = (idx // p) % m
                k = idx % p
                
                coeff = rng.uniform(0.5, 1.5)
                U[i*m+j, r] += coeff
                V[j*p+k, r] += coeff
                W[k*n+i, r] += coeff
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        from scipy.optimize import minimize
        
        def objective(x):
            u = x[:U_dim*R].reshape(U_dim, R)
            v = x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
            w = x[U_dim*R+V_dim*R:].reshape(W_dim, R)
            return compute_loss(u, v, w)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})
            
            if res.fun < 1e-3:
                u_opt = res.x[:U_dim*R].reshape(U_dim, R)
                v_opt = res.x[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
                w_opt = res.x[U_dim*R+V_dim*R:].reshape(W_dim, R)
                
                valid, U_snap, V_snap, W_snap = validate(u_opt, v_opt, w_opt)
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    break
        except Exception:
            pass
    
    if best_solution:
        U_snap, V_snap, W_snap = best_solution
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    # Try to find a rank-45 decomposition
    result = _try_rank_optimization(n, m, p, 45, time_budget=60.0)
    if result is not None:
        return result
    
    # Try rank 46
    if time.time() < deadline - 40:
        result = _try_rank_optimization(n, m, p, 46, time_budget=35.0)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
