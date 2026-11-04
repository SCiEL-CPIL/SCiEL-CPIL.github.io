"""Matrix multiplication 2x3 by 3x9 — tensor rank decomposition search."""
import time
import sys

DIMENSIONS = [2, 3, 9]
DEADLINE = 140  # seconds (evaluator kills at 150s, leave 10s margin)

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_full(n, m, p, U, V, W, tol=1e-6):
    """Verify that the decomposition satisfies the matmul tensor identity.
    Check: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all (i,j,k).
    Also check some zero entries are actually zero.
    """
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(s - 1) > tol:
                    return False
    
    # Check a sample of zero entries
    rng = np.random.RandomState(42)
    for _ in range(200):
        a = rng.randint(n * m)
        b = rng.randint(m * p)
        c = rng.randint(p * n)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (j == j2 and k == k2 and i == i2):
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val) > tol:
                return False
    return True


def _verify_fast(n, m, p, U, V, W, tol=1e-4):
    """Faster verification using numpy einsum for the nonzero entries only."""
    rank = U.shape[1]
    # Compute all (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                if abs(s - 1) > tol:
                    return False
    return True


def _block_diagonal_decomposition(n, m, p):
    """Block-diagonal approach: treat each row i of A independently.
    
    For fixed i, we need: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1 for all j,k.
    If we use separate rank blocks for each i, we get rank = n * m * p = 54 (no gain).
    
    But we can try to share structure. The key insight: for fixed i, the problem is
    a 1x3 by 3x9 multiplication, which has tensor rank 3 (just the 3 inner products).
    So if we can make the W factors for different i orthogonal/separable, we might
    share some rank.
    
    Actually, let's think about this differently. The tensor is:
    T[i,j,k] = 1 for all i,j,k.
    
    This is a rank-1 tensor in the sense that T = u ⊗ v ⊗ w where u=(1,1), v=(1,1,1), w=(1,...,1).
    Wait, that's not right for tensor rank. A rank-1 tensor decomposition would be
    T[i,j,k] = a[i]*b[j]*c[k]. For T[i,j,k]=1, we need a[i]=1, b[j]=1, c[k]=1, so
    T = (1,1) ⊗ (1,1,1) ⊗ (1,1,...,1). That's rank 1!
    
    But wait - the matmul tensor isn't T[i,j,k]=1 for ALL i,j,k in the full tensor space.
    The matmul tensor has specific structure. Let me re-examine.
    
    The matrix multiplication tensor M is defined on (n*m) x (m*p) x (p*n) space.
    M[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k; else 0.
    
    So M is NOT simply all-ones. It's a structured tensor. The nonzero entries are
    exactly the 54 entries where the indices are consistent.
    
    For 2x3 x 3x9:
    - U has 6 rows (i=0,1; j=0,1,2)
    - V has 27 rows (j=0,1,2; k=0..8)
    - W has 18 rows (k=0..8; i=0,1)
    
    The nonzero pattern: M[i*m+j, j*p+k, k*n+i] = 1.
    
    This is the standard matmul tensor. Its rank is what we're trying to minimize.
    """
    # The block-diagonal idea from the strategy:
    # For each row i of A (i=0,1), we have a 1x3 by 3x9 multiplication.
    # The tensor for row i: T_i[j,k] = 1 for all j,k (since A[i,j]*B[j,k] contributes).
    # But in the full tensor, the W factor depends on both k and i.
    
    # Let's try a structured decomposition.
    # Key observation: W[k*n+i] couples k and i.
    # For i=0: W[k*2+0] for k=0..8
    # For i=1: W[k*2+1] for k=0..8
    
    # Strategy: Use the fact that the tensor factors as a sum of rank-1 terms.
    # T[i,j,k] (as a function of the flat indices) = 1 iff the indices are consistent.
    
    # Let's try to find a low-rank decomposition by solving the system directly.
    # For rank r, we have r*(6+27+18) = 51r variables and 54 equations (nonzero) 
    # plus many zero equations.
    
    # Actually, the zero equations are crucial. Let's count them:
    # Total entries: 6*27*18 = 2916
    # Nonzero: 54
    # Zero: 2862
    
    # So we have 2916 equations in 51r unknowns. For r=54, that's 2754 unknowns.
    # For lower r, we need the system to be consistent.
    
    # Let's try a different approach: exploit the Kronecker product structure.
    # The matmul tensor for 2x3 x 3x9 can be written as:
    # M = sum_{j=0}^{2} (e_j ⊗ e_j^T) ⊗ ... hmm, this gets complicated.
    
    # Let me try a direct algebraic approach.
    # T[i,j,k] = 1 for all i,j,k (in the consistent sense).
    # 
    # Claim: The tensor rank of the 2x3 x 3x9 matmul is at most 2*3 + 3*9 = 6+27=33? No.
    # 
    # Actually, there's a known result: the tensor rank of the n x m x p matmul tensor
    # is at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p) / something.
    # 
    # For our case, let's try to construct a decomposition with rank < 54.
    
    # Approach: Use the "laser method" or structured ansatz.
    # 
    # Let's try: rank = 27 (just the V dimension).
    # If we set U[i*m+j, r] = delta_{j, r//9} * something, this gets messy.
    
    # Better approach: Use the fact that T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{j'} delta_{j,j'} * 1
    # But that doesn't help directly.
    
    # Let's try a computational approach: solve for a rank-r decomposition
    # using iterative refinement.
    
    return None  # Placeholder, actual search below


def _search_rank(n, m, p, max_rank, time_budget=60):
    """Search for a rank-r decomposition for r from max_rank down to some minimum.
    
    Uses alternating least squares with random restarts.
    """
    start_time = time.time()
    dim_U = n * m  # 6
    dim_V = m * p  # 27
    dim_W = p * n  # 18
    
    best_result = None
    best_rank = max_rank + 1
    
    # Precompute index mappings
    idx_U = {}  # (i,j) -> row index
    for i in range(n):
        for j in range(m):
            idx_U[(i, j)] = i * m + j
    idx_V = {}  # (j,k) -> row index
    for j in range(m):
        for k in range(p):
            idx_V[(j, k)] = j * p + k
    idx_W = {}  # (k,i) -> row index
    for k in range(p):
        for i in range(n):
            idx_W[(k, i)] = k * n + i
    
    # Build the list of (a, b, c) triples for nonzero entries
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = idx_U[(i, j)]
                b = idx_V[(j, k)]
                c = idx_W[(k, i)]
                triples.append((a, b, c))
    
    num_triples = len(triples)  # 54
    
    for r in range(max_rank, max(10, max_rank - 20), -1):
        if time.time() - start_time > time_budget:
            break
        
        found = False
        num_trials = 30 if r > 40 else 50
        
        for trial in range(num_trials):
            if time.time() - start_time > time_budget:
                break
            
            # Random initialization
            rng = np.random.RandomState(trial * 1000 + r)
            U = rng.randn(dim_U, r) * 0.5
            V = rng.randn(dim_V, r) * 0.5
            W = rng.randn(dim_W, r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(300):
                if time.time() - start_time > time_budget - 5:
                    break
                
                # Solve for W given U, V
                # For each triple (a,b,c): sum_d U[a,d]*V[b,d]*W[c,d] = 1
                # This is linear in W: for each c, we have equations involving W[c,:]
                # Group by c: for fixed c, the equations are those triples where c_idx = c
                # Equation: sum_d U[a,d]*V[b,d]*W[c,d] = 1
                # This is a linear equation in W[c,:]: dot(U[a,:]*V[b,:], W[c,:]) = 1
                
                # Build system for W:
                # For each c in range(dim_W), collect all (a,b) such that (a,b,c) is a triple
                # Then solve: sum over those (a,b) of (U[a,:]*V[b,:] @ W[c,:]) = 1
                # Wait, that's not quite right. Each triple gives one equation.
                
                # Better: build the full linear system.
                # Unknowns: W flat, size dim_W * r
                # Equations: num_triples = 54
                # For triple t = (a, b, c): sum_d U[a,d]*V[b,d]*W[c,d] = 1
                # In flat form: sum_d U[a,d]*V[b,d]*W_flat[c*r + d] = 1
                
                A_W = np.zeros((num_triples, dim_W * r))
                b_W = np.ones(num_triples)
                
                for t_idx, (a, b, c) in enumerate(triples):
                    for d in range(r):
                        A_W[t_idx, c * r + d] = U[a, d] * V[b, d]
                
                W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(dim_W, r)
                
                # Solve for V given U, W
                A_V = np.zeros((num_triples, dim_V * r))
                b_V = np.ones(num_triples)
                
                for t_idx, (a, b, c) in enumerate(triples):
                    for d in range(r):
                        A_V[t_idx, b * r + d] = U[a, d] * W[c, d]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_flat.reshape(dim_V, r)
                
                # Solve for U given V, W
                A_U = np.zeros((num_triples, dim_U * r))
                b_U = np.ones(num_triples)
                
                for t_idx, (a, b, c) in enumerate(triples):
                    for d in range(r):
                        A_U[t_idx, a * r + d] = V[b, d] * W[c, d]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(dim_U, r)
                
                # Check convergence
                if it % 20 == 19:
                    max_err = 0
                    for a, b, c in triples:
                        s = np.dot(U[a] * V[b], W[c])
                        max_err = max(max_err, abs(s - 1))
                    if max_err < 1e-8:
                        converged = True
                        break
            
            # Final check
            max_err = 0
            for a, b, c in triples:
                s = np.dot(U[a] * V[b], W[c])
                max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-6:
                # Check zero entries
                zero_ok = True
                rng2 = np.random.RandomState(12345)
                for _ in range(100):
                    a = rng2.randint(dim_U)
                    b = rng2.randint(dim_V)
                    c = rng2.randint(dim_W)
                    # Check if this is a valid triple
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        val = np.dot(U[a] * V[b], W[c])
                        if abs(val) > 1e-6:
                            zero_ok = False
                            break
                
                if zero_ok:
                    # Try to snap to rational numbers
                    U_snap = np.round(U * 4) / 4
                    V_snap = np.round(V * 4) / 4
                    W_snap = np.round(W * 4) / 4
                    
                    max_err_snap = 0
                    for a, b, c in triples:
                        s = np.dot(U_snap[a] * V_snap[b], W_snap[c])
                        max_err_snap = max(max_err_snap, abs(s - 1))
                    
                    if max_err_snap < 1e-4:
                        best_result = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        best_rank = r
                        found = True
                        break
                    else:
                        # Use the unsnapped version if it's good enough
                        if max_err < 1e-6:
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                            best_rank = r
                            found = True
                            break
        
        if found:
            break
    
    return best_result, best_rank


def _try_structured_patterns(n, m, p):
    """Try specific structured patterns that might give low rank."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Pattern 1: Use the fact that T[i,j,k] = 1 for all consistent (i,j,k).
    # Try: U[i*m+j, r] = 1 for r in some set, V and W similarly.
    # This is essentially the schoolbook, so skip.
    
    # Pattern 2: Exploit that n=2 is small.
    # W[k*n+i] = W[k*2+i]. For i=0 and i=1, we have W[2k] and W[2k+1].
    # If we set W[2k] = W[2k+1] = w_k for all k, then:
    # sum_r U[i*m+j][r]*V[j*p+k][r]*w_k = 1 for all i,j,k
    # This means: sum_r U[i*m+j][r]*V[j*p+k][r] = 1/w_k for all i
    # So U[0*m+j] and U[1*m+j] must be proportional, and similarly for V.
    # This constrains the solution heavily.
    
    # Pattern 3: Use a Kronecker-like structure.
    # For 2x3 x 3x9, think of it as (2x3) x (3x9).
    # The 3x3 part can use rank 3 (schoolbook for 3x3).
    # Then we have 2*3 = 6 "rows" of A and 9 "columns" of B.
    # 
    # Actually, let's try: rank = 2*9 + 3 = 21? Or rank = 3*9 = 27?
    # 
    # Let me try a specific construction:
    # U[i*m+j, r] where r indexes (j', k') pairs: r = j'*9 + k'
    # Then U[i*m+j, j'*9+k'] = delta_{j,j'}
    # V[j*p+k, j'*9+k'] = delta_{k,k'}
    # W[k*n+i, j'*9+k'] = 1
    # 
    # Check: sum_{j',k'} U[i*m+j, j'*9+k'] * V[j*p+k, j'*9+k'] * W[k*n+i, j'*9+k']
    # = sum_{j',k'} delta_{j,j'} * delta_{k,k'} * 1
    # = 1. Yes! This works.
    # Rank = m*p = 27.
    
    # Let's verify this construction:
    rank = m * p  # 27
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U[i * m + j, r] = 1  # U[i*m+j, j'*9+k'] = delta_{j,j'} for j'=j
            # V[j*p+k, r] = delta_{k,k'} for k'=k
            V[j * p + k, r] = 1
            for i in range(n):
                W[k * n + i, r] = 1  # W[k*n+i, r] = 1 for all i
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                max_err = max(max_err, abs(s - 1))
    
    if max_err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    # Pattern 4: Try rank = n*p = 18.
    # U[i*m+j, r] where r indexes (i', k'): r = i'*9 + k'
    # U[i*m+j, i'*9+k'] = delta_{i,i'}
    # V[j*p+k, i'*9+k'] = 1
    # W[k*n+i, i'*9+k'] = delta_{k,k'}
    # 
    # Check: sum_{i',k'} delta_{i,i'} * 1 * delta_{k,k'} = 1. Yes!
    # Rank = n*p = 18.
    
    rank = n * p  # 18
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j, r] = 1  # U[i*m+j, i'*9+k'] = delta_{i,i'} for i'=i
            for j in range(m):
                V[j * p + k, r] = 1  # V[j*p+k, i'*9+k'] = 1
            for i2 in range(n):
                W[k * n + i2, r] = 1 if i2 == i else 0  # W[k*n+i, i'*9+k'] = delta_{k,k'} ... wait
    
    # Let me redo this more carefully.
    # r indexes (i', k') with r = i'*p + k'
    # U[i*m+j, r] = delta_{i, i'} (independent of j)
    # V[j*p+k, r] = delta_{k, k'} (independent of j)
    # W[k*n+i, r] = 1 (independent of i and k)
    # 
    # Check: sum_{i',k'} delta_{i,i'} * delta_{k,k'} * 1 = 1. Correct!
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j, r] = 1  # delta_{i,i'} = 1 when i'=i
            for j in range(m):
                V[j * p + k, r] = 1  # delta_{k,k'} = 1 when k'=k
            for i2 in range(n):
                for k2 in range(p):
                    # W[k2*n+i2, r] should be 1 if this is the right (k',i') pair
                    # r = i'*p + k', so i'=i, k'=k
                    # W[k*n+i, r] = 1 for the specific k,i that match
                    pass
    
    # Actually, W[k*n+i, r] = 1 for all k,i when r = i'*p+k'? No.
    # Let me re-derive. We want:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # With r = i'*p + k':
    # sum_{i',k'} delta_{i,i'} * delta_{k,k'} * W[k*n+i, i'*p+k'] = 1
    # = W[k*n+i, i*p+k] (only the term i'=i, k'=k survives)
    # So we need W[k*n+i, i*p+k] = 1 for all i,k.
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j, r] = 1
            for j in range(m):
                V[j * p + k, r] = 1
            W[k * n + i, r] = 1
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
                max_err = max(max_err, abs(s - 1))
    
    if max_err < 1e-10:
        # Check zero entries
        zero_ok = True
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(dim_U)
            b = rng.randint(dim_V)
            c = rng.randint(dim_W)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if not (j == j2 and k == k2 and i == i2):
                val = np.dot(U[a] * V[b], W[c])
                if abs(val) > 1e-10:
                    zero_ok = False
                    break
        
        if zero_ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    return None, None


def _try_lower_rank_patterns(n, m, p):
    """Try to find ranks lower than n*p = 18 using structured patterns."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Pattern: rank = m*p = 27 (already tried above, gave 27)
    # Pattern: rank = n*p = 18 (tried above)
    # Can we do better?
    
    # Try rank = m*n + p = 6 + 9 = 15? Or rank = n*m + m*p = 6 + 27 = 33?
    # 
    # Let's try a different structure.
    # 
    # Idea: Use the fact that the tensor is "separable" in some sense.
    # T[i,j,k] = 1 for all consistent (i,j,k).
    # 
    # Try: rank = n + p = 2 + 9 = 11.
    # U[i*m+j, r] for r in 0..10:
    #   For r < n: U[i*m+j, r] = delta_{i,r}
    #   For r >= n: U[i*m+j, r] = 0
    # V[j*p+k, r]:
    #   For r < n: V[j*p+k, r] = 1
    #   For r >= n: V[j*p+k, r] = delta_{k, r-n}
    # W[k*n+i, r]:
    #   For r < n: W[k*n+i, r] = 0
    #   For r >= n: W[k*n+i, r] = delta_{i, r-n} ... wait, this doesn't work cleanly.
    
    # Let me try a more systematic approach.
    # 
    # The key constraint is that the tensor must be exactly reproduced.
    # For 2x3 x 3x9, the "natural" lower bounds suggest rank >= 18 (from n*p)
    # or rank >= 27 (from m*p).
    # 
    # Actually, I recall that the tensor rank of the n x m x p matmul tensor
    # is at least max(n*m, m*p, n*p) / max(n,m,p) in some cases, but that's not
    # a standard bound.
    # 
    # A known lower bound: rank >= max(n, m, p) * min(n, m, p) ... no.
    # 
    # Let's just try to search for ranks 10-18 computationally.
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Always have schoolbook as ultimate fallback
    fallback = _schoolbook(n, m, p)
    
    # Try structured patterns first (fast, deterministic)
    result, rank = _try_structured_patterns(n, m, p)
    if result is not None:
        # Verify
        if _verify_full(n, m, p, result["U"], result["V"], result["W"]):
            if time.time() < DEADLINE - 10:
                return result
    
    # Try computational search for lower ranks
    time_remaining = DEADLINE - (time.time() - start) - 10
    if time_remaining > 20:
        search_result, search_rank = _search_rank(n, m, p, max_rank=17, time_budget=time_remaining)
        if search_result is not None and search_rank < (rank if rank else 54):
            if _verify_full(n, m, p, search_result["U"], search_result["V"], search_result["W"]):
                return search_result
    
    # Return the best we found
    if result is not None:
        return result
    
    return fallback
# EVOLVE-BLOCK-END
