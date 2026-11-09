"""Rank-minimal exact tensor decomposition for 2x3 times 3x7 matrix multiplication.

Uses a structured approach: exploit the small dimension n=2 to find decompositions
where rank-1 components are supported on combinatorial rectangles. Each component
covers a subset of (i,j,k) triples where the index sets form a product structure.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]
DEADLINE = 130.0  # Safety margin before 150s kill


def _seed():
    """Schoolbook rank-42 decomposition (fallback)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the tensor using Fraction arithmetic."""
    rank = len(U[0])
    # Check consistent triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = Fraction(0)
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 1:
                    return False
    # Check inconsistent triples
    for u in range(n * m):
        i, j = divmod(u, m)
        for v in range(m * p):
            j2, k = divmod(v, p)
            for w in range(p * n):
                k2, i2 = divmod(w, n)
                if j == j2 and k == k2 and i == i2:
                    continue
                s = Fraction(0)
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 0:
                    return False
    return True


def _try_structured_decomposition(rank, time_limit):
    """
    Search for structured rank-`rank` decomposition.
    
    Strategy: Each rank-1 component r is associated with:
    - A subset I_r ⊆ {0,1} (rows of A)
    - A subset J_r ⊆ {0,1,2} (cols of A / rows of B)  
    - A subset K_r ⊆ {0,...,6} (cols of B)
    - Scalars a_r, b_r, c_r
    
    The component contributes to T[i,j,k] iff i∈I_r, j∈J_r, k∈K_r.
    For the decomposition to work:
    - For each valid (i,j,k): sum over r with i∈I_r, j∈J_r, k∈K_r of a_r*b_r*c_r = 1
    - For invalid (u,v,w): the sum must be 0
    
    We'll search over partitions of the index sets.
    """
    n, m, p = DIMENSIONS
    rows_u = n * m  # 6
    rows_v = m * p  # 21
    rows_w = p * n  # 14
    
    end_time = time.time() + time_limit
    
    # Precompute valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid_triples.append((u, v, w, i, j, k))
    
    # Try different structured approaches
    
    # Approach 1: For each i, try to decompose the 3x7 sub-tensor
    # The sub-tensor for fixed i has entries T[j,k] = 1 for all j,k
    # This is just the all-ones matrix, which has rank 1!
    # So for fixed i, we can use 1 rank-1 component: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # But this would give rank 2 (one for i=0, one for i=1), which is too optimistic.
    # The issue: cross-terms. When i=0 component has U[0*m+j, r]=1, it also affects
    # W[k*n+0, r] which is shared with i=0 only. But U[0*m+j, r] for j in 0..2 affects
    # all V[j*p+k, r] for all k. So the i=0 component contributes to T[0,j,k] for all j,k,
    # but also to T[1,j,k] if U[1*m+j, r] is nonzero.
    
    # Let me try: separate the i=0 and i=1 cases completely.
    # For i=0: U[0*m+j, r] nonzero only for r in R0, U[1*m+j, r] = 0 for r in R0
    # For i=1: U[0*m+j, r] = 0 for r in R1, U[1*m+j, r] nonzero only for r in R1
    # Similarly for W: W[k*n+0, r] nonzero only for r in R0, W[k*n+1, r] = 0 for r in R0
    # W[k*n+0, r] = 0 for r in R1, W[k*n+1, r] nonzero only for r in R1
    
    # Then the i=0 and i=1 sub-problems are decoupled.
    # Each sub-problem is: decompose the 3x7 all-ones tensor.
    # The 3x7 all-ones matrix has rank 1, so we need only 1 component per i.
    # Total rank: 2.
    
    # But wait: the V matrix is shared. V[j*p+k, r] for r in R0 must satisfy:
    # For i=0: sum_{r in R0} U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1 for all j,k
    # If R0 has 1 component r0: U[0*m+j, r0] * V[j*p+k, r0] * W[k*n+0, r0] = 1 for all j,k
    # This requires U[0*m+j, r0] * V[j*p+k, r0] * W[k*n+0, r0] to be constant in j,k.
    # Let U[0*m+j, r0] = a_j, V[j*p+k, r0] = b_{j,k}, W[k*n+0, r0] = c_k
    # Then a_j * b_{j,k} * c_k = 1 for all j,k.
    # This implies b_{j,k} = 1/(a_j * c_k), so V[j*p+k, r0] = 1/(a_j * c_k).
    # For this to be a valid entry (integer/half-integer), we need a_j * c_k to divide 1 nicely.
    # If we set a_j = 1, c_k = 1, then b_{j,k} = 1. This works!
    
    # So rank-2 decomposition:
    # Component r0 (for i=0):
    #   U[0*m+j, r0] = 1 for j=0,1,2; U[1*m+j, r0] = 0
    #   V[j*p+k, r0] = 1 for all j,k
    #   W[k*n+0, r0] = 1 for k=0..6; W[k*n+1, r0] = 0
    # Component r1 (for i=1):
    #   U[0*m+j, r1] = 0; U[1*m+j, r1] = 1 for j=0,1,2
    #   V[j*p+k, r1] = 1 for all j,k
    #   W[k*n+0, r1] = 0; W[k*n+1, r1] = 1 for k=0..6
    
    # Let me verify this:
    U = [[0] * 2 for _ in range(rows_u)]
    V = [[0] * 2 for _ in range(rows_v)]
    W = [[0] * 2 for _ in range(rows_w)]
    
    # Component 0: i=0
    for j in range(m):
        U[j][0] = 1  # U[0*m + j, 0]
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        W[k * n][0] = 1  # W[k*n + 0, 0]
    
    # Component 1: i=1
    for j in range(m):
        U[m + j][1] = 1  # U[1*m + j, 1]
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1
    for k in range(p):
        W[k * n + 1][1] = 1  # W[k*n + 1, 1]
    
    # Convert to Fraction
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    if _check_decomposition(Uf, Vf, Wf, n, m, p):
        return {"U": Uf, "V": Vf, "W": Wf}
    
    # If the simple rank-2 didn't work, try other structured approaches
    # But given the analysis, it should work. Let me double-check the cross-terms.
    
    # For an invalid triple (u,v,w) where u=i*m+j, v=j2*p+k, w=k2*n+i2:
    # Sum_r U[u,r]*V[v,r]*W[w,r]
    # = U[i*m+j, 0]*V[j2*p+k, 0]*W[k2*n+i2, 0] + U[i*m+j, 1]*V[j2*p+k, 1]*W[k2*n+i2, 1]
    
    # Term 0: U[i*m+j, 0] is 1 if i=0, else 0. W[k2*n+i2, 0] is 1 if i2=0, else 0.
    # So term 0 is 1 if i=0 and i2=0, else 0. If i=0 and i2=0, then term 0 = V[j2*p+k, 0] = 1.
    # But this is an invalid triple, so (j, k) != (j2, k2) or i != i2.
    # If i=0, i2=0, but j != j2 or k != k2, then term 0 = 1, which is wrong!
    
    # So the simple rank-2 decomposition does NOT work because of cross-terms.
    # The V matrix is shared between i=0 and i=1, causing interference.
    
    # We need V to be different for i=0 and i=1, but V is shared.
    # So we need more components to separate the cases.
    
    # Let me try: for each (i, j) pair, have a separate component?
    # That gives 6 components, but might not be enough.
    
    # Alternative: use the fact that V[j*p+k, r] can depend on j.
    # For the i=0 case: we need sum_r U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
    # For the i=1 case: we need sum_r U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] = 1
    # And cross-terms must vanish.
    
    # Let me try a rank-3 decomposition with more careful structure.
    
    # Given time constraints, let me just try random search with better initialization.
    
    rng = np.random.default_rng()
    
    def compute_error(U, V, W, sample_size=100):
        """Compute approximation error."""
        err = 0.0
        for (u, v, w, i, j, k) in valid_triples:
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            err += (s - 1.0) ** 2
        for _ in range(sample_size):
            u = rng.integers(0, rows_u)
            v = rng.integers(0, rows_v)
            w = rng.integers(0, rows_w)
            i, j = divmod(u, m)
            j2, k2 = divmod(v, p)
            k3, i2 = divmod(w, n)
            if j == j2 and k2 == k3 and i == i2:
                continue
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            err += s ** 2
        return err
    
    # Try random search for the given rank
    for attempt in range(500):
        if time.time() > end_time:
            break
        
        U = np.zeros((rows_u, rank), dtype=float)
        V = np.zeros((rows_v, rank), dtype=float)
        W = np.zeros((rows_w, rank), dtype=float)
        
        # Initialize with structured pattern: each component covers a random subset
        for r in range(rank):
            # Random support pattern
            i_support = rng.integers(0, 2)  # Which i this component targets
            j_support = set(rng.choice(m, size=rng.integers(1, m + 1), replace=False))
            k_support = set(rng.choice(p, size=rng.integers(1, p + 1), replace=False))
            
            for j in j_support:
                U[i_support * m + j, r] = 1.0
            for j in j_support:
                for k in k_support:
                    V[j * p + k, r] = 1.0
            for k in k_support:
                W[k * n + i_support, r] = 1.0
        
        # Local optimization
        for it in range(10):
            if time.time() > end_time:
                break
            for r in range(rank):
                U_trial = U.copy()
                U_trial[:, r] += rng.normal(0, 0.1, rows_u)
                if compute_error(U_trial, V, W, 20) < compute_error(U, V, W, 20):
                    U = U_trial
                
                V_trial = V.copy()
                V_trial[:, r] += rng.normal(0, 0.1, rows_v)
                if compute_error(U, V_trial, W, 20) < compute_error(U, V, W, 20):
                    V = V_trial
                
                W_trial = W.copy()
                W_trial[:, r] += rng.normal(0, 0.1, rows_w)
                if compute_error(U, W_trial, W, 20) < compute_error(U, V, W, 20):
                    W = W_trial
        
        err = compute_error(U, V, W, 200)
        if err < 1e-6:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            Uf = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in U_rounded]
            Vf = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in V_rounded]
            Wf = [[Fraction(int(round(x * 2))) / 2 for x in row] for row in W_rounded]
            
            if _check_decomposition(Uf, Vf, Wf, n, m, p):
                return {"U": Uf, "V": Vf, "W": Wf}
    
    return None


def construct():
    start_time = time.time()
    
    # Try ranks from 3 down to 41
    for rank in range(3, 42):
        if time.time() > DEADLINE - 10:
            break
        time_left = min(10.0, DEADLINE - 10 - (time.time() - start_time))
        if time_left <= 0:
            break
        res = _try_structured_decomposition(rank, time_left)
        if res is not None:
            return res
    
    # Fallback to schoolbook
    return _seed()
