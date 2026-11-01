"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n=2, m=3, p=10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n*m: return False
    if len(V) != m*p: return False
    if len(W) != p*n: return False
    for _ in range(len(U)):
        if len(U[_]) != rank: return False
    for _ in range(len(V)):
        if len(V[_]) != rank: return False
    for _ in range(len(W)):
        if len(W[_]) != rank: return False
    
    # For each (i,j,k), the coefficient should be 1 if this is a valid mult term, 0 otherwise
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Our decomposition: C[i,k] = sum_c (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k'])
    # We need: for each i,k: sum_c (sum_{i',j'} U[i'*m+j',c] A[i',j']) * (sum_{j',k'} V[j'*p+k',c] B[j',k']) = sum_j A[i,j] B[j,k]
    # This must hold for ALL A, B, so the tensor must match exactly.
    # The tensor T[i,k] as a bilinear form: T[i,k](A,B) = sum_j A[i,j] B[j,k]
    # Our decomposition gives: T_decomp[i,k](A,B) = sum_c (sum_{i',j'} U[i'*m+j',c] A[i',j']) (sum_{j',k'} V[j'*p+k',c] B[j',k'])
    # For this to equal T[i,k], we need for each i,k,j,j': the coefficient of A[i',j']B[j',k'] in T_decomp[i,k] matches T[i,k]
    # T[i,k] has coefficient 1 for (i',j')=(i,j) and (j',k')=(j,k), i.e., A[i,j]B[j,k], and 0 otherwise.
    # T_decomp[i,k] coefficient of A[i',j']B[j',k'] = sum_c U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i,c]
    # Wait, W is indexed by W[k*n+i, c] per the convention.
    # So: sum_c U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i,c] should equal:
    #   1 if i'=i, j'=j, k'=k for some j (i.e., i'=i, k'=k, j'=j) -- but actually for fixed i,k, the nonzero terms are A[i,j]B[j,k] for j=0..m-1
    #   So coefficient of A[i',j']B[j',k'] in T[i,k] is 1 if i'=i AND k'=k AND j'=j' (any j')... no.
    #   T[i,k] = sum_j A[i,j] B[j,k]. So the coefficient of A[i',j']B[j',k'] is 1 if i'=i, k'=k, and j'=j' (the same j'). Actually it's 1 if i'=i AND k'=k AND j'=j' where j' is free... 
    #   More precisely: coeff of A[a,b]B[c,d] in T[i,k] is 1 if a=i, d=k, and b=c (since we need A[i,b]B[b,k]). Otherwise 0.
    # So: sum_c U[a*m+b, c] * V[c*p+d, c] * W[d*n+i, c] = (1 if a==i and d==k and b==c else 0)
    # Wait, W index is W[k'*n+i, c] where k' is the row of B / column of C. So W index = d*n+i.
    # And V index = c*p+d where c is the column of A / row of B.
    # U index = a*m+b.
    # Condition: sum_c U[a*m+b, c] * V[c*p+d, c] * W[d*n+i, c] = delta(a,i)*delta(d,k)*delta(b,c)
    # for all a,b,c,d,i,k.
    
    # Let's verify a sample of entries to save time, but for exactness we should check all.
    # n*m*p = 60, rank up to 60, so checking all (a,b,c,d) for each (i,k) is 2*10 * 2*3*3*10 = 20*180 = 3600 sums of up to 60 terms = 216000 ops. Fine.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            # Expected: for each (a,b,c,d), coeff = 1 if a==i, d==k, b==c else 0
            # Computed: sum_c U[a*m+b, c] * V[c*p+d, c] * W[d*n+i, c]
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and d == k and b == c) else 0.0
                            # Compute sum over rank
                            s = 0.0
                            for r in range(rank):
                                s += U_arr[a*m+b, r] * V_arr[c*p+d, r] * W_arr[d*n+i, r]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _schoolbook(n=2, m=3, p=10):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_random_rank(r, n=2, m=3, p=10, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition via random init + least squares."""
    rng = np.random.default_rng(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the linear system. We have variables U (nm x r), V (mp x r), W (pn x r).
    # The constraint is: for each (i,k,a,b,c,d): sum_r U[a*m+b,r] V[c*p+d,r] W[d*n+i,r] = target
    # This is nonlinear. Instead, use iterative approach: fix two, solve for third.
    
    # Initialize
    U = rng.standard_normal((nm, r))
    V = rng.standard_normal((mp, r))
    W = rng.standard_normal((pn, r))
    
    # Normalize to avoid overflow
    U /= np.linalg.norm(U) * np.sqrt(nm)
    V /= np.linalg.norm(V) * np.sqrt(mp)
    W /= np.linalg.norm(W) * np.sqrt(pn)
    
    best_err = float('inf')
    best = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iters):
        # Fix V and W, solve for U (linear least squares)
        # For each (a,b): sum_r U[a*m+b,r] * (V[c*p+d,r] * W[d*n+i,r]) = target for all (i,k,c,d)
        # This is messy. Let's do a simpler alternating approach.
        
        # Fix V, W: for each (a,b), U[a*m+b, :] should satisfy:
        # For all i,k,c,d: sum_r U[a*m+b,r] * V[c*p+d,r] * W[d*n+i,r] = delta(a,i)*delta(d,k)*delta(b,c)
        # Let M_{(i,k,c,d), r} = V[c*p+d, r] * W[d*n+i, r]
        # Then M @ U[a*m+b, :]^T = target_{(i,k,c,d)}
        # But this couples across (i,k) for the same (a,b).
        
        # Simpler: use a different parameterization. Let's just do gradient descent on the error.
        pass
    
    # This is getting complex. Let's use a different strategy: 
    # For small r, do a structured search.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, get the schoolbook solution as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = n * m * p  # 60
    
    # Try to find lower rank decompositions
    # The tensor rank for (n x m) * (m x p) is at least max(n*m, m*p, n*p) and at most n*m*p
    # Lower bound: max(n*m, m*p, n*p) = max(6, 30, 20) = 30
    # So try ranks from 30 to 59
    
    # Strategy: Use the structure. For each of the p=10 columns, we have a 2x3 by 3x1 multiplication.
    # The 2x3 by 3x1 tensor has rank 6. But maybe we can share across columns.
    
    # Alternative: use a known result or try random search with verification.
    # Given time constraints, let's try a few ranks with random restarts.
    
    for r in range(30, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.default_rng(1000 + r * 100 + attempt)
            
            # Random initialization
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            W = rng.standard_normal((p*n, r))
            
            # Normalize
            for mat in [U, V, W]:
                norm = np.linalg.norm(mat)
                if norm > 0:
                    mat /= norm * np.sqrt(mat.size / r)
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                found = True
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
        
        if found:
            # Found a lower rank, continue searching for even lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
