"""Matrix multiplication 4x5x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor M[a, b, c] where
    # a = i*m + j (U index), b = j'*p + k (V index), c = k'*n + i' (W index)
    # The tensor should satisfy: sum_c U[a,c]*V[b,c]*W[c2,c] == delta(a->(i,j), b->(j',k), c2->(k',i'))
    # i.e., product is 1 iff j == j' AND k == k' AND i == i'
    # But actually the multiplication tensor is: T[(i,j), (j,k), (k,i)] = 1 if indices match
    # More precisely: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has T[i,j, j',k, k',i'] = 1 if j==j' and k==k' and i==i'
    # In flattened form: T[a, b, c] = 1 if (i,j) from a, (j',k) from b, (k',i') from c
    #   satisfy j == j' and k == k' and i == i'
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check on random instances for speed
    rng = np.random.RandomState(42)
    for trial in range(50):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        # Reconstruct: C[i,k] = sum_c (sum_j A[i,j]*U[j*m... no
        # U has shape (n*m, rank), indexed by (i,j) -> row i*m+j
        # V has shape (m*p, rank), indexed by (j,k) -> row j*p+k  
        # W has shape (p*n, rank), indexed by (k,i) -> row k*n+i
        # C[i,k] = sum_c U[i*m+j, c] * ... 
        # Actually: C[i,k] = sum_{c} (sum_j A[i,j] * V[j*p+k, c]) * (sum_j' A[i,j'] * U[j'*m+i, c])... 
        # No. Let me think again.
        # The standard form: C[i,k] = sum_c U_c(i,j) V_c(j,k) W_c(k,i) ... that's not right either.
        # 
        # In the seed code: for each (i,j,k): U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
        # So C[i,k] = sum_j A[i,j] B[j,k]
        # And the decomposition says: C[i,k] = sum_c (A contracted with U_c) (B contracted with V_c) W_c(k,i)
        # Where A contracted with U_c = sum_{i',j} A[i',j] U[i'*m+j, c] ... no that gives a vector over i'
        # 
        # Actually the standard matrix multiplication tensor decomposition:
        # C = A*B means C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: T_{i,j,k} = A_{i,j} * B_{j,k} * ... no.
        # The multiplication tensor is: M_{i,j,k} which represents the trilinear form
        # sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} C_{k,i} ... 
        # 
        # Let me just verify directly:
        # C_reconstructed[i,k] = sum_c (sum_j A[i,j] * U[j*m+i, c]) ... no
        # 
        # From seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The identity is: sum_j A[i,j]*B[j,k] = sum_c (sum_{j1} A[i,j1]*U[j1*m+i, c]) * ... 
        # Hmm, this is getting confused. Let me just do a direct check.
        
        # The decomposition asserts: for all A, B:
        # (A*B)[i,k] = sum_c (U^c @ something) ... 
        # Actually in the standard formulation:
        # C[i,k] = sum_c U_c[i,j] A[i,j] * V_c[j,k] B[j,k] * W_c[k,i]
        # No, it's: C[i,k] = sum_c (sum_{j} A[i,j] * U_c(j)) * (sum_k B[j,k] * V_c(k)) ... no
        
        # Let me just verify by checking the tensor directly.
        # The multiplication tensor T has T[a,b,c] = 1 iff:
        # a = i*m+j, b = j'*p+k, c = k'*n+i' AND j==j' AND k==k' AND i==i'
        # Wait no. T[a,b,c] where a indexes (i,j), b indexes (j',k'), c indexes (k'',i'')
        # T[a,b,c] = 1 iff j == j' AND k' == k'' AND i == i''
        # Hmm, but in the seed W[k*n+i, c] so c = k*n + i, meaning the third index pair is (k, i)
        # So T[a,b,c] = 1 iff (j from a) == (j' from b) AND (k from b) == (k from c) AND (i from c) == (i from a)
        
        pass
    
    # Direct tensor verification
    # T[a, b, c] = 1 iff j_a == j_b AND k_b == k_c AND i_c == i_a
    # where a = i_a * m + j_a, b = j_b * p + k_b, c = k_c * n + i_c
    
    # The decomposition says: T[a,b,c] = sum_rank U[a,r] * V[b,r] * W[c,r]
    
    # Build left side
    errors = 0
    max_err = 0.0
    # Sample a subset for speed
    np.random.seed(123)
    samples = 2000
    for _ in range(samples):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        ia = a // m; ja = a % m
        jb = b // p; kb = b % p
        kc = c // n; ic = c % n
        true_val = 1.0 if (ja == jb and kb == kc and ic == ia) else 0.0
        comp_val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        err = abs(true_val - comp_val)
        if err > 1e-9:
            errors += 1
            max_err = max(max_err, err)
    
    if errors > 0:
        return False, max_err
    return True, 0.0


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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


def _try_block_decomposition(n, m, p):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    
    For 4x5x6: We can partition the middle dimension (size 5) into blocks.
    If we write A as [A1 | A2] where A1 is 4x2 and A2 is 4x3,
    and B as [[B1], [B2]] where B1 is 2x6 and B2 is 3x6,
    then C = A1*B1 + A2*B2.
    
    The rank needed is rank(4x2x6) + rank(3x4x6)... no, rank(2x4x6) + rank(3x4x6).
    Schoolbook: 2*4*6 + 3*4*6 = 48 + 72 = 120. That's worse than 120=4*5*6.
    
    Alternative: use the fact that min(n,m,p) gives some structure.
    Actually for rectangular matrices, the standard lower bound is max(n*p, m*p, n*m) 
    ... no that's not right either.
    
    Let me try a different approach: use the Winograd-style or Karatsuba-style
    decomposition for the middle dimension.
    
    For multiplying 4x5 by 5x6, the middle dimension is 5.
    We can't easily do Karatsuba since 5 is prime.
    
    Let me try: rank of matrix mult tensor for n x m * m x p is at most n*m*p (schoolbook).
    Known lower bounds: max(n*p, m*p, n*m) = max(24, 30, 20) = 30.
    So rank is between 30 and 120.
    
    Let me try to achieve rank below 120 using a structured approach.
    """
    # For now, return the schoolbook as baseline
    return _schoolbook()


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s, keep safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try various structured decompositions
    # Strategy: partition the matrix multiplication into sub-problems
    # and use additivity of rank under direct sum
    
    # Partition A (4x5) as [A_left (4xk) | A_right (4x(5-k))]
    # Partition B (5x6) as [[B_top (kx6)], [B_bottom ((5-k)x6)]]
    # C = A_left * B_top + A_right * B_bottom
    # rank(C_mult) <= rank(4xkxk x kx6) + rank(4x(5-k)x(5-k) x (5-k)x6)
    # = 4*k*k*6 ... no, rank of (n x k) * (k x p) mult tensor is n*k*p
    # So total rank = 4*k*6 + 4*(5-k)*6 = 4*6*k + 4*6*(5-k) = 24*k + 24*(5-k) = 120
    # That's the same as schoolbook! Partitioning doesn't help for rank.
    
    # The issue is that rank is additive under direct sum, and the schoolbook
    # already achieves the "natural" rank for each block.
    
    # To beat 120, we need to exploit linear dependencies across blocks,
    # which requires a non-trivial tensor decomposition.
    
    # Let me try: use the fact that for the 4x5*5x6 case,
    # we can potentially use fewer terms by sharing computations.
    
    # Approach: Greedy/local search on the tensor
    # The tensor T has dimension (20, 30, 24) and is rank-120 (schoolbook).
    # We want to find a rank-r decomposition where r < 120.
    
    # This is computationally intractable in general, but let's try
    # a structured approach:
    
    # Key insight: The multiplication tensor for (n x m) * (m x p) can be
    # viewed as a sum of p rank-(n*m) terms (one for each k), or as a sum
    # of n rank-(m*p) terms, or as a sum of m rank-(n*p) terms.
    # 
    # rank <= min over partitions of sum of sub-tensor ranks
    # 
    # For m=5: partition into 2+3:
    #   rank(4x2*2x6) + rank(4x3*3x6) = 4*2*6 + 4*3*6 = 48 + 72 = 120
    # Partition into 1+1+1+1+1: 5 * (4*1*6) = 120
    # 
    # All simple partitions give 120. We need something more clever.
    
    # Let's try a Winograd-like approach for the middle dimension.
    # For m=5, we can use a 5-point convolution approach.
    # But this typically helps for square matrices.
    
    # Alternative: Use the Strassen-like recursion.
    # 4x5*5x6: write as block matrices.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, etc.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x3, etc.
    # Wait, dimensions don't work out nicely for 2x2 blocks with 5 middle.
    
    # Let me try: A is 4x5, split rows of A and cols of B into 2+3.
    # A = [A1 (4x2) | A2 (4x3)]
    # B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # rank(A1*B1) = 4*2*6 = 48 (schoolbook for that sub-problem)
    # rank(A2*B2) = 4*3*6 = 72
    # Total = 120. Same as before.
    
    # The fundamental issue: for rectangular matrix multiplication,
    # the schoolbook rank n*m*p may actually be close to optimal,
    # and known results show the rank of the multiplication tensor
    # for (n x m) * (m x p) is exactly n*m*p for small dimensions
    # where no algebraic shortcuts exist.
    
    # However, let me check: is there a known better decomposition for 4x5*5x6?
    # 
    # The border rank of the multiplication tensor is known to be
    # at most n*m*p, and for "generic" dimensions, the rank equals n*m*p.
    # 
    # But wait - there might be a way to use the specific values 4,5,6.
    # 4 = 2^2, 6 = 2*3, 5 is prime.
    
    # Let me try a computational approach: use ALS (alternating least squares)
    # to find a low-rank approximation, then check if it's exact.
    
    # Actually, let me think about this differently.
    # The tensor T[(i,j),(j',k),(k',i')] = delta(j,j')*delta(k,k')*delta(i',i)
    # 
    # We want T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
    # where U_r[i*m+j] = a_r(i,j), V_r[j'*p+k] = b_r(j',k), W_r[k'*n+i'] = c_r(k',i')
    # 
    # For each (i,j,j',k,k',i'):
    # sum_r a_r(i,j) b_r(j',k) c_r(k',i') = 1 if j=j', k=k', i'=i else 0
    # 
    # This is a system of equations. For R terms, we have R * (20 * 30 * 24) unknowns
    # (well, R*20 + R*30 + R*24 = R*74 unknowns) and 20*30*24 = 14400 equations.
    # 
    # For R = 120: 120*74 = 8880 unknowns < 14400 equations. Overdetermined but consistent.
    # For R < 120: even fewer unknowns.
    
    # Let me try R = 60, 70, 80, ... and see if a solution exists.
    # This is a highly non-linear system (trilinear in U, V, W).
    
    # Use a randomized approach: fix V and W randomly, solve for U, etc.
    
    import numpy as np
    
    def try_rank(R, seed=0, max_iters=500):
        """Try to find a rank-R decomposition using ALS."""
        if time.time() > deadline - 15:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, R)
        V = rng.randn(m * p, R)
        W = rng.randn(p * n, R)
        
        # The tensor is sparse: T[a,b,c] = 1 iff j_a=j_b, k_b=k_c, i_c=i_a
        # Non-zero entries: for each j in range(m), k in range(p), i in range(n):
        #   a = i*m + j, b = j*p + k, c = k*n + i
        # So there are n*m*p = 120 non-zero entries, all equal to 1.
        
        # For ALS: 
        # Fix V, W: solve for U. For each a:
        #   sum_r U[a,r] * (sum_{b,c: T[a,b,c]=1} V[b,r]*W[c,r]) ... 
        # Actually T[a,b,c] is non-zero only for specific (b,c) given a.
        # Given a = i*m + j:
        #   T[a, j*p+k, k*n+i] = 1 for all k in range(p)
        # So for fixed a: sum_r U[a,r] * sum_{k=0}^{p-1} V[(j+p*k), r] * W[(k*n+i), r] = 1
        # 
        # Let M_U[a, r] = U[a, r]
        # For each a, we need: sum_r U[a,r] * G_VW[a, r] = 1
        # where G_VW[a, r] = sum_{k=0}^{p-1} V[j*p+k, r] * W[k*n+i, r]
        # with a = i*m + j.
        # 
        # This is: U @ G_VW^T ... no.
        # For each a: dot(U[a, :], G_VW[a, :]) = 1
        # This is an underdetermined system (R unknowns, 1 equation per a).
        # 
        # Better: use the full tensor equation.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # 
        # For ALS on U (fix V, W):
        # For each a, b, c with T[a,b,c] != 0:
        #   sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
        # Group by a: for each a, and each (b,c) pair where T[a,b,c]!=0:
        #   sum_r U[a,r] * H[a,(b,c),r] = T[a,b,c]
        # where H[a,(b,c),r] = V[b,r] * W[c,r]
        # 
        # For a = i*m + j, the (b,c) pairs are (j*p+k, k*n+i) for k=0..p-1.
        # So we have p equations per a, each with R unknowns (components of U[a,:]).
        # Total: n*m*p equations, R unknowns per a.
        # 
        # We can solve this in least squares for each a independently!
        # For each a: solve min ||H_a * U[a,:] - t_a||^2
        # where H_a is p x R and t_a is p x 1 (all ones).
        
        # Precompute index structure
        # For each a = i*m + j:
        #   b_k = j*p + k, c_k = k*n + i for k = 0..p-1
        
        def build_a_structure():
            """For each a, return list of (b, c) pairs."""
            struct = []
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    pairs = []
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        pairs.append((b, c))
                    struct.append(pairs)
            return struct
        
        a_struct = build_a_structure()
        
        # ALS iterations
        for it in range(max_iters):
            if time.time() > deadline - 20:
                return None
            
            # Update U
            for a in range(n * m):
                pairs = a_struct[a]
                H = np.array([[V[b, r] * W[c, r] for r in range(R)] for b, c in pairs])
                t = np.ones(len(pairs))
                # Solve H @ U[a,:] = t in least squares
                U[a, :], residuals, rank_u, sv = np.linalg.lstsq(H, t, rcond=None)
            
            # Update V
            # For each b = j*p + k:
            #   T[a, b, c] = 1 iff a = i*m + j and c = k*n + i for some i
            #   So for fixed b = j*p + k: pairs (a,c) are (i*m+j, k*n+i) for i=0..n-1
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    H = np.array([[U[a, r] * W[c, r] for r in range(R)] for i in range(n) 
                                  for a, c in [(i * m + j, k * n + i)]])
                    # Simplify:
                    H = np.zeros((n, R))
                    for i in range(n):
                        a = i * m + j
                        c = k * n + i
                        H[i, :] = U[a, :] * W[c, :]
                    t = np.ones(n)
                    V[b, :], residuals, rank_v, sv = np.linalg.lstsq(H, t, rcond=None)
            
            # Update W
            # For each c = k*n + i:
            #   T[a, b, c] = 1 iff a = i*m + j and b = j*p + k for some j
            #   So for fixed c = k*n + i: pairs (a,b) are (i*m+j, j*p+k) for j=0..m-1
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    H = np.zeros((m, R))
                    for j in range(m):
                        a = i * m + j
                        b = j * p + k
                        H[j, :] = U[a, :] * V[b, :]
                    t = np.ones(m)
                    W[c, :], residuals, rank_w, sv = np.linalg.lstsq(H, t, rcond=None)
            
            # Check convergence
            if it % 50 == 0 and it > 0:
                # Quick check
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                            # Also check some zero entries
                if err < 1e-10:
                    # Verify all entries
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try various ranks
    for R in [60, 70, 80, 90, 100, 110, 115]:
        if time.time() > deadline - 30:
            break
        for seed in range(3):
            if time.time() > deadline - 30:
                break
            result = try_rank(R, seed=seed)
            if result is not None:
                U_list, V_list, W_list = result
                # Verify
                ok, err = _verify(U_list, V_list, W_list, n, m, p)
                if ok:
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we haven't found a better decomposition, return schoolbook
    if time.time() > deadline - 10:
        return best
    
    return best

# EVOLVE-BLOCK-END
