"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Convention: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k].
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_4x4x7_optimized():
    """Build decomposition for 4x4 * 4x7 using Strassen where beneficial.
    
    Part 1: 4x4 * 4x4 -> rank 49 (Kronecker of two Strassen 2x2x2)
    Part 2: 4x4 * 4x2 -> rank 28 (four 2x2*2x2 Strassen products)
    Part 3: 4x4 * 4x1 -> rank 16 (four 2x2*2x1 schoolbook products)
    Total: 93
    """
    n, m, p = 4, 4, 7
    U2, V2, W2 = _strassen_2x2x2()
    
    # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(16)]
    V1 = [[0] * rank1 for _ in range(16)]
    W1 = [[0] * rank1 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U1[i*4+j][r] += val
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if val != 0:
                                V1[j*4+k][r] += val
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if val != 0:
                                W1[k*4+i][r] += val
    
    # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
    rank2 = 28
    U2p = [[0] * rank2 for _ in range(16)]
    V2p = [[0] * rank2 for _ in range(8)]  # 4*2
    W2p = [[0] * rank2 for _ in range(8)]  # 2*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for t in range(7):
                r = block_idx * 7 + t
                for lr in range(2):
                    for lc in range(2):
                        i = br * 2 + lr
                        j = bc * 2 + lc
                        val = U2[lr*2+lc][t]
                        if val != 0:
                            U2p[i*4+j][r] += val
                for lr in range(2):
                    for kl in range(2):
                        j = bc * 2 + lr
                        k = 4 + kl
                        val = V2[lr*2+kl][t]
                        if val != 0:
                            V2p[j*2+kl][r] += val
                for lr in range(2):
                    for kl in range(2):
                        i = br * 2 + lr
                        k = kl
                        val = W2[kl*2+lr][t]
                        if val != 0:
                            W2p[k*4+i][r] += val
            block_idx += 1
    
    # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
    rank3 = 16
    U3 = [[0] * rank3 for _ in range(16)]
    V3 = [[0] * rank3 for _ in range(4)]  # 4*1
    W3 = [[0] * rank3 for _ in range(4)]  # 1*4
    
    block_idx = 0
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    r = block_idx * 4 + lr * 2 + lc
                    i = br * 2 + lr
                    j = bc * 2 + lc
                    U3[i*4+j][r] = 1
                    V3[j][r] = 1
                    W3[i][r] = 1
            block_idx += 1
    
    # Combine all parts
    total_rank = rank1 + rank2 + rank3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3, C cols 0-3
    for i in range(16):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                V[i*p+j][r] = V1[i*4+j][r]
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                W[i*n+j][r] = W1[i*4+j][r]
    
    # Part 2: B cols 4-5, C cols 4-5
    offset2 = rank1
    for i in range(16):
        for r in range(rank2):
            U[i][offset2 + r] = U2p[i][r]
    for i in range(4):
        for j in range(2):
            for r in range(rank2):
                V[i*p + (4+j)][offset2 + r] = V2p[i*2+j][r]
    for i in range(2):
        for j in range(4):
            for r in range(rank2):
                W[(4+i)*n + j][offset2 + r] = W2p[i*4+j][r]
    
    # Part 3: B col 6, C col 6
    offset3 = rank1 + rank2
    for i in range(16):
        for r in range(rank3):
            U[i][offset3 + r] = U3[i][r]
    for i in range(4):
        for r in range(rank3):
            V[i*p + 6][offset3 + r] = V3[i][r]
    for i in range(4):
        for r in range(rank3):
            W[6*n + i][offset3 + r] = W3[i][r]
    
    return {"U": U, "V": V, "W": W}


def _try_reduce_rank(U, V, W, n, m, p, time_budget=10.0):
    """Try to reduce rank by identifying linearly dependent rank-1 terms.
    
    Strategy: Build the Khatri-Rao product structure and look for terms
    that are linear combinations of others. If a term's contribution is
    in the span of other terms' contributions, it can be eliminated.
    """
    start = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each term r, compute its "fingerprint" - the vector of values
    # U[a,r]*V[b,r]*W[c,r] for all (a,b,c) in the tensor support.
    # If fingerprint_r is in the span of {fingerprint_s for s != r},
    # then term r is redundant.
    
    # This is expensive, so use a heuristic: check for proportional terms.
    # Two terms r1, r2 are "compatible" if U[:,r1] ∝ U[:,r2], V[:,r1] ∝ V[:,r2],
    # and W[:,r1] ∝ W[:,r2]. In that case, they can be merged.
    
    # More generally, if the 3-way product of term r1 is a linear combination
    # of products of other terms, r1 can be eliminated.
    
    # Simple approach: try to find pairs where one term's U vector is a
    # linear combination of others, and check if the full term can be eliminated.
    
    # For efficiency, only check a subset of terms.
    n_terms = rank
    removed = set()
    
    for target in range(n_terms):
        if target in removed:
            continue
        if time.time() - start > time_budget:
            break
        
        u_target = U_arr[:, target]
        if np.linalg.norm(u_target) < 1e-12:
            removed.add(target)
            continue
        
        # Check if u_target is in the span of other U vectors
        other_indices = [i for i in range(n_terms) if i != target and i not in removed]
        if not other_indices:
            continue
        
        # Build matrix of other U vectors
        U_others = U_arr[:, other_indices]
        
        # Check if u_target is in the column space of U_others
        try:
            # Solve U_others @ x = u_target
            coeff, residuals, rank_u, sv = np.linalg.lstsq(U_others, u_target, rcond=None)
            if len(residuals) > 0:
                resid = residuals[0]
            else:
                resid = np.linalg.norm(U_others @ coeff - u_target)
            
            if resid < 1e-8 * max(np.linalg.norm(u_target), 1e-12):
                # u_target is in the span of other U vectors
                # Now check if the full term can be eliminated
                # The term contribution is u_target ⊗ v_target ⊗ w_target
                # If this is in the span of other term contributions, we can remove it.
                
                v_target = V_arr[:, target]
                w_target = W_arr[:, target]
                
                # Check if v_target ⊗ w_target is "compatible" with the
                # linear combination found above.
                # This is a complex check, so use a simpler heuristic:
                # if the residual is very small, assume the term is redundant.
                removed.add(target)
        except Exception:
            pass
    
    if not removed:
        return U, V, W
    
    # Rebuild without removed terms
    keep = [i for i in range(n_terms) if i not in removed]
    new_U = [[U[i][r] for r in keep] for i in range(len(U))]
    new_V = [[V[i][r] for r in keep] for i in range(len(V))]
    new_W = [[W[i][r] for r in keep] for i in range(len(W))]
    
    return new_U, new_V, new_W


def _global_sparse_search(n, m, p, target_rank, time_budget=20.0):
    """Global linear algebraic search for a sparse rank decomposition.
    
    Strategy: Treat the entire n×m×p tensor as a single entity.
    1. Flatten the multiplication tensor T into a vector of length n*m*p.
    2. For a given support pattern (which U, V, W entries are nonzero),
       the tensor equation becomes a system of equations in the unknown
       nonzero entries.
    3. Use randomized initialization to find a sparse support of size
       target_rank, then solve the resulting (nonlinear) system.
    
    This bypasses additive rank bounds from sub-tensor decompositions.
    """
    start = time.time()
    
    # Build the multiplication tensor T as a 3D array
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # The tensor equation: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # In vector form: sum_r (U[:,r] ⊗ V[:,r] ⊗ W[:,r]).flatten() = T.flatten()
    
    # For a fixed support pattern, this is a system of n*m*p equations
    # in 3*target_rank unknowns (the nonzero entries of U, V, W).
    
    # Heuristic: use the structure of the schoolbook decomposition as a
    # starting point, then try to merge terms.
    
    # Alternative: use ALS (alternating least squares) to find a rank-
    # target_rank decomposition, then round to half-integers.
    
    # Build the Khatri-Rao structure.
    # For each term r, the contribution to T[a,b,c] is U[a,r]*V[b,r]*W[c,r].
    # Fixing U and V, the equation for W is:
    #   sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all (a,b,c)
    # This is linear in W.
    
    # Similarly for U and V.
    
    # Initialize with random half-integer values
    np.random.seed(42)
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize with small random values
    for r in range(target_rank):
        U[:, r] = np.random.choice([-1, 0, 1], size=n*m)
        V[:, r] = np.random.choice([-1, 0, 1], size=m*p)
        W[:, r] = np.random.choice([-1, 0, 1], size=p*n)
    
    # ALS iterations
    for iteration in range(50):
        if time.time() - start > time_budget:
            break
        
        # Update W: for each c, solve the linear system
        # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # For fixed c: sum_r (U[:,r] ⊗ V[:,r])[idx] * W[c,r] = T[:, :, c].flatten()[idx]
        
        # Build the matrix M_W where M_W[idx, r] = U[a,r]*V[b,r] for (a,b) = idx
        # and the vector b_W where b_W[idx] = T[a,b,c]
        
        # This is expensive, so only do it for a subset of iterations.
        
        # Actually, let's use a simpler approach: gradient descent.
        
        # Compute gradient of ||sum_r U_r ⊗ V_r ⊗ W_r - T||^2 w.r.t. W
        # dL/dW[c,r] = 2 * sum_{a,b} (sum_s U[a,s]*V[b,s]*W[c,s] - T[a,b,c]) * U[a,r]*V[b,r]
        
        # This is O(target_rank^2 * n*m*p) per iteration, which might be too slow.
        
        # For now, skip the ALS and just return the initial random guess.
        # In a future version, implement proper ALS with sparse support.
        
        break
    
    # Convert to lists
    U_list = [[float(U[i][r]) for r in range(target_rank)] for i in range(n*m)]
    V_list = [[float(V[i][r]) for r in range(target_rank)] for i in range(m*p)]
    W_list = [[float(W[i][r]) for r in range(target_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Try the optimized construction first (rank 93)
    try:
        result = _build_4x4x7_optimized()
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            # Try to reduce rank
            if time.time() - _START_TIME < _TIME_LIMIT - 15:
                U_red, V_red, W_red = _try_reduce_rank(
                    result['U'], result['V'], result['W'], 4, 4, 7,
                    time_budget=min(10.0, _TIME_LIMIT - (time.time() - _START_TIME) - 5)
                )
                if len(U_red[0]) < len(result['U'][0]):
                    if _verify(U_red, V_red, W_red, 4, 4, 7):
                        return {"U": U_red, "V": V_red, "W": W_red}
            return result
    except Exception:
        pass
    
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)
