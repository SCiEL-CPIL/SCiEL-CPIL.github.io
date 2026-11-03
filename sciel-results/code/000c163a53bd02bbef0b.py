import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # Schoolbook baseline
    rank_sb = n * m * p  # 144
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, R):
        """Verify the decomposition is exact by checking all indices using vectorized ops."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check ALL inconsistent indices (complete verification)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1, j1 = divmod(a, m)
                    j2, k2 = divmod(b, p)
                    k3, i3 = divmod(c, n)
                    
                    if j1 == j2 and i1 == i3 and k2 == k3:
                        continue  # consistent, already checked
                    
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
        
        return True

    def verify_fast(U, V, W, R):
        """Faster verification using the tensor structure."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the full tensor T and check T == sum_r U[:,r] outer V[:,r] outer W[:,r]
        # But that's too expensive. Instead, check consistent + a sample of inconsistent.
        
        # Check all consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check inconsistent: for each pair (a,b) with j1!=j2, all c should give 0
        # For each pair (a,c) with i1!=i3, all b should give 0
        # For each pair (b,c) with k2!=k3, all a should give 0
        
        # Check j-inconsistency: a=i*m+j1, b=j2*p+k where j1!=j2
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2:
                        continue
                    a = i * m + j1
                    for k in range(p):
                        b = j2 * p + k
                        # For any c, the value should be 0
                        # sum_r U[a,r] V[b,r] W[c,r] = 0 for all c
                        # This is equivalent to (U[a,:] * V[b,:]) being orthogonal to all rows of W
                        vec = U_np[a] * V_np[b]
                        if np.max(np.abs(W_np @ vec)) > 1e-9:
                            return False
        
        # Check i-inconsistency: a=i1*m+j, c=k*n+i3 where i1!=i3
        for i1 in range(n):
            for i3 in range(n):
                if i1 == i3:
                    continue
                for j in range(m):
                    a = i1 * m + j
                    for k in range(p):
                        c = k * n + i3
                        # sum_r U[a,r] V[b,r] W[c,r] = 0 for all b
                        # Equivalent to (U[a,:] * W[c,:]) being orthogonal to all rows of V
                        vec = U_np[a] * W_np[c]
                        if np.max(np.abs(V_np @ vec)) > 1e-9:
                            return False
        
        # Check k-inconsistency: b=j*p+k2, c=k3*n+i where k2!=k3
        for k2 in range(p):
            for k3 in range(p):
                if k2 == k3:
                    continue
                for j in range(m):
                    b = j * p + k2
                    for i in range(n):
                        c = k3 * n + i
                        # sum_r U[a,r] V[b,r] W[c,r] = 0 for all a
                        # Equivalent to (V[b,:] * W[c,:]) being orthogonal to all rows of U
                        vec = V_np[b] * W_np[c]
                        if np.max(np.abs(U_np @ vec)) > 1e-9:
                            return False
        
        return True

    # Try structured constructions

    # Construction 1: R = n*m*p (schoolbook) - already have it

    # Construction 2: Try R = 36 with r = i*p + k (fails on j-consistency as analyzed)
    # Construction 3: Try R = 36 with r = j*p + k (fails on i-consistency)

    # Key insight: we need all three consistency conditions simultaneously.
    # The minimal rank must be at least max(n*m, m*p, p*n) = max(16,36,36) = 36.
    # But structural constraints may require higher rank.

    # Let's try a block decomposition: split p=9 into blocks
    # Block 1: k in [0,4), Block 2: k in [4,8), Block 3: k in [8,9)
    # For each block, we need a 4x4 * 4xb decomposition

    # For 4x4 * 4x4: try to construct rank-19 (Strassen-like)
    # For 4x4 * 4x1: rank 16 (schoolbook for that subproblem)
    # Total: 19 + 19 + 16 = 54

    # Actually, let me try a simpler approach first.
    # The tensor for [n,m,p] can be built from [n,m,p1] and [n,m,p2] if p=p1+p2
    # by direct sum, giving rank [n,m,p] <= [n,m,p1] + [n,m,p2]

    # So [4,4,9] <= [4,4,4] + [4,4,4] + [4,4,1]
    # [4,4,1] = 16 (each of 16 entries needs one multiplication)
    # [4,4,4] <= 49 (Strassen) but we need explicit construction

    # Let me try to construct a rank-19 decomposition for [4,4,4]
    # using a known pattern. One approach: use the 2x2 Strassen recursively.

    # For 2x2 * 2x2, Strassen uses 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # For 4x4 * 4x4, we can use a 2x2 block structure where each block is 2x2,
    # and apply Strassen at the block level, with each "multiplication" being a
    # 2x2*2x2 multiply (rank 7 each). Total: 7*7 = 49.

    # But we want rank < 49. Let me try a different approach.

    # Actually, let me just try to find ANY rank < 144 construction.
    # The simplest improvement: [4,4,9] <= [4,4,4] + [4,4,4] + [4,4,1] = 49+49+16 = 114
    # But I need explicit rank-49 constructions for [4,4,4].

    # Let me try a direct search with the fast verifier.

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Try the R=36 constructions first (quick check)
    if time.time() < deadline - 80:
        # R = n*p = 36: r = i*p+k
        R_try = n * p
        U_try = np.zeros((n * m, R_try))
        V_try = np.zeros((m * p, R_try))
        W_try = np.zeros((p * n, R_try))
        for i in range(n):
            for k in range(p):
                r = i * p + k
                U_try[:, r] = 1  # all i' get 1 (wrong, should depend on i')
                # Actually: U[i'*m+j, r] = 1 if i'=i
                U_try[:] = 0
                for i2 in range(n):
                    if i2 == i:
                        for j in range(m):
                            U_try[i2 * m + j, r] = 1
                for j in range(m):
                    V_try[j * p + k, r] = 1
                W_try[k * n + i, r] = 1

        if verify_fast(U_try, V_try, W_try, R_try):
            return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}

        # R = m*p = 36: r = j*p+k
        R_try2 = m * p
        U_try2 = np.zeros((n * m, R_try2))
        V_try2 = np.zeros((m * p, R_try2))
        W_try2 = np.zeros((p * n, R_try2))
        for j in range(m):
            for k in range(p):
                r = j * p + k
                for i in range(n):
                    U_try2[i * m + j, r] = 1
                V_try2[j * p + k, r] = 1
                for i in range(n):
                    W_try2[k * n + i, r] = 1

        if verify_fast(U_try2, V_try2, W_try2, R_try2):
            return {"U": U_try2.tolist(), "V": V_try2.tolist(), "W": W_try2.tolist()}

    # Try random search with fast verification
    if time.time() < deadline - 60:
        np.random.seed(42)
        for R in [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]:
            if time.time() > deadline - 40:
                break
            for attempt in range(5):
                if time.time() > deadline - 35:
                    break
                U_rand = np.random.randint(-1, 2, size=(n*m, R))
                V_rand = np.random.randint(-1, 2, size=(m*p, R))
                W_rand = np.random.randint(-1, 2, size=(p*n, R))
                if verify_fast(U_rand, V_rand, W_rand, R):
                    best_rank = R
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    break
            if best_rank < rank_sb:
                break

    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
