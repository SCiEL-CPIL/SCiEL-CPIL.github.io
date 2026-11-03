import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n*m, m*p, p*n  # 8, 28, 14
    
    # Build target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[(u_idx, v_idx, w_idx)] = 1
    
    # Build rank-50 block decomposition (known correct)
    def build_rank50():
        rank = 50
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        
        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
            a00 = a_rows[0]*m + a_cols[0]
            a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]
            a11 = a_rows[1]*m + a_cols[1]
            b00 = b_rows[0]*p + b_cols[0]
            b01 = b_rows[0]*p + b_cols[1]
            b10 = b_rows[1]*p + b_cols[0]
            b11 = b_rows[1]*p + b_cols[1]
            c00 = c_cols[0]*n + c_rows[0]
            c01 = c_cols[1]*n + c_rows[0]
            c10 = c_cols[0]*n + c_rows[1]
            c11 = c_cols[1]*n + c_rows[1]
            col = start_col
            # Strassen's 7 multiplications for 2x2*2x2
            U[a00][col] += 1; U[a11][col] += 1; V[b00][col] += 1; V[b11][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[a10][col] += 1; U[a11][col] += 1; V[b00][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a00][col] += 1; V[b01][col] += 1; V[b11][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[a11][col] += 1; V[b10][col] += 1; V[b00][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a00][col] += 1; U[a01][col] += 1; V[b11][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[a10][col] += 1; U[a00][col] -= 1; V[b00][col] += 1; V[b01][col] += 1; W[c11][col] += 1; col += 1
            U[a01][col] += 1; U[a11][col] -= 1; V[b10][col] += 1; V[b11][col] += 1; W[c00][col] += 1; col += 1
            return col
        
        def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
            a00 = a_rows[0]*m + a_cols[0]; a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]; a11 = a_rows[1]*m + a_cols[1]
            b0 = b_rows[0]*p + b_col; b1 = b_rows[1]*p + b_col
            c0 = c_col*n + c_rows[0]; c1 = c_col*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1; col += 1
            U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1; col += 1
            U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1; col += 1
            U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1
            return col
        
        col = 0
        # 6 Strassen blocks (2x2 x 2x2) + 2 schoolbook blocks (2x2 x 2x1)
        col = add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], col)       # A[:2,:2] * B[:2,:2] -> C[:2,:2]
        col = add_schoolbook_2x2_2x1([0,1],[0,1],[0,1],2,[0,1],2, col)       # A[:2,:2] * B[:2,2] -> C[:2,2]
        col = add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], col)         # A[:2,2:4] * B[2:4,:2] -> C[:2,:2]
        col = add_schoolbook_2x2_2x1([0,1],[2,3],[2,3],2,[0,1],2, col)       # A[:2,2:4] * B[2:4,2] -> C[:2,2]
        col = add_strassen([0,1],[0,1],[0,1],[3,4],[0,1],[3,4], col)         # A[:2,:2] * B[:2,3:5] -> C[:2,3:5]
        col = add_strassen([0,1],[0,1],[0,1],[5,6],[0,1],[5,6], col)         # A[:2,:2] * B[:2,5:7] -> C[:2,5:7]
        col = add_strassen([0,1],[2,3],[2,3],[3,4],[0,1],[3,4], col)         # A[:2,2:4] * B[2:4,3:5] -> C[:2,3:5]
        col = add_strassen([0,1],[2,3],[2,3],[5,6],[0,1],[5,6], col)         # A[:2,2:4] * B[2:4,5:7] -> C[:2,5:7]
        return {"U": U, "V": V, "W": W}
    
    # Verify decomposition
    def verify(U, V, W, rank):
        for u in range(Na):
            for v in range(Nb):
                for w in range(Nc):
                    s = 0.0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    expected = target.get((u,v,w), 0)
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Try to find rank-49 by looking for linear dependencies in rank-50
    def try_rank49():
        """Try to find a rank-49 decomposition by checking for linear dependencies."""
        result50 = build_rank50()
        U = np.array(result50["U"], dtype=float)
        V = np.array(result50["V"], dtype=float)
        W = np.array(result50["W"], dtype=float)
        
        # First: check pairwise proportional terms (fast)
        for r1 in range(50):
            for r2 in range(r1+1, 50):
                U1, U2 = U[:, r1], U[:, r2]
                V1, V2 = V[:, r1], V[:, r2]
                W1, W2 = W[:, r1], W[:, r2]
                
                ratios = []
                for idx in range(Na):
                    if abs(U1[idx]) > 1e-12 and abs(U2[idx]) > 1e-12:
                        ratios.append(U1[idx] / U2[idx])
                if not ratios:
                    continue
                    
                alpha = ratios[0]
                consistent = True
                for ratio in ratios[1:]:
                    if abs(ratio - alpha) > 1e-9:
                        consistent = False
                        break
                if not consistent:
                    continue
                
                v_ok = True
                for idx in range(Nb):
                    if abs(V1[idx] - alpha * V2[idx]) > 1e-9:
                        v_ok = False
                        break
                if not v_ok:
                    continue
                    
                w_ok = True
                for idx in range(Nc):
                    if abs(W1[idx] - alpha * W2[idx]) > 1e-9:
                        w_ok = False
                        break
                if not w_ok:
                    continue
                
                # Found proportional terms - try dropping each
                new_rank = 49
                for drop_r in [r1, r2]:
                    new_U2 = np.zeros((Na, new_rank), dtype=float)
                    new_V2 = np.zeros((Nb, new_rank), dtype=float)
                    new_W2 = np.zeros((Nc, new_rank), dtype=float)
                    
                    for r in range(50):
                        if r == drop_r:
                            continue
                        new_r = r if r < drop_r else r - 1
                        new_U2[:, new_r] = U[:, r]
                        new_V2[:, new_r] = V[:, r]
                        new_W2[:, new_r] = W[:, r]
                    
                    if verify(new_U2.tolist(), new_V2.tolist(), new_W2.tolist(), new_rank):
                        return {"U": new_U2.tolist(), "V": new_V2.tolist(), "W": new_W2.tolist()}
        
        # Second: check for 3-term linear dependencies
        # If c1*term1 + c2*term2 + c3*term3 = 0, we can eliminate one term
        # Build the "flattened" representation of each term as a vector
        # term_r flattened = U[:,r] ⊗ V[:,r] ⊗ W[:,r] but that's too large
        # Instead, for each target entry (u,v,w), the contribution is U[u,r]*V[v,r]*W[w,r]
        # A linear dependency means: sum_r c_r * U[u,r]*V[v,r]*W[w,r] = 0 for all (u,v,w)
        
        # This is equivalent to: the 50 vectors in the space of (Na*Nb*Nc) are linearly dependent
        # But that space is 8*28*14 = 3136 dimensional, and we have 50 vectors
        # We can check by building a matrix where each column is a "flattened term"
        # But flattening gives a 3136-dim vector per term, making a 3136x50 matrix
        
        # Alternative: use the fact that if there's a dependency, we can find it via
        # solving a system. Let's try a randomized approach:
        # Pick random (u,v,w) samples and check if the 50x50 system has a null space
        
        # Actually, let's just try: for each pair (r1, r2), check if term_r1 - c*term_r2
        # can be expressed as a combination of other terms. This is expensive.
        
        # Simpler approach: check if the 50 rank-1 tensors are linearly dependent
        # by evaluating on a sufficient number of random points
        
        np.random.seed(42)
        # Sample enough points to detect dependencies
        num_samples = 200
        samples = []
        for _ in range(num_samples):
            u = np.random.randint(0, Na)
            v = np.random.randint(0, Nb)
            w = np.random.randint(0, Nc)
            samples.append((u, v, w))
        
        # Build evaluation matrix: num_samples x 50
        eval_mat = np.zeros((num_samples, 50))
        for si, (u, v, w) in enumerate(samples):
            for r in range(50):
                eval_mat[si, r] = U[u, r] * V[v, r] * W[w, r]
        
        # Check rank of eval_mat
        try:
            rank_eval = np.linalg.matrix_rank(eval_mat, tol=1e-10)
        except:
            rank_eval = 50
        
        if rank_eval < 50:
            # There's a linear dependency! Find the null vector
            try:
                _, s, vh = np.linalg.svd(eval_mat)
                # The smallest singular value should be ~0
                null_vec = vh[-1]  # Last row of Vh
                # Find the index with the largest absolute value
                max_idx = np.argmax(np.abs(null_vec))
                
                # Try dropping that term
                new_rank = 49
                drop_r = max_idx
                new_U2 = np.zeros((Na, new_rank), dtype=float)
                new_V2 = np.zeros((Nb, new_rank), dtype=float)
                new_W2 = np.zeros((Nc, new_rank), dtype=float)
                
                for r in range(50):
                    if r == drop_r:
                        continue
                    new_r = r if r < drop_r else r - 1
                    new_U2[:, new_r] = U[:, r]
                    new_V2[:, new_r] = V[:, r]
                    new_W2[:, new_r] = W[:, r]
                
                if verify(new_U2.tolist(), new_V2.tolist(), new_W2.tolist(), new_rank):
                    return {"U": new_U2.tolist(), "V": new_V2.tolist(), "W": new_W2.tolist()}
            except:
                pass
        
        return None
    
    # Fallback: return rank-50
    result = build_rank50()
    
    # Verify it's correct
    if verify(result["U"], result["V"], result["W"], 50):
        # Try to reduce
        reduced = try_rank49()
        if reduced is not None:
            return reduced
        return result
    else:
        # Should not happen, but return anyway
        return result
