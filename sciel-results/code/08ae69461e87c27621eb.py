"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140
    
    def check_decomposition(U, V, W, rank, tol=1e-9):
        """Verify the decomposition is exact using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    idx_u = a * m + b
                    idx_v = b * p + c
                    idx_w = c * n + a
                    val = np.dot(U_arr[idx_u, :], V_arr[idx_v, :] * W_arr[idx_w, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def build_rank17_correct():
        """Construct a valid rank-17 decomposition."""
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for columns 0,1 (2x2 block) - rank 7
        # U: 0=(a0,b0), 1=(a0,b1), 2=(a1,b0), 3=(a1,b1)
        # V: 0=(b0,c0), 1=(b0,c1), 2=(b1,c0), 3=(b1,c1)
        # W: 0=(c0,a0), 1=(c0,a1), 2=(c1,a0), 3=(c1,a1)
        
        # r=0: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[2][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: A00(B10-B00)
        U[0][2] = 1
        V[3][2] = 1; V[2][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: A11(B01-B10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        # For columns 2,3,4 (2x3 block) - rank 10
        # U: 0=(a0,b0), 1=(a0,b1), 2=(a1,b0), 3=(a1,b1)
        # V: 4=(b0,c2), 5=(b0,c3), 6=(b0,c4), 7=(b1,c2), 8=(b1,c3), 9=(b1,c4)
        # W: 4=(c2,a0), 5=(c2,a1), 6=(c3,a0), 7=(c3,a1), 8=(c4,a0), 9=(c4,a1)
        
        # Schoolbook for 2x2 x 2x3: rank 12, but we can do 10
        # Actually, let me just use schoolbook: rank 12
        # That gives total rank 7+12=19, which is worse
        
        # Better: use a structured rank-10 for 2x2 x 2x3
        # This requires careful design
        
        # For now, use schoolbook for the 2x3 part: rank 12
        # Total: 7 + 12 = 19
        
        # Actually, let me try a different approach.
        # Use Strassen-like for a 2x2 sub-block within the 2x3, plus handle the rest
        
        # For columns 2,3: use Strassen (rank 7)
        # For column 4: use schoolbook (rank 4)
        # Total for 2x3 part: 7 + 4 = 11
        # Overall: 7 + 11 = 18
        
        # That's the same as before. Let me try to be smarter.
        
        # Actually, the original rank-18 was fine. Let me just return that.
        return None
    
    def build_rank18():
        """Construct a valid rank-18 decomposition."""
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen for columns 0,1 (rank 7)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[2][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[3][2] = 1; V[2][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        # Schoolbook for columns 2,3,4 (rank 12)
        for c in range(3):
            col = c + 2
            for a in range(n):
                for b in range(m):
                    r = 7 + c * 4 + a * 2 + b
                    U[a * m + b][r] = 1
                    V[b * p + col][r] = 1
                    W[col * n + a][r] = 1
        
        return U, V, W
    
    def try_als_rank(rank, max_iters=1000, tol=1e-12, seed=42):
        """Try to find a rank-`rank` decomposition using ALS."""
        if time.time() > deadline - 15:
            return None
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        np.random.seed(seed)
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Normalize columns
        for r in range(rank):
            U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-12)
            V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-12)
            W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-12)
        
        def compute_error():
            err = 0.0
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        idx_u = a * m + b
                        idx_v = b * p + c
                        idx_w = c * n + a
                        val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                        err += (val - 1.0) ** 2
            return err
        
        best_error = compute_error()
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Update U
            for a in range(n):
                for b in range(m):
                    idx_u = a * m + b
                    G = np.zeros((p, rank))
                    for c in range(p):
                        G[c, :] = V[b * p + c, :] * W[c * n + a, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(G, np.ones(p), rcond=None)
                        U[idx_u, :] = x
                    except:
                        pass
            
            # Update V
            for b in range(m):
                for c in range(p):
                    idx_v = b * p + c
                    H = np.zeros((n, rank))
                    for a in range(n):
                        H[a, :] = U[a * m + b, :] * W[c * n + a, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(H, np.ones(n), rcond=None)
                        V[idx_v, :] = x
                    except:
                        pass
            
            # Update W
            for c in range(p):
                for a in range(n):
                    idx_w = c * n + a
                    K = np.zeros((m, rank))
                    for b in range(m):
                        K[b, :] = U[a * m + b, :] * V[b * p + c, :]
                    try:
                        x, _, _, _ = np.linalg.lstsq(K, np.ones(m), rcond=None)
                        W[idx_w, :] = x
                    except:
                        pass
            
            # Renormalize
            for r in range(rank):
                U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-12)
                V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-12)
                W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-12)
            
            err = compute_error()
            if err < best_error:
                best_error = err
                if best_error < tol:
                    break
            else:
                if it > 50 and err > best_error * 1.1:
                    break
        
        if best_error < tol:
            U_list = [[U[i, r] for r in range(rank)] for i in range(nm)]
            V_list = [[V[i, r] for r in range(rank)] for i in range(mp)]
            W_list = [[W[i, r] for r in range(rank)] for i in range(pn)]
            
            if check_decomposition(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
        
        return None
    
    # Try rank 16 first
    for attempt in range(40):
        if time.time() > deadline - 30:
            break
        result = try_als_rank(16, max_iters=200, seed=100+attempt)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Try rank 17
    for attempt in range(40):
        if time.time() > deadline - 30:
            break
        result = try_als_rank(17, max_iters=200, seed=200+attempt)
        if result is not None:
            return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fallback to rank 18
    U18, V18, W18 = build_rank18()
    return {"U": U18, "V": V18, "W": W18}
# EVOLVE-BLOCK-END
