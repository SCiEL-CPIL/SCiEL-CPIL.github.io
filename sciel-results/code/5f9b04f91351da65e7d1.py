import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 150.0  # Evaluator kills at 150s
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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

    def verify(U, V, W, n, m, p):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def find_decomposition(n, m, p, target_rank, time_budget=10.0):
        """
        Attempts to find a decomposition of M_{n,m,p} with rank <= target_rank.
        Returns (U, V, W) as lists of lists if found, else None.
        """
        start = time.time()
        nm, mp, pn = n*m, m*p, p*n
        
        for seed in range(20):
            if time.time() - start > time_budget:
                break
            np.random.seed(1000 + seed + n + m + p)
            
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for step in range(100):
                # ALS for U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        try:
                            U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = U_col
                        except:
                            return None
                            
                # ALS for V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        try:
                            V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = V_col
                        except:
                            return None
                            
                # ALS for W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        try:
                            W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = W_col
                        except:
                            return None
                            
                # Check loss
                loss = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            loss += (val - 1.0)**2
                if loss < 1e-12:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
                    
                    # Try rounding to integers?
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if verify(U_i.tolist(), V_i.tolist(), W_i.tolist(), n, m, p):
                        return U_i.tolist(), V_i.tolist(), W_i.tolist()
                        
        return None

    def combine_decompositions(U3, V3, W3, U2, V2, W2, n, m):
        """
        Combines two decompositions:
        1. M_{n,m,3} with rank r3
        2. M_{n,m,2} with rank r2
        into M_{n,m,5} with rank r3+r2.
        Wait, we need M_{3,3,8}.
        We will combine:
        1. M_{3,3,3} (rank r3) for k=0,1,2
        2. M_{3,3,3} (rank r3) for k=3,4,5
        3. M_{3,3,2} (rank r2) for k=6,7
        """
        r3 = len(U3[0])
        r2 = len(U2[0])
        total_rank = 2 * r3 + r2
        
        # Initialize U, V, W
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * 8)]
        W = [[0] * total_rank for _ in range(8 * n)]
        
        # Part 1: k=0,1,2 using (U3, V3, W3)
        for r in range(r3):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = U3[i*m+j][r]
            for j in range(m):
                for k in range(3):
                    V[j*8+k][r] = V3[j*3+k][r]
            for k in range(3):
                for i in range(n):
                    W[k*n+i][r] = W3[k*n+i][r]
                    
        # Part 2: k=3,4,5 using (U3, V3, W3) again
        for r in range(r3):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r+r3] = U3[i*m+j][r]
            for j in range(m):
                for k in range(3):
                    V[j*8+(k+3)][r+r3] = V3[j*3+k][r]
            for k in range(3):
                for i in range(n):
                    W[(k+3)*n+i][r+r3] = W3[k*n+i][r]
                    
        # Part 3: k=6,7 using (U2, V2, W2)
        for r in range(r2):
            for i in range(n):
                for j in range(m):
                    U[i*m+j][2*r3+r] = U2[i*m+j][r]
            for j in range(m):
                for k in range(2):
                    V[j*8+(k+6)][2*r3+r] = V2[j*2+k][r]
            for k in range(2):
                for i in range(n):
                    W[(k+6)*n+i][2*r3+r] = W2[k*n+i][r]
                    
        return {"U": U, "V": V, "W": W}

    # Search for M_{3,3,3} decomposition
    best_333 = None
    best_rank_333 = 27 # Schoolbook
    for r in [26, 25, 24]:
        remaining = timeout - (time.time() - start_time)
        if remaining < 20: break
        res = find_decomposition(3, 3, 3, r, time_budget=min(10, remaining - 5))
        if res:
            best_333 = res
            best_rank_333 = r
            break

    # Search for M_{3,3,2} decomposition
    best_332 = None
    best_rank_332 = 18 # Schoolbook
    for r in [17, 16, 15]:
        remaining = timeout - (time.time() - start_time)
        if remaining < 20: break
        res = find_decomposition(3, 3, 2, r, time_budget=min(10, remaining - 5))
        if res:
            best_332 = res
            best_rank_332 = r
            break

    # Construct final result
    remaining = timeout - (time.time() - start_time)
    
    # If we found improvements
    if best_333 and best_332:
        # Check if combined rank is better than schoolbook (72)
        total_rank = 2 * best_rank_333 + best_rank_332
        if total_rank < 72:
            try:
                return combine_decompositions(best_333[0], best_333[1], best_333[2], 
                                              best_332[0], best_332[1], best_332[2], 3, 3)
            except Exception as e:
                pass
    elif best_333:
        # Use found 333 and schoolbook 332
        total_rank = 2 * best_rank_333 + 18
        if total_rank < 72:
             # Need schoolbook 332
             sb_332 = find_decomposition(3, 3, 2, 18, time_budget=1) # Should be fast
             if sb_332:
                 return combine_decompositions(best_333[0], best_333[1], best_333[2], 
                                               sb_332[0], sb_332[1], sb_332[2], 3, 3)
    elif best_332:
        # Use schoolbook 333 and found 332
        total_rank = 2 * 27 + best_rank_332
        if total_rank < 72:
             sb_333 = find_decomposition(3, 3, 3, 27, time_budget=1)
             if sb_333:
                 return combine_decompositions(sb_333[0], sb_333[1], sb_333[2], 
                                               best_332[0], best_332[1], best_332[2], 3, 3)

    # Fall back to schoolbook
    return schoolbook()
