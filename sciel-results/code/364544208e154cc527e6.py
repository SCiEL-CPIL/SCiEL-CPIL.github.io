import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 6
    
    # Build the 3x3x3 tensor T333
    T333 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T333[a, b, c] = 1.0
    
    # ALS to find rank-19 decomposition of 3x3x3
    def als_333(rank, max_iter=2000, tol=1e-8):
        """Alternating Least Squares for 3x3x3 tensor decomposition - vectorized."""
        U = np.random.randn(9, rank) * 0.5
        V = np.random.randn(9, rank) * 0.5
        W = np.random.randn(9, rank) * 0.5
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute current approximation
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Update W
            for r in range(rank):
                # R[a,b,c] = T333[a,b,c] - sum_{s!=r} U[a,s]V[b,s]W[c,s]
                # = T333[a,b,c] - (approx[a,b,c] - U[a,r]V[b,r]W[c,r])
                R = T333 - approx + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = U[:, r] * V[:, r]
                num = np.einsum('ab,abc->c', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-12:
                    W[:, r] = num / den
                else:
                    W[:, r] = 0.0
            
            # Recompute approx after W update
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Update U
            for r in range(rank):
                R = T333 - approx + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = V[:, r] * W[:, r]
                num = np.einsum('bc,abc->a', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-12:
                    U[:, r] = num / den
                else:
                    U[:, r] = 0.0
            
            # Recompute approx after U update
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Update V
            for r in range(rank):
                R = T333 - approx + np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                coeff = U[:, r] * W[:, r]
                num = np.einsum('ac,abc->b', coeff, R)
                den = np.sum(coeff**2)
                if den > 1e-12:
                    V[:, r] = num / den
                else:
                    V[:, r] = 0.0
            
            # Compute error
            approx = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T333 - approx)**2)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        if _verify_333(U, V, W):
            return U, V, W
        return None
    
    def _verify_333(U, V, W, tol=1e-4):
        """Verify 3x3x3 decomposition using matrix operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    if abs(approx[a, b, c] - 1.0) > tol:
                        return False
        
        for a in range(9):
            i_a = a // 3
            j_a = a % 3
            for b in range(9):
                j_b = b // 3
                k_b = b % 3
                for c in range(9):
                    k_c = c // 3
                    i_c = c % 3
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        if abs(approx[a, b, c]) > tol:
                            return False
        return True
    
    def _verify_full(U, V, W, n=3, m=3, p=6, tol=1e-4):
        """Verify full 3x3x6 decomposition using matrix operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n*m: return False
        if V.shape[0] != m*p: return False
        if W.shape[0] != p*n: return False
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]: return False
        
        # Build full tensor approximation
        approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    if abs(approx[a, b, c] - 1.0) > tol:
                        return False
        
        # Check that non-required entries are zero
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_c == i_a):
                        if abs(approx[a, b, c]) > tol:
                            return False
        return True
    
    # Try to find rank-19 3x3x3 decomposition
    U333 = V333 = W333 = None
    for rank in range(19, 26):
        if time.time() > deadline - 15:
            break
        
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 1000 + rank)
            U333, V333, W333 = als_333(rank, max_iter=800)
            if U333 is not None:
                break
        
        if U333 is not None:
            break
    
    # Also try direct ALS on 3x3x6
    if time.time() < deadline - 20:
        T336 = np.zeros((9, 18, 18))
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    a = i * 3 + j
                    b = j * 6 + k
                    c = k * 3 + i
                    T336[a, b, c] = 1.0
        
        def als_336(rank, max_iter=500, tol=1e-8):
            """ALS for 3x3x6 tensor."""
            np.random.seed(12345 + rank)
            U = np.random.randn(9, rank) * 0.1
            V = np.random.randn(18, rank) * 0.1
            W = np.random.randn(18, rank) * 0.1
            
            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update W
                for r in range(rank):
                    R = T336.copy()
                    for s in range(rank):
                        if s != r:
                            R -= np.einsum('as,bs,cs->abc', U[:, s], V[:, s], W[:, s])
                    coeff = U[:, r] * V[:, r]
                    num = np.einsum('ab,abc->c', coeff, R)
                    den = np.sum(coeff**2)
                    if den > 1e-12:
                        W[:, r] = num / den
                    else:
                        W[:, r] = 0.0
                
                # Update U
                for r in range(rank):
                    R = T336.copy()
                    for s in range(rank):
                        if s != r:
                            R -= np.einsum('as,bs,cs->abc', U[:, s], V[:, s], W[:, s])
                    coeff = V[:, r] * W[:, r]
                    num = np.einsum('bc,abc->a', coeff, R)
                    den = np.sum(coeff**2)
                    if den > 1e-12:
                        U[:, r] = num / den
                    else:
                        U[:, r] = 0.0
                
                # Update V
                for r in range(rank):
                    R = T336.copy()
                    for s in range(rank):
                        if s != r:
                            R -= np.einsum('as,bs,cs->abc', U[:, s], V[:, s], W[:, s])
                    coeff = U[:, r] * W[:, r]
                    num = np.einsum('ac,abc->b', coeff, R)
                    den = np.sum(coeff**2)
                    if den > 1e-12:
                        V[:, r] = num / den
                    else:
                        V[:, r] = 0.0
                
                approx = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.sum((T336 - approx)**2)
                
                if abs(prev_err - err) < tol:
                    break
                prev_err = err
            
            if _verify_full(U, V, W, tol=1e-4):
                return U, V, W
            return None
        
        for rank in range(28, 36):
            if time.time() > deadline - 10:
                break
            
            for seed in range(5):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(seed * 10000 + rank)
                result = als_336(rank, max_iter=300)
                if result is not None:
                    U, V, W = result
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Build 3x3x6 decomposition from 3x3x3
    if U333 is not None:
        rank_333 = U333.shape[1]
        rank_total = 2 * rank_333
        
        U = np.zeros((9, rank_total))
        V = np.zeros((18, rank_total))
        W = np.zeros((18, rank_total))
        
        # First half: k in [0,3)
        for r in range(rank_333):
            U[:, r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + k, r] = V333[j*3 + k, r]
            # W for first half: c = k*3 + i where k in [0,3)
            for k in range(3):
                for i in range(3):
                    W[k*3 + i, r] = W333[k*3 + i, r]
        
        # Second half: k in [3,6)
        for r in range(rank_333):
            U[:, rank_333 + r] = U333[:, r]
            for j in range(3):
                for k in range(3):
                    V[j*6 + (k+3), rank_333 + r] = V333[j*3 + k, r]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3 + i, rank_333 + r] = W333[k*3 + i, r]
        
        # Quick verification
        if _verify_full(U, V, W, tol=1e-4):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
