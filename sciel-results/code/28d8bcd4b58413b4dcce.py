import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20

    def build_naive():
        """Naive decomposition: rank = m*p = 20? Let's see."""
        # For each (j, k) pair, we need to handle the i dimension
        # Actually, let me build it properly
        rank = m * p  # 2*10 = 20
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        col = 0
        for j in range(m):
            for k in range(p):
                # This term: A[i,j] * B[j,k] contributes to C[i,k]
                # U index: i*m + j for i in {0,1}
                # V index: j*p + k
                # W index: k*n + i for i in {0,1}
                # 
                # We need 2 rank-1 terms for this (j,k) pair (one for each i)
                # Or can we do it in 1? No, because U and W each have 2 nonzeros.
                # So we need 2 terms per (j,k) pair.
                pass
        
        # Actually naive is 2*m*p = 40 rank-1 terms
        # But we can do better: for fixed (j,k), the contribution is:
        # T[j, j*p+k, k*2+0] = 1 (for i=0)
        # T[2+j, j*p+k, k*2+1] = 1 (for i=1)
        # This is rank 2, so total naive rank = 2 * m * p = 40
        
        # Let me just build it
        rank = 2 * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        col = 0
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-9):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += Ua[idx_u, r] * Va[idx_v, r] * Wa[idx_w, r]
                    expected = 1.0
                    if abs(total - expected) > tol:
                        return False
        return True

    def build_strassen_rank35():
        rank = 35
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        strassen = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
            ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
            ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
            ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        col = 0
        for block in range(5):
            for a, b, c in strassen:
                U[0, col] = a[0]
                U[1, col] = a[1]
                U[2, col] = a[2]
                U[3, col] = a[3]
                
                V[0*p + 2*block, col] = b[0]
                V[0*p + 2*block+1, col] = b[1]
                V[1*p + 2*block, col] = b[2]
                V[1*p + 2*block+1, col] = b[3]
                
                W[(2*block)*2 + 0, col] = c[0]
                W[(2*block)*2 + 1, col] = c[1]
                W[(2*block+1)*2 + 0, col] = c[2]
                W[(2*block+1)*2 + 1, col] = c[3]
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Start with Strassen rank 35
    best = build_strassen_rank35()
    
    # Try to find lower rank solutions
    # Use a local search: start from a candidate and try to zero out columns
    
    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_iter=5000):
        """Try to find a valid decomposition with exactly target_rank nonzero columns."""
        Ua = np.array(U_init, dtype=float).copy()
        Va = np.array(V_init, dtype=float).copy()
        Wa = np.array(W_init, dtype=float).copy()
        
        # Randomly permute columns
        idx = np.random.permutation(Ua.shape[1])
        Ua = Ua[:, idx]
        Va = Va[:, idx]
        Wa = Wa[:, idx]
        
        # Try to zero out columns from the end
        for r in range(Ua.shape[1] - target_rank, -1, -1):
            if time.time() > deadline - 5:
                break
            # Try to zero out column r by adjusting other columns
            # This is a linear system: we want to find adjustments to columns 0..r-1
            # such that the contribution of column r is absorbed
            # For now, just try setting it to zero and checking
            Ua_save = Ua[r].copy()
            Va_save = Va[r].copy()
            Wa_save = Wa[r].copy()
            
            Ua[r] = 0
            Va[r] = 0
            Wa[r] = 0
            
            if verify(Ua.tolist(), Va.tolist(), Wa.tolist()):
                # Found a valid solution with fewer effective columns
                Ua = Ua[:, :r]
                Va = Va[:, :r]
                Wa = Wa[:, :r]
                return Ua.tolist(), Va.tolist(), Wa.tolist()
            
            Ua[r] = Ua_save
            Va[r] = Va_save
            Wa[r] = Wa_save
        
        return None

    # Try various approaches
    best_rank = 35
    best_solution = best
    
    # Try random search for lower ranks
    if time.time() < deadline - 30:
        for target_rank in range(34, 19, -1):
            if time.time() > deadline - 10:
                break
            found = False
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                # Random initialization with small integers
                U_try = np.random.randint(-2, 3, (nm, target_rank))
                V_try = np.random.randint(-2, 3, (mp, target_rank))
                W_try = np.random.randint(-2, 3, (pn, target_rank))
                
                if verify(U_try.tolist(), V_try.tolist(), W_try.tolist()):
                    best_solution = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    return best_solution
