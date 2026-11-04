"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n=3, m=4, p=7):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0.0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    def make_schoolbook():
        rank = nmp
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
        return U, V, W
    
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    total = np.dot(u_row, v_row * w_row)
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()
    
    # Try structured approaches first
    # Approach: Use the fact that we can sometimes save by combining terms
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Try to find small rank reductions by modifying the schoolbook
    # Key insight: if two schoolbook terms can be merged or if a linear combination
    # of terms gives the same result with fewer terms, we can reduce rank.
    
    # For now, try a limited random search focused on near-schoolbook ranks
    # with a smarter initialization that starts from schoolbook and perturbs
    
    target_ranks = [83, 82, 81, 80]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Start from schoolbook and try to merge two terms
            # or find a dependency
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with schoolbook-like structure
            # Use the first target_rank of the 84 schoolbook terms
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                        else:
                            break
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            # Now try to fix the missing terms by adjusting existing ones
            # This is heuristic - try random small perturbations
            for _ in range(3):
                # Pick a random column and try to adjust
                c = random.randint(0, target_rank - 1)
                # Try adding a small random contribution
                U[random.randint(0, n*m-1), c] += random.choice([-0.5, 0.5])
                V[random.randint(0, m*p-1), c] += random.choice([-0.5, 0.5])
                W[random.randint(0, p*n-1), c] += random.choice([-0.5, 0.5])
            
            # Round to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            if verify_np(U.tolist(), V.tolist(), W.tolist()):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < nmp:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
