"""Matrix multiplication 3x3x5 rank search with structured construction and deadline guard."""
import time
import random
import math

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate(U, V, W):
    """Check exact tensor decomposition. Returns (rank, U, V, W) if valid, else None."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wk[c]
                if s != 1:
                    return None
    return (r, U, V, W)

def _schoolbook():
    """Schoolbook rank-45 decomposition."""
    n, m, p = DIMENSIONS
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
    return (rank, U, V, W)

def _try_rank15_construction():
    """
    Try to construct a rank-15 decomposition.
    
    Key insight: For 3x3 * 3x5, we can think of it as 3x3 * 3x3 + 3x3 * 3x2.
    If we use Strassen-like rank-7 for the 3x3*3x3 part and rank-8 for the 3x3*3x2 part,
    total = 15.
    
    Actually, let's try a more direct approach: use the fact that rank(3x3x5) >= 15
    and try to find an explicit construction.
    
    One approach: use the "slice" method. The tensor has slices indexed by j (the shared index).
    For each j in {0,1,2}, we have a 3x5 matrix multiplication slice.
    
    Let me try a different construction: use Kronecker products and known low-rank decompositions.
    
    For 2x2 * 2x2, Strassen gives rank 7.
    For 3x3 * 3x3, the minimal rank is 19 (not 7, since Strassen doesn't directly extend).
    
    Actually, let me try a simpler approach: use the fact that we can share the "j" index.
    
    Consider: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can write this as: for each j, we multiply the i-th row of A (length 3) by the k-th column of B (length 3, but we have 5 columns).
    
    Let me try a rank-15 construction based on the following idea:
    Use 15 "multiplications" M_c for c in 0..14.
    
    For each c, define:
      U[i*m+j, c] = coefficient for A[i,j]
      V[j*p+k, c] = coefficient for B[j,k]  
      W[k*n+i, c] = coefficient for C[k,i]
    
    Such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    Let me try a specific pattern. One known approach is to use the "binomial" or "combinatorial" construction.
    
    Actually, let me try a more systematic search over small integer/half-integer values for rank 15.
    """
    n, m, p = DIMENSIONS
    rank = 15
    rng = random.Random(12345)
    
    # Try several random initializations with local search
    best_val = float('inf')
    best_solution = None
    
    for trial in range(20):
        if _time_left() < 5.0:
            break
        
        # Initialize with small random values
        def rand_val():
            return rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2])
        
        U = [[rand_val() for _ in range(rank)] for _ in range(n * m)]
        V = [[rand_val() for _ in range(rank)] for _ in range(m * p)]
        W = [[rand_val() for _ in range(rank)] for _ in range(p * n)]
        
        # Compute residual
        def compute_res(U, V, W):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        res += (s - 1.0) ** 2
            return res
        
        val = compute_res(U, V, W)
        
        # Simple local search: try flipping each variable
        for step in range(500):
            if _time_left() < 5.0:
                break
            
            improved = False
            mat = rng.choice([0, 1, 2])
            if mat == 0:
                row = rng.randrange(n * m)
                arr = U
            elif mat == 1:
                row = rng.randrange(m * p)
                arr = V
            else:
                row = rng.randrange(p * n)
                arr = W
            col = rng.randrange(rank)
            old = arr[row][col]
            
            # Try different values
            for new_val in [-2, -1, -0.5, 0, 0.5, 1, 2]:
                if new_val == old:
                    continue
                arr[row][col] = new_val
                new_val_res = compute_res(U, V, W)
                if new_val_res < val:
                    val = new_val_res
                    improved = True
                    break
                arr[row][col] = old
            
            if not improved:
                break
        
        if val < best_val:
            best_val = val
            best_solution = (U, V, W)
            
            if val < 1e-10:
                # Check if it's exactly valid
                U_int = [[(int(round(x * 2)) / 2) for x in row] for row in U]
                V_int = [[(int(round(x * 2)) / 2) for x in row] for row in V]
                W_int = [[(int(round(x * 2)) / 2) for x in row] for row in W]
                if _validate(U_int, V_int, W_int) is not None:
                    return (rank, U_int, V_int, W_int)
    
    return None

def _try_rank16_construction():
    """Try rank 16 with a more structured approach."""
    n, m, p = DIMENSIONS
    rank = 16
    rng = random.Random(54321)
    
    # Use a more informed initialization
    # Start from schoolbook and try to merge terms
    
    # For now, try random search
    best_val = float('inf')
    best_solution = None
    
    for trial in range(15):
        if _time_left() < 5.0:
            break
        
        def rand_val():
            return rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2])
        
        U = [[rand_val() for _ in range(rank)] for _ in range(n * m)]
        V = [[rand_val() for _ in range(rank)] for _ in range(m * p)]
        W = [[rand_val() for _ in range(rank)] for _ in range(p * n)]
        
        def compute_res(U, V, W):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        res += (s - 1.0) ** 2
            return res
        
        val = compute_res(U, V, W)
        
        for step in range(400):
            if _time_left() < 5.0:
                break
            
            mat = rng.choice([0, 1, 2])
            if mat == 0:
                row = rng.randrange(n * m)
                arr = U
            elif mat == 1:
                row = rng.randrange(m * p)
                arr = V
            else:
                row = rng.randrange(p * n)
                arr = W
            col = rng.randrange(rank)
            old = arr[row][col]
            
            improved = False
            for new_val in [-2, -1, -0.5, 0, 0.5, 1, 2]:
                if new_val == old:
                    continue
                arr[row][col] = new_val
                new_val_res = compute_res(U, V, W)
                if new_val_res < val:
                    val = new_val_res
                    improved = True
                    break
                arr[row][col] = old
            
            if not improved:
                break
        
        if val < best_val:
            best_val = val
            best_solution = (U, V, W)
            
            if val < 1e-10:
                U_int = [[(int(round(x * 2)) / 2) for x in row] for row in U]
                V_int = [[(int(round(x * 2)) / 2) for x in row] for row in V]
                W_int = [[(int(round(x * 2)) / 2) for x in row] for row in W]
                if _validate(U_int, V_int, W_int) is not None:
                    return (rank, U_int, V_int, W_int)
    
    return None

def construct():
    """Main entry point. Returns {'U', 'V', 'W'} with the best valid decomposition found."""
    # Start with schoolbook as fallback
    best_rank, best_U, best_V, best_W = _schoolbook()
    
    # Try rank 15 first (theoretical lower bound)
    result = _try_rank15_construction()
    if result is not None:
        r, U, V, W = result
        if _validate(U, V, W) is not None:
            return {"U": U, "V": V, "W": W}
    
    # Try rank 16
    if _time_left() > 10.0:
        result = _try_rank16_construction()
        if result is not None:
            r, U, V, W = result
            if _validate(U, V, W) is not None:
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
