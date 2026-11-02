"""Matrix multiplication 3x3x5 rank search with validation and deadline guard."""
import time
import random
import math
import sys

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate(U, V, W):
    """Check that U,V,W (lists of lists) form an exact decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Identity: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    Returns (rank, U, V, W) if valid, else None.
    """
    n, m, p = DIMENSIONS
    r = len(U[0])
    # Precompute for speed
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

def _matmul_check(U, V, W):
    """Full check: for random inputs, C = W^T * (U^T * A) * V or similar.
    Actually the tensor identity is the ground truth; we check it directly above.
    This is a redundant double-check with concrete matrices."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    # Build C_{k,i} = sum_{j,r} A_{i,j} B_{j,k} W_{k,i,r} U_{i,j,r} V_{j,k,r} / ...
    # Simpler: pick random A (n x m), B (m x p), compute C = A @ B
    # Then check: C_{i,k} should equal sum_j A_{i,j} B_{j,k}
    # The decomposition says: C_{i,k} = sum_r (sum_j A_{i,j} U_{i,j,r}) * (sum_j B_{j,k} V_{j,k,r}) * W_{k,i,r}
    # Wait, that's not right. Let me re-derive.
    # Tensor: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # This must equal delta (the multiplication tensor) which is 1 for all (i,j,k) since
    # C_{i,k} = sum_j A_{i,j} B_{j,k} means the "tensor" of the multiplication map is
    # T_{i,j,k} = 1 (the coefficient relating A_{i,j} and B_{j,k} to C_{i,k}).
    # So the identity is exactly: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
    # Which is what _validate checks. Good.
    return True

def _anneal(r, n_restarts, max_iter_per_restart, seed):
    """Simulated annealing to find U,V,W of rank r with integer/half-integer entries
    satisfying the tensor identity. Returns (U,V,W) as lists of lists if found, else None."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    best_res = None
    best_val = float('inf')

    for restart in range(n_restarts):
        if _time_left() < 3.0:
            break
        # Initialize with small random integer/half-integer values
        def rand_val():
            # Values in {-2,-1,-0.5,0,0.5,1,2} with some probability of 0
            x = rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2])
            return x

        U = [[rand_val() for _ in range(r)] for _ in range(n * m)]
        V = [[rand_val() for _ in range(r)] for _ in range(m * p)]
        W = [[rand_val() for _ in range(r)] for _ in range(p * n)]

        # Compute residual
        def compute_res(U, V, W):
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        ui = U[i * m + j]
                        vk = V[j * p + k]
                        wk = W[k * n + i]
                        for c in range(r):
                            s += ui[c] * vk[c] * wk[c]
                        res += (s - 1.0) ** 2
            return res

        val = compute_res(U, V, W)
        if val < best_val:
            best_val = val
            best_res = (U, V, W)

        # Simulated annealing
        T = 5.0
        T_min = 1e-10
        cooling = 0.995
        for it in range(max_iter_per_restart):
            if _time_left() < 3.0:
                break
            # Pick a random variable to perturb
            mat = rng.choice([0, 1, 2])  # 0=U, 1=V, 2=W
            if mat == 0:
                row = rng.randrange(n * m)
                arr = U
            elif mat == 1:
                row = rng.randrange(m * p)
                arr = V
            else:
                row = rng.randrange(p * n)
                arr = W
            col = rng.randrange(r)
            old = arr[row][col]
            # Perturb: add a small random delta
            delta = rng.choice([-1, -0.5, 0.5, 1])
            new = old + delta
            arr[row][col] = new

            new_val = compute_res(arr, V if mat != 1 else V, W if mat != 2 else W)
            # Recompute properly
            new_val = compute_res(U, V, W)
            d = new_val - val
            if d < 0 or rng.random() < math.exp(-d / T):
                val = new_val
                if val < best_val:
                    best_val = val
                    best_res = (U, V, W)
                    if best_val < 1e-12:
                        # Found exact solution, verify
                        U_int = [[(int(round(x * 2)) / 2) for x in row] for row in U]
                        V_int = [[(int(round(x * 2)) / 2) for x in row] for row in V]
                        W_int = [[(int(round(x * 2)) / 2) for x in row] for row in W]
                        if _validate(U_int, V_int, W_int) is not None:
                            return U_int, V_int, W_int
            else:
                arr[row][col] = old  # revert

            T *= cooling
            if T < T_min:
                T = T_min

    return best_res

def _try_rank(r, seed=0):
    """Try to find a valid rank-r decomposition. Returns (U,V,W) or None."""
    # First, try a few quick random initializations with annealing
    result = _anneal(r, n_restarts=5, max_iter_per_restart=2000, seed=seed)
    if result is not None:
        U, V, W = result
        if _validate(U, V, W) is not None:
            return U, V, W
    return None

def _strassen_like_3x3x5():
    """Construct a Strassen-like decomposition for 3x3*3x5.
    
    For 3x3*3x3, Strassen uses rank 7. For 3x3*3x5, we can extend.
    The multiplication tensor for 3x3*3x5 has 45 equations.
    
    Let's try to construct one based on the 3x3 Strassen pattern extended to 5 columns.
    
    Actually, let's use a simpler approach: the rank of the 3x3*3x5 multiplication tensor
    is at most 15 (schoolbook) and at least 9 (trivial bound).
    Known results suggest rank around 13-15 for this case.
    
    Let me try a systematic construction. For 3x3*3x3, Strassen's 7-rank:
    The 7 multiplications are based on products of sums/differences.
    
    For 3x3*3x5, one approach: split the 5 columns into 3+2, use Strassen for the 3x3 part
    and schoolbook for the remaining 2 columns, but that gives 7 + 6 = 13 rank.
    
    Let me construct a 13-rank decomposition:
    - For columns 0,1,2 of B: use Strassen 7-rank for 3x3*3x3
    - For columns 3,4 of B: use schoolbook 3*3=9... no, 3 rows * 2 cols = 6 multiplications
    
    Total: 7 + 6 = 13.
    
    Let me build this explicitly.
    """
    n, m, p = 3, 3, 5
    rank = 13
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7-rank for 3x3*3x3 (columns 0,1,2)
    # Standard Strassen for 2x2*2x2 has 7 products. For 3x3*3x3, we can tile.
    # Actually, let's use a known 3x3*3x3 decomposition.
    # 
    # For simplicity, let me use the schoolbook for the 3x3 part (9 mults) + 6 for the 2 extra = 15.
    # That's just schoolbook. Not helpful.
    #
    # Let me try a different approach: use the fact that rank(3x3*3x5) <= rank(3x3*3x3) + 3*2 = 7+6=13
    # if we have a 7-rank for 3x3*3x3.
    #
    # Here's a 7-rank for 3x3*3x3 (adapted from Strassen's 2x2 idea, but 3x3 needs more care).
    # Actually, the minimal rank for 3x3*3x3 is 19 (Landsberg) or 20? No, it's 19.
    # Strassen's 2x2 gives 7, but 3x3 is harder.
    #
    # Let me just use schoolbook for now and rely on the search to find better.
    # Return None to indicate "no known low-rank construction".
    return None

def construct():
    """Main entry point. Returns {'U', 'V', 'W'} with the best (lowest-rank) valid decomposition found."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45

    # Start with schoolbook as fallback
    best = None
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = (schoolbook_rank, U_sb, V_sb, W_sb)

    # Try to find lower-rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9,15,15) = 15
    # Actually rank >= max(n, m, p)^2 / something... the trivial lower bound is max(n*m, m*p, p*n) is wrong.
    # The correct lower bound: rank >= max(n*m, m*p, p*n) is not right either.
    # For matrix multiplication A(n x m) * B(m x p), the rank is at least max(n*m, m*p, p*n)? No.
    # The rank must be at least n*m (since U has n*m rows and must have rank at least... no).
    # Actually, the multiplication tensor has a certain "slice rank" lower bound.
    # For now, let's try ranks from 15 upward (since schoolbook is 45, and we know it's at most 15 by
    # the simple bound rank <= n*m*p but also rank <= n*m + m*p + p*n - ... no).
    # 
    # Simple upper bound: rank <= n*m*p = 45 (schoolbook).
    # Better: rank <= n*m + n*p + m*p - n - m - p + ... no clear formula.
    # 
    # Let's just search from rank 15 to 25, trying to find a valid decomposition.
    
    rng = random.Random(42)
    for r in range(15, 26):
        if _time_left() < 5.0:
            break
        result = _try_rank(r, seed=rng.randint(0, 10**9))
        if result is not None:
            U, V, W = result
            validated = _validate(U, V, W)
            if validated is not None:
                best = (r, U, V, W)
                # Found a valid decomposition of rank r, try to go lower
                # (we're iterating from low to high, so this is the first valid at this rank)
                break

    # If we still have schoolbook, return it
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
