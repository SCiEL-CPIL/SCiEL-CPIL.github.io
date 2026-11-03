"""Matrix multiplication 4,4,8 — rank-minimal exact tensor decomposition."""
import time
import random
import math
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Returns U (16 x rank), V (32 x rank), W (32 x rank) with integer/half-integer entries.
    Score is the rank.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 120.0  # safety margin before 150s kill

    # The tensor T has entries T[i, j, k] = 1 iff (i, j, k) corresponds to a valid
    # multiplication term. For A (n x m) times B (m x p), output C (n x p):
    # C[a, c] = sum_b A[a, b] * B[b, c]
    # Tensor index: U index = a*m + b (n*m = 16), V index = b*p + c (m*p = 32),
    # W index = c*n + a (p*n = 32) — official trace convention W[k*n+i, r] where
    # k = c (row of B / col of C), i = a (row of A / row of C).
    # T[a*m+b, b*p+c, c*n+a] = 1

    # Build the true tensor as a list of (i, j, k) triples
    tensor_entries = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                tensor_entries.append((i, j, k))

    # Schoolbook rank = n*m*p = 128
    rank_schoolbook = n * m * p

    def make_schoolbook():
        """Return (U, V, W) as lists of lists for schoolbook decomposition."""
        r = rank_schoolbook
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for (i, j, k) in tensor_entries:
            U[i][col] = 1
            V[j][col] = 1
            W[k][col] = 1
            col += 1
        return U, V, W

    def verify(U, V, W):
        """Check that U @ V^T (broadcast) matches the tensor exactly."""
        r = len(U[0])
        # Build tensor from U, V, W: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Check each entry
        for (i, j, k) in tensor_entries:
            val = sum(U[i][rr] * V[j][rr] * W[k][rr] for rr in range(r))
            if abs(val - 1) > 1e-9:
                return False
        # Also check that non-entries are zero (sample a few)
        # For efficiency, check all non-entries
        entry_set = set(tensor_entries)
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    if (i, j, k) not in entry_set:
                        val = sum(U[i][rr] * V[j][rr] * W[k][rr] for rr in range(r))
                        if abs(val) > 1e-9:
                            return False
        return True

    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        r = U_np.shape[1]
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Compute via einsum
        T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        # Build true tensor
        T_true = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for (i, j, k) in tensor_entries:
            T_true[i, j, k] = 1.0
        return np.allclose(T, T_true, atol=1e-9, rtol=0)

    def round_to_half(arr):
        """Round array entries to nearest half-integer."""
        return [[round(x * 2) / 2 for x in row] for row in arr]

    def try_strassen_based():
        """
        Try to construct a decomposition using Strassen-like ideas.
        For 4x4 * 4x8, we can think of it as two 4x4 * 4x4 multiplications.
        If we have a rank-25 decomposition for 4x4*4x4, we could potentially
        share some structure. But without explicit rank-25 matrices, let's try
        a different approach: use the fact that 4x4*4x8 can be decomposed by
        splitting the 8-wide output into two 4-wide outputs.
        
        Actually, let's try a more direct approach: use known small-rank
        decompositions for sub-blocks.
        """
        pass

    def local_search(start_rank, max_iters=5000):
        """
        Local search to find a valid decomposition at the given rank.
        Start from a random or structured initialization.
        """
        r = start_rank
        # Initialize with random values in [-1, 1]
        U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p * n)]
        
        best_err = float('inf')
        best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute error
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
            T_true = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for (i, j, k) in tensor_entries:
                T_true[i, j, k] = 1.0
            err = np.sum((T - T_true) ** 2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
            
            if err < 1e-10:
                # Try rounding
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                if verify_fast(U_r, V_r, W_r):
                    return U_r, V_r, W_r, r
                # Also try integer rounding
                U_i = [[int(round(x)) for x in row] for row in U]
                V_i = [[int(round(x)) for x in row] for row in V]
                W_i = [[int(round(x)) for x in row] for row in W]
                if verify_fast(U_i, V_i, W_i):
                    return U_i, V_i, W_i, r
                # Try rounding to 1/4
                U_q = [[round(x * 4) / 4 for x in row] for row in U]
                V_q = [[round(x * 4) / 4 for x in row] for row in V]
                W_q = [[round(x * 4) / 4 for x in row] for row in W]
                if verify_fast(U_q, V_q, W_q):
                    return U_q, V_q, W_q, r
                # If close but not exact, do a few more iterations with smaller steps
                for _ in range(100):
                    which = random.choice(['U', 'V', 'W'])
                    if which == 'U':
                        idx = random.randint(0, n * m - 1)
                        rr = random.randint(0, r - 1)
                        U[idx][rr] += random.uniform(-0.05, 0.05)
                    elif which == 'V':
                        idx = random.randint(0, m * p - 1)
                        rr = random.randint(0, r - 1)
                        V[idx][rr] += random.uniform(-0.05, 0.05)
                    else:
                        idx = random.randint(0, p * n - 1)
                        rr = random.randint(0, r - 1)
                        W[idx][rr] += random.uniform(-0.05, 0.05)
                    U_np = np.array(U)
                    V_np = np.array(V)
                    W_np = np.array(W)
                    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
                    err = np.sum((T - T_true) ** 2)
                    if err < 1e-10:
                        U_r = round_to_half(U)
                        V_r = round_to_half(V)
                        W_r = round_to_half(W)
                        if verify_fast(U_r, V_r, W_r):
                            return U_r, V_r, W_r, r
                        U_i = [[int(round(x)) for x in row] for row in U]
                        V_i = [[int(round(x)) for x in row] for row in V]
                        W_i = [[int(round(x)) for x in row] for row in W]
                        if verify_fast(U_i, V_i, W_i):
                            return U_i, V_i, W_i, r
                        break
            
            # Update: random perturbation
            which = random.choice(['U', 'V', 'W'])
            if which == 'U':
                idx = random.randint(0, n * m - 1)
                rr = random.randint(0, r - 1)
                U[idx][rr] += random.uniform(-0.5, 0.5)
            elif which == 'V':
                idx = random.randint(0, m * p - 1)
                rr = random.randint(0, r - 1)
                V[idx][rr] += random.uniform(-0.5, 0.5)
            else:
                idx = random.randint(0, p * n - 1)
                rr = random.randint(0, r - 1)
                W[idx][rr] += random.uniform(-0.5, 0.5)
        
        return None

    def structured_search(rank):
        """
        Try a structured initialization: start from schoolbook and try to
        merge columns or set some entries to zero to reduce rank.
        """
        r = rank
        if r >= rank_schoolbook:
            # Just return schoolbook
            return make_schoolbook()
        
        # Try random restarts with structured init
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with some structure: use first r of the 128 schoolbook columns
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # Fill with random values
            for i in range(n * m):
                for rr in range(r):
                    U[i][rr] = random.choice([-1, 0, 1])
            for j in range(m * p):
                for rr in range(r):
                    V[j][rr] = random.choice([-1, 0, 1])
            for k in range(p * n):
                for rr in range(r):
                    W[k][rr] = random.choice([-1, 0, 1])
            
            # Local search from here
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)
            T_true = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for (i, j, k) in tensor_entries:
                T_true[i, j, k] = 1.0
            
            best_err = float('inf')
            for it in range(2000):
                if time.time() > deadline - 10:
                    break
                T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
                err = np.sum((T - T_true) ** 2)
                if err < best_err:
                    best_err = err
                    best_U_np = U_np.copy()
                    best_V_np = V_np.copy()
                    best_W_np = W_np.copy()
                
                if err < 1e-10:
                    U_r = round_to_half(U_np.tolist())
                    V_r = round_to_half(V_np.tolist())
                    W_r = round_to_half(W_np.tolist())
                    if verify_fast(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                
                # Gradient-like update
                # dT/dU[i,rr] = V[j,rr] * W[k,rr]
                # Simplified: random perturbation
                which = random.choice(['U', 'V', 'W'])
                if which == 'U':
                    idx = random.randint(0, n * m - 1)
                    rr = random.randint(0, r - 1)
                    U_np[idx, rr] += random.uniform(-0.3, 0.3)
                elif which == 'V':
                    idx = random.randint(0, m * p - 1)
                    rr = random.randint(0, r - 1)
                    V_np[idx, rr] += random.uniform(-0.3, 0.3)
                else:
                    idx = random.randint(0, p * n - 1)
                    rr = random.randint(0, r - 1)
                    W_np[idx, rr] += random.uniform(-0.3, 0.3)
            
            # Final check
            if best_err < 1e-8:
                U_r = round_to_half(best_U_np.tolist())
                V_r = round_to_half(best_V_np.tolist())
                W_r = round_to_half(best_W_np.tolist())
                if verify_fast(U_r, V_r, W_r):
                    return U_r, V_r, W_r
        
        return None

    # Strategy: try ranks from low to high
    # Lower bound: at least n*m*p / (something), but for now try 25, 30, 35, 40, 45, 50, ...
    # For 4x4*4x4, rank 25 is known. For 4x4*4x8, a reasonable upper bound is 50 (two 4x4*4x4).
    # Try to find something below 128.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try a few ranks
    for target_rank in [25, 28, 30, 32, 35, 40, 45, 50, 60, 80, 100]:
        if time.time() > deadline - 20:
            break
        
        if target_rank >= rank_schoolbook:
            break
        
        # Try structured search
        result = structured_search(target_rank)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W)
                best_rank = target_rank
                break
        
        # Try local search
        result = local_search(target_rank, max_iters=3000)
        if result is not None:
            U, V, W, r = result
            if verify_fast(U, V, W):
                best_result = (U, V, W)
                best_rank = r
                break
    
    # If no better result found, return schoolbook
    if best_result is None:
        U, V, W = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
