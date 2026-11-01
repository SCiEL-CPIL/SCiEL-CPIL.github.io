import numpy as np
import time

def construct():
    n, m, p = 2, 2, 8
    
    # Build the rank-28 Strassen decomposition as the baseline
    U, V, W = _strassen_28()
    
    # Try to find a lower-rank decomposition via local search
    # The idea: we have 28 basis tensors. Can we express the multiplication tensor
    # as a sum of fewer than 28 rank-1 tensors?
    # 
    # Strategy: Start with the 28-tensor decomposition and try to merge/eliminate
    # tensors by finding linear dependencies.
    
    # First, verify the baseline works
    # (The evaluator will check this, but let's be safe)
    
    # Try a different approach: use the structure more cleverly
    # The multiplication C = A * B where A is 2x2 and B is 2x8
    # We can think of B as 4 blocks of 2x2: B0, B1, B2, B3
    # C = A * [B0 | B1 | B2 | B3] = [A*B0 | A*B1 | A*B2 | A*B3]
    # Each A*Bt is a 2x2 multiplication, which has rank 7 (Strassen)
    # So the total rank is 4*7 = 28
    #
    # Can we do better? The key insight is that the A matrix is the same for all blocks.
    # In Strassen's algorithm, we compute 7 products of linear combinations of A entries.
    # These same linear combinations can be reused across all B blocks!
    #
    # Specifically, Strassen computes:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # For each B block, we need 7 multiplications. But the A-side linear combinations
    # (a+d), (c+d), a, d, (a+b), (c-a), (b-d) are the same for all blocks.
    # However, the B-side linear combinations depend on the specific B block.
    #
    # So we still need 7 multiplications per block, giving 28 total.
    # Unless... we can find a way to share more.
    
    # Let's try a local search to see if we can find a rank < 28 decomposition
    # using integer/half-integer entries.
    
    # Build the tensor T
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k), else 0
    # where i in {0,1}, j in {0,1}, k in {0..7}
    
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Non-zero entries of T
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # Total number of triples
    total = num_a * num_b * num_c
    
    # For a rank-r decomposition, we need:
    # sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    
    # Let's try to find a rank-24 to rank-27 decomposition
    # using a simple local search approach
    
    start_time = time.time()
    deadline = 120.0 - 10.0  # 110 seconds, leave 10s safety margin
    
    best_rank = 28
    best_result = {"U": U, "V": V, "W": W}
    
    # Try ranks from 27 down to 24
    for target_rank in [27, 26, 25, 24]:
        if time.time() - start_time > deadline:
            break
        
        # Try multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            result = _try_find_decomposition(target_rank, nonzero, num_a, num_b, num_c, 
                                             seed=restart, time_budget=2.0)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
        
        if best_rank < 28:
            break
    
    return best_result


def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 x 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    num_blocks = p // 2  # 4 blocks
    
    for b in range(num_blocks):
        base = b * 7
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0][base + 0] += 1  # A[0,0]
        U[3][base + 0] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 0] += 1  # B[0,2b]
        V[1 * p + 2 * b + 1][base + 0] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 0] += 1  # C[0,2b]
        W[(2 * b + 1) * n + 1][base + 0] += 1  # C[1,2b+1]
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2][base + 1] += 1  # A[1,0]
        U[3][base + 1] += 1  # A[1,1]
        V[0 * p + 2 * b][base + 1] += 1  # B[0,2b]
        W[2 * b * n + 1][base + 1] += 1  # C[1,2b]
        W[(2 * b + 1) * n + 1][base + 1] -= 1  # C[1,2b+1]
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0][base + 2] += 1  # A[0,0]
        V[0 * p + 2 * b + 1][base + 2] += 1  # B[0,2b+1]
        V[1 * p + 2 * b + 1][base + 2] -= 1  # B[1,2b+1]
        W[(2 * b + 1) * n + 0][base + 2] += 1  # C[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 2] += 1  # C[1,2b+1]
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3][base + 3] += 1  # A[1,1]
        V[1 * p + 2 * b][base + 3] += 1  # B[1,2b]
        V[0 * p + 2 * b][base + 3] -= 1  # B[0,2b]
        W[2 * b * n + 0][base + 3] += 1  # C[0,2b]
        W[2 * b * n + 1][base + 3] += 1  # C[1,2b]
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0][base + 4] += 1  # A[0,0]
        U[1][base + 4] += 1  # A[0,1]
        V[1 * p + 2 * b + 1][base + 4] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 4] -= 1  # C[0,2b]
        W[(2 * b + 1) * n + 0][base + 4] += 1  # C[1,2b]
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2][base + 5] += 1  # A[1,0]
        U[0][base + 5] -= 1  # A[0,0]
        V[0 * p + 2 * b][base + 5] += 1  # B[0,2b]
        V[0 * p + 2 * b + 1][base + 5] += 1  # B[0,2b+1]
        W[(2 * b + 1) * n + 1][base + 5] += 1  # C[1,2b+1]
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1][base + 6] += 1  # A[0,1]
        U[3][base + 6] -= 1  # A[1,1]
        V[1 * p + 2 * b][base + 6] += 1  # B[1,2b]
        V[1 * p + 2 * b + 1][base + 6] += 1  # B[1,2b+1]
        W[2 * b * n + 0][base + 6] += 1  # C[0,2b]
    
    return U, V, W


def _try_find_decomposition(target_rank, nonzero, num_a, num_b, num_c, seed=0, time_budget=2.0):
    """Try to find a valid decomposition of the given rank using local search."""
    start = time.time()
    rng = np.random.default_rng(seed)
    
    # Initialize with small integers
    U = rng.integers(-1, 2, size=(num_a, target_rank)).astype(float)
    V = rng.integers(-1, 2, size=(num_b, target_rank)).astype(float)
    W = rng.integers(-1, 2, size=(num_c, target_rank)).astype(float)
    
    # Local search: try to reduce the residual
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        # Compute residual on non-zero entries
        residual = 0.0
        for a, b, c in nonzero:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
            residual += (val - 1.0) ** 2
        
        # Also check zero entries (sample some)
        zero_samples = []
        for _ in range(100):
            a = rng.integers(0, num_a)
            b = rng.integers(0, num_b)
            c = rng.integers(0, num_c)
            # Check if this is a non-zero entry
            is_nonzero = any((a, b, c) == nz for nz in nonzero)
            if not is_nonzero:
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
                residual += val ** 2
        
        if residual < 1e-10:
            # Check if entries are integer/half-integer
            if _check_half_integer(U) and _check_half_integer(V) and _check_half_integer(W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try a random perturbation
        idx_u = rng.integers(0, num_a)
        idx_r_u = rng.integers(0, target_rank)
        delta = rng.choice([-1, -0.5, 0.5, 1])
        U[idx_u, idx_r_u] += delta
        
        # Occasionally try V or W
        choice = rng.integers(0, 3)
        if choice == 1:
            idx_v = rng.integers(0, num_b)
            idx_r_v = rng.integers(0, target_rank)
            delta = rng.choice([-1, -0.5, 0.5, 1])
            V[idx_v, idx_r_v] += delta
        elif choice == 2:
            idx_w = rng.integers(0, num_c)
            idx_r_w = rng.integers(0, target_rank)
            delta = rng.choice([-1, -0.5, 0.5, 1])
            W[idx_w, idx_r_w] += delta
    
    return None


def _check_half_integer(arr):
    """Check if all entries are integer or half-integer."""
    for row in arr:
        for val in row:
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
    return True
