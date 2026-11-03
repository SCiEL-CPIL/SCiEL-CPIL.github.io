"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    if n == 0:
        return []
    
    # Precompute symmetric part for faster delta calculation
    # R[i, k] = A[i, k] + A[k, i]
    R = a + a.T
    
    # 1. Initialization: Rank Matching
    row_sums_a = a.sum(axis=1)
    row_sums_b = b.sum(axis=1)
    
    # Sort facilities by descending flow
    fac_order = np.argsort(-row_sums_a, kind='stable')
    # Sort locations by ascending distance
    loc_order = np.argsort(row_sums_b, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Calculate initial cost
    # Cost = sum(a[i, j] * b[p[i], p[j]])
    # Using matrix multiplication: sum(a * b[np.ix_(p, p)])
    current_cost = np.sum(a * b[np.ix_(p, p)])
    
    best_p = p.copy()
    best_cost = current_cost
    
    # Helper function to calculate delta cost for swapping facilities i and j
    def delta_swap(i, j, p, R, a, b):
        x = p[i]
        y = p[j]
        
        # Vectors
        Ri = R[i, :]
        Rj = R[j, :]
        By = b[y, :]
        Bx = b[x, :]
        
        # Sum over all k
        # dot(Ri - Rj, By - Bx)
        diff_R = Ri - Rj
        diff_B = By - Bx
        total_sum = np.dot(diff_R, diff_B)
        
        # Subtract k=i term
        # Coeff_i = R[i, i] - R[j, i] = 2*a[i, i] - (a[j, i] + a[i, j])
        coeff_i = 2 * a[i, i] - a[j, i] - a[i, j]
        diff_B_i = b[y, i] - b[x, i]
        total_sum -= coeff_i * diff_B_i
        
        # Subtract k=j term
        # Coeff_j = R[i, j] - R[j, j] = (a[i, j] + a[j, i]) - 2*a[j, j]
        coeff_j = a[i, j] + a[j, i] - 2 * a[j, j]
        diff_B_j = b[y, j] - b[x, j]
        total_sum -= coeff_j * diff_B_j
        
        # Add diagonal and cross corrections
        # Aii * (Byy - Bxx)
        diag_i = a[i, i] * (b[y, y] - b[x, x])
        # Ajj * (Bxx - Byy)
        diag_j = a[j, j] * (b[x, x] - b[y, y])
        # Aij * (Byx - Bxy)
        cross_ij = a[i, j] * (b[y, x] - b[x, y])
        # Aji * (Bxy - Byx)
        cross_ji = a[j, i] * (b[x, y] - b[y, x])
        
        return total_sum + diag_i + diag_j + cross_ij + cross_ji

    # 2. Local Search Pre-optimization
    start_time = time.time()
    max_pre_time = 5.0
    np.random.seed(42) # Fixed seed for reproducibility
    
    improved = True
    while improved and (time.time() - start_time) < max_pre_time:
        improved = False
        # Perform a batch of swaps
        for _ in range(2000):
            i, j = np.random.choice(n, 2, replace=False)
            dc = delta_swap(i, j, p, R, a, b)
            if dc < 0:
                p[i], p[j] = p[j], p[i]
                current_cost += dc
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()

    # 3. Simulated Annealing
    T = 1000.0
    cooling = 0.9995
    start_time = time.time()
    deadline = 115.0 # Safety margin
    
    no_improve_count = 0
    threshold = 20000
    
    # We can restart SA if we get stuck for a long time
    # For now, single long run
    
    step = 0
    while (time.time() - start_time) < deadline:
        i, j = np.random.choice(n, 2, replace=False)
        dc = delta_swap(i, j, p, R, a, b)
        
        if dc < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += dc
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Accept with probability exp(-dc/T)
            # Optimization: if dc > T * 20, exp is ~0, skip
            if dc < T * 20:
                if np.random.rand() < np.exp(-dc / T):
                    p[i], p[j] = p[j], p[i]
                    current_cost += dc
            no_improve_count += 1
        
        T *= cooling
        step += 1
        
        # Check time periodically
        if step % 1000 == 0:
            if (time.time() - start_time) >= deadline:
                break
        
        # Optional: Restart logic
        # If no improvement for a long time, we might be stuck.
        # We could reset T and randomize p slightly.
        # But for simplicity and stability, we just continue cooling.
        # If T becomes too small, we are essentially doing local search.
        if T < 0.1:
             # Restart with higher T?
             # Let's do a soft restart: randomize 10% of the permutation
             # and reset T
             if no_improve_count > 50000:
                 # Randomize
                 idx = np.random.choice(n, n//10, replace=False)
                 np.random.shuffle(idx)
                 # This is complex to do correctly with p. 
                 # Simpler: just reset T and keep going.
                 T = 500.0
                 no_improve_count = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
