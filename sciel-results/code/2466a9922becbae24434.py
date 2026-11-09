import numpy as np
import random
import time

def construct():
    base = 51975
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    n_classes = len(divisors)
    
    # Sort divisors for deterministic processing, maybe larger first helps?
    # Let's stick to the order generated or sorted.
    
    def evaluate(residues):
        """Calculate uncovered count for a given set of residues."""
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(divisors):
            r = residues[i]
            counts[r::m] += 1
        return int(np.sum(counts == 0))

    def local_search(initial_residues, time_limit=10):
        """
        Perform local search to minimize uncovered count.
        Returns best residues and score found.
        """
        residues = initial_residues.copy()
        
        # Initialize counts
        counts = np.zeros(base, dtype=np.int32)
        for i, m in enumerate(divisors):
            r = residues[i]
            counts[r::m] += 1
        
        unc = int(np.sum(counts == 0))
        
        start_time = time.time()
        improved = True
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Shuffle order to avoid bias in deterministic loops?
            # Or just iterate. Let's iterate fixed for speed, restarts handle diversity.
            for i in range(n_classes):
                m = divisors[i]
                r_old = residues[i]
                
                # 1. Remove old residue
                counts[r_old::m] -= 1
                # Count how many became uncovered (count dropped to 0)
                new_uncovered = int(np.sum(counts[r_old::m] == 0))
                unc_temp = unc + new_uncovered
                
                best_r = r_old
                best_unc = unc_temp
                
                # 2. Try all new residues
                # Optimization: We can vectorize the check for all residues?
                # For a fixed m, we want to find r in [0, m) that minimizes
                # unc_temp - sum(counts[r::m] == 0)
                # This is equivalent to maximizing sum(counts[r::m] == 0)
                
                # We can compute the "gain" for each residue r.
                # gain[r] = number of indices j such that j % m == r and counts[j] == 0
                # We can compute this by iterating r? No, that's O(m * N/m) = O(N).
                # Since N is small (51975), O(N) per modulus is fine.
                # Total per pass: sum(N) over all moduli? No.
                # For each modulus i, we do O(N) work.
                # Total work per pass: n_classes * N.
                # 47 * 51975 ~ 2.4M operations. This is fast in Python/NumPy.
                
                # Let's try to vectorize the gain calculation for all r for this m.
                # We have `counts` (length N).
                # We want `gains` (length m).
                # gains[r] = sum(counts[r::m] == 0)
                
                # We can create a boolean array `is_zero = counts == 0`
                is_zero = (counts == 0).astype(np.int32)
                
                # For each r, sum is_zero[r::m]
                # This is a reduction.
                # We can reshape?
                # If m divides N, we can reshape to (m, N//m) and sum along axis 1.
                # 51975 is divisible by all its divisors.
                # So N // m is an integer.
                # Reshape is_zero to (m, N//m) is not directly possible if the stride is 1 in original but m in new?
                # Original array is 1D.
                # is_zero[r::m] takes elements at r, r+m, r+2m...
                # This is exactly the columns if we reshape to (m, N//m) with order='F' (Fortran/column-major)?
                # Let's check.
                # A = [a0, a1, ..., a_{N-1}]
                # Reshape to (m, N//m) with order='F':
                # Col 0: a0, a1, ... a_{m-1}
                # Col 1: a_m, a_{m+1}, ... a_{2m-1}
                # This groups by index modulo m?
                # Index k in flat array goes to (k % m, k // m) in Fortran order?
                # No, Fortran order fills column by column.
                # Element (i, j) in matrix is at flat index i + j*m.
                # So flat index k corresponds to i = k % m, j = k // m.
                # So the matrix M[i, j] = A[i + j*m].
                # We want sum over j for fixed i.
                # Sum of M[i, :] is sum of A[i::m].
                # So yes, reshape to (m, N//m) with order='F' and sum along axis 1 gives the gains.
                
                try:
                    # N is divisible by m, so this works
                    is_zero_reshaped = is_zero.reshape((m, base // m), order='F')
                    gains = np.sum(is_zero_reshaped, axis=1)
                except ValueError:
                    # Fallback if somehow not divisible (shouldn't happen)
                    gains = np.array([np.sum(is_zero[r::m]) for r in range(m)])
                
                # Find best r
                max_gain = np.max(gains)
                best_r = int(np.argmax(gains))
                best_unc = unc_temp - max_gain
                
                # 3. Apply best
                if best_r != r_old:
                    counts[best_r::m] += 1
                    residues[i] = best_r
                    unc = best_unc
                    improved = True
                else:
                    # Restore old
                    counts[best_r::m] += 1
                    # unc should be unchanged if best_r == r_old and logic is consistent
                    # Let's verify:
                    # unc_temp = unc + new_uncovered
                    # best_unc = unc_temp - gains[r_old]
                    # gains[r_old] = sum(counts[r_old::m] == 0) AFTER decrement.
                    # new_uncovered was sum(counts[r_old::m] == 0) AFTER decrement.
                    # So gains[r_old] == new_uncovered.
                    # best_unc = (unc + new_uncovered) - new_uncovered = unc.
                    # Correct.
                    pass
        
        return residues, unc

    best_unc = base
    best_res = None
    
    # Multiple restarts
    # 1. Greedy init (all 0s)
    # 2. Random inits
    num_restarts = 5
    time_per_restart = 15 # 5 * 15 = 75s, leaving buffer
    
    for k in range(num_restarts):
        if k == 0:
            init_res = [0] * n_classes
        else:
            init_res = [random.randint(0, m-1) for m in divisors]
        
        res, unc = local_search(init_res, time_limit=time_per_restart)
        
        if unc < best_unc:
            best_unc = unc
            best_res = res
    
    if best_res is None:
        best_res = [0] * n_classes
        best_unc = evaluate(best_res)

    classes = [[divisors[i], best_res[i]] for i in range(n_classes)]
    return classes
