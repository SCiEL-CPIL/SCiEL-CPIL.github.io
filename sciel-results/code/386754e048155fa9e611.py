import numpy as np
import scipy.optimize as opt
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    # Choose K. 5000 is a good balance between solution quality and solver time/memory.
    K = 5000
    
    # Variables are x_2, ..., x_K
    n_vars = K - 1
    
    # Objective: minimize sum c_i x_i where c_i = log(i)/i
    # We want to maximize -sum c_i x_i, so minimize sum c_i x_i
    indices = np.arange(2, K + 1)
    c = np.log(indices) / indices
    
    # Constraints:
    # 1. M(j) <= 1.0001 for j in 1..10K
    # M(j) = sum_{i=2}^K x_i (floor(j/i) - j/i) + x_1 * j
    # x_1 = - sum_{i=2}^K x_i / i
    # M(j) = sum_{i=2}^K x_i (floor(j/i) - j/i - j/i) ... wait.
    # M(j) = x_1 * floor(j/1) + sum_{i=2}^K x_i floor(j/i)
    # x_1 = - sum_{i=2}^K x_i / i
    # M(j) = -j * sum_{i=2}^K x_i / i + sum_{i=2}^K x_i floor(j/i)
    # M(j) = sum_{i=2}^K x_i (floor(j/i) - j/i)
    
    # We need to build the matrix A where A[j-1, i-2] = floor(j/i) - j/i
    # j goes from 1 to 10*K
    # i goes from 2 to K
    
    max_j = 10 * K
    A_rows = []
    b_rows = []
    
    # Precompute for efficiency
    # We can compute row by row
    # j is the row index
    # i is the column index (2..K)
    
    # To avoid building a huge dense matrix if K is large, we can use sparse.
    # But for K=5000, dense is 10000 x 4999 ~ 50M entries. ~400MB.
    # Let's try sparse to be safe.
    
    row_ind = []
    col_ind = []
    data = []
    
    for j in range(1, max_j + 1):
        for i in range(2, K + 1):
            val = np.floor(j / i) - j / i
            if abs(val) > 1e-12: # It's rarely exactly 0 except when i divides j
                row_ind.append(j - 1)
                col_ind.append(i - 2)
                data.append(val)
        
        # Check time
        if (j % 1000 == 0) and (time.time() - start_time > 500):
            # If we are running out of time, break and use what we have?
            # No, we need the full system.
            # If K=5000, this loop takes 5000 * 5000 = 25M iterations.
            # In Python, this might be slow. 25M iterations ~ 1-2 seconds?
            # Let's vectorize row construction.
            pass

    # Let's vectorize the row construction to be faster.
    row_ind = []
    col_ind = []
    data = []
    
    for j in range(1, max_j + 1):
        # i from 2 to K
        i_vals = np.arange(2, K + 1)
        vals = np.floor(j / i_vals) - j / i_vals
        # Find non-zeros
        mask = np.abs(vals) > 1e-12
        cols = np.where(mask)[0]
        if len(cols) > 0:
            row_ind.extend([j-1] * len(cols))
            col_ind.extend(cols)
            data.extend(vals[mask])
            
        if (j % 1000 == 0) and (time.time() - start_time > 500):
             # Fallback to smaller K if construction is too slow?
             # For now, assume it's fast enough.
             pass

    A = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(max_j, n_vars))
    b = np.full(max_j, 1.0001)
    
    # 2. Box constraints: -10 <= x_i <= 10
    # 3. Sum constraint: -10 <= sum x_i / i <= 10
    # We can add these as linear constraints or use bounds.
    # Bounds handle box constraints.
    bounds = [(-10, 10)] * n_vars
    
    # Add sum constraint to A and b
    # sum x_i / i <= 10
    # -sum x_i / i <= 10  => sum x_i / i >= -10
    sum_coeff = 1.0 / indices
    # Append to A
    # A is sparse, so we can add rows
    A_new = scipy.sparse.vstack([A, scipy.sparse.csr_matrix(sum_coeff)], format='csr')
    b_new = np.concatenate([b, [10.0]])
    
    # -sum x_i / i <= 10
    A_new = scipy.sparse.vstack([A_new, scipy.sparse.csr_matrix(-sum_coeff)], format='csr')
    b_new = np.concatenate([b_new, [10.0]])
    
    # Solve LP
    # Minimize c^T x
    res = opt.linprog(c, A_ub=A_new, b_ub=b_new, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    x = res.x
    # x corresponds to x_2, ..., x_K
    x_1 = -np.sum(x * (1.0 / indices))
    
    # Construct full domain and values
    full_indices = np.arange(1, K + 1)
    full_values = np.empty(K)
    full_values[0] = x_1
    full_values[1:] = x
    
    # Clip to [-10, 10] as per task description
    full_values = np.clip(full_values, -10, 10)
    
    # Recalculate x_1 adjustment?
    # The task says: "values are clipped... Then f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0"
    # Our LP ensured x_1 was within [-10, 10] and sum was 0.
    # Clipping might change the sum if any value was clipped.
    # But our bounds were [-10, 10], so clipping is a no-op for the LP solution.
    # So sum is still 0.
    # S = sum full_values / full_indices
    # S should be ~0.
    # full_values[0] = full_values[0] - S
    
    # Return
    domain = full_indices.astype(np.int64)
    values = full_values.astype(np.float64)
    
    return domain, values
