# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # Keep 30s margin
    
    # Strategy: Use LP with larger K and sparse constraint matrix
    # The constraint M(x) <= 1 becomes sum_{k>1} f(k)(floor(x/k) - x/k) <= 1
    # since f(1) = -sum_{k>1} f(k)/k ensures sum f(k)/k = 0
    
    # Choose K to balance LP size and quality
    # With K=5000, we have ~3000 square-free numbers in [2,5000]
    # and 50000 constraints - this is large but HiGHS can handle it
    # Let's try K=3000 first for safety
    
    K = 3000
    
    # Compute Möbius function up to K
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0.0
    
    # Domain for variables: square-free numbers in [2, K]
    sq_free = np.array([k for k in range(2, K + 1) if mu[k] != 0], dtype=np.int64)
    m = len(sq_free)
    
    if m == 0:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    # Check time
    if time.time() > deadline - 60:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    # Objective: minimize sum g(k) log(k)/k (to maximize -sum g(k) log(k)/k)
    c = np.array([np.log(k) / k for k in sq_free], dtype=np.float64)
    
    # Build sparse constraint matrix A_ub
    # A_ub[i, j] = floor(x_i / k_j) - x_i / k_j for x_i in [1, 10K], k_j in sq_free
    # This is sparse because floor(x/k) - x/k is non-zero only when k doesn't divide x
    
    n_x = 10 * K
    x_vals = np.arange(1, n_x + 1, dtype=np.float64)
    
    # Build COO matrix
    rows = []
    cols = []
    data = []
    
    # Process in batches to manage memory
    batch_size = 1000
    for start in range(0, n_x, batch_size):
        end = min(start + batch_size, n_x)
        x_batch = x_vals[start:end]
        n_batch = end - start
        
        # For each x in batch, compute floor(x/k) - x/k for all k in sq_free
        # Only add non-zero entries
        for i in range(n_batch):
            x = x_batch[i]
            # Compute for all k
            ratios = x / sq_free
            floors = np.floor(ratios)
            diff = floors - ratios
            # Find non-zero entries
            nz = np.nonzero(np.abs(diff) > 1e-15)[0]
            if len(nz) > 0:
                rows.extend([start + i] * len(nz))
                cols.extend(nz.tolist())
                data.extend(diff[nz].tolist())
        
        if time.time() > deadline - 120:
            break
    
    if not rows:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    A_ub = coo_matrix((data, (rows, cols)), shape=(n_x, m)).tocsr()
    b_ub = np.ones(n_x, dtype=np.float64)
    
    # Bounds
    bounds = [(-10.0, 10.0)] * m
    
    # Check time
    if time.time() > deadline - 90:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    if time.time() > deadline - 30:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
    
    if result.success:
        g_vals = result.x
        
        # Compute f(1)
        f1 = -np.sum(g_vals / sq_free)
        
        # Construct final domain and values
        domain = np.concatenate(([1], sq_free)).astype(np.int64)
        values = np.concatenate(([f1], g_vals)).astype(np.float64)
        
        # Verification: simulate evaluator
        clipped_values = np.clip(values, -10.0, 10.0)
        S = np.sum(clipped_values / domain)
        clipped_values[0] = clipped_values[0] - S
        
        # Check M(x) <= 1.0001
        K_final = int(domain.max())
        # Check a subset of x values for speed
        x_check = np.arange(1, 10 * K_final + 1, max(1, K_final // 50))
        
        M_max = -np.inf
        for x in x_check:
            floors = np.floor(x / domain)
            Mx = np.dot(clipped_values, floors)
            if Mx > M_max:
                M_max = Mx
        
        if M_max <= 1.0001:
            return domain, values
    
    # Fallback
    domain = np.array([1, 2], dtype=np.int64)
    values = np.array([1.0, -2.0], dtype=np.float64)
    return domain, values

# EVOLVE-BLOCK-END
