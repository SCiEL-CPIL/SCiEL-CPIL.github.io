# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def _compute_mobius(n):
    """Compute the Mobius function mu(k) for k=1..n using a sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            sq = i * i
            if sq <= n:
                for j in range(sq, n + 1, sq):
                    mu[j] = 0
    return mu

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # 1. Compute Mobius function up to a reasonable limit.
    # K=200000 is the max key allowed.
    # The constraint M(x) <= 1.0001 is very tight.
    # M(x) = sum_{k|? no} sum_k f(k) floor(x/k).
    # If f = mu, M(x) = 1 for all x >= 1.
    # We want to maximize -sum f(k) log(k) / k.
    # Since mu(k) is -1 for squarefree with odd number of primes, 1 for even, 0 else.
    # The term -mu(k) log(k) / k is positive when mu(k) = -1, negative when mu(k) = 1.
    # To maximize the sum, we want to include as many k with mu(k) = -1 as possible,
    # and as few with mu(k) = 1 as possible.
    # However, we must maintain the constraint M(x) <= 1.0001.
    # If we drop some terms, M(x) will deviate from 1.
    # Specifically, if we drop a term f(k)mu(k), the new M(x) becomes 1 - sum_{dropped} mu(k) floor(x/k).
    # We need 1 - sum_{dropped} mu(k) floor(x/k) <= 1.0001 => sum_{dropped} mu(k) floor(x/k) >= -0.0001.
    # This must hold for all x in [1, 10K_max].
    
    # Let's try to include all k with mu(k) = -1 up to some limit N, and a subset of k with mu(k) = 1.
    # Actually, the LP approach suggested in the knowledge is robust.
    # But solving an LP with 200,000 variables and constraints might be slow or memory intensive.
    # Constraints: M(x) <= 1.0001 for x=1..10K. That's 2,000,000 constraints.
    # Variables: f(k) for k in domain.
    
    # Let's estimate the gain from extending the domain.
    # a = -sum_{k=1}^N mu(k) log(k) / k.
    # As N -> infinity, a -> 1.
    # The current score is 1.0 (seed). The LP found 1.009.
    # Let's try to compute the sum for larger N to see the potential.
    
    # We will use a greedy or heuristic approach to select the domain and values,
    # or a small LP if feasible.
    
    # Given the 600s limit, a full LP over 200k variables is risky.
    # Let's try a domain of size ~5000 or 10000 first.
    
    # Strategy:
    # 1. Compute mu up to N_max = 10000.
    # 2. Define variables f(k) for k=1..N_max.
    # 3. Objective: maximize -sum f(k) log(k)/k.
    #    Minimize sum f(k) log(k)/k.
    # 4. Constraints:
    #    M(x) = sum_{k=1}^{N_max} f(k) floor(x/k) <= 1.0001 for x=1..10*N_max.
    #    Also, we should probably bound f(k) to avoid numerical issues, e.g., |f(k)| <= 10.
    #    And f(1) is adjusted by the evaluator, but in the LP we can treat f(1) as a free variable
    #    subject to the sum f(k)/k = 0 constraint? No, the evaluator clips and adjusts f(1).
    #    The prompt says: "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0."
    #    This means the evaluator enforces sum f(k)/k = 0.
    #    So in our LP, we should include the constraint sum f(k)/k = 0?
    #    Wait, the evaluator *modifies* f(1) to make this true.
    #    If we assume f(1) is determined by the other f(k), we can eliminate f(1).
    #    f(1) = - sum_{k>1} f(k)/k.
    #    Then M(x) = f(1) * x + sum_{k=2}^{N} f(k) floor(x/k) = -x * sum_{k=2}^N f(k)/k + sum_{k=2}^N f(k) floor(x/k).
    #    This is a linear function of f(k) for k>1.
    #    Objective: -f(1) log(1)/1 - sum_{k=2}^N f(k) log(k)/k = - sum_{k=2}^N f(k) log(k)/k (since log(1)=0).
    #    So we only need to optimize over f(k) for k>1.
    
    N_max = 5000
    mu = _compute_mobius(N_max)
    
    # Variables: f(2), ..., f(N_max). Count = N_max - 1.
    n_vars = N_max - 1
    
    # Objective coefficients: c_j = log(j+1) / (j+1) for j=0..n_vars-1 (corresponding to k=2..N_max)
    ks = np.arange(2, N_max + 1)
    c = np.log(ks) / ks
    
    # Constraints:
    # 1. M(x) <= 1.0001 for x=1..10*N_max.
    #    M(x) = -x * sum_{k=2}^N f(k)/k + sum_{k=2}^N f(k) floor(x/k)
    #    M(x) = sum_{k=2}^N f(k) * (floor(x/k) - x/k)
    #    Let A[x-1, k-2] = floor(x/k) - x/k.
    #    We need A @ f <= 1.0001.
    
    #    Note: floor(x/k) - x/k is negative or zero.
    #    If f(k) is negative, the term is positive.
    
    #    Number of constraints: 10 * N_max = 50,000.
    #    This is a large sparse matrix.
    
    #    Let's check if we can construct this efficiently.
    #    A is 50000 x 4999.
    
    #    Also, we might want to bound f(k).
    #    The evaluator clips to [-10, 10].
    #    So we can add constraints -10 <= f(k) <= 10.
    
    #    Let's try to solve this LP.
    #    If it's too slow, we fall back to a heuristic.
    
    if time.time() > deadline - 300:
        # Not enough time for LP, use heuristic.
        # Heuristic: use mu(k) for k up to some small N, and 0 otherwise.
        # This guarantees feasibility (M(x)=1) if we include all k up to N?
        # No, if we truncate mu, M(x) is not 1.
        # But for small N, the deviation might be small.
        # Actually, the seed program uses a small domain and specific values.
        # Let's just return the seed program's values if we run out of time.
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct A matrix
    # x ranges from 1 to 10*N_max
    xs = np.arange(1, 10 * N_max + 1)
    # k ranges from 2 to N_max
    # A[i, j] = floor(xs[i] / ks[j]) - xs[i] / ks[j]
    
    # This might be memory heavy: 50000 * 5000 * 8 bytes = 2 GB. Too much.
    # We need to use sparse matrices or block processing.
    # scipy.sparse.csr_matrix can help, but constructing it efficiently is key.
    
    # Let's try a smaller N_max first, e.g., 1000.
    N_max = 1000
    n_vars = N_max - 1
    ks = np.arange(2, N_max + 1)
    c = np.log(ks) / ks
    xs = np.arange(1, 10 * N_max + 1)
    
    # Construct A in CSR format
    # We can build it row by row or use block operations.
    # Let's try to build it using numpy broadcasting in chunks to avoid memory blowup.
    
    rows = []
    cols = []
    data = []
    
    chunk_size = 1000
    for i_start in range(0, len(xs), chunk_size):
        i_end = min(i_start + chunk_size, len(xs))
        x_chunk = xs[i_start:i_end]
        # x_chunk shape (chunk,), ks shape (n_vars,)
        # floor(x_chunk[:, None] / ks[None, :])
        # This creates a (chunk, n_vars) array.
        
        # To save memory, we can process in smaller sub-chunks if needed.
        # 1000 * 1000 = 1e6 floats = 8 MB. This is fine.
        
        floor_part = np.floor(x_chunk[:, None] / ks[None, :])
        real_part = x_chunk[:, None] / ks[None, :]
        A_chunk = floor_part - real_part
        
        # Only keep non-zeros? A_chunk is dense mostly.
        # But many entries might be small? No, floor(x/k) - x/k is in [-1, 0).
        
        # Append to sparse lists
        # This is inefficient. Let's try to build the full matrix if memory allows.
        # 10000 x 999 = 10M entries = 80 MB. This is OK.
        
        if i_start == 0:
            A_full = np.zeros((len(xs), n_vars), dtype=np.float64)
        
        A_full[i_start:i_end, :] = A_chunk
    
    # A_full is 10000 x 999.
    
    # Convert to sparse
    A_sparse = csr_matrix(A_full)
    
    # Constraints: A_sparse @ f <= 1.0001
    b = np.full(len(xs), 1.0001)
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10, 10)] * n_vars
    
    # Solve LP
    # Minimize c @ f
    # Subject to A @ f <= b
    # Bounds on f
    
    # linprog might be slow for 1000 vars and 10000 constraints.
    # Let's check time.
    
    try:
        res = linprog(c, A_ub=A_sparse, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            f_vals = res.x
            # f_vals corresponds to k=2..N_max
            # f(1) is determined by sum f(k)/k = 0 => f(1) = -sum_{k=2}^N f(k)/k
            f1 = -np.sum(f_vals / ks)
            
            # Construct domain and values
            # We should probably only include k where f(k) is not close to 0 to save space?
            # The problem says "finite partial function".
            # We can include all k=1..N_max.
            
            domain = np.arange(1, N_max + 1, dtype=np.int64)
            values = np.zeros(N_max + 1, dtype=np.float64)
            values[0] = f1
            values[1:] = f_vals
            
            # Clip values to [-10, 10] just in case (LP bounds should handle this)
            values = np.clip(values, -10, 10)
            
            # Return
            return domain, values
        else:
            print("LP failed:", res.message)
    except Exception as e:
        print("LP error:", e)
    
    # Fallback: Heuristic based on Mobius
    # Use mu(k) for k up to N_fallback
    N_fallback = 100
    mu = _compute_mobius(N_fallback)
    domain = np.arange(1, N_fallback + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Adjust f(1) to satisfy sum f(k)/k = 0
    S = np.sum(values / domain)
    values[0] = -S
    
    return domain, values

# EVOLVE-BLOCK-END
