import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    # We formulate the problem as finding a subset S of {0, ..., N-1}
    # that maximizes K / lambda, where K = |S| and lambda is the max multiplicity of sums.
    # This is equivalent to maximizing the ratio for uniform weights on S.
    
    # Variables:
    # x_i for i in 0..N-1 (binary)
    # y_{i,j} for i,j in 0..N-1 such that i+j < 2N (binary)
    # Actually, we only need y_{i,j} for pairs that appear in the convolution sums.
    # The sum for index k is sum_{i} x_i x_{k-i}.
    # The terms are x_i x_j where i+j=k.
    
    # To keep the model size manageable, we iterate lambda.
    best_score = 0.0
    best_h = None
    
    # Max possible lambda is N (if all x_i=1, sum at center is N).
    # But we expect small lambda.
    max_lambda = 10 
    
    # Pre-compute pairs for each k
    # pairs[k] = list of (i, j) such that i+j=k, 0<=i,j<N
    # Note: i and j are distinct in the product x_i x_j, but in the sum they are just terms.
    # If i == j, it's x_i^2 = x_i (since binary).
    # If i != j, it's x_i x_j.
    
    # We will use scipy.optimize.milp
    # Variables:
    # 0 .. N-1: x_i
    # N .. N*M-1: y_{i,j}
    
    # To simplify, we can just define y_{i,j} for all i,j in 0..N-1.
    # M = N*N = 2500. Total vars = 2550.
    # This is small enough for MILP.
    
    M = N * N
    total_vars = N + M
    
    # Indices for y_{i,j}
    # index = N + i * N + j
    
    # Objective: Maximize sum(x_i)
    # c = [-1 for x, 0 for y] (milp minimizes, so we minimize -sum)
    c = np.zeros(total_vars)
    c[:N] = -1.0
    
    # Constraints:
    # 1. x_i in {0,1}
    # 2. y_{i,j} in {0,1}
    # 3. y_{i,j} <= x_i
    # 4. y_{i,j} <= x_j
    # 5. y_{i,j} >= x_i + x_j - 1
    # 6. For each k, sum_{i+j=k} y_{i,j} <= lambda
    #    Note: if i==j, y_{i,i} is x_i.
    #    If i!=j, y_{i,j} and y_{j,i} are both in the sum?
    #    The convolution sum is sum_{i=0}^{N-1} h_i h_{k-i}.
    #    If we use binary x, term is x_i x_{k-i}.
    #    If i != k-i, the term x_i x_{k-i} appears once in the sum over i?
    #    Yes, for a fixed k, the index i runs 0..N-1.
    #    The term for i is x_i x_{k-i}.
    #    The term for k-i is x_{k-i} x_i.
    #    These are the same product.
    #    So the sum is sum_{i} x_i x_{k-i}.
    #    If we define y_{i,j} = x_i x_j, then the constraint is sum_{i: 0<=k-i<N} y_{i, k-i} <= lambda.
    #    This sums each product once for each i.
    #    So if i != j, the product appears twice in the sum (once as i, once as j=k-i).
    #    Wait.
    #    Let's check k=3, N=5.
    #    i=0: x0 x3
    #    i=1: x1 x2
    #    i=2: x2 x1
    #    i=3: x3 x0
    #    Sum = 2 x0 x3 + 2 x1 x2.
    #    So the constraint is sum_{i} y_{i, k-i} <= lambda.
    #    This correctly accounts for the multiplicity.
    
    # We iterate lambda
    for lam in range(1, max_lambda + 1):
        # Build constraints
        
        # List of linear constraints
        # A_ub x <= b_ub
        # A_eq x = b_eq
        
        # Inequality constraints:
        # y_{i,j} - x_i <= 0
        # y_{i,j} - x_j <= 0
        # -y_{i,j} + x_i + x_j >= 1  =>  y_{i,j} - x_i - x_j <= -1
        # sum_{i} y_{i, k-i} <= lam
        
        # We'll use LinearConstraint for milp
        
        # Number of constraints:
        # N*N for y<=x_i
        # N*N for y<=x_j
        # N*N for y>=x_i+x_j-1
        # 2N-1 for sum constraints
        
        n_constraints = 3 * M + (2 * N - 1)
        
        # We can build the sparse matrix, but for N=50, dense is fine (2550 vars, ~8000 constraints)
        # 2550 * 8000 = 20 million floats. ~160MB. Might be too much for memory/time.
        # Let's use sparse or build carefully.
        
        # Actually, we can just use `linprog` if we relax binaries? No, we need binaries.
        # `scipy.optimize.milp` uses HiGHS, which is efficient.
        
        # Let's try to construct the constraint matrix.
        # It might be faster to just use a simple heuristic if MILP is too slow.
        # But N=50 is small.
        
        # Alternative: Use `pulp`? Not in stdlib.
        # `scipy.optimize.milp` is the way.
        
        # To save time, we can break early if K/lam <= best_score.
        # Max possible K is N.
        # If N/lam <= best_score, break.
        if N / lam <= best_score:
            break
            
        # Construct A_ub, b_ub
        # We'll use a list of (row, col, val) for sparse, or just dense if memory allows.
        # 2550 vars.
        # Let's try dense for simplicity, but check memory.
        # 8000 rows * 2550 cols * 8 bytes = 160 MB.
        # Python overhead might be high.
        
        # Let's use scipy.sparse
        from scipy.sparse import lil_matrix, csr_matrix
        
        A_ub = lil_matrix((n_constraints, total_vars))
        b_ub = np.zeros(n_constraints)
        
        row = 0
        
        # y_{i,j} - x_i <= 0
        for i in range(N):
            for j in range(N):
                idx_y = N + i * N + j
                A_ub[row, idx_y] = 1
                A_ub[row, i] = -1
                b_ub[row] = 0
                row += 1
                
        # y_{i,j} - x_j <= 0
        for i in range(N):
            for j in range(N):
                idx_y = N + i * N + j
                A_ub[row, idx_y] = 1
                A_ub[row, j] = -1
                b_ub[row] = 0
                row += 1
                
        # y_{i,j} - x_i - x_j <= -1
        for i in range(N):
            for j in range(N):
                idx_y = N + i * N + j
                A_ub[row, idx_y] = 1
                A_ub[row, i] = -1
                A_ub[row, j] = -1
                b_ub[row] = -1
                row += 1
                
        # Sum constraints
        for k in range(2 * N - 1):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    idx_y = N + i * N + j
                    A_ub[row, idx_y] = 1
            b_ub[row] = lam
            row += 1
            
        # Integrities
        integrality = np.ones(total_vars)
        
        # Bounds
        bounds = Bounds(lb=0, ub=1)
        
        # Constraints
        constraints = LinearConstraint(A_ub, ub=b_ub)
        
        # Solve
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0: # Optimal
                x_opt = res.x[:N]
                K = int(np.sum(x_opt))
                if K > 0:
                    score = K / lam
                    if score > best_score:
                        best_score = score
                        # Construct h
                        h = np.zeros(N)
                        h[x_opt > 0.5] = 1.0 / K
                        best_h = h
        except Exception:
            pass

    if best_h is None:
        # Fallback
        best_h = np.ones(N) / N
        
    return list(best_h)
