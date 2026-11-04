import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, leave 5s margin for overhead and output
    
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    # Total number of equations: n * m * p = 84
    num_eqs = n * m * p
    
    # Lower bound for rank
    min_rank = max(dim_U, dim_V, dim_W) # 28
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Pre-compute indices for equations
    # Equation (i, j, k) corresponds to index eq_idx
    # Variables are W[k*n+i, c]
    # We can vectorize the construction of A for given U, V
    
    # Strategy:
    # 1. Try ranks from min_rank upwards.
    # 2. For each rank, try random integer initializations for U and V.
    # 3. Solve for W using least squares.
    # 4. Check if W is close to half-integers.
    # 5. If so, verify with exact Fraction arithmetic.
    
    # Optimization:
    # Instead of building A in a loop, we can use broadcasting if we reshape.
    # But U and V change every iteration.
    # A[i,j,k, row_W, c] = U[i*m+j, c] * V[j*p+k, c]
    # This is a bit complex to vectorize directly without large intermediate tensors.
    # dim_U=12, dim_V=28, dim_W=21.
    # r is small (28-83).
    # num_eqs = 84.
    # A size: 84 x (21*r).
    # For r=50, A is 84 x 1050.
    # Building A in Python loop: 84 * 50 = 4200 ops. Very fast.
    # Solving lstsq: 84x1050. Fast.
    
    # The bottleneck is the number of tries.
    # We can increase the number of tries per rank if we are fast.
    
    # Let's implement a helper to build A and b
    def build_system(U, V, r):
        A = np.zeros((num_eqs, dim_W * r))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_W = k * n + i
                    # Vectorized update for this equation
                    # A[eq_idx, row_W*r : row_W*r+r] = U[i*m+j, :] * V[j*p+k, :]
                    A[eq_idx, row_W*r : (row_W+1)*r] = U[i*m+j, :] * V[j*p+k, :]
                    eq_idx += 1
        b = np.ones(num_eqs)
        return A, b

    # Search for better rank
    # We start from min_rank.
    # We limit the total number of attempts or time per rank.
    
    # Known lower bounds:
    # For 3x4x7, the rank is likely > 28.
    # We'll try to find any rank < 84.
    
    # To improve search efficiency, we can use random integers from a larger set?
    # Or stick to {-1, 0, 1} which is good for sparsity and exactness.
    # Maybe {-1, 1} is better? Or {-2, -1, 0, 1, 2}?
    # Let's stick to {-1, 0, 1} first, but maybe try {-1, 1} as well.
    
    # Let's iterate ranks.
    # We will allocate time budget to each rank.
    # Lower ranks are harder, so maybe give them more tries?
    # Or just do a fixed number of tries per rank.
    
    # Let's try to be smarter.
    # We can stop early if we find a solution.
    
    # To maximize chance of finding a solution, we should try ranks that are "plausible".
    # But we don't know the minimal rank.
    # Let's just sweep from min_rank to 83.
    
    # We need to be careful with time.
    # Let's say we have 110 seconds.
    # If we do 10 tries per rank, and each try takes 1ms (optimistic, maybe 10ms),
    # 10 * 10ms = 100ms per rank.
    # 55 ranks * 100ms = 5.5s.
    # So we can afford many more tries.
    
    # Let's increase tries.
    # If r < 40, try 50 times.
    # If 40 <= r < 60, try 20 times.
    # If r >= 60, try 5 times.
    
    # Actually, higher ranks are easier to solve (more variables), so they are more likely to have a solution if one exists.
    # But we want the MINIMAL rank. So we must check lower ranks first.
    
    # Let's use a fixed number of tries, say 20, for all ranks, and see how long it takes.
    # If it's too slow, we reduce.
    
    for r in range(min_rank, 84):
        if time.time() > deadline - 5:
            break
        
        # Determine number of tries
        # We can adapt based on time remaining
        # But let's start with a heuristic
        num_tries = 30
        
        for _ in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            # Random initialization
            # Using {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(dim_U, r))
            V = np.random.choice([-1, 0, 1], size=(dim_V, r))
            
            # Build system
            A, b = build_system(U, V, r)
            
            # Solve
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-8:
                continue
                
            # Reshape
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check half-integer
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-5:
                # Potential candidate
                W_int = W_rounded / 2.0
                
                # Verify with Fraction
                # Convert to Fraction for exactness
                # U, V are integers
                # W_int are half-integers
                # We can check sum U[i,j,c] * V[j,k,c] * W_int[k,i,c] == 1
                # Multiply by 2 to avoid fractions?
                # 2 * sum ... = 2
                # Or just use Fraction.
                
                is_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = Fraction(0)
                            for c in range(r):
                                u_val = U[i*m+j, c]
                                v_val = V[j*p+k, c]
                                w_val = Fraction(W_int[k*n+i, c], 1) # W_int is float, but represents half-int
                                # Better: W_int is rounded/2. So W_int * 2 is integer.
                                # Let W2 = W_rounded. Then W = W2/2.
                                # Term: U * V * W2/2
                                # Sum: 1/2 * sum(U * V * W2)
                                # We want sum(U * V * W2/2) == 1 => sum(U * V * W2) == 2
                                term = u_val * v_val * W_rounded[k*n+i, c]
                                total += term
                            if total != 2:
                                is_valid = False
                                break
                    if not is_valid:
                        break
                        # break outer loop? No, break inner
                    if not is_valid:
                        break
                # Need to break all loops
                # Python doesn't have multi-break.
                # Let's refactor verification to a function.
                
                def verify(U, V, W2, r, n, m, p):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0
                                for c in range(r):
                                    total += U[i*m+j, c] * V[j*p+k, c] * W2[k*n+i, c]
                                if total != 2:
                                    return False
                    return True

                if verify(U, V, W_rounded, r, n, m, p):
                    # Valid!
                    # Convert to lists
                    # U, V are integers. W is half-integers.
                    # The task says "integer/half-integer entries".
                    # Returning floats is usually fine if they are exact representations.
                    # Or we can return Fractions? No, standard library types are safer.
                    # Floats for .0 and .5 are exact.
                    result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_int.tolist()
                    }
                    return result

    # If no better rank found, return schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the interface is correct.
# The prompt says: Interface: construct() -> {'U', 'V', 'W'}
# My construct returns exactly that.
# EVOLVE-BLOCK-END
