"""Program that searches for a lower-rank tensor decomposition of matrix multiplication 3x5x7."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n=3, m=5, p=7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 5, 7
    rank = len(U[0]) if U else 0
    # Reconstruct the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _validate_decomposition_f2(U, V, W, n=3, m=5, p=7):
    """Validate decomposition over F_2 (mod 2 arithmetic)."""
    n, m, p = 3, 5, 7
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = (s + (U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])) % 2
                if s != 1 % 2:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for evaluation overhead
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook
    U_school = [[0] * base_rank for _ in range(n * m)]
    V_school = [[0] * base_rank for _ in range(m * p)]
    W_school = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = base_rank
    
    # Strategy: Use F_2 algebraic search to find lower-rank decompositions
    # The multiplication tensor over F_2 may have lower rank than over reals
    
    # Build the multiplication tensor over F_2
    # T[i,j,k] = 1 for all valid (i,j,k)
    # We want to find rank-r decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # Approach: Randomized greedy rank-1 peeling over F_2
    # 1. Start with full tensor
    # 2. At each step, find a rank-1 term that covers many remaining 1s
    # 3. Subtract it (XOR in F_2)
    # 4. Repeat until tensor is zero or budget exhausted
    
    # Represent tensor as a 3D array over F_2
    tensor = np.zeros((n, m, p), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i, j, k] = 1
    
    # Greedy rank-1 peeling over F_2
    U_factors = []
    V_factors = []
    W_factors = []
    
    max_peels = 80  # Try to find decomposition with rank <= 80
    current_tensor = tensor.copy()
    
    for peel_idx in range(max_peels):
        if time.time() > deadline:
            break
        
        # Find the best rank-1 term to peel
        # A rank-1 term is defined by vectors u (length n*m), v (length m*p), w (length p*n)
        # Contribution to T[i,j,k] = sum over all (i',j',k') where u[i',j']=1, v[j',k']=1, w[k',i']=1
        # Actually, for a single rank-1 term: U_c[i,j] * V_c[j,k] * W_c[k,i]
        # We want to maximize the number of 1s in current_tensor that are covered
        
        best_score = -1
        best_u = None
        best_v = None
        best_w = None
        
        # Try random rank-1 terms
        num_trials = 500
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Generate random sparse vectors
            # u: n*m entries, v: m*p entries, w: p*n entries
            # Use moderate sparsity
            u = np.zeros(n * m, dtype=np.int8)
            v = np.zeros(m * p, dtype=np.int8)
            w = np.zeros(p * n, dtype=np.int8)
            
            # Set random positions to 1
            u_positions = random.sample(range(n * m), random.randint(3, 8))
            for pos in u_positions:
                u[pos] = 1
            
            v_positions = random.sample(range(m * p), random.randint(3, 8))
            for pos in v_positions:
                v[pos] = 1
            
            w_positions = random.sample(range(p * n), random.randint(3, 8))
            for pos in w_positions:
                w[pos] = 1
            
            # Compute the rank-1 term's contribution to the tensor
            # T_contrib[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i] mod 2
            # Count how many 1s in current_tensor are matched
            score = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        contrib = (u[i * m + j] * v[j * p + k] * w[k * n + i]) % 2
                        if contrib == 1 and current_tensor[i, j, k] == 1:
                            score += 1
                        elif contrib == 1 and current_tensor[i, j, k] == 0:
                            # This would create a 1 where we don't want one - penalize
                            score -= 1
            
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None or best_score <= 0:
            break
        
        # Peel this rank-1 term
        U_factors.append(best_u.tolist())
        V_factors.append(best_v.tolist())
        W_factors.append(best_w.tolist())
        
        # Update current tensor (XOR in F_2)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib = (best_u[i * m + j] * best_v[j * p + k] * best_w[k * n + i]) % 2
                    if contrib == 1:
                        current_tensor[i, j, k] = (current_tensor[i, j, k] + 1) % 2
        
        # Check if tensor is now zero
        if np.all(current_tensor == 0):
            break
    
    # Check if we found a complete decomposition
    if len(U_factors) < base_rank and np.all(current_tensor == 0):
        # Convert to the required format
        rank = len(U_factors)
        U_new = [[0] * rank for _ in range(n * m)]
        V_new = [[0] * rank for _ in range(m * p)]
        W_new = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            for i in range(n * m):
                U_new[i][c] = U_factors[c][i]
            for j in range(m * p):
                V_new[j][c] = V_factors[c][j]
            for k in range(p * n):
                W_new[k][c] = W_factors[c][k]
        
        # Validate over reals (entries are 0 or 1, so F_2 decomposition is also valid over reals
        # if all entries are 0 or 1 and the sum works out)
        # Actually, F_2 decomposition may not be valid over reals because of carries
        # We need to check
        if _validate_decomposition(U_new, V_new, W_new, n, m, p):
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = rank
    
    # If F_2 approach didn't work, try a different strategy:
    # Use structured decompositions based on the prime dimensions
    
    # Try Kronecker product structure
    # 3x5x7: Can we exploit that 3, 5, 7 are primes?
    # Not directly helpful for rank reduction
    
    # Try random search with small integer entries for rank = base_rank - 1
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        test_rank = base_rank - 1
        # Randomly generate sparse factors with small integer entries
        U_test = [[0] * test_rank for _ in range(n * m)]
        V_test = [[0] * test_rank for _ in range(m * p)]
        W_test = [[0] * test_rank for _ in range(p * n)]
        
        # Fill randomly with -1, 0, 1
        for i in range(n * m):
            for c in range(test_rank):
                U_test[i][c] = random.choice([0, 0, 0, 1, -1])
        for j in range(m * p):
            for c in range(test_rank):
                V_test[j][c] = random.choice([0, 0, 0, 1, -1])
        for k in range(p * n):
            for c in range(test_rank):
                W_test[k][c] = random.choice([0, 0, 0, 1, -1])
        
        # Quick validation check on a subset of entries
        valid = True
        for _ in range(100):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            s = 0
            for c in range(test_rank):
                s += U_test[i * m + j][c] * V_test[j * p + k][c] * W_test[k * n + i][c]
            if abs(s - 1.0) > 1e-9:
                valid = False
                break
        
        if valid:
            # Full validation
            if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                best_result = {"U": U_test, "V": V_test, "W": W_test}
                best_rank = test_rank
                break
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is kept for structural compatibility.
# EVOLVE-BLOCK-END
