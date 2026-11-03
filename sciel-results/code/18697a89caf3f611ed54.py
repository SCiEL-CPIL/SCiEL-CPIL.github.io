"""Matrix multiplication 3x4x7: rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 3, 4, 7
    
    # We'll search for a decomposition of rank r.
    # The tensor T[i,j,k] = 1 if we multiply A[i,j] * B[j,k] -> C[i,k]
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, c] follows the trace convention
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Schoolbook gives rank = n*m*p = 84
    # Best known for 3x4x7 is likely around 28-35
    
    # Strategy: Try ranks from lower bound upward, search for valid decomposition
    # Use integer coefficients in {-1, 0, 1} for speed, then try {-1, 0, 1, 2}
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # The tensor entry is 1 for valid (i,j,k)
                    actual = 0
                    for c in range(rank):
                        actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if actual != expected:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, rank):
        """Faster check using vectorized operations."""
        import numpy as np
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Reshape to 3D
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        # Compute tensor product sum
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with a known valid decomposition (schoolbook) as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find better decompositions
    # For small ranks, we can try systematic search
    
    best_result = schoolbook()
    
    # Try rank values from 28 upward
    for rank in range(28, 40):
        if time.time() > deadline - 20:
            break
        
        # For each rank, try to find a decomposition
        # This is a hard problem, so we use a heuristic approach
        
        # Approach: Use the fact that we need integer/half-integer entries
        # Try small integer coefficients {-1, 0, 1}
        
        # For rank <= 30, do a more systematic search
        if rank <= 32:
            result = _search_for_rank(rank, deadline - 20)
            if result is not None:
                best_result = result
                break
    
    return best_result

def _search_for_rank(rank, time_limit):
    """Search for a decomposition of given rank."""
    n, m, p = 3, 4, 7
    start_time = time.time()
    
    # For a valid decomposition, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is equivalent to finding rank-r matrices that factor the multiplication tensor
    
    # Heuristic: Try to build the decomposition column by column
    # Each column c contributes a rank-1 tensor
    
    # A rank-1 tensor is: u_i * v_j * w_k where u is in R^(n*m), v in R^(m*p), w in R^(p*n)
    # But we need u[i*m+j] * v[j*p+k] * w[k*n+i] = some pattern
    
    # For a valid rank-1 contribution, we need the outer product structure
    # Let's think of it differently:
    # The tensor T has T[i,j,k] = 1
    # We want T = sum_c U_c ⊗ V_c ⊗ W_c (appropriately indexed)
    
    # Key insight: For the standard matrix multiplication tensor,
    # each "slice" in one dimension gives a matrix
    # T[i,j,k] = A[i,j] * B[j,k]
    
    # A rank-1 term: f(i,j) * g(j,k) * h(i,k)
    # where f(i,j) = U[i*m+j], g(j,k) = V[j*p+k], h(i,k) = W[k*n+i]
    
    # For this to work: f(i,j) * g(j,k) * h(i,k) should sum to 1 over c
    
    # Try a structured approach:
    # Let f_c(i,j) depend only on i and j in a simple way
    # Let g_c(j,k) depend only on j and k
    # Let h_c(i,k) depend only on i and k
    
    # Simplest: f_c(i,j) = a_c(i) * b_c(j), etc.
    # Then term = a_c(i)*b_c(j)*c_c(j)*d_c(k)*e_c(i)*f_c(k)
    # = [a_c(i)*e_c(i)] * [b_c(j)*c_c(j)] * [d_c(k)*f_c(k)]
    # This is a product of three functions, one in i, one in j, one in k
    # But we need it to be 1 for all (i,j,k), which means each must be constant
    # That doesn't work for non-trivial decompositions
    
    # Better approach: Allow f_c(i,j) to be more general
    # For rank r, we need r terms that sum to 1
    
    # Let's try: f_c(i,j) = delta(i, i_c) * delta(j, j_c)  [sparse in i,j]
    # g_c(j,k) = delta(j, j_c) * delta(k, k_c)  [sparse in j,k]  
    # h_c(i,k) = delta(i, i_c) * delta(k, k_c)  [sparse in i,k]
    # Product: delta(i,i_c)*delta(j,j_c)*delta(j,j_c)*delta(k,k_c)*delta(i,i_c)*delta(k,k_c)
    # = delta(i,i_c)^2 * delta(j,j_c)^2 * delta(k,k_c)^2
    # = delta(i,i_c) * delta(j,j_c) * delta(k,k_c)
    # Sum over c: sum_c delta(i,i_c)*delta(j,j_c)*delta(k,k_c)
    # This equals 1 iff there's exactly one c with (i_c,j_c,k_c) = (i,j,k)
    # This is just the schoolbook decomposition with rank = n*m*p
    
    # For better decompositions, we need overlapping support
    
    # Try: f_c(i,j) = 1 for a set S_c of (i,j) pairs
    # g_c(j,k) = 1 for a set T_c of (j,k) pairs
    # h_c(i,k) = 1 for a set R_c of (i,k) pairs
    # Product is 1 iff (i,j) in S_c AND (j,k) in T_c AND (i,k) in R_c
    # Sum over c should be 1 for all (i,j,k)
    
    # This means: for each (i,j,k), exactly one c has all three conditions
    # This is a set cover / exact cover problem
    
    # Let's try a different parameterization:
    # f_c(i,j) = alpha_c(i) * beta_c(j)
    # g_c(j,k) = gamma_c(j) * delta_c(k)
    # h_c(i,k) = epsilon_c(i) * zeta_c(k)
    # Product = alpha_c(i)*epsilon_c(i) * beta_c(j)*gamma_c(j) * delta_c(k)*zeta_c(k)
    # Let A_c(i) = alpha_c(i)*epsilon_c(i), B_c(j) = beta_c(j)*gamma_c(j), C_c(k) = delta_c(k)*zeta_c(k)
    # Term = A_c(i) * B_c(j) * C_c(k)
    # Sum_c A_c(i) * B_c(j) * C_c(k) = 1 for all i,j,k
    
    # This means the tensor with entries A_c(i)*B_c(j)*C_c(k) summed over c equals the all-ones tensor
    # The all-ones tensor has rank 1, so we need rank >= 1
    # But the constraint is that A_c, B_c, C_c must come from factorizations of the U, V, W entries
    
    # Actually this parameterization is too restrictive. Let's try a more general search.
    
    # Given time constraints, let's try a randomized approach with local search
    
    import numpy as np
    import random
    
    random.seed(42)
    np.random.seed(42)
    
    # Try random initializations and local optimization
    for attempt in range(50):
        if time.time() > start_time + (time_limit - (time.time() - start_time)) * 0.9:
            break
        
        # Initialize with random small integers
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        for c in range(rank):
            # Random sparse initialization
            for idx in range(n * m):
                if random.random() < 0.3:
                    U[idx, c] = random.choice([-1, 1])
            for idx in range(m * p):
                if random.random() < 0.3:
                    V[idx, c] = random.choice([-1, 1])
            for idx in range(p * n):
                if random.random() < 0.3:
                    W[idx, c] = random.choice([-1, 1])
        
        # Local search: try to fix violations
        for iteration in range(1000):
            if time.time() > start_time + 10:
                break
            
            # Compute current tensor values
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        errors.append((i, j, k, val - 1.0))
            
            # Check if we've found a solution
            max_err = max(abs(e[3]) for e in errors)
            if max_err < 1e-9:
                # Convert to list of lists with integer/half-integer check
                U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (int(round(2*x))/2) for x in row] for row in U]
                V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (int(round(2*x))/2) for x in row] for row in V]
                W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (int(round(2*x))/2) for x in row] for row in W]
                
                # Verify with exact arithmetic
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for c in range(rank):
                                total += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                            if total != 1:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            # Find worst error and try to fix it
            worst = max(errors, key=lambda x: abs(x[3]))
            i, j, k, err = worst
            if abs(err) < 1e-9:
                continue
            
            # Try modifying one entry in U, V, or W for a random column
            c = random.randint(0, rank - 1)
            choice = random.choice(['U', 'V', 'W'])
            if choice == 'U':
                idx = i * m + j
                old_val = U[idx, c]
                U[idx, c] = old_val + random.choice([-1, 1])
            elif choice == 'V':
                idx = j * p + k
                old_val = V[idx, c]
                V[idx, c] = old_val + random.choice([-1, 1])
            else:
                idx = k * n + i
                old_val = W[idx, c]
                W[idx, c] = old_val + random.choice([-1, 1])
            
            # Accept if it reduces total error
            new_errors = []
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        val = 0
                        for cc in range(rank):
                            val += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                        new_errors.append(abs(val - 1.0))
            
            old_total = sum(abs(e[3]) for e in errors)
            new_total = sum(new_errors)
            
            if new_total >= old_total:
                # Revert
                if choice == 'U':
                    U[idx, c] = old_val
                elif choice == 'V':
                    V[idx, c] = old_val
                else:
                    W[idx, c] = old_val
    
    return None

# Fallback: return schoolbook if nothing better found
def construct():
    start_time = time.time()
    
    n, m, p = 3, 4, 7
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find better decomposition
    result = None
    for rank in range(28, 36):
        if time.time() - start_time > 120:
            break
        result = _search_for_rank(rank, 120 - (time.time() - start_time))
        if result is not None:
            break
    
    if result is None:
        result = schoolbook()
    
    return result
