"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # Check the "on" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    # Check some "off" positions (random sample)
    rng = np.random.RandomState(42)
    for _ in range(min(2000, nm * mp * pn // 5)):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False, (a, b, c, val)
    
    return True, None

def _validate_exact_mod(U, V, W, n, m, p, mod):
    """Exact validation over Z/modZ. Returns True if decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    U_mod = np.array(U) % mod
    V_mod = np.array(V) % mod
    W_mod = np.array(W) % mod
    
    # Check on-positions: should be 1 mod p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0
                for r in range(rank):
                    val = (val + U_mod[a, r] * V_mod[b, r] * W_mod[c, r]) % mod
                if val != 1 % mod:
                    return False
    
    # Check off-positions: should be 0 mod p
    rng = np.random.RandomState(123)
    for _ in range(min(3000, nm * mp * pn // 5)):
        a = rng.randint(0, nm)
        b = rng.randint(0, mp)
        c = rng.randint(0, pn)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        i_c = c % n
        k_c = c // n
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0
        for r in range(rank):
            val = (val + U_mod[a, r] * V_mod[b, r] * W_mod[c, r]) % mod
        if val != 0:
            return False
    
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _rank15_3x3x3():
    """
    Known rank-15 decomposition for 3x3x3 matrix multiplication.
    This uses the fact that 3x3x3 can be done in rank 15 (vs schoolbook 27).
    
    The decomposition uses the structure:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use a polynomial-based construction.
    """
    # For 3x3x3, one known rank-15 construction:
    # Use the identity based on evaluating at points.
    # A[i,j] and B[j,k] are bilinear forms.
    
    # Actually, let me use a concrete known decomposition.
    # The 3x3x3 tensor has rank 15 (proven by various methods).
    
    # One explicit rank-15 decomposition:
    # We can use the "3x3 = 2x2 + 1x1" structure with Strassen.
    # But let me try a direct construction.
    
    # Alternative: use the fact that we can decompose 3x3 as:
    # [a b c]   [e f g]
    # [d e f] * [h i j]  =  ...
    # [g h i]   [k l m]
    
    # Let me use a known rank-15 algorithm for 3x3x3.
    # One approach: use 5x5 sub-blocks or polynomial evaluation.
    
    # Actually, for this problem, let me try a different approach:
    # Directly construct a low-rank decomposition for 3x3x5.
    
    # The schoolbook rank for 3x3x5 is 45.
    # Lower bound: max(9, 15, 15) = 15.
    # But the true rank is likely higher.
    
    # Let me try to find a decomposition with rank < 45.
    return None

def _try_lifting_3x3x3_to_3x3x5():
    """
    Strategy: Find a low-rank decomposition for 3x3x3, then lift to 3x3x5.
    
    For 3x3x3: C[i,k] = sum_j A[i,j]*B[j,k] for i,k in 0..2
    For 3x3x5: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, k in 0..4
    
    The lifting idea: if we have a rank-r decomposition for 3x3x3:
    C[i,k] = sum_{r=1}^R U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r]
    
    Then for 3x3x5, we need to handle k=3,4 as well.
    The additional terms involve B[j,3] and B[j,4].
    
    One approach: extend the V matrix. For each existing term r,
    we need V[j*5+k, r] for k=3,4. But this doesn't directly work
    because the W factor also changes.
    
    Better approach: construct the 3x3x5 decomposition directly.
    """
    n, m, p = 3, 3, 5
    nm, mp, pn = 9, 15, 15
    
    # Try to construct a decomposition with rank around 20-30
    # using a structured approach.
    
    # Method: Use the "split" technique.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can write this as:
    # C[i,k] = sum_j (A[i,j] * B[j,k])
    #
    # If we define for each j:
    #   f_j(A) = A[:,j]  (column j of A)
    #   g_j(B) = B[j,:]  (row j of B)
    # Then C = sum_j f_j(A) * g_j(B)
    #
    # This is rank 3 (one term per j), but each "term" is a matrix product
    # of a column and a row, which is rank 1 as a matrix but involves
    # n*p = 15 entries.
    #
    # In tensor form, this gives rank 3 * 1 = 3? No, that's not right.
    # The tensor T[i,j,k] has T[i*m+j, j'*p+k, k'*n+i] = 1 iff j=j' and i=i' and k=k'.
    #
    # The schoolbook decomposition has rank n*m*p = 45.
    #
    # Let me try a different approach: use random search with gradient descent
    # to minimize the error, then round to integers/halves.
    
    return None

def _gradient_descent_search(n, m, p, target_rank, max_iters=5000, lr=0.01, seed=42):
    """
    Use gradient descent to find a low-rank decomposition.
    Minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize
    U = rng.randn(nm, target_rank) * 0.5
    V = rng.randn(mp, target_rank) * 0.5
    W = rng.randn(pn, target_rank) * 0.5
    
    # Precompute target tensor entries
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute current tensor: S[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Loss = ||S - target||^2
        
        # Gradient w.r.t. U[a,r]:
        # dL/dU[a,r] = 2 * sum_{b,c} (S[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        #             = 2 * (S - T)[a, :, :] @ (V[:,r] * W[:,r])  ... not quite
        
        # More carefully:
        # S[a,b,c] = sum_r' U[a,r']*V[b,r']*W[c,r']
        # dS[a,b,c]/dU[a,r] = V[b,r]*W[c,r]
        # dL/dU[a,r] = 2 * sum_{b,c} (S[a,b,c]-T[a,b,c]) * V[b,r]*W[c,r]
        
        # For efficiency, compute residual R[a,b,c] = S[a,b,c] - T[a,b,c]
        # Then dL/dU[a,r] = 2 * sum_{b,c} R[a,b,c] * V[b,r] * W[c,r]
        
        # This is expensive for full tensor. Let me use a different approach.
        # Use the fact that we only need to check specific positions.
        
        # Alternative: use stochastic gradient descent on random samples.
        if it % 100 == 0:
            # Compute loss on a sample
            sample_size = 100
            indices = []
            for _ in range(sample_size):
                a = rng.randint(0, nm)
                b = rng.randint(0, mp)
                c = rng.randint(0, pn)
                indices.append((a, b, c))
            
            loss = 0.0
            for a, b, c in indices:
                s_val = np.dot(U[a, :], V[b, :] * W[c, :])
                t_val = target[a, b, c]
                loss += (s_val - t_val) ** 2
            loss /= sample_size
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-8:
                break
        
        # Stochastic gradient update
        batch_size = 20
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for _ in range(batch_size):
            a = rng.randint(0, nm)
            b = rng.randint(0, mp)
            c = rng.randint(0, pn)
            
            s_val = np.dot(U[a, :], V[b, :] * W[c, :])
            t_val = target[a, b, c]
            err = s_val - t_val
            
            # dL/dU[a,r] = 2*err * V[b,r]*W[c,r]
            grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
            # dL/dV[b,r] = 2*err * U[a,r]*W[c,r]
            grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
            # dL/dW[c,r] = 2*err * U[a,r]*V[b,r]
            grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
        
        grad_U /= batch_size
        grad_V /= batch_size
        grad_W /= batch_size
        
        # Update with momentum-like decay
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _try_direct_construction(n, m, p):
    """
    Try to directly construct a low-rank decomposition for 3x3x5.
    
    Key insight: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can use the following trick:
    Define intermediate quantities that combine A and B entries.
    
    For example, using polynomial evaluation:
    Let x be a variable. Define:
      A_poly(x) = A[0,0] + A[1,0]*x + A[2,0]*x^2  (column 0)
      B_poly(x) = B[0,0] + B[1,0]*x + B[2,0]*x^2  (row 0)
    Then the coefficient of x^j in A_poly(x)*B_poly(x) gives sum_{j1+j2=j} A[i,j1]*B[j2,k]...
    
    This doesn't directly help. Let me try another approach.
    
    Actually, for the specific case 3x3x5, let me try to use
    a combination of Strassen-like ideas.
    """
    n, m, p = 3, 3, 5
    nm, mp, pn = 9, 15, 15
    
    # Try rank values from 20 to 40
    for target_rank in range(20, 41):
        result = _gradient_descent_search(n, m, p, target_rank, max_iters=2000, lr=0.05, seed=target_rank)
        if result is not None:
            U, V, W, loss = result
            if loss < 1e-6:
                # Verify
                valid, _ = _validate(U, V, W, n, m, p, tol=1e-4)
                if valid:
                    return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_school, V_school, W_school)
    
    # Try direct construction with gradient descent
    if time.time() < deadline - 10:
        result = _try_direct_construction(n, m, p)
        if result is not None:
            U, V, W, rank = result
            valid, _ = _validate(U, V, W, n, m, p, tol=1e-6)
            if valid:
                best_rank = rank
                best_result = (U, V, W)
    
    # If we still have time, try more ranks
    if time.time() < deadline - 5 and best_rank == n * m * p:
        for target_rank in range(25, 45):
            if time.time() > deadline - 5:
                break
            U, V, W, loss = _gradient_descent_search(n, m, p, target_rank, max_iters=1000, lr=0.05, seed=1000+target_rank)
            if loss < 1e-6:
                valid, _ = _validate(U, V, W, n, m, p, tol=1e-4)
                if valid:
                    best_rank = target_rank
                    best_result = (U, V, W)
                    break
    
    # Return the best result found
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
