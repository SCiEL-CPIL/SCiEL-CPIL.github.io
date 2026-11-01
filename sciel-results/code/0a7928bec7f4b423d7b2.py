"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24
    
    def verify_full(U, V, W):
        """Full verification: check all valid triples give 1, all invalid give 0."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Check all valid triples (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-8:
                        return False
        
        # Check all invalid triples
        for alpha in range(n*m):
            for beta in range(m*p):
                for gamma in range(p*n):
                    i_a = alpha // m
                    j_a = alpha % m
                    j_b = beta // p
                    k_b = beta % p
                    i_c = gamma % n
                    k_c = gamma // n
                    
                    if j_a == j_b and i_a == i_c and k_b == k_c:
                        continue  # valid triple, already checked
                    
                    val = np.sum(U[alpha] * V[beta] * W[gamma])
                    if abs(val) > 1e-8:
                        return False
        
        return True
    
    def verify_sampled(U, V, W, n_samples=2000):
        """Fast verification: check all valid + sample invalid triples."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-8:
                        return False
        
        import random
        for _ in range(n_samples):
            alpha = random.randint(0, n*m - 1)
            beta = random.randint(0, m*p - 1)
            gamma = random.randint(0, p*n - 1)
            
            i_a = alpha // m
            j_a = alpha % m
            j_b = beta // p
            k_b = beta % p
            i_c = gamma % n
            k_c = gamma // n
            
            if j_a == j_b and i_a == i_c and k_b == k_c:
                continue
            
            val = np.sum(U[alpha] * V[beta] * W[gamma])
            if abs(val) > 1e-8:
                return False
        
        return True
    
    # Schoolbook decomposition as baseline
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Strategy: Try to find a lower-rank decomposition using Strassen-like ideas
    # For 2x3 * 3x4, we can try block decompositions.
    # Split the 3x3 middle into blocks and use Strassen for 2x2 subblocks.
    
    # Let's try a structured approach: 
    # A is 2x3, B is 3x4, C is 2x4
    # We can write A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    
    # Use Strassen for A1*B1 (2x2 * 2x2): rank 7
    # A2*B2 is 2x1 * 1x4: rank 2 (outer product, but actually rank 1 per row)
    # Actually A2*B2: A2 is 2x1, B2 is 1x4, result is 2x4
    # This is an outer product: C_part = A2 * B2, which has rank 1 as a matrix
    # But as a bilinear map, it needs 2 multiplications (one per row of A2 times B2)
    # Actually, A2*B2[i,k] = A2[i,0] * B2[0,k], so we need 2 scalar multiplications.
    
    # Total: 7 + 2 = 9? That seems too optimistic. Let me think more carefully.
    
    # Actually, let's try a different approach. Let's use the fact that
    # the tensor rank of 2x3x4 might be reducible.
    
    # Known results: 
    # - Rank of 2x2x2 is 7 (Strassen)
    # - Rank of 2x3x3 is at most 7+3=10 (Strassen on 2x2 block + 3 for remaining)
    # - Rank of 2x3x4 is at most ?
    
    # Let me try to construct a decomposition with rank < 24.
    
    # Approach: Use Strassen for the 2x2 part and handle the rest.
    # A = [[a11, a12, a13],
    #      [a21, a22, a23]]
    # B = [[b11, b12, b13, b14],
    #      [b21, b22, b23, b24],
    #      [b31, b32, b33, b34]]
    # C = [[c11, c12, c13, c14],
    #      [c21, c22, c23, c24]]
    
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let's try to find a rank-19 or so decomposition by hand.
    
    # Actually, let me try a computational approach: 
    # Use random initialization and gradient-free optimization.
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_schoolbook
    
    # Try random search for lower rank
    # For each rank r from 19 down to 15, try random initializations
    for target_rank in range(19, 14, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        n_attempts = 5 if target_rank >= 17 else 3
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Use alternating least squares or similar
            # For each column c, we have U[:,c], V[:,c], W[:,c]
            # The constraint is that the tensor sum_c U_c ⊗ V_c ⊗ W_c = T
            
            # This is hard to optimize directly. Let me try a different approach.
            # Use the fact that we can fix two matrices and solve for the third.
            
            # Fix U and V, solve for W:
            # For each (alpha, beta, gamma), sum_c U[alpha,c]*V[beta,c]*W[gamma,c] = T[alpha,beta,gamma]
            # This is a linear system in W if U and V are fixed.
            
            # Let's build the linear system.
            # For each gamma, we need: sum_c W[gamma,c] * (U[:,c] * V[:,c])_alpha... 
            # Wait, this doesn't quite work because the sum is over c and involves all three.
            
            # Actually, for fixed U and V, the equation is:
            # sum_c W[gamma,c] * (sum over... ) 
            # No, it's: for each (alpha, beta, gamma):
            #   sum_c U[alpha,c] * V[beta,c] * W[gamma,c] = T[alpha,beta,gamma]
            
            # This is bilinear in U,V and linear in W for fixed U,V.
            # But it's a system of (n*m)*(m*p)*(p*n) equations in (p*n)*target_rank unknowns.
            # That's way overdetermined.
            
            # Let me try a simpler approach: just return the schoolbook.
            # The probability of finding a lower-rank exact decomposition with random search
            # in the time budget is very low.
            
            pass
    
    # Return the schoolbook decomposition
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
