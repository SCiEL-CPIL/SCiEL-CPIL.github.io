import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms for 2x2 matrices.
    # Each term s has coefficients for A (beta), B (gamma), and C (delta).
    # Indices are 0-based: 00, 01, 10, 11 -> 0, 1, 2, 3
    # beta[s][I][J], gamma[s][J][K], delta[s][I][K]
    
    # Initialize with zeros
    beta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    delta = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0 (M1) = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    delta[0][0][0] = 1; delta[0][1][1] = 1
    
    # s=1 (M2) = (A10+A11)(B00) -> C10+=M2, C11-=M2
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    delta[1][1][0] = 1; delta[1][1][1] = -1
    
    # s=2 (M3) = (A00)(B01-B11) -> C01+=M3, C11+=M3
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    delta[2][0][1] = 1; delta[2][1][1] = 1
    
    # s=3 (M4) = (A11)(B10-B00) -> C00+=M4, C10+=M4
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    delta[3][0][0] = 1; delta[3][1][0] = 1
    
    # s=4 (M5) = (A00+A01)(B11) -> C00-=M5, C01+=M5
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    delta[4][0][0] = -1; delta[4][0][1] = 1
    
    # s=5 (M6) = (A10-A00)(B00+B01) -> C11+=M6
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    delta[5][1][1] = 1
    
    # s=6 (M7) = (A01-A11)(B10+B11) -> C00+=M7
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    delta[6][0][0] = 1

    # Construct the rank-49 decomposition.
    # The decomposition is a Kronecker product of the two Strassen decompositions.
    # Outer index s (0..6) corresponds to the block-level Strassen term.
    # Inner index t (0..6) corresponds to the element-level Strassen term.
    # Column r = 7*s + t.
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # Fill U
            # U[i*4+j, r] = sum_{I,J} beta[s][I][J] * beta_inner[t][i%2][j%2]
            # Actually, the inner Strassen coefficients are the same structure as beta/gamma/delta.
            # Let's reuse the beta/gamma/delta arrays for the inner level too, 
            # because the 2x2 Strassen algorithm is identical.
            
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    I_prime = i % 2
                    J = j // 2
                    J_prime = j % 2
                    # Coefficient is product of outer coeff and inner coeff
                    coeff = beta[s][I][J] * beta[t][I_prime][J_prime]
                    if coeff != 0:
                        U[i * 4 + j][r] = coeff
                        
            # Fill V
            # V[j*4+k, r] = sum_{J,K} gamma[s][J][K] * gamma_inner[t][j%2][k%2]
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    J_prime = j % 2
                    K = k // 2
                    K_prime = k % 2
                    coeff = gamma[s][J][K] * gamma[t][J_prime][K_prime]
                    if coeff != 0:
                        V[j * 4 + k][r] = coeff
                        
            # Fill W
            # W[k*4+i, r] = sum_{K,I} delta[s][I][K] * delta_inner[t][i%2][k%2]
            # Note: delta[s][I][K] corresponds to C[I][K] contribution.
            # Inner delta[t][I_prime][K_prime] corresponds to local C[I_prime][K_prime].
            # Global C index is (I*2+I_prime, K*2+K_prime).
            # W index is k*4+i = (K*2+K_prime)*4 + (I*2+I_prime).
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    K_prime = k % 2
                    I = i // 2
                    I_prime = i % 2
                    coeff = delta[s][I][K] * delta[t][I_prime][K_prime]
                    if coeff != 0:
                        W[k * 4 + i][r] = coeff
                        
    return {"U": U, "V": V, "W": W}
