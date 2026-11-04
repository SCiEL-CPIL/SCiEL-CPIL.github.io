"""
Winograd-style decomposition for 4x5 by 5x6 matrix multiplication.
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = 4, 5, 6
    
    # 4-point DFT matrix
    omega = np.exp(-2j * np.pi / 4)
    F4 = np.array([[omega**(a*b) for b in range(4)] for a in range(4)])
    F4_inv = F4.conj().T / 4.0
    
    # Partition:
    # A (4x5) -> A1 (4x4) [cols 0-3], A2 (4x1) [col 4]
    # B (5x6) -> B1 (4x6) [rows 0-3], B2 (1x6) [row 4]
    # C (4x6) = A1 @ B1 + A2 @ B2
    
    # Part 1: A1 (4x4) @ B1 (4x6)
    # Split B1 into B1a (4x4) and B1b (4x2)
    # A1 @ B1a: use FFT-based Winograd
    # A1 @ B1b: schoolbook
    
    # Part 2: A2 (4x1) @ B2 (1x6): schoolbook
    
    # Collect all (U_col, V_col, W_col) triples
    triples = []
    
    # === Part 1a: A1 (4x4) @ B1a (4x4) via 4-point FFT ===
    # C[i,k] = sum_j A1[i,j] * B1a[j,k]
    # Using FFT: 
    #   A_hat = F4 @ A1.T  -> A_hat[omega, i] = sum_j A1[i,j] * omega^(j*omega)
    #   B_hat = B1a @ F4_inv.T -> B_hat[k, omega] = sum_j B1a[j,k] * omega^(-j*omega) / 4
    #   Wait, let me be more careful.
    
    # For circular convolution of length 4:
    # (a * b)[t] = sum_{j} a[j] b[(t-j) mod 4]
    # But matrix mult is: C[i,k] = sum_j A[i,j] B[j,k]
    # This is NOT a standard circular convolution unless we re-index.
    
    # Let me just do: for each (i,k), C[i,k] = sum_j A1[i,j] B1a[j,k]
    # FFT approach: 
    #   Take FFT of A1[i,:] for each i: A_fft[i, freq]
    #   Take FFT of B1a[:,k] for each k: B_fft[freq, k]  
    #   Product: P[i, freq, k] = A_fft[i, freq] * B_fft[freq, k]
    #   Inverse FFT over freq: C[i,k] = IFFT_f(P[i, freq, k])
    
    # This gives rank = 4 (i) * 4 (freq) * 4 (k) = 64 for the 4x4x4 core.
    # Same as schoolbook. Not helpful.
    
    # Let me just use schoolbook for everything to ensure correctness.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
