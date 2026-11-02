"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Use DFT-based construction with N = m = 5
    # C[i,k] = (1/N) sum_f A_hat[i,f] * B_hat[f,k]
    # where A_hat[i,f] = sum_j A[i,j] * omega^(f*j)
    #       B_hat[f,k] = sum_j B[j,k] * omega^(-f*j)
    # 
    # This gives rank = n * N * m = 3 * 5 * 5 = 75 if we expand each product
    # But actually, each (i,f,k) term requires summing over j1,j2, giving m^2 terms.
    # 
    # Instead, use the direct schoolbook: rank = n*m*p = 105
    # 
    # To improve, note that we can use the DFT to combine the j dimension.
    # For each (i,k,f), the contribution is sum_j A[i,j] B[j,k] omega^(f*j) omega^(-f*j)
    # = sum_j A[i,j] B[j,k]
    # 
    # The key: use N=5 DFT. For each frequency f, define:
    #   U[i*m+j, (i,f)] = omega^(f*j) / sqrt(N)
    #   V[j*p+k, (i,f)] = omega^(-f*j) / sqrt(N)  
    #   W[k*n+i, (i,f)] = 1
    # 
    # Then sum over j and f: sum_f sum_j U*V*W = sum_f sum_j (omega^(f*j)/sqrt(N)) (omega^(-f*j)/sqrt(N)) * 1
    # = sum_f (1/N) * m * delta_{f,0}... no this is wrong.
    #
    # Let me use the correct schoolbook decomposition:
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
# EVOLVE-BLOCK-END
