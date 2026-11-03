"""6x6 matrix multiplication tensor decomposition using Strassen 2x2 block
partition with schoolbook 3x3 sub-blocks. Total rank = 7 * 27 = 189."""
import time as _time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2_schoolbook_3x3():
    """
    Builds a rank-189 decomposition for 6x6 matrix multiplication.
    Uses tensor product of:
    - Rank-7 Strassen decomposition for 2x2 matrix multiplication
    - Rank-27 schoolbook decomposition for 3x3 matrix multiplication
    
    The 6x6 matrices are partitioned into 2x2 blocks of 3x3 submatrices.
    """
    n = 6
    rank = 189
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # Strassen 2x2 coefficients
    # beta[s][I][J]: coefficient of A[I,J] block in left factor of step s
    # gamma[s][K][L]: coefficient of B[K,L] block in right factor of step s  
    # alpha[s][I][K]: coefficient of result M[s] in C[I,K] block
    
    # s=0: M0 = (A00+A11)(B00+B11) -> C00+=M0, C11+=M0
    beta0 = {(0,0): 1, (1,1): 1}
    gamma0 = {(0,0): 1, (1,1): 1}
    alpha0 = {(0,0): 1, (1,1): 1}
    
    # s=1: M1 = (A10+A11)B00 -> C10+=M1, C11-=M1
    beta1 = {(1,0): 1, (1,1): 1}
    gamma1 = {(0,0): 1}
    alpha1 = {(1,0): 1, (1,1): -1}
    
    # s=2: M2 = A00(B01-B11) -> C01+=M2, C11+=M2
    beta2 = {(0,0): 1}
    gamma2 = {(0,1): 1, (1,1): -1}
    alpha2 = {(0,1): 1, (1,1): 1}
    
    # s=3: M3 = A11(B10-B00) -> C00+=M3, C10+=M3
    beta3 = {(1,1): 1}
    gamma3 = {(1,0): 1, (0,0): -1}
    alpha3 = {(0,0): 1, (1,0): 1}
    
    # s=4: M4 = (A00+A01)B11 -> C00-=M4, C01+=M4
    beta4 = {(0,0): 1, (0,1): 1}
    gamma4 = {(1,1): 1}
    alpha4 = {(0,0): -1, (0,1): 1}
    
    # s=5: M5 = (A10-A00)(B00+B01) -> C11+=M5
    beta5 = {(1,0): 1, (0,0): -1}
    gamma5 = {(0,0): 1, (0,1): 1}
    alpha5 = {(1,1): 1}
    
    # s=6: M6 = (A01-A11)(B10+B11) -> C00+=M6
    beta6 = {(0,1): 1, (1,1): -1}
    gamma6 = {(1,0): 1, (1,1): 1}
    alpha6 = {(0,0): 1}
    
    strassen_steps = [
        (beta0, gamma0, alpha0),
        (beta1, gamma1, alpha1),
        (beta2, gamma2, alpha2),
        (beta3, gamma3, alpha3),
        (beta4, gamma4, alpha4),
        (beta5, gamma5, alpha5),
        (beta6, gamma6, alpha6),
    ]
    
    block_size = 3
    
    for s_idx, (beta, gamma, alpha) in enumerate(strassen_steps):
        for a in range(block_size):
            for b in range(block_size):
                for c in range(block_size):
                    r = s_idx * 27 + a * 9 + b * 3 + c
                    
                    # Fill U: A coefficients
                    for (I, J), coeff in beta.items():
                        i_glob = I * block_size + a
                        j_glob = J * block_size + b
                        U[i_glob * 6 + j_glob][r] = coeff
                    
                    # Fill V: B coefficients
                    for (K, L), coeff in gamma.items():
                        j_glob = K * block_size + b
                        k_glob = L * block_size + c
                        V[j_glob * 6 + k_glob][r] = coeff
                    
                    # Fill W: C coefficients (trace convention: W[k*n+i, r])
                    for (CI, CK), coeff in alpha.items():
                        i_glob = CI * block_size + a
                        k_glob = CK * block_size + c
                        W[k_glob * 6 + i_glob][r] = coeff
    
    return U, V, W


def _verify_decomposition(U, V, W, n=6, m=6, p=6):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if i == k else 0
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if actual != expected:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    t_start = _time.time()
    deadline = t_start + 100  # Safety margin before 150s limit
    
    try:
        U, V, W = _build_strassen_2x2_schoolbook_3x3()
        rank = len(U[0])
        
        # Verify correctness
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to trivial schoolbook rank 216
            rank_fb = 216
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * 36 + j * 6 + k
                        U_fb[i * m + j][r] = 1
                        V_fb[j * p + k][r] = 1
                        W_fb[k * n + i][r] = 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
            
    except Exception as e:
        # Emergency fallback
        rank_fb = 216
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * 36 + j * 6 + k
                    U_fb[i * m + j][r] = 1
                    V_fb[j * p + k][r] = 1
                    W_fb[k * n + i][r] = 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
