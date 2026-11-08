"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_start_time = time.time()
_TIME_LIMIT = 150  # seconds

def _verify_decomposition(n, m, p, U, V, W, num_tests=5):
    """
    Verify if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
    """
    R = len(U[0])
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_direct = A @ B
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_reshaped = U_arr.reshape(n, m, R)
        V_reshaped = V_arr.reshape(m, p, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        LA = np.einsum('ij,ijr->ir', A, U_reshaped)
        LB = np.einsum('jk,jkr->kr', B, V_reshaped)
        C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
        
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2 * 2x2 multiplication.
    C[i,k] = sum_j A[i,j] B[j,k] for i,j,k in {0,1}
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U indices: i*m + j = i*2 + j for i,j in {0,1}
    # V indices: j*p + k = j*2 + k for j,k in {0,1}
    # W indices: k*n + i = k*2 + i for k,i in {0,1}
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> U[0][0]=1, U[3][0]=1
    # V: B00=1, B11=1 -> V[0][0]=1, V[3][0]=1
    # W: C00 += 1, C11 += 1 -> W[0][0]=1, W[3][0]=1
    
    # M2 = (A10+A11)*B00
    # U: A10=1, A11=1 -> U[2][1]=1, U[3][1]=1
    # V: B00=1 -> V[0][1]=1
    # W: C10 += 1 -> W[2][1]=1
    
    # M3 = A00*(B01-B11)
    # U: A00=1 -> U[0][2]=1
    # V: B01=1, B11=-1 -> V[1][2]=1, V[3][2]=-1
    # W: C01 += 1 -> W[1][2]=1
    
    # M4 = A11*(B10-B00)
    # U: A11=1 -> U[3][3]=1
    # V: B10=1, B00=-1 -> V[2][3]=1, V[0][3]=-1
    # W: C00 += 1, C10 += 1 -> W[0][3]=1, W[2][3]=1
    
    # M5 = (A00+A01)*B11
    # U: A00=1, A01=1 -> U[0][4]=1, U[1][4]=1
    # V: B11=1 -> V[3][4]=1
    # W: C00 -= 1, C01 += 1 -> W[0][4]=-1, W[1][4]=1
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1 -> U[2][5]=1, U[0][5]=-1
    # V: B00=1, B01=1 -> V[0][5]=1, V[1][5]=1
    # W: C11 += 1 -> W[3][5]=1
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1 -> U[1][6]=1, U[3][6]=-1
    # V: B10=1, B11=1 -> V[2][6]=1, V[3][6]=1
    # W: C00 += 1, C11 += 1 -> W[0][6]=1, W[3][6]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    global _start_time
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Schoolbook solution (always valid fallback)
    schoolbook_rank = n * m * p  # 36
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Structured decomposition: split 6 columns into 3 pairs
    # For each pair of columns, use rank-1 for A[:,0]*B[0,:] + rank-7 Strassen for A[:,1:3]*B[1:3,:]
    # Total rank: 3 * (1 + 7) = 24
    
    # Build the structured solution
    U_struct = [[0] * 24 for _ in range(n * m)]  # (6, 24)
    V_struct = [[0] * 24 for _ in range(m * p)]  # (18, 24)
    W_struct = [[0] * 24 for _ in range(p * n)]  # (12, 24)
    
    col = 0
    for block in range(3):
        k0 = 2 * block  # first column in this block
        k1 = 2 * block + 1  # second column in this block
        
        # Part 1: A[:,0] * B[0, k0:k1+1] (rank 1)
        # U: A[i,0] for i in {0,1} -> U[i*3 + 0][col] = 1
        # V: B[0, k] for k in {k0, k1} -> V[0*6 + k][col] = 1
        # W: C[i, k] for i in {0,1}, k in {k0, k1} -> W[k*2 + i][col] = 1
        for i in range(n):
            U_struct[i * m + 0][col] = 1
        for k in range(k0, k1 + 1):
            V_struct[0 * p + k][col] = 1
        for i in range(n):
            for k in range(k0, k1 + 1):
                W_struct[k * n + i][col] = 1
        col += 1
        
        # Part 2: A[:,1:3] * B[1:3, k0:k1+1] using Strassen (rank 7)
        # This is a 2x2 * 2x2 multiplication
        # Local indices: i' in {0,1} corresponds to global i in {0,1}
        #                 j' in {0,1} corresponds to global j in {1,2}
        #                 k' in {0,1} corresponds to global k in {k0, k1}
        # 
        # Strassen U,V,W for 2x2*2x2:
        # U_local[i'*2 + j'][r] for i',j' in {0,1}
        # V_local[j'*2 + k'][r] for j',k' in {0,1}
        # W_local[k'*2 + i'][r] for k',i' in {0,1}
        
        U_str, V_str, W_str = _strassen_2x2x2()
        
        for r in range(7):
            # U: local (i', j') -> global (i, j=j'+1)
            for i_prime in range(2):
                for j_prime in range(2):
                    global_i = i_prime
                    global_j = j_prime + 1
                    u_val = U_str[i_prime * 2 + j_prime][r]
                    if u_val != 0:
                        U_struct[global_i * m + global_j][col + r] = u_val
            
            # V: local (j', k') -> global (j=j'+1, k=k0+k')
            for j_prime in range(2):
                for k_prime in range(2):
                    global_j = j_prime + 1
                    global_k = k0 + k_prime
                    v_val = V_str[j_prime * 2 + k_prime][r]
                    if v_val != 0:
                        V_struct[global_j * p + global_k][col + r] = v_val
            
            # W: local (k', i') -> global (k=k0+k', i=i')
            for k_prime in range(2):
                for i_prime in range(2):
                    global_k = k0 + k_prime
                    global_i = i_prime
                    w_val = W_str[k_prime * 2 + i_prime][r]
                    if w_val != 0:
                        W_struct[global_k * n + global_i][col + r] = w_val
        col += 7
    
    # Verify the structured solution
    if _verify_decomposition(n, m, p, U_struct, V_struct, W_struct, num_tests=10):
        best_solution = {"U": U_struct, "V": V_struct, "W": W_struct}
        best_rank = 24
    
    # Try to improve further with random search if time allows
    if time.time() - _start_time < _TIME_LIMIT - 30:
        # Try random search for rank < best_rank
        for rank in range(best_rank - 1, 15, -1):
            if time.time() - _start_time > _TIME_LIMIT - 20:
                break
            
            max_candidates = 200
            for _ in range(max_candidates):
                if time.time() - _start_time > _TIME_LIMIT - 10:
                    break
                
                # Generate random sparse matrices
                U_rand = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
                V_rand = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
                W_rand = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
                
                if _verify_decomposition(n, m, p, U_rand, V_rand, W_rand, num_tests=3):
                    best_solution = {"U": U_rand, "V": V_rand, "W": W_rand}
                    best_rank = rank
                    break
            if best_rank < rank:
                break
    
    return best_solution
# EVOLVE-BLOCK-END
