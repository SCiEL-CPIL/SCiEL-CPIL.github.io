import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # We construct a decomposition of rank 81.
    # Part 1: 4x4 * 4x4 using Strassen (Rank 49)
    # Part 2: 4x4 * 4x2 using Schoolbook (Rank 32)
    
    rank_4x4 = 49
    rank_4x2 = 32
    total_rank = rank_4x4 + rank_4x2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Strassen 4x4x4x4 ---
    
    # Define Strassen coefficients for 2x2x2
    # U_in[L, i, j] is coeff of A[i,j] in L-th term
    U_in = np.zeros((7, 2, 2))
    U_in[0, 0, 0] = 1; U_in[0, 1, 1] = 1
    U_in[1, 1, 0] = 1; U_in[1, 1, 1] = 1
    U_in[2, 0, 0] = 1
    U_in[3, 1, 1] = 1
    U_in[4, 0, 0] = 1; U_in[4, 0, 1] = 1
    U_in[5, 1, 0] = 1; U_in[5, 0, 0] = -1
    U_in[6, 0, 1] = 1; U_in[6, 1, 1] = -1
    
    # V_in[L, j, k] is coeff of B[j,k] in L-th term
    V_in = np.zeros((7, 2, 2))
    V_in[0, 0, 0] = 1; V_in[0, 1, 1] = 1
    V_in[1, 0, 0] = 1
    V_in[2, 1, 0] = 1; V_in[2, 1, 1] = -1
    V_in[3, 0, 1] = 1; V_in[3, 0, 0] = -1
    V_in[4, 1, 1] = 1
    V_in[5, 0, 0] = 1; V_in[5, 0, 1] = 1
    V_in[6, 1, 0] = 1; V_in[6, 1, 1] = 1
    
    # W_in[L, i, k] is coeff of C[i,k] in L-th term
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    W_in = np.zeros((7, 2, 2))
    W_in[0, 0, 0] = 1; W_in[0, 1, 1] = 1
    W_in[1, 1, 0] = 1; W_in[1, 1, 1] = -1
    W_in[2, 0, 1] = 1; W_in[2, 1, 1] = 1
    W_in[3, 0, 0] = 1; W_in[3, 1, 0] = 1
    W_in[4, 0, 0] = -1; W_in[4, 0, 1] = 1
    W_in[5, 1, 1] = 1
    W_in[6, 0, 0] = 1
    
    # Outer Strassen coefficients for 4x4 blocks (2x2 blocks of 2x2)
    # Alpha_K[u, v] is coeff of Block A[u,v] in K-th outer term
    Alpha = np.zeros((7, 2, 2))
    Alpha[0, 0, 0] = 1; Alpha[0, 1, 1] = 1
    Alpha[1, 1, 0] = 1; Alpha[1, 1, 1] = 1
    Alpha[2, 0, 0] = 1
    Alpha[3, 1, 1] = 1
    Alpha[4, 0, 0] = 1; Alpha[4, 0, 1] = 1
    Alpha[5, 1, 0] = 1; Alpha[5, 0, 0] = -1
    Alpha[6, 0, 1] = 1; Alpha[6, 1, 1] = -1
    
    # Beta_K[u, v] is coeff of Block B[u,v] in K-th outer term
    Beta = np.zeros((7, 2, 2))
    Beta[0, 0, 0] = 1; Beta[0, 1, 1] = 1
    Beta[1, 0, 0] = 1
    Beta[2, 1, 0] = 1; Beta[2, 1, 1] = -1
    Beta[3, 0, 1] = 1; Beta[3, 0, 0] = -1
    Beta[4, 1, 1] = 1
    Beta[5, 0, 0] = 1; Beta[5, 0, 1] = 1
    Beta[6, 1, 0] = 1; Beta[6, 1, 1] = 1
    
    # Gamma_K[u, v] is coeff of Block C[u,v] in K-th outer term
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    Gamma = np.zeros((7, 2, 2))
    Gamma[0, 0, 0] = 1; Gamma[0, 1, 1] = 1
    Gamma[1, 1, 0] = 1; Gamma[1, 1, 1] = -1
    Gamma[2, 0, 1] = 1; Gamma[2, 1, 1] = 1
    Gamma[3, 0, 0] = 1; Gamma[3, 1, 0] = 1
    Gamma[4, 0, 0] = -1; Gamma[4, 0, 1] = 1
    Gamma[5, 1, 1] = 1
    Gamma[6, 0, 0] = 1
    
    for K in range(7):
        for L in range(7):
            R = K * 7 + L
            # Fill U, V, W for column R
            
            # U: coeff of A[i,j]
            # A_factor = sum_{i',j'} U_in[L, i', j'] * A^K[i', j']
            # A^K[i', j'] = sum_{u,v} Alpha[K, u, v] * A[2u+i', 2v+j']
            for i_prime in range(2):
                for j_prime in range(2):
                    coeff_in = U_in[L, i_prime, j_prime]
                    if coeff_in == 0: continue
                    for u in range(2):
                        for v in range(2):
                            coeff_out = Alpha[K, u, v]
                            if coeff_out == 0: continue
                            i = 2 * u + i_prime
                            j = 2 * v + j_prime
                            U[i * m + j, R] += coeff_in * coeff_out
            
            # V: coeff of B[j,k]
            # B_factor = sum_{j',k'} V_in[L, j', k'] * B^K[j', k']
            # B^K[j', k'] = sum_{u,v} Beta[K, u, v] * B[2u+j', 2v+k']
            for j_prime in range(2):
                for k_prime in range(2):
                    coeff_in = V_in[L, j_prime, k_prime]
                    if coeff_in == 0: continue
                    for u in range(2):
                        for v in range(2):
                            coeff_out = Beta[K, u, v]
                            if coeff_out == 0: continue
                            j = 2 * u + j_prime
                            k = 2 * v + k_prime
                            V[j * p + k, R] += coeff_in * coeff_out
            
            # W: coeff of C[i,k]
            # C^K[i'', k''] = sum_L W_in[L, i'', k''] * M_{K,L}
            # C[u'', v''] = sum_K Gamma[K, u'', v''] * C^K[u'', v'']
            # So C[2u''+i'', 2v''+k''] gets W_in[L, i'', k''] * Gamma[K, u'', v'']
            for i_double in range(2):
                for k_double in range(2):
                    coeff_in = W_in[L, i_double, k_double]
                    if coeff_in == 0: continue
                    for u_double in range(2):
                        for v_double in range(2):
                            coeff_out = Gamma[K, u_double, v_double]
                            if coeff_out == 0: continue
                            i = 2 * u_double + i_double
                            k = 2 * v_double + k_double
                            W[k * n + i, R] += coeff_in * coeff_out
    
    # --- Part 2: Schoolbook 4x4 * 4x2 ---
    # Columns 49 to 80
    # i in 0..3, j in 0..3, k in 0..1
    col_offset = rank_4x4
    for i in range(n):
        for j in range(m):
            for k in range(2): # p=6, but we only handle k=0,1 here? 
                # Wait, the task is 4x4 * 4x6.
                # My split was B = [B_1 (4x4), B_2 (4x2)].
                # B_1 corresponds to k=0..3.
                # B_2 corresponds to k=4..5.
                # My Part 1 handled the multiplication of A by the first 4 columns of B?
                # No, Part 1 was 4x4 * 4x4. This produces the first 4 columns of C.
                # Part 2 is 4x4 * 4x2. This produces the last 2 columns of C.
                
                # So for Part 1, k should be 0..3.
                # But my Strassen construction above assumed B is 4x4.
                # The indices in V for Part 1 should only cover k=0..3.
                # The indices in W for Part 1 should only cover k=0..3.
                
                # Let's re-verify the index mapping.
                # V shape (m*p, rank) = (24, rank).
                # In Part 1, we used V[j*p+k, R].
                # If p=6, and we are computing the first 4 columns, k is 0..3.
                # So V[j*6+k, R] for k in 0..3.
                # My code above: V[j * p + k, R]. With p=6, this is correct for k=0..3.
                
                # W shape (p*n, rank) = (24, rank).
                # In Part 1, we used W[k*n+i, R].
                # k is 0..3.
                # So W[k*4+i, R].
                # My code above: W[k * n + i, R]. With n=4, this is correct for k=0..3.
                
                pass
    
    # Now fill Part 2: Schoolbook for columns k=4,5
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                R = col_offset + (i * m + j) * 2 + (k - 4)
                # i: 0..3, j: 0..3, k: 4,5
                # Total 4*4*2 = 32 terms.
                U[i * m + j, R] = 1
                V[j * p + k, R] = 1
                W[k * n + i, R] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
