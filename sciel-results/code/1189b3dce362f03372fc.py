import time
import numpy as np

def construct():
    start_time = time.time()
    
    n, m, p = 3, 4, 5
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val
        col += 1

    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds 7 Strassen terms for a 2x2x2 block.
        Local indices i', j', k' in {0,1}.
        Global i = i_base + i', j = j_base + j', k = k_base + k'.
        """
        # Strassen coefficients K_q for C_ik
        # K_q is a 2x2 matrix where K_q[i'][k'] is the coefficient of M_q in C_{i' k'}
        
        # M1: (A00+A11)(B00+B11) -> C00:1, C11:1. K1 = [[1,0],[0,1]]
        K1 = np.array([[1, 0], [0, 1]])
        # M2: (A10+A11)B00 -> C10:1, C11:-1. K2 = [[0,0],[1,-1]]
        K2 = np.array([[0, 0], [1, -1]])
        # M3: A00(B01-B11) -> C01:1. K3 = [[0,1],[0,0]]
        K3 = np.array([[0, 1], [0, 0]])
        # M4: A11(B10-B00) -> C00:-1, C10:1. K4 = [[-1,0],[1,0]]
        K4 = np.array([[-1, 0], [1, 0]])
        # M5: (A00+A01)B11 -> C01:1, C11:-1. K5 = [[0,1],[0,-1]]
        K5 = np.array([[0, 1], [0, -1]])
        # M6: (A10-A00)(B00+B01) -> C11:1. K6 = [[0,0],[0,1]]
        K6 = np.array([[0, 0], [0, 1]])
        # M7: (A01-A11)(B10+B11) -> C00:1, C10:-1. K7 = [[1,-1],[0,0]]
        K7 = np.array([[1, -1], [0, 0]])
        
        Ks = [K1, K2, K3, K4, K5, K6, K7]
        
        for K in Ks:
            # We use U_q = K, V_q = I, W_q = 1
            # U is indexed by (i, j). K is indexed by (i', k').
            # Wait. U_q[i', j'] * V_q[j', k'] = K[i', k'].
            # If V_q = I, then V_q[j', k'] = 1 if j'=k' else 0.
            # Then (U_q V_q)[i', k'] = sum_j' U_q[i', j'] delta_{j', k'} = U_q[i', k'].
            # So U_q must be K.
            
            for i_prime in range(2):
                for j_prime in range(2):
                    u_val = K[i_prime, j_prime]
                    if u_val != 0:
                        i_global = i_base + i_prime
                        j_global = j_base + j_prime
                        u_idx = i_global * m + j_global
                        # V_q is Identity: V[j', k'] = 1 if j'=k'
                        # So for this term, V is non-zero only when j_prime == k_prime?
                        # No, V is a matrix. We set V_q[j', k'] = 1 if j'=k'.
                        # But we are adding to a specific column.
                        # For this column, V_q[j_global, k_global] should be 1 if j_prime == k_prime.
                        # This means V is non-zero only on the diagonal of the block.
                        
                        # Let's just fill U, V, W explicitly for this term.
                        
                        # U part
                        U[u_idx, col] = u_val
                        
                        # V part: V[j', k'] = 1 if j'=k' else 0
                        for k_prime in range(2):
                            if j_prime == k_prime:
                                v_idx = j_global * p + (k_base + k_prime)
                                V[v_idx, col] = 1.0
                            else:
                                # V is 0, so we don't need to add it (already 0)
                                pass
                        
                        # W part: W[k', i'] = 1 for all k', i' in block
                        for k_prime in range(2):
                            for i_prime_w in range(2):
                                w_idx = (k_base + k_prime) * n + (i_base + i_prime_w)
                                W[w_idx, col] = 1.0
            
            col += 1

    # Define the 4 blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    
    # Add schoolbook terms for remaining parts
    # Remaining i: 2
    # Remaining k: 4 (for i in 0,1)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term was covered by Strassen
                # Strassen covered i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
                if i in [0, 1] and k in [0, 1, 2, 3]:
                    continue
                # Add schoolbook term
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    
    # Verify rank
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
