import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = 50
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def a(i, j): return i*m + j
    def b(j, k): return j*p + k
    def c(k, i): return k*n + i
    
    col = 0
    # Process 4 blocks of 2 columns in B (k=0..7)
    for b_idx in range(4):
        k0 = 2 * b_idx
        k1 = k0 + 1
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2]
        # M1..M7
        
        # M1 = (A00+A11)(B0k0+B1k1)
        U[a(0,0), col] += 1; U[a(1,1), col] += 1
        V[b(0,k0), col] += 1; V[b(1,k1), col] += 1
        W[c(k0,0), col] += 1; W[c(k1,1), col] += 1
        
        # M2 = (A10+A11)B0k0
        U[a(1,0), col+1] += 1; U[a(1,1), col+1] += 1
        V[b(0,k0), col+1] += 1
        W[c(k0,1), col+1] += 1
        
        # M3 = A00(B1k0-B0k1) -- Wait, standard Strassen:
        # M3 = A00 (B0k1 - B1k0) ? No.
        # Standard Strassen for C = AB:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B10)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Let's stick to the exact indices in the original code to avoid errors.
        # The original code was verified to work. I will replicate its logic exactly in Numpy.
        
        # Re-initialize to be safe and clear
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0

        for b_idx in range(4):
            k0 = 2 * b_idx
            k1 = 2 * b_idx + 1
            
            v_e = b(0, k0)
            v_f = b(0, k1)
            v_g = b(1, k0)
            v_h = b(1, k1)
            
            u_a = a(0, 0)
            u_b = a(0, 1)
            u_c = a(1, 0)
            u_d = a(1, 1)
            
            # M1
            U[u_a, col] += 1.0; U[u_d, col] += 1.0
            V[v_e, col] += 1.0; V[v_h, col] += 1.0
            # M2
            U[u_c, col+1] += 1.0; U[u_d, col+1] += 1.0
            V[v_e, col+1] += 1.0
            # M3
            U[u_a, col+2] += 1.0
            V[v_f, col+2] += 1.0; V[v_h, col+2] -= 1.0
            # M4
            U[u_d, col+3] += 1.0
            V[v_g, col+3] += 1.0; V[v_e, col+3] -= 1.0
            # M5
            U[u_a, col+4] += 1.0; U[u_b, col+4] += 1.0
            V[v_h, col+4] += 1.0
            # M6
            U[u_c, col+5] += 1.0; U[u_a, col+5] -= 1.0
            V[v_e, col+5] += 1.0; V[v_f, col+5] += 1.0
            # M7
            U[u_b, col+6] += 1.0; U[u_d, col+6] -= 1.0
            V[v_g, col+6] += 1.0; V[v_h, col+6] += 1.0
            
            w_00 = c(k0, 0)
            w_01 = c(k1, 0)
            w_10 = c(k0, 1)
            w_11 = c(k1, 1)
            
            W[w_00, col]   += 1.0
            W[w_00, col+3] += 1.0
            W[w_00, col+4] -= 1.0
            W[w_00, col+6] += 1.0
            W[w_01, col+2] += 1.0
            W[w_01, col+4] += 1.0
            W[w_10, col+1] += 1.0
            W[w_10, col+3] += 1.0
            W[w_11, col]   += 1.0
            W[w_11, col+1] -= 1.0
            W[w_11, col+2] += 1.0
            W[w_11, col+5] += 1.0
            
            col += 7

        # k=8
        k8 = 8
        v_e = b(0, k8)
        v_g = b(1, k8)
        
        U[a(0,0), col]   += 1.0; V[v_e, col]   += 1.0
        U[a(0,1), col+1] += 1.0; V[v_g, col+1] += 1.0
        U[a(1,0), col+2] += 1.0; V[v_e, col+2] += 1.0
        U[a(1,1), col+3] += 1.0; V[v_g, col+3] += 1.0
        
        W[c(k8,0), col]   += 1.0
        W[c(k8,0), col+1] += 1.0
        W[c(k8,1), col+2] += 1.0
        W[c(k8,1), col+3] += 1.0
        col += 4

        # Part 2
        for k in range(9):
            v_b = b(2, k)
            U[a(0,2), col]   += 1.0; V[v_b, col]   += 1.0
            W[c(k,0), col] += 1.0
            U[a(1,2), col+1] += 1.0; V[v_b, col+1] += 1.0
            W[c(k,1), col+1] += 1.0
            col += 2

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
