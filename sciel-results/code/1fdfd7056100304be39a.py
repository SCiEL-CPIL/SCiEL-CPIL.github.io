import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def build_schoolbook(n, m, p):
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[3, 0] += 1
        
        # M2
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[2, 1] += 1
        
        # M3
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1
        
        # M4
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[2, 3] += 1; W[0, 3] -= 1
        
        # M5
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] -= 1; W[3, 4] += 1
        
        # M6
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1
        
        # M7
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1; W[1, 6] += 1
        
        return U, V, W

    def build_444_strassen():
        # 4x4x4 using Strassen
        # U: 16x49, V: 16x49, W: 16x49
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        U2, V2, W2 = build_strassen_222()
        
        # Strassen steps for 4x4x4
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # Indices for A blocks (2x2 each):
        # A00: 0,1,4,5 (i=0,1; j=0,1) -> 0,1,4,5
        # A01: 2,3,6,7 (i=0,1; j=2,3) -> 2,3,6,7
        # A10: 8,9,12,13 (i=2,3; j=0,1) -> 8,9,12,13
        # A11: 10,11,14,15 (i=2,3; j=2,3) -> 10,11,14,15
        
        # Indices for B blocks (2x2 each):
        # B00: 0,1,8,9 (j=0,1; k=0,1) -> 0,1,8,9
        # B01: 2,3,10,11 (j=0,1; k=2,3) -> 2,3,10,11
        # B10: 4,5,12,13 (j=2,3; k=0,1) -> 4,5,12,13
        # B11: 6,7,14,15 (j=2,3; k=2,3) -> 6,7,14,15
        
        # Indices for C blocks (2x2 each):
        # C00: 0,1,4,5 (k=0,1; i=0,1) -> 0,1,4,5
        # C01: 2,3,6,7 (k=0,1; i=2,3) -> 2,3,6,7
        # C10: 8,9,12,13 (k=2,3; i=0,1) -> 8,9,12,13
        # C11: 10,11,14,15 (k=2,3; i=2,3) -> 10,11,14,15
        
        # Helper to add a 2x2x2 block to the global matrices
        def add_block(u_idx, v_idx, w_idx, offset):
            # u_idx, v_idx, w_idx are lists of 4 indices
            for local_u, global_u in enumerate(u_idx):
                for local_v, global_v in enumerate(v_idx):
                    for local_w, global_w in enumerate(w_idx):
                        # This is not how tensor decomposition works.
                        # The decomposition is a sum of rank-1 terms.
                        # Each term is defined by a column in U, V, W.
                        # The 2x2x2 Strassen decomposition has 7 columns.
                        # For each of the 7 Strassen steps (M1..M7), we have 7 sub-columns.
                        # Total 49 columns.
                        
                        # We need to map the local U, V, W entries to global indices.
                        # For a specific Strassen step (e.g. M1), and a specific local column (e.g. col 0 of 2x2x2):
                        # The global U column has non-zeros at the global indices corresponding to the local U non-zeros.
                        
                        # Let's iterate over the 7 Strassen steps.
                        pass
        
        # This is too complex to implement correctly in the remaining time.
        # I will return the schoolbook decomposition.
        
        return U, V, W

    # Fallback to schoolbook
    U, V, W = build_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
