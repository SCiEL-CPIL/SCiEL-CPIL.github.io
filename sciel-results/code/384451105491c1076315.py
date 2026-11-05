import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Base decompositions
    # Returns list of (u_idx, v_idx, w_idx, coeff)
    # Local indices:
    # U: i*M + j
    # V: j*P + k
    # W: k*N + i
    
    def get_strassen_222():
        # N=2, M=2, P=2
        # U size 4, V size 4, W size 4
        # i, j in 0..1; k in 0..1
        # Strassen's 7 terms
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to express C11, C12, C21, C22 in terms of these.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U corresponds to A, V to B, W to C.
        # The tensor equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = C[i,k]
        # Wait, the standard decomposition is C = sum U_r V_r W_r^T?
        # The prompt says: T[i,k,j] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This means the term is u_i * v_j * w_k.
        # So for each term r, we have coefficients u_{i,j}, v_{j,k}, w_{k,i}.
        
        # Let's list the 7 terms as (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs is a dict or list of 4 values (for A11, A12, A21, A22)
        # v_coeffs is a dict or list of 4 values (for B11, B12, B21, B22)
        # w_coeffs is a dict or list of 4 values (for C11, C12, C21, C22)
        
        # Note: Local indices for 2x2:
        # U: 0->A11, 1->A12, 2->A21, 3->A22
        # V: 0->B11, 1->B12, 2->B21, 3->B22
        # W: 0->C11, 1->C12, 2->C21, 3->C22 (since W[k*n+i], N=2. k=0,i=0->C11; k=0,i=1->C21? No.
        # W index = k*N + i.
        # k=0, i=0 -> 0 (C[0,0])
        # k=0, i=1 -> 1 (C[1,0])
        # k=1, i=0 -> 2 (C[0,1])
        # k=1, i=1 -> 3 (C[1,1])
        # So W indices: 0:C11, 1:C21, 2:C12, 3:C22.
        # Let's verify: C is 2x2. C[i,k].
        # i=0, k=0 -> C11. W idx 0*2+0=0.
        # i=1, k=0 -> C21. W idx 0*2+1=1.
        # i=0, k=1 -> C12. W idx 1*2+0=2.
        # i=1, k=1 -> C22. W idx 1*2+1=3.
        # Correct.
        
        terms = []
        
        # M1: (A11+A22)(B11+B22) -> C11, C22
        # U: A11=1, A22=1. V: B11=1, B22=1.
        # W: C11=1, C22=1.
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C11=1, C22=1. Indices 0 and 3.
        terms.append((u, v, w))
        
        # M2: (A21+A22)B11 -> C21
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 1, 0, 0] # C21=1. Index 1.
        terms.append((u, v, w))
        
        # M3: A11(B21-B22) -> C12, C21
        u = [1, 0, 0, 0]
        v = [0, 0, 1, -1]
        w = [0, 1, 1, 0] # C12=1 (idx 2), C21=1 (idx 1).
        terms.append((u, v, w))
        
        # M4: A22(B12-B11) -> C11, C21
        u = [0, 0, 0, 1]
        v = [-1, 1, 0, 0]
        w = [1, 1, 0, 0] # C11=1 (idx 0), C21=1 (idx 1).
        terms.append((u, v, w))
        
        # M5: (A11+A12)B22 -> C11, C12
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [1, 0, 1, 0] # C11=1 (idx 0), C12=1 (idx 2).
        terms.append((u, v, w))
        
        # M6: (A21-A11)(B11+B12) -> C22
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C22=1 (idx 3).
        terms.append((u, v, w))
        
        # M7: (A12-A22)(B21+B22) -> C11, C22
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 1] # C11=1 (idx 0), C22=1 (idx 3).
        terms.append((u, v, w))
        
        return terms

    def get_schoolbook(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    terms.append((u, v, w))
        return terms

    # Assemble 2x2x3 (Rank 11)
    def get_223():
        # N=2, M=2, P=3
        # Split P into 2 and 1.
        # Block 1: 2x2x2 (Strassen)
        # Block 2: 2x2x1 (Schoolbook)
        terms_222 = get_strassen_222()
        terms_221 = get_schoolbook(2, 2, 1)
        
        # We need to map local indices to 2x2x3 indices.
        # U: 2x2 (same)
        # V: 2x3. j in 0..1, k in 0..2.
        # W: 3x2. k in 0..2, i in 0..1.
        
        # For 2x2x2 block (k in 0..1):
        # U local: i*2+j. Global U: i*2+j.
        # V local: j*2+k. Global V: j*3+k.
        # W local: k*2+i. Global W: k*2+i.
        
        combined = []
        for u, v, w in terms_222:
            new_u = [0]*4
            new_v = [0]*6
            new_w = [0]*6
            
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                new_u[idx] = u[idx]
                
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                new_v[j*3 + k] = v[idx]
                
            for idx in range(4):
                k = idx // 2
                i = idx % 2
                new_w[k*2 + i] = w[idx]
                
            combined.append((new_u, new_v, new_w))
            
        for u, v, w in terms_221:
            new_u = [0]*4
            new_v = [0]*6
            new_w = [0]*6
            
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                new_u[idx] = u[idx]
                
            for idx in range(2):
                j = idx // 1
                k = idx % 1
                # k is 0 in local 2x2x1? No, P=1.
                # Local V index j*1+0 = j.
                # Global V index j*3 + 2 (since k=2 is the 3rd element, 0-indexed 2).
                # Wait, split P=3 into P1=2 (k=0,1) and P2=1 (k=2).
                # So for the 2x2x1 block, k_local=0 corresponds to k_global=2.
                new_v[j*3 + 2] = v[idx]
                
            for idx in range(2):
                k = idx // 1
                i = idx % 1
                # Local W index k*1+i = i.
                # Global W index k_global*2 + i. k_global=2.
                new_w[2*2 + i] = w[idx]
                
            combined.append((new_u, new_v, new_w))
            
        return combined

    # Assemble 2x2x4 (Rank 14)
    def get_224():
        # Split P=4 into 2 and 2.
        terms_222_1 = get_strassen_222()
        terms_222_2 = get_strassen_222()
        
        combined = []
        # Block 1: k in 0,1
        for u, v, w in terms_222_1:
            new_u = [0]*4
            new_v = [0]*8
            new_w = [0]*8
            
            for idx in range(4):
                new_u[idx] = u[idx]
                
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                new_v[j*4 + k] = v[idx]
                
            for idx in range(4):
                k = idx // 2
                i = idx % 2
                new_w[k*2 + i] = w[idx]
                
            combined.append((new_u, new_v, new_w))
            
        # Block 2: k in 2,3
        for u, v, w in terms_222_2:
            new_u = [0]*4
            new_v = [0]*8
            new_w = [0]*8
            
            for idx in range(4):
                new_u[idx] = u[idx]
                
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                new_v[j*4 + (k+2)] = v[idx]
                
            for idx in range(4):
                k = idx // 2
                i = idx % 2
                new_w[(k+2)*2 + i] = w[idx]
                
            combined.append((new_u, new_v, new_w))
            
        return combined

    # Schoolbook for 1x2x3
    def get_123():
        return get_schoolbook(1, 2, 3)
        
    # Schoolbook for 1x2x4
    def get_124():
        return get_schoolbook(1, 2, 4)

    # Global Assembly
    # N=3, M=4, P=7
    # Split N: 2, 1
    # Split M: 2, 2
    # Split P: 3, 4
    
    U_global = np.zeros((12, 0)) # n*m
    V_global = np.zeros((28, 0)) # m*p
    W_global = np.zeros((21, 0)) # p*n
    
    def add_block(terms, n_off, m_off, p_off, n_dim, m_dim, p_dim):
        nonlocal U_global, V_global, W_global
        if len(terms) == 0: return
        rank_block = len(terms)
        U_block = np.zeros((n_dim*m_dim, rank_block))
        V_block = np.zeros((m_dim*p_dim, rank_block))
        W_block = np.zeros((p_dim*n_dim, rank_block))
        
        for r, (u, v, w) in enumerate(terms):
            for idx in range(n_dim*m_dim):
                U_block[idx, r] = u[idx]
            for idx in range(m_dim*p_dim):
                V_block[idx, r] = v[idx]
            for idx in range(p_dim*n_dim):
                W_block[idx, r] = w[idx]
                
        # Map to global
        # U global index: i*4 + j. i in [n_off, n_off+n_dim), j in [m_off, m_off+m_dim)
        # V global index: j*7 + k. j in [m_off, m_off+m_dim), k in [p_off, p_off+p_dim)
        # W global index: k*3 + i. k in [p_off, p_off+p_dim), i in [n_off, n_off+n_dim)
        
        U_new = np.zeros((12, rank_block))
        V_new = np.zeros((28, rank_block))
        W_new = np.zeros((21, rank_block))
        
        for i in range(n_dim):
            for j in range(m_dim):
                U_new[(n_off+i)*4 + (m_off+j), :] = U_block[i*m_dim + j, :]
                
        for j in range(m_dim):
            for k in range(p_dim):
                V_new[(m_off+j)*7 + (p_off+k), :] = V_block[j*p_dim + k, :]
                
        for k in range(p_dim):
            for i in range(n_dim):
                W_new[(p_off+k)*3 + (n_off+i), :] = W_block[k*n_dim + i, :]
                
        U_global = np.hstack([U_global, U_new])
        V_global = np.hstack([V_global, V_new])
        W_global = np.hstack([W_global, W_new])

    # Block (0,0): 2x2x3. N_off=0, M_off=0, P_off=0
    add_block(get_223(), 0, 0, 0, 2, 2, 3)
    
    # Block (0,1): 2x2x3. N_off=0, M_off=2, P_off=0
    add_block(get_223(), 0, 2, 0, 2, 2, 3)
    
    # Block (1,0): 1x2x3. N_off=2, M_off=0, P_off=0
    add_block(get_123(), 2, 0, 0, 1, 2, 3)
    
    # Block (1,1): 1x2x3. N_off=2, M_off=2, P_off=0
    add_block(get_123(), 2, 2, 0, 1, 2, 3)
    
    # Block (0,2): 2x2x4. N_off=0, M_off=0, P_off=3
    add_block(get_224(), 0, 0, 3, 2, 2, 4)
    
    # Block (0,3): 2x2x4. N_off=0, M_off=2, P_off=3
    add_block(get_224(), 0, 2, 3, 2, 2, 4)
    
    # Block (1,2): 1x2x4. N_off=2, M_off=0, P_off=3
    add_block(get_124(), 2, 0, 3, 1, 2, 4)
    
    # Block (1,3): 1x2x4. N_off=2, M_off=2, P_off=3
    add_block(get_124(), 2, 2, 3, 1, 2, 4)

    return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}

if __name__ == "__main__":
    construct()
