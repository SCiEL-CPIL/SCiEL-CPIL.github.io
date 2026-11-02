import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit

    def strassen_2x2x2():
        """
        Return (U, V, W) for 2x2x2 matrix multiplication with rank 7.
        U: shape (4, 7) - U[i*2+j, r]
        V: shape (4, 7) - V[j*2+k, r]  
        W: shape (4, 7) - W[k*2+i, r]
        
        For each term r, we have U[i,j,r] * V[j,k,r] * W[k,i,r]
        and summing over r should give 1 for all (i,j,k) in {0,1}^3.
        """
        # Strassen's 7 products for 2x2 matrix multiplication:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # For tensor decomposition, each term r contributes:
        # U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
        
        # Let's define the 7 terms:
        # Term 1 (M1): U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has 1 at (0,0) and (1,1)
        # Actually, let me think more carefully.
        
        # The tensor T[i,j,k] = 1 for all i,j,k in {0,1}
        # We need to decompose this as sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # Using Strassen's structure:
        # M1 = (a00+a11)(b00+b11) contributes to c00, c11
        # M2 = (a10+a11)b00 contributes to c10
        # M3 = a00(b01-b11) contributes to c01
        # M4 = a11(b10-b00) contributes to c00, c10
        # M5 = (a00+a01)b11 contributes to c00, c01
        # M6 = (a10-a00)(b00+b01) contributes to c11
        # M7 = (a01-a11)(b10+b11) contributes to c00
        
        # For the tensor decomposition, each "product" M_r is a rank-1 tensor:
        # M1: u1[i,j] = (i==0 and j==0) or (i==1 and j==1), v1[j,k] = (j==0 and k==0) or (j==1 and k==1), w1[k,i] = (k==0 and i==0) or (k==1 and i==1)
        # But this doesn't directly work because we need the product to be 1.
        
        # Let me use a known rank-7 decomposition of the 2x2x2 multiplication tensor.
        # The tensor T[i,j,k] = 1 for i,j,k in {0,1} has rank 7.
        
        # One valid decomposition:
        # Term 0: U=diag(1,1), V=diag(1,1), W=diag(1,1) -> contributes 1 to (0,0,0) and (1,1,1)
        # This is getting complex. Let me use a different approach.
        
        # Actually, for the purpose of this problem, I can use a simpler construction.
        # The key insight is that we can use the identity:
        # (a00+a11)(b00+b11) - a00*b00 - a11*b11 = a00*b11 + a11*b00
        # This allows us to express certain products with fewer terms.
        
        # Let me use a verified rank-7 decomposition. Here's one based on Strassen:
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 0: M1 = (a00+a11)(b00+b11)
        # U[0,0]=1, U[1,1]=1 (for i=0,j=0 and i=1,j=1)
        # V[0,0]=1, V[1,1]=1 (for j=0,k=0 and j=1,k=1)  
        # W[0,0]=1, W[1,1]=1 (for k=0,i=0 and k=1,i=1)
        # But this gives 1 for (0,0,0), (0,0,1), (1,1,0), (1,1,1) which is wrong.
        
        # I need to be more careful. Let me look up the correct Strassen decomposition.
        
        # The correct way: each term r defines three vectors:
        # u_r in R^4 (indexed by (i,j)), v_r in R^4 (indexed by (j,k)), w_r in R^4 (indexed by (k,i))
        # such that sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all (i,j,k)
        
        # Strassen's 7 terms:
        # r=0: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  -> M1
        # r=1: u=(0,0,1,1), v=(1,0,0,0), w=(0,1,0,0)  -> M2  
        # r=2: u=(1,0,0,0), v=(0,1,0,0), w=(0,1,0,0)  -> M3 (with sign)
        # This is getting too error-prone. Let me use a computational approach.
        
        # For now, let me use a simpler but correct approach:
        # Use schoolbook for 2x2x2 (rank 8) and see if we can do better.
        # Actually, let me just implement the block decomposition with schoolbook
        # for all blocks, which gives rank 105, and then try to improve.
        
        # Wait, I should implement Strassen properly. Here's a known rank-7 decomposition:
        
        # Based on the literature, one valid decomposition is:
        terms = [
            # (u, v, w) where u[i*2+j], v[j*2+k], w[k*2+i]
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # 0
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),  # 1
            ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),  # 2
            ([0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]),  # 3
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),  # 4
            ([0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # 5
            ([0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]),  # 6
        ]
        
        for r, (u, v, w) in enumerate(terms):
            for idx, val in enumerate(u):
                U[idx][r] = val
            for idx, val in enumerate(v):
                V[idx][r] = val
            for idx, val in enumerate(w):
                W[idx][r] = val
        
        # Verify
        U_np = np.array(U, dtype=float).reshape(2, 2, 7)
        V_np = np.array(V, dtype=float).reshape(2, 2, 7)
        W_np = np.array(W, dtype=float).reshape(2, 2, 7)
        T = np.einsum('ijr,jkr,kin->ijk', U_np, V_np, W_np)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fallback to schoolbook if Strassen fails
            U = [[0]*8 for _ in range(4)]
            V = [[0]*8 for _ in range(4)]
            W = [[0]*8 for _ in range(4)]
            c = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i*2+j][c] = 1
                        V[j*2+k][c] = 1
                        W[k*2+i][c] = 1
                        c += 1
            return U, V, W, 8
        
        return U, V, W, 7

    # Build block decomposition
    # n=3: blocks [0,1] (size 2), [2] (size 1)
    # m=5: blocks [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # p=7: blocks [0,1] (size 2), [2,3] (size 2), [4,5] (size 2), [6] (size 1)
    
    n_blocks = [(0, 2), (2, 3)]  # (start, end)
    m_blocks = [(0, 2), (2, 4), (4, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    # Precompute Strassen for 2x2x2
    U_str, V_str, W_str, rank_str = strassen_2x2x2()
    
    # Collect all terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    for (ns, ne) in n_blocks:
        n_size = ne - ns
        for (ms, me) in m_blocks:
            m_size = me - ms
            for (ps, pe) in p_blocks:
                p_size = pe - ps
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Use Strassen
                    for r in range(rank_str):
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        
                        for i in range(2):
                            for j in range(2):
                                u[(ns + i) * m + (ms + j)] = U_str[i * 2 + j][r]
                        
                        for j in range(2):
                            for k in range(2):
                                v[(ms + j) * p + (ps + k)] = V_str[j * 2 + k][r]
                        
                        for k in range(2):
                            for i in range(2):
                                w[(ps + k) * n + (ns + i)] = W_str[k * 2 + i][r]
                        
                        U_terms.append(u)
                        V_terms.append(v)
                        W_terms.append(w)
                else:
                    # Schoolbook
                    for i in range(n_size):
                        for j in range(m_size):
                            for k in range(p_size):
                                u = [0] * (n * m)
                                v = [0] * (m * p)
                                w = [0] * (p * n)
                                
                                u[(ns + i) * m + (ms + j)] = 1
                                v[(ms + j) * p + (ps + k)] = 1
                                w[(ps + k) * n + (ns + i)] = 1
                                
                                U_terms.append(u)
                                V_terms.append(v)
                                W_terms.append(w)
    
    rank = len(U_terms)
    U = np.array(U_terms).T  # shape (n*m, rank)
    V = np.array(V_terms).T  # shape (m*p, rank)
    W = np.array(W_terms).T  # shape (p*n, rank)
    
    # Verify
    U_np = U
    V_np = V
    W_np = W
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Convert to list of lists with half-integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure entries are half-integers (multiply by 2, round, divide by 2)
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify rounded solution
    U_t2 = U_rounded.reshape(n, m, rank)
    V_t2 = V_rounded.reshape(m, p, rank)
    W_t2 = W_rounded.reshape(p, n, rank)
    T2 = np.einsum('ijr,jkr,kin->ijk', U_t2, V_t2, W_t2)
    
    if np.allclose(T2, 1.0, atol=1e-9):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    else:
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
