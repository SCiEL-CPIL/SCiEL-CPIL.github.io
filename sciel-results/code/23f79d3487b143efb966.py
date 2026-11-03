import numpy as np
import time

def construct():
    """
    Constructs a rank-84 exact tensor decomposition for 3x5 * 5x6 matrix multiplication
    using Strassen's algorithm for 2x2x2 sub-blocks.
    """
    n, m, p = 3, 5, 6
    
    # Define the partitions
    # n: [2, 1]
    # m: [2, 2, 1]
    # p: [2, 2, 2]
    
    n_splits = [(0, 2), (2, 3)]
    m_splits = [(0, 2), (2, 4), (4, 5)]
    p_splits = [(0, 2), (2, 4), (4, 6)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def get_strassen_2x2():
        """
        Returns U, V, W for 2x2x2 Strassen decomposition.
        Shapes: (4, 7), (4, 7), (4, 7)
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_schoolbook(n_b, m_b, p_b):
        """
        Returns U, V, W for n_b x m_b x p_b schoolbook decomposition.
        Shapes: (n_b*m_b, rank), (m_b*p_b, rank), (p_b*n_b, rank)
        """
        rank = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, rank))
        V = np.zeros((m_b * p_b, rank))
        W = np.zeros((p_b * n_b, rank))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j, col] = 1
                    V[j * p_b + k, col] = 1
                    W[k * n_b + i, col] = 1
                    col += 1
        return U, V, W

    # Iterate through all blocks
    for n_start, n_end in n_splits:
        n_b = n_end - n_start
        for m_start, m_end in m_splits:
            m_b = m_end - m_start
            for p_start, p_end in p_splits:
                p_b = p_end - p_start
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    Ub, Vb, Wb = get_strassen_2x2()
                else:
                    Ub, Vb, Wb = get_schoolbook(n_b, m_b, p_b)
                
                rank_b = Ub.shape[1]
                
                # Create full-sized zero matrices for this block's contribution
                U_block = np.zeros((n * m, rank_b))
                V_block = np.zeros((m * p, rank_b))
                W_block = np.zeros((p * n, rank_b))
                
                # Fill U_block
                for i in range(n_b):
                    for j in range(m_b):
                        U_block[(n_start + i) * m + (m_start + j), :] = Ub[i * m_b + j, :]
                        
                # Fill V_block
                for j in range(m_b):
                    for k in range(p_b):
                        V_block[(m_start + j) * p + (p_start + k), :] = Vb[j * p_b + k, :]
                        
                # Fill W_block
                for k in range(p_b):
                    for i in range(n_b):
                        W_block[(p_start + k) * n + (n_start + i), :] = Wb[k * n_b + i, :]
                        
                U_cols.append(U_block)
                V_cols.append(V_block)
                W_cols.append(W_block)
    
    U = np.hstack(U_cols)
    V = np.hstack(V_cols)
    W = np.hstack(W_cols)
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

if __name__ == '__main__':
    # Simple verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = 3, 5, 6
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    print("Max error:", np.max(np.abs(T - 1.0)))
    print("Rank:", r)
