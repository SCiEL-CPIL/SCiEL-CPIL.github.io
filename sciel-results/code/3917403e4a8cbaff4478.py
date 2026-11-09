import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 5, 6
    
    # We will construct a decomposition for M_{4,6,6} and restrict it to M_{4,5,6}
    # M_{4,6,6} = M_{2,2,3} (otimes) M_{2,3,2}
    
    # 1. Decomposition for M_{2,2,3}
    # Use schoolbook for M_{2,2,3} (Rank 12)
    # Or use M_{2,2,4} (Rank 14) restricted? 12 is better.
    # Let's use schoolbook for M_{2,2,3} to be safe and simple.
    n1, m1, p1 = 2, 2, 3
    rank1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
                
    # 2. Decomposition for M_{2,3,2}
    # Use Kronecker M_{2,1,2} (Rank 2) (otimes) M_{1,3,1} (Rank 3)
    # M_{2,1,2}
    n2a, m2a, p2a = 2, 1, 2
    rank2a = n2a
    U2a = np.zeros((n2a * m2a, rank2a))
    V2a = np.zeros((m2a * p2a, rank2a))
    W2a = np.zeros((p2a * n2a, rank2a))
    for i in range(n2a):
        U2a[i * m2a + 0, i] = 1
        for k in range(p2a):
            V2a[0 * p2a + k, i] = 1 if k == i else 0 # Wait, V index is j*p+k. j=0.
            # V[0*k, i] = 1 if k==i?
            # Check M_{2,1,2}:
            # U[i, i]=1.
            # V[k, i]=1 if k==i?
            # W[2i+k, i]=1.
            # Sum U[i,r] V[k,r] W[2i+k,r].
            # r=i: 1 * V[k,i] * W[2i+k,i].
            # Need V[k,i] W[2i+k,i] = 1 if k=k (always) and i=i.
            # So V[k,i] must be 1 for all k?
            # If V[k,i]=1, then W[2i+k,i] must be 1.
            # Then sum is 1.
            # Zero conditions:
            # If i != i', U[i',r]=0.
            # So V[k,i] can be 1 for all k.
            pass
    # Let's re-verify M_{2,1,2} rank 2.
    # U[i, i] = 1.
    # V[k, i] = 1 for all k.
    # W[2i+k, i] = 1.
    # Sum for (i', k', w'):
    # r=i'. U[i', i']=1.
    # Term: V[k', i'] W[w', i'].
    # V[k', i'] = 1.
    # W[w', i'] = 1 if w' = 2i' + k'.
    # So sum is 1 if w' = 2i' + k'.
    # This matches the tensor condition.
    # So:
    for i in range(n2a):
        U2a[i, i] = 1
        for k in range(p2a):
            V2a[k, i] = 1
            W2a[2 * i + k, i] = 1
            
    # M_{1,3,1}
    n2b, m2b, p2b = 1, 3, 1
    rank2b = m2b
    U2b = np.zeros((n2b * m2b, rank2b))
    V2b = np.zeros((m2b * p2b, rank2b))
    W2b = np.zeros((p2b * n2b, rank2b))
    for j in range(m2b):
        U2b[0 * m2b + j, j] = 1
        V2b[j * p2b + 0, j] = 1
        W2b[0 * n2b + 0, j] = 1
        
    # Combine M_{2,1,2} and M_{1,3,1} to get M_{2,3,2}
    n2, m2, p2 = n2a * n2b, m2a * m2b, p2a * p2b
    rank2 = rank2a * rank2b
    U2 = np.kron(U2a, U2b)
    V2 = np.kron(V2a, V2b)
    W2 = np.kron(W2a, W2b)
    
    # Combine M_{2,2,3} and M_{2,3,2} to get M_{4,6,6}
    n_full, m_full, p_full = n1 * n2, m1 * m2, p1 * p2
    rank_full = rank1 * rank2
    U_full = np.kron(U1, U2)
    V_full = np.kron(V1, V2)
    W_full = np.kron(W1, W2)
    
    # Restrict M_{4,6,6} to M_{4,5,6}
    # m_full = 6, target m = 5.
    # Set j=5 block to 0 in U and V.
    # U index: i * 6 + j. j=5 -> i*6+5.
    for i in range(n_full):
        U_full[i * m_full + 5, :] = 0
    # V index: j * 6 + k. j=5 -> 5*6+k = 30+k.
    for k in range(p_full):
        V_full[5 * p_full + k, :] = 0
        
    # Convert to lists of lists
    U_out = U_full.tolist()
    V_out = V_full.tolist()
    W_out = W_full.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
