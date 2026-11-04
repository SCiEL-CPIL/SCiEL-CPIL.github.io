import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape to 3D for einsum
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    T = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3)
    
    M = np.ones((n, m, p))
    err = np.max(np.abs(T - M))
    return err < tol

def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[0,3]=-1; V2[2,3]=1; W2[0,3]=1; W2[1,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    U2[0,5]=-1; U2[2,5]=1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    
    # Reshape U2, V2, W2 to (2, 2, 7)
    # Index mapping: u = 2*i + j
    # U2[i, j, r]
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # Construct 4x4x4 factors (rank 49)
    # U4[i, j, r_o, r_i] = U2[i_o, j_o, r_o] * U2[i_i, j_i, r_i]
    # i = 2*i_o + i_i, j = 2*j_o + j_i
    # r = 7*r_o + r_i
    
    # U4 shape: (4, 4, 49) -> (2, 2, 2, 2, 7, 7)
    # einsum: 'ijr,kl s->ikjl rs'
    # U2_3d is (i_o, j_o, r_o)
    # U2_3d is (i_i, j_i, r_i)
    # Result: (i_o, i_i, j_o, j_i, r_o, r_i)
    U4_3d = np.einsum('ijr,kl s->ikjl rs', U2_3d, U2_3d)
    U4 = U4_3d.reshape(16, 49)
    
    # V4[j, k, r_o, r_i] = V2[j_o, k_o, r_o] * V2[j_i, k_i, r_i]
    # j = 2*j_o + j_i, k = 2*k_o + k_i
    # V2_3d is (j, k, r)
    # Result: (j_o, j_i, k_o, k_i, r_o, r_i)
    V4_3d = np.einsum('jk r,lm s->jlkm rs', V2_3d, V2_3d)
    V4 = V4_3d.reshape(16, 49)
    
    # W4[k, i, r_o, r_i] = W2[k_o, i_o, r_o] * W2[k_i, i_i, r_i]
    # k = 2*k_o + k_i, i = 2*i_o + i_i
    # W2_3d is (k, i, r)
    # Result: (k_o, k_i, i_o, i_i, r_o, r_i)
    W4_3d = np.einsum('ki r,lm s->klmi rs', W2_3d, W2_3d)
    W4 = W4_3d.reshape(16, 49)
    
    # Assemble 4x4x8 (rank 98)
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left half
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
            W[k*4 + j, 0:49] = W4[k*4 + j, :]
            
    # Right half
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + 4 + k, 49:98] = V4[j*4 + k, :]
            W[(4+k)*4 + j, 49:98] = W4[k*4 + j, :]
            
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Tensor decomposition verification failed")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
