import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    t_start = time.time()
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # The rank of the tensor M_{n,m,p} is at least max(n*m, m*p, p*n).
    # For n=2, m=2, p=13, lower bound is max(4, 26, 26) = 26.
    # However, the tensor is the direct sum of p copies of M_{n,m,1}.
    # M_{n,m,1} corresponds to multiplying an n x m matrix by an m x 1 vector.
    # The rank of M_{n,m,1} is n*m.
    # Proof: The tensor is T_{i,j,j',i'} = delta_{j,j'} delta_{i,i'}.
    # This requires n*m rank-1 terms (one for each pair (i,j) effectively, 
    # or rather, it's the identity map on the m-dimension tensor product with 
    # identity on n-dimension? No, it's the standard matrix-vector mult).
    # Specifically, c_i = sum_j A_{i,j} b_j.
    # This decomposes into n*m terms: sum_{i,j} (e_i e_j^T) (e_j) (e_i^T).
    # Wait, rank 1 term: U_{i',j'} = delta_{i',i}delta_{j',j}, V_{j'',k''}=delta_{j'',j}, W_{k''',i''''}=delta_{i'''',i}.
    # Sum over i,j.
    # So rank(M_{n,m,1}) = n*m.
    # Here n=2, m=2, so rank(M_{2,2,1}) = 4.
    # Total rank = p * rank(M_{2,2,1}) = 13 * 4 = 52.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Fast verification
    # T[i*m+j, j'*p+k', k''*n+i'] = 1 if j==j' and k==k' and i==i'
    # We check if reconstruction matches this.
    
    # Instead of full 4x26x26 tensor construction which is small anyway,
    # we can just check a few random points or the whole thing.
    # 4*26*26 = 2704 elements. Very fast.
    
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(recon, T, atol=1e-10):
        raise RuntimeError("Decomposition verification failed")
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
