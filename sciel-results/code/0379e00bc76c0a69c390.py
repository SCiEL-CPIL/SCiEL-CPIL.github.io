"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START

def _verify_strassen(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    Convention: W[k*n + i, r]
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i, j, k (since (AB)_{ik} = sum_j A_{ij} B_{jk}).
    """
    rank = U.shape[1]
    # Build the tensor from the decomposition
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i * m + j, r]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j * p + k, r]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i, r]
                    T[i, j, k] += u_val * v_val * w_val
    
    # Check against identity tensor (all ones)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=1e-10)


def _strassen_decomposition():
    """Return the Strassen rank-7 decomposition for 2x2 matrix multiplication.
    
    Strassen's algorithm computes C = A * B where:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    
    The 7 intermediate products are:
    p1 = (a + d)(e + h)
    p2 = (c + d)e
    p3 = a(f - h)
    p4 = d(g - e)
    p5 = (a + b)h
    p6 = (c + a)(e + f)
    p7 = (b + d)(g + h)
    
    Then:
    i = p1 + p4 - p5 + p7
    j = p3 + p5
    k = p2 + p4
    l = p1 - p2 + p3 + p6
    
    In the tensor decomposition framework:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where n=m=p=2.
    
    For each product p_r, we need to express it as a product of a linear form in A,
    a linear form in B, and a linear form that extracts the coefficient for C.
    
    Let's index:
    A indices: i=0,1; j=0,1 -> A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    B indices: j=0,1; k=0,1 -> B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    C indices: k=0,1; i=0,1 -> C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l
    
    W[k*n+i, r] extracts the contribution to C[k,i] from product r.
    
    For p1 = (a+d)(e+h):
      U contributes: a+d -> U[0,0]=1, U[1,1]=1 (for i=0,j=0 and i=1,j=1)
      V contributes: e+h -> V[0,0]=1, V[1,1]=1 (for j=0,k=0 and j=1,k=1)
      W: p1 contributes to i (C[0,0]) with +1, j (C[0,1]) with 0, k (C[1,0]) with 0, l (C[1,1]) with -1
        Wait, let me re-check:
        i = p1 + p4 - p5 + p7
        j = p3 + p5
        k = p2 + p4
        l = p1 - p2 + p3 + p6
        
        So p1 contributes: C[0,0]+=1, C[0,1]+=0, C[1,0]+=0, C[1,1]+=1
        W[k*n+i, r]: W[0*2+0, r]=W[0,r]=1 (for C[0,0]=i), W[0*2+1,r]=W[1,r]=0 (for C[0,1]=j),
                     W[1*2+0,r]=W[2,r]=0 (for C[1,0]=k), W[1*2+1,r]=W[3,r]=1 (for C[1,1]=l)
    
    Let me build this carefully.
    
    n=m=p=2
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    A flat index: i*m+j -> i*2+j
      A[0,0]=a: idx 0
      A[0,1]=b: idx 1
      A[1,0]=c: idx 2
      A[1,1]=d: idx 3
    
    B flat index: j*p+k -> j*2+k
      B[0,0]=e: idx 0
      B[0,1]=f: idx 1
      B[1,0]=g: idx 2
      B[1,1]=h: idx 3
    
    C flat index: k*n+i -> k*2+i
      C[0,0]=i: idx 0
      C[0,1]=j: idx 1
      C[1,0]=k: idx 2
      C[1,1]=l: idx 3
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # p1 = (a + d)(e + h)
    # U: a=U[0,0], d=U[3,0]
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    # V: e=V[0,0], h=V[3,0]
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    # W: p1 -> C[0,0]+=1 (i), C[1,1]+=1 (l)
    W[0, 0] = 1  # C[0,0] = i
    W[3, 0] = 1  # C[1,1] = l
    
    # p2 = (c + d)e
    # U: c=U[2,1], d=U[3,1]
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    # V: e=V[0,1]
    V[0, 1] = 1  # e
    # W: p2 -> C[1,0]+=1 (k), C[1,1]+=-1 (l)
    W[2, 1] = 1  # C[1,0] = k
    W[3, 1] = -1  # C[1,1] = l
    
    # p3 = a(f - h)
    # U: a=U[0,2]
    U[0, 2] = 1  # a
    # V: f=V[1,2], h=V[3,2]
    V[1, 2] = 1  # f
    V[3, 2] = -1  # -h
    # W: p3 -> C[0,1]+=1 (j), C[1,1]+=1 (l)
    W[1, 2] = 1  # C[0,1] = j
    W[3, 2] = 1  # C[1,1] = l
    
    # p4 = d(g - e)
    # U: d=U[3,3]
    U[3, 3] = 1  # d
    # V: g=V[2,3], e=V[0,3]
    V[2, 3] = 1  # g
    V[0, 3] = -1  # -e
    # W: p4 -> C[0,0]+=1 (i), C[1,0]+=1 (k)
    W[0, 3] = 1  # C[0,0] = i
    W[2, 3] = 1  # C[1,0] = k
    
    # p5 = (a + b)h
    # U: a=U[0,4], b=U[1,4]
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    # V: h=V[3,4]
    V[3, 4] = 1  # h
    # W: p5 -> C[0,0]+=-1 (i), C[0,1]+=1 (j)
    W[0, 4] = -1  # C[0,0] = i
    W[1, 4] = 1  # C[0,1] = j
    
    # p6 = (c + a)(e + f)
    # U: c=U[2,5], a=U[0,5]
    U[2, 5] = 1  # c
    U[0, 5] = 1  # a
    # V: e=V[0,5], f=V[1,5]
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    # W: p6 -> C[1,1]+=1 (l)
    W[3, 5] = 1  # C[1,1] = l
    
    # p7 = (b + d)(g + h)
    # U: b=U[1,6], d=U[3,6]
    U[1, 6] = 1  # b
    U[3, 6] = 1  # d
    # V: g=V[2,6], h=V[3,6]
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    # W: p7 -> C[0,0]+=1 (i)
    W[0, 6] = 1  # C[0,0] = i
    
    return U, V, W


def _schoolbook_decomposition():
    """Return the schoolbook rank-8 decomposition."""
    n, m, p = 2, 2, 2
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # First, try the Strassen rank-7 decomposition
    try:
        U, V, W = _strassen_decomposition()
        if _verify_strassen(n, m, p, U, V, W):
            # Convert to lists for the return interface
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback to schoolbook rank-8
    U, V, W = _schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
