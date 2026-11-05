import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    
    # Initialize the tensor T with Fraction(1, 1) at the appropriate indices
    # T is a 3D array of Fractions
    # Dimensions: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = Fraction(1, 1)
    
    U_list = []
    V_list = []
    W_list = []
    
    # We will try to peel off rank-1 terms
    # We search for u, v, w with entries in a small set of rationals
    # To keep it fast, we limit the search space
    
    # Define a small set of candidate values for u, v, w entries
    # We can try integers and half-integers
    # But the space is (n*m) * (m*p) * (p*n) = 30 * 15 * 10 = 4500 variables.
    # This is too large for brute force.
    
    # Alternative: Use the fact that the tensor is sparse.
    # The non-zero entries of T are at 30 specific locations.
    # A rank-1 term u ⊗ v ⊗ w has non-zero entries at (a, b, c) if u[a]!=0, v[b]!=0, w[c]!=0.
    # To cancel out the entries of T, we need the sum of rank-1 terms to match T.
    
    # Let's try a specific known decomposition if available, or fall back to schoolbook.
    # Since I don't have a guaranteed low-rank decomposition for 2x3x5, 
    # I will implement a simple improvement: 
    # If we can find a rank-29 decomposition, it's better than 30.
    # 
    # Actually, let's just return the schoolbook solution to be safe, 
    # unless I can quickly find a better one.
    # 
    # Wait, the prompt says "score is the rank". Lower is better.
    # If I return rank 30, score is 30.
    # If I return rank 28, score is 28.
    # 
    # Let's try to construct a rank-28 solution by combining two schoolbook terms 
    # if possible.
    # 
    # Consider two terms:
    # T1: i=0, j=0, k=0 -> U[0,0], V[0,0], W[0,0]
    # T2: i=0, j=0, k=1 -> U[0,0], V[0,1], W[1,0]
    # 
    # Can we replace these two with one rank-1 term?
    # No, a rank-1 term is a single product.
    # 
    # However, we can use a **linear combination** of the basis vectors.
    # Let's define new basis vectors for U, V, W.
    # 
    # For now, I will return the schoolbook solution. 
    # This ensures correctness. 
    # Improving this requires a more sophisticated search.
    
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
