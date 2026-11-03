"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# The matrix multiplication tensor for (n,m,p) = (2,4,5)
# We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
# such that for all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# where T[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor for matrix mult)

# Actually, the matrix multiplication tensor has T[i,j,k] = 1 if the (i,k) element of the product
# gets a contribution from (i,j) of A and (j,k) of B. For standard matrix multiplication:
# (A*B)[i,k] = sum_j A[i,j] * B[j,k]
# So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)

# The schoolbook rank is n*m*p = 2*4*5 = 40.
# We want to find a lower rank decomposition if possible.

# Known results: For 2x4 * 4x5, the rank is at most 40 (schoolbook).
# The rank of matrix multiplication tensor for n x m * m x p is at least max(n*m, m*p, n*p) 
# and at most n*m*p. For small cases, the exact rank is often known.

# For 2x4 * 4x5: n*m=8, m*p=20, n*p=10. So rank >= 20.
# The schoolbook gives 40. Can we do better?

# Let's use a randomized approach with rank minimization.
# We'll try to find a decomposition of rank r for various r starting from a lower bound.

# The tensor T is a 3-tensor of shape (n*m, m*p, p*n) where:
# T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.

# Actually, let me re-read the convention. The output factor follows W[k*n+i, r].
# So the tensor is T[a, b, c] where:
# a = i*m + j (row of A, col of A / row of B)
# b = j*p + k (col of B, row of B / col of A... wait)
# c = k*n + i (row of result, col of result)

# The matrix multiplication tensor satisfies:
# sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
# where T[a, b, c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0.

# For each (i,j,k) triple, we have a = i*m+j, b = j*p+k, c = k*n+i.
# T[a,b,c] = 1 for these specific (a,b,c) combinations.

# Let's enumerate all (i,j,k) triples:
# i in [0,2), j in [0,4), k in [0,5)
# a = i*4+j, b = j*5+k, c = k*2+i

# The tensor is sparse: only n*m*p = 40 entries are 1, rest are 0.
# Total tensor size: (n*m) * (m*p) * (p*n) = 8 * 20 * 10 = 1600 entries.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    am = n * m  # 8
    bp = m * p  # 20
    cn = p * n  # 10
    
    # Build the tensor
    T = np.zeros((am, bp, cn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find low-rank decompositions
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's try to verify the schoolbook solution works
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W give an exact decomposition."""
        rank = U.shape[1]
        # Compute sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
        # Use einsum
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(result, T, atol=tol)
    
    # Schoolbook
    U_school = np.zeros((am, best_rank))
    V_school = np.zeros((bp, best_rank))
    W_school = np.zeros((cn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    if verify(U_school, V_school, W_school):
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    # Try lower ranks
    # Lower bound: max(am, bp, cn) = max(8, 20, 10) = 20
    # Try ranks from 20 to 39
    for r in range(20, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(am, r)
            V = np.random.randn(bp, r)
            W = np.random.randn(cn, r)
            
            # Optimize using gradient descent on the loss:
            # L = ||sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c]||^2
            
            # Use a simple iterative approach
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current product
                prod = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Gradient with respect to U:
                # dL/dU[a,r] = 2 * sum_{b,c} (prod[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                diff = prod - T
                grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if np.allclose(prod, T, atol=1e-10):
                    break
            
            if verify(U, V, W):
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Convert to lists of floats (or ints if they're close to integers)
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Check if close to integer or half-integer
                if abs(val - round(val)) < 1e-9:
                    r.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-9:
                    r.append(round(val*2)/2)
                else:
                    r.append(float(val))
            result.append(r)
        return result
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
