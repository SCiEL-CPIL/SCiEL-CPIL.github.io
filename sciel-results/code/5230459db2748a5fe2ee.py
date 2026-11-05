import numpy as np
from functools import lru_cache

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Term 1: (a+d)(e+h)
    U[0, 0] = 1; U[2, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4: d(g-c)
    U[3, 3] = 1
    V[2, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6: (c-a)(e+f)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1
    
    return U, V, W

def _combine_decomps(n1, m1, p1, U1, V1, W1, n2, m2, p2, U2, V2, W2, dim_split):
    """
    Combine two decompositions.
    dim_split: 'n', 'm', or 'p' indicating which dimension was split.
    """
    if dim_split == 'm':
        # A is split into A1 (n x m1) and A2 (n x m2)
        # B is split into B1 (m1 x p) and B2 (m2 x p)
        # C is sum of C1 and C2
        
        # U: (n*m1 + n*m2) x (R1+R2)
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R_total = R1 + R2
        
        U = np.zeros((n1 * (m1 + m2), R_total))
        # Rows 0..n1*m1-1 from U1
        U[:n1*m1, :R1] = U1
        # Rows n1*m1..n1*(m1+m2)-1 from U2
        U[n1*m1:, R1:] = U2
        
        # V: (m1*p1 + m2*p2) x (R1+R2)
        # Note: p1 == p2 == p
        V = np.zeros(((m1 + m2) * p1, R_total))
        V[:m1*p1, :R1] = V1
        V[m1*p1:, R1:] = V2
        
        # W: (p*n) x (R1+R2)
        # W is concatenated horizontally
        W = np.hstack([W1, W2])
        
        return U, V, W

    elif dim_split == 'n':
        # A split into A1 (n1 x m) and A2 (n2 x m)
        # B is same (m x p)
        # C split into C1 (n1 x p) and C2 (n2 x p)
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R_total = R1 + R2
        
        # U: (n1*m + n2*m) x R_total
        U = np.zeros(((n1 + n2) * m1, R_total))
        U[:n1*m1, :R1] = U1
        U[n1*m1:, R1:] = U2
        
        # V: (m*p) x R_total
        # V1 and V2 are the same shape (m*p, R1) and (m*p, R2)
        # But they correspond to the same B.
        # In the combined tensor, the V rows are shared?
        # No, the decomposition is a sum of rank-1 terms.
        # Term r from decomp 1 uses V1[:, r].
        # Term r from decomp 2 uses V2[:, r].
        # Since B is the same, we just need to place V1 and V2 in the same rows?
        # Wait. The tensor is T = sum u v w.
        # If we split n, we have T = T1 + T2.
        # T1 uses U1, V1, W1.
        # T2 uses U2, V2, W2.
        # The V matrix for the combined problem has shape (m*p, R_total).
        # The first R1 columns are V1.
        # The next R2 columns are V2.
        V = np.hstack([V1, V2])
        
        # W: (p*n1 + p*n2) x R_total
        W = np.zeros((p1 * (n1 + n2), R_total))
        W[:p1*n1, :R1] = W1
        W[p1*n1:, R1:] = W2
        
        return U, V, W

    elif dim_split == 'p':
        # A is same (n x m)
        # B split into B1 (m x p1) and B2 (m x p2)
        # C split into C1 (n x p1) and C2 (n x p2)
        
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R_total = R1 + R2
        
        # U: (n*m) x R_total
        U = np.hstack([U1, U2])
        
        # V: (m*p1 + m*p2) x R_total
        V = np.zeros((m1 * (p1 + p2), R_total))
        V[:m1*p1, :R1] = V1
        V[m1*p1:, R1:] = V2
        
        # W: (p1*n + p2*n) x R_total
        W = np.zeros(((p1 + p2) * n1, R_total))
        W[:p1*n1, :R1] = W1
        W[p1*n1:, R1:] = W2
        
        return U, V, W

def construct():
    n, m, p = 3, 5, 6
    
    # We will use a recursive search for the best decomposition.
    # To keep it simple and fast, we hardcode the best splits found by the logic above.
    # Split m=5 into 2 and 3.
    
    # Get decomp for 3x2x6 and 3x3x6
    
    # Helper to get decomp for small dims
    def get_decomp(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return np.array([[1]]), np.array([[1]]), np.array([[1]])
        
        if n == 2 and m == 2 and p == 2:
            return _get_strassen_222()
            
        # Try splits
        best_rank = float('inf')
        best_decomp = None
        
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            rank = U1.shape[1] + U2.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_decomp = _combine_decomps(n, m1, p, U1, V1, W1, n, m2, p, U2, V2, W2, 'm')
                
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            rank = U1.shape[1] + U2.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_decomp = _combine_decomps(n1, m, p, U1, V1, W1, n2, m, p, U2, V2, W2, 'n')
                
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            rank = U1.shape[1] + U2.shape[1]
            if rank < best_rank:
                best_rank = rank
                best_decomp = _combine_decomps(n, m, p1, U1, V1, W1, n, m, p2, U2, V2, W2, 'p')
                
        return best_decomp

    U, V, W = get_decomp(n, m, p)
    
    # Convert to lists as required by interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The return type is dict of U, V, W.
    # The seed program returned lists of lists.
    # I will return numpy arrays, but ensure they are compatible.
    # The evaluator likely accepts numpy arrays or lists.
    # To be safe, convert to lists of lists if necessary, or just return arrays.
    # The seed returned lists. I'll return lists to be safe.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
