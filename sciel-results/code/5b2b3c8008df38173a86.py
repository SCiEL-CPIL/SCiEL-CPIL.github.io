import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = get_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_decomp(n, m, p):
    # Base case: 1 in any dimension
    if n == 1 or m == 1 or p == 1:
        U = np.ones((n, 1))
        V = np.ones((m, 1))
        W = np.ones((p, 1))
        return U, V, W
    
    # Base case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        # Strassen decomposition for 2x2x2
        # Returns U (4x7), V (4x7), W (4x7)
        U = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0], # Placeholder, need correct Strassen U
            [0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        # Correct Strassen U, V, W construction is complex.
        # For now, use Trivial for 2x2x2 to ensure correctness, or implement Strassen properly.
        # Let's use Trivial for 2x2x2 to be safe? Rank 8 vs 7.
        # 8 is still better than the alternatives if we don't implement Strassen.
        # But 7 is better.
        # Let's implement a simple Strassen for 2x2x2.
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is U[0], A22 is U[3]
        # B11 is V[0], B22 is V[3]
        # C11 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        
        # M2 = (A21+A22)B11
        # A21 is U[2], A22 is U[3]
        # B11 is V[0]
        # C12 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = A11(B12-B22)
        # A11 is U[0]
        # B12 is V[1], B22 is V[3]
        # C21 += M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = A22(B21-B11)
        # A22 is U[3]
        # B21 is V[2], B11 is V[0]
        # C22 += M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5 = (A11+A12)B22
        # A11 is U[0], A12 is U[1]
        # B22 is V[3]
        # C22 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21 is U[2], A11 is U[0]
        # B11 is V[0], B12 is V[1]
        # C21 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12 is U[1], A22 is U[3]
        # B21 is V[2], B22 is V[3]
        # C12 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    # Recursive Case
    U_list = []
    V_list = []
    W_list = []
    
    # Split n if odd
    if n % 2 != 0:
        n1 = n - 1
        n2 = 1
        U1, V1, W1 = get_decomp(n1, m, p)
        U2, V2, W2 = get_decomp(n2, m, p)
        U_list.append(U1)
        V_list.append(V1)
        W_list.append(W1)
        U_list.append(U2)
        V_list.append(V2)
        W_list.append(W2)
        n = 0 # Mark as processed
    else:
        n1 = n // 2
        n2 = n // 2
        U1, V1, W1 = get_decomp(n1, m, p)
        U2, V2, W2 = get_decomp(n2, m, p)
        U_list.append(U1)
        V_list.append(V1)
        W_list.append(W1)
        U_list.append(U2)
        V_list.append(V2)
        W_list.append(W2)

    # The above logic is flawed because it splits n, then m, then p independently.
    # We need to split only one dimension at a time and recurse on the resulting sub-problems.
    
    # Correct Recursive Logic:
    if n % 2 != 0:
        n1 = n - 1
        n2 = 1
        U1, V1, W1 = get_decomp(n1, m, p)
        U2, V2, W2 = get_decomp(n2, m, p)
        return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])
    elif m % 2 != 0:
        m1 = m - 1
        m2 = 1
        U1, V1, W1 = get_decomp(n, m1, p)
        U2, V2, W2 = get_decomp(n, m2, p)
        return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])
    elif p % 2 != 0:
        p1 = p - 1
        p2 = 1
        U1, V1, W1 = get_decomp(n, m, p1)
        U2, V2, W2 = get_decomp(n, m, p2)
        return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])
    else:
        # All even
        # Split n into two halves
        n1 = n // 2
        n2 = n // 2
        U1, V1, W1 = get_decomp(n1, m, p)
        U2, V2, W2 = get_decomp(n2, m, p)
        return np.hstack([U1, U2]), np.hstack([V1, V2]), np.hstack([W1, W2])

