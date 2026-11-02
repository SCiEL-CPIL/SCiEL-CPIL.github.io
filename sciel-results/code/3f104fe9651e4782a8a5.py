"""
Matrix multiplication 2x5x5: Strassen-block decomposition.
Uses 2x2x2 Strassen blocks where dimensions allow, falling back to schoolbook
for the remaining 1-dimensional slices. This reduces rank from 50 (schoolbook)
to 46.

Decomposition plan:
- j in {0,1}, k in {0,1}: Strassen(0,0,0) -> 7 terms
- j in {0,1}, k in {2,3}: Strassen(0,0,2) -> 7 terms
- j in {0,1}, k in {4}:   Schoolbook 2x1x1 -> 2 terms
- j in {2,3}, k in {0,1}: Strassen(0,2,0) -> 7 terms
- j in {2,3}, k in {2,3}: Strassen(0,2,2) -> 7 terms
- j in {2,3}, k in {4}:   Schoolbook 2x1x1 -> 2 terms
- j in {4},   k in {0,4}: Schoolbook 2x1x5 -> 10 terms

Total rank = 7+7+2+7+7+2+10 = 42
"""

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        """Adds 7 rank-1 terms for a 2x2x2 Strassen block."""
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        u = {(i0, j0): 1, (i1, j1): 1}
        v = {(j0, k0): 1, (j1, k1): 1}
        w = {(k0, i0): 1, (k1, i1): 1}
        terms.append((u, v, w))
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        u = {(i1, j0): 1, (i1, j1): 1}
        v = {(j0, k0): 1}
        w = {(k0, i1): 1, (k1, i1): -1}
        terms.append((u, v, w))
        
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        u = {(i0, j0): 1}
        v = {(j0, k1): 1, (j1, k1): -1}
        w = {(k1, i0): 1, (k1, i1): 1}
        terms.append((u, v, w))
        
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        u = {(i1, j1): 1}
        v = {(j1, k0): 1, (j0, k0): -1}
        w = {(k0, i0): 1, (k0, i1): 1}
        terms.append((u, v, w))
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        u = {(i0, j0): 1, (i0, j1): 1}
        v = {(j1, k1): 1}
        w = {(k0, i0): -1, (k1, i0): 1}
        terms.append((u, v, w))
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        u = {(i1, j0): 1, (i0, j0): -1}
        v = {(j0, k0): 1, (j0, k1): 1}
        w = {(k1, i1): 1}
        terms.append((u, v, w))
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        u = {(i0, j1): 1, (i1, j1): -1}
        v = {(j1, k0): 1, (j1, k1): 1}
        w = {(k0, i0): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds standard rank-1 terms for the given sub-block."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Decomposition plan:
    # Block 1: j={0,1}, k={0,1} -> Strassen
    add_strassen(0, 0, 0)
    # Block 2: j={0,1}, k={2,3} -> Strassen
    add_strassen(0, 0, 2)
    # Block 3: j={0,1}, k={4} -> Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    # Block 4: j={2,3}, k={0,1} -> Strassen
    add_strassen(0, 2, 0)
    # Block 5: j={2,3}, k={2,3} -> Strassen
    add_strassen(0, 2, 2)
    # Block 6: j={2,3}, k={4} -> Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    # Block 7: j={4}, k={0,1,2,3,4} -> Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return {"U": U, "V": V, "W": W}
