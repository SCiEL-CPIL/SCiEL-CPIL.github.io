import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(u_offset, v_offset, w_offset, base_rank):
        # u_offset: index of A[0,0] in U
        # v_offset: index of B[0,0] in V
        # w_offset: index of C[0,0] in W
        
        # M1
        r = base_rank + 0
        U[u_offset, r] += 1
        U[u_offset + 4, r] += 1 # A[1,1] is u_offset + 1*m + 1? No.
        # A is 2x3. Subblock is cols 0,1.
        # A[0,0] is u_offset.
        # A[0,1] is u_offset + 1.
        # A[1,0] is u_offset + 3.
        # A[1,1] is u_offset + 4.
        
        U[u_offset, r] += 1
        U[u_offset + 4, r] += 1
        V[v_offset, r] += 1
        V[v_offset + 5, r] += 1 # B[1,1] is v_offset + 1*4 + 1 = v_offset + 5.
        W[w_offset, r] += 1
        W[w_offset + 5, r] += 1 # C[1,1] is w_offset + 1*2 + 1 = w_offset + 3?
        # W index is k*n + i.
        # C[0,0] is 0*2+0 = 0.
        # C[0,1] is 1*2+0 = 2.
        # C[1,0] is 0*2+1 = 1.
        # C[1,1] is 1*2+1 = 3.
        # So W indices are w_offset + 0, w_offset + 2, w_offset + 1, w_offset + 3.
        
        # Correct W mapping:
        # C[0,0] -> w_offset + 0
        # C[0,1] -> w_offset + 2
        # C[1,0] -> w_offset + 1
        # C[1,1] -> w_offset + 3
        
        # Redo M1
        W[w_offset, r] += 1
        W[w_offset + 3, r] += 1

        # M2
        r = base_rank + 1
        U[u_offset + 3, r] += 1 # A[1,0]
        U[u_offset + 4, r] += 1 # A[1,1]
        V[v_offset, r] += 1     # B[0,0]
        W[w_offset + 1, r] += 1 # C[1,0]

        # M3
        r = base_rank + 2
        U[u_offset, r] += 1     # A[0,0]
        V[v_offset + 1, r] += 1 # B[0,1]
        V[v_offset + 5, r] -= 1 # B[1,1]
        W[w_offset + 2, r] += 1 # C[0,1]

        # M4
        r = base_rank + 3
        U[u_offset + 4, r] += 1 # A[1,1]
        V[v_offset + 4, r] += 1 # B[1,0]
        V[v_offset, r] -= 1     # B[0,0]
        W[w_offset, r] -= 1     # C[0,0]
        W[w_offset + 1, r] += 1 # C[1,0]

        # M5
        r = base_rank + 4
        U[u_offset, r] += 1     # A[0,0]
        U[u_offset + 1, r] += 1 # A[0,1]
        V[v_offset + 5, r] += 1 # B[1,1]
        W[w_offset + 2, r] += 1 # C[0,1]
        W[w_offset + 3, r] += 1 # C[1,1]

        # M6
        r = base_rank + 5
        U[u_offset + 3, r] += 1 # A[1,0]
        U[u_offset, r] -= 1     # A[0,0]
        V[v_offset, r] += 1     # B[0,0]
        V[v_offset + 1, r] += 1 # B[0,1]
        W[w_offset, r] += 1     # C[0,0]
        W[w_offset + 1, r] -= 1 # C[1,0]

        # M7
        r = base_rank + 6
        U[u_offset + 1, r] += 1 # A[0,1]
        U[u_offset + 4, r] -= 1 # A[1,1]
        V[v_offset + 4, r] += 1 # B[1,0]
        V[v_offset + 5, r] += 1 # B[1,1]
        W[w_offset + 2, r] -= 1 # C[0,1]
        W[w_offset + 3, r] += 1 # C[1,1]

    # Block 1: j in {0,1}, k in {0,1}
    # A cols 0,1 -> u_offset = 0
    # B rows 0,1, cols 0,1 -> v_offset = 0
    # C rows 0,1, cols 0,1 -> w_offset = 0
    add_strassen_block(0, 0, 0, 0)

    # Block 2: j in {0,1}, k in {2,3}
    # A cols 0,1 -> u_offset = 0
    # B rows 0,1, cols 2,3 -> v_offset = 2
    # C rows 0,1, cols 2,3 -> w_offset = 4*2 + 0 = 8?
    # W index for C[i,k] is 4k+i.
    # C[0,2] -> 8. C[0,3] -> 12. C[1,2] -> 9. C[1,3] -> 13.
    # So w_offset = 8.
    add_strassen_block(0, 2, 8, 7)

    # Block 3: j=2, k in {0,1}
    # A col 2 -> u_offset = 2 (A[0,2]=2, A[1,2]=5)
    # B row 2, cols 0,1 -> v_offset = 2*4 + 0 = 8
    # C rows 0,1, cols 0,1 -> w_offset = 0
    # Trivial 4 terms
    for i in range(2):
        for k in range(2):
            r = 14 + i * 2 + k
            U[i * 3 + 2, r] = 1
            V[2 * 4 + k, r] = 1
            W[4 * k + i, r] = 1

    # Block 4: j=2, k in {2,3}
    # A col 2 -> u_offset = 2
    # B row 2, cols 2,3 -> v_offset = 2*4 + 2 = 10
    # C rows 0,1, cols 2,3 -> w_offset = 8
    for i in range(2):
        for k in range(2):
            r = 18 + i * 2 + k
            U[i * 3 + 2, r] = 1
            V[2 * 4 + (k + 2), r] = 1
            W[4 * (k + 2) + i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
