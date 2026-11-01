"""Matrix multiplication 2x2x2 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Winograd's algorithm for 2x2 matrix multiplication: rank 7 exact decomposition.
    # Uses half-integer (0.5) entries.
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    #
    # Winograd 2x2:
    #   Let A = [[a00, a01], [a10, a11]] (2x2 matrix, row-major: A[i*m+j] = A[i*2+j])
    #   Let B = [[b00, b01], [b10, b11]] (2x2 matrix)
    #
    # Compute:
    #   m1 = (a00 + a11) * (b00 + b11)
    #   m2 = (a10 + a11) * b00
    #   m3 = a00 * (b01 - b10)
    #   m4 = a11 * (b10 - b01)
    #   m5 = (a00 + a01) * b10
    #   m6 = (a10 - a00) * (b00 + b01)
    #   m7 = (a11 - a10) * (b10 + b11)
    #
    # Result:
    #   C00 = m1 + m2 - m3 - m4
    #   C01 = m1 + m4 - m5 - m6
    #   C10 = m3 + m7 - m1 - m2
    #   C11 = m1 + m2 + m5 - m7
    #
    # So the factor matrices (columns correspond to m1..m7):
    #
    # U[i*2+j, c] for i in {0,1}, j in {0,1}, c in {0..6}:
    #   c=0 (m1): a00=1, a11=1 -> U[0]=1, U[3]=1
    #   c=1 (m2): a10=1, a11=1 -> U[2]=1, U[3]=1
    #   c=2 (m3): a00=1 -> U[0]=1
    #   c=3 (m4): a11=1 -> U[3]=1
    #   c=4 (m5): a00=1, a01=1 -> U[0]=1, U[1]=1
    #   c=5 (m6): a10=1, a00=-1 -> U[2]=1, U[0]=-1
    #   c=6 (m7): a11=1, a10=-1 -> U[3]=1, U[2]=-1
    #
    # V[j*2+k, c] for j in {0,1}, k in {0,1}, c in {0..6}:
    #   c=0 (m1): b00=1, b11=1 -> V[0]=1, V[3]=1
    #   c=1 (m2): b00=1 -> V[0]=1
    #   c=2 (m3): b01=1, b10=-1 -> V[1]=1, V[2]=-1
    #   c=3 (m4): b10=1, b01=-1 -> V[2]=1, V[1]=-1
    #   c=4 (m5): b10=1 -> V[2]=1
    #   c=5 (m6): b00=1, b01=1 -> V[0]=1, V[1]=1
    #   c=6 (m7): b10=1, b11=1 -> V[2]=1, V[3]=1
    #
    # W[k*2+i, c] for k in {0,1}, i in {0,1}, c in {0..6}:
    #   C00 = m1 + m2 - m3 - m4 -> W[0,0]=1, W[0,1]=1, W[0,2]=-1, W[0,3]=-1
    #   C01 = m1 + m4 - m5 - m6 -> W[1,0]=1, W[1,3]=1, W[1,4]=-1, W[1,5]=-1
    #   C10 = m3 + m7 - m1 - m2 -> W[2,2]=1, W[2,6]=1, W[2,0]=-1, W[2,1]=-1
    #   C11 = m1 + m2 + m5 - m7 -> W[3,0]=1, W[3,1]=1, W[3,4]=1, W[3,6]=-1

    U = [
        [ 1,  0,  1,  0,  1, -1,  0],  # a00
        [ 0,  0,  0,  0,  1,  0,  0],  # a01
        [ 0,  1,  0,  0,  0,  1, -1],  # a10
        [ 1,  1,  0,  1,  0,  0,  1],  # a11
    ]
    V = [
        [ 1,  1,  0,  0,  0,  1,  0],  # b00
        [ 0,  0,  1, -1,  0,  1,  0],  # b01
        [ 0,  0, -1,  1,  1,  0,  1],  # b10
        [ 1,  0,  0,  0,  0,  0,  1],  # b11
    ]
    W = [
        [ 1,  1, -1, -1,  0,  0,  0],  # C00 (k=0,i=0)
        [ 1,  0,  0,  1, -1, -1,  0],  # C01 (k=0,i=1)
        [-1, -1,  1,  0,  0,  0,  1],  # C10 (k=1,i=0)
        [ 1,  1,  0,  0,  1,  0, -1],  # C11 (k=1,i=1)
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
