# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 9

    # Build M_{2,1,9} via schoolbook: rank = 18
    U1 = []
    V1 = []
    W1 = []
    for i in range(n):
        for k in range(p):
            U1.append([1] * n)
            V1.append([1] * p)
            W1.append([1] * (p * n))
    U1 = np.array(U1, dtype=float).T  # (n, 18)
    V1 = np.array(V1, dtype=float).T  # (p, 18)
    W1 = np.array(W1, dtype=float).T  # (p*n, 18)

    # Build M_{2,2,2} via Strassen: rank = 7
    # Strassen matrices for 2x2 * 2x2
    # C = A*B where A is 2x2, B is 2x2
    # Standard Strassen decomposition
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

    # U for M_{2,2,2}: rows indexed by (i,j) in {0,1}x{0,1}
    # A indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    # B indices: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)

    # For each Strassen term, we have a product of a linear combination of A entries
    # and a linear combination of B entries, contributing to C entries.

    # Let's build the decomposition explicitly.
    # Term r: u_r * v_r * w_r
    # u_r is a vector in R^4 (for A entries), v_r in R^4 (for B entries), w_r in R^4 (for C entries)

    # We need to express each term as outer products.
    # Actually, for the tensor decomposition, we need:
    # T_{i,j,k,l} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the indices are: A is n x m = 2 x 2, so A[i,j] with i in {0,1}, j in {0,1}
    # B is m x p = 2 x 2, so B[j,k] with j in {0,1}, k in {0,1}
    # C is n x p = 2 x 2, so C[i,k] with i in {0,1}, k in {0,1}
    # C[i,k] = sum_j A[i,j] * B[j,k]

    # So the tensor is T_{i,j,k} = delta_{i,k}... no wait.
    # The multiplication tensor T_{i,j,k} where i indexes C row, j indexes A col / B row, k indexes B col.
    # Actually, the standard convention: T_{i,j,k} = 1 if C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} = delta_{i,i} * delta_{k,k}... no.
    # T_{i,j,k} = 1 for all i,j,k? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T_{i,j,k} = 1 for all valid i,j,k.
    # Wait, that's not right either. The tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if the (i,k) entry of C depends on A_{i,j} and B_{j,k}
    # So T_{i,j,k} = 1 for all i in {0,...,n-1}, j in {0,...,m-1}, k in {0,...,p-1}
    # That's the schoolbook decomposition with rank n*m*p.

    # For Strassen, we're using a different decomposition.
    # Let me just use the known Strassen rank-7 decomposition for 2x2.

    # U: (n*m, 7) = (4, 7)
    # V: (m*p, 7) = (4, 7)
    # W: (p*n, 7) = (4, 7)

    # A entries: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B entries: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C entries: C[0,0], C[0,1], C[1,0], C[1,1]

    # Strassen terms:
    # m1 = (a+d)(e+h) -> contributes to C00 (+1), C01 (-1), C10 (-1), C11 (+1)
    # Wait, let me re-derive.
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-d)(e+g)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    U22 = np.zeros((4, 7), dtype=float)
    V22 = np.zeros((4, 7), dtype=float)
    W22 = np.zeros((4, 7), dtype=float)

    # Term 1: m1 = (a+d)(e+h)
    # U: a=1, d=1 -> [1,0,0,1]
    # V: e=1, h=1 -> [1,0,0,1]
    # W: C00=+1, C01=-1, C10=-1, C11=+1 -> [1,-1,-1,1]
    U22[:, 0] = [1, 0, 0, 1]
    V22[:, 0] = [1, 0, 0, 1]
    W22[:, 0] = [1, -1, -1, 1]

    # Term 2: m2 = (c+d)e
    # U: c=1, d=1 -> [0,0,1,1]
    # V: e=1 -> [1,0,0,0]
    # W: C10=+1, C11=-1 -> [0,0,1,-1]
    U22[:, 1] = [0, 0, 1, 1]
    V22[:, 1] = [1, 0, 0, 0]
    W22[:, 1] = [0, 0, 1, -1]

    # Term 3: m3 = a(f-h)
    # U: a=1 -> [1,0,0,0]
    # V: f=1, h=-1 -> [0,1,0,-1]
    # W: C01=+1, C11=+1 -> [0,1,0,1]
    U22[:, 2] = [1, 0, 0, 0]
    V22[:, 2] = [0, 1, 0, -1]
    W22[:, 2] = [0, 1, 0, 1]

    # Term 4: m4 = d(g-e)
    # U: d=1 -> [0,0,0,1]
    # V: g=1, e=-1 -> [-1,0,1,0]
    # W: C00=+1, C10=+1 -> [1,0,1,0]
    U22[:, 3] = [0, 0, 0, 1]
    V22[:, 3] = [-1, 0, 1, 0]
    W22[:, 3] = [1, 0, 1, 0]

    # Term 5: m5 = (a+b)h
    # U: a=1, b=1 -> [1,1,0,0]
    # V: h=1 -> [0,0,0,1]
    # W: C00=-1, C01=+1 -> [-1,1,0,0]
    U22[:, 4] = [1, 1, 0, 0]
    V22[:, 4] = [0, 0, 0, 1]
    W22[:, 4] = [-1, 1, 0, 0]

    # Term 6: m6 = (c-d)(f+h)
    # U: c=1, d=-1 -> [0,0,1,-1]
    # V: f=1, h=1 -> [0,1,0,1]
    # W: C11=+1 -> [0,0,0,1]
    U22[:, 5] = [0, 0, 1, -1]
    V22[:, 5] = [0, 1, 0, 1]
    W22[:, 5] = [0, 0, 0, 1]

    # Term 7: m7 = (b-d)(e+g)
    # U: b=1, d=-1 -> [0,1,0,-1]
    # V: e=1, g=1 -> [1,0,1,0]
    # W: C00=+1 -> [1,0,0,0]
    U22[:, 6] = [0, 1, 0, -1]
    V22[:, 6] = [1, 0, 1, 0]
    W22[:, 6] = [1, 0, 0, 0]

    # Build M_{2,2,1} via schoolbook: rank = 4
    # n=2, m=2, p=1
    U21 = np.zeros((4, 4), dtype=float)
    V21 = np.zeros((2, 4), dtype=float)
    W21 = np.zeros((2, 4), dtype=float)
    r = 0
    for i in range(2):
        for j in range(2):
            U21[i * 2 + j, r] = 1
            V21[j, r] = 1
            W21[i, r] = 1
            r += 1

    # Now build M_{2,2,9} by splitting B into B_{:,0:2} and B_{:,2}
    # B_{:,0:2} is 2x2, so M_{2,2,2} part
    # B_{:,2} is 2x1, so M_{2,2,1} part
    # But we need M_{2,2,9}, so B is 2x9.
    # Split B into three 2x3 blocks? Or use the Kronecker product approach.
    # Actually, let's use: M_{2,2,9} = M_{2,2,3} tensor M_{1,1,3}
    # M_{2,2,3} rank: we can split B (2x3) into B_{:,0:2} (2x2) and B_{:,2} (2x1)
    # So M_{2,2,3} = M_{2,2,2} + M_{2,2,1}, rank = 7 + 4 = 11
    # Then M_{2,2,9} = M_{2,2,3} tensor M_{1,1,3}, rank = 11 * 3 = 33

    # M_{1,1,3}: rank 3
    U113 = np.ones((1, 3), dtype=float)
    V113 = np.ones((3, 3), dtype=float)
    W113 = np.ones((3, 3), dtype=float)

    # M_{2,2,3}: rank 11
    # U: (n*m, 11) = (4, 11)
    # V: (m*p, 11) = (2*3, 11) = (6, 11)
    # W: (p*n, 11) = (3*2, 11) = (6, 11)
    # For the M_{2,2,2} part (first 7 terms):
    #   A is 2x2, B is 2x2 (columns 0,1 of the 2x3 B)
    #   U: same as U22
    #   V: B has 6 entries (2x3), but we only use first 2 columns
    #   W: C is 2x2 (rows 0,1 of the 2x3 C? No, C is n x p = 2 x 3)
    #   Wait, for M_{2,2,3}, C is 2x3.
    #   The M_{2,2,2} part computes C[:,0:2] = A * B[:,0:2]
    #   The M_{2,2,1} part computes C[:,2] = A * B[:,2]

    U223 = np.zeros((4, 11), dtype=float)
    V223 = np.zeros((6, 11), dtype=float)
    W223 = np.zeros((6, 11), dtype=float)

    # First 7 terms: M_{2,2,2} part
    U223[:, :7] = U22
    # V: B is 2x3, so V has 6 rows. For the 2x2 part, we use rows 0,1 (col 0) and 2,3 (col 1)
    # V22 has 4 rows (for 2x2 B). Map to V223:
    # B22[0,0] -> V223[0, :], B22[0,1] -> V223[2, :], B22[1,0] -> V223[1, :], B22[1,1] -> V223[3, :]
    V223[0, :7] = V22[0, :]
    V223[2, :7] = V22[1, :]
    V223[1, :7] = V22[2, :]
    V223[3, :7] = V22[3, :]
    # W: C is 2x3, so W has 6 rows. For the 2x2 part, C[:,0:2]
    # W22 has 4 rows (for 2x2 C). Map to W223:
    # C22[0,0] -> W223[0, :], C22[0,1] -> W223[2, :], C22[1,0] -> W223[1, :], C22[1,1] -> W223[3, :]
    W223[0, :7] = W22[0, :]
    W223[2, :7] = W22[1, :]
    W223[1, :7] = W22[2, :]
    W223[3, :7] = W22[3, :]

    # Last 4 terms: M_{2,2,1} part
    U223[:, 7:] = U21
    # V: B is 2x3, for the 2x1 part, we use column 2
    # V21 has 2 rows (for 2x1 B). Map to V223:
    # B21[0,0] -> V223[4, :], B21[1,0] -> V223[5, :]
    V223[4, 7:] = V21[0, :]
    V223[5, 7:] = V21[1, :]
    # W: C is 2x3, for the 2x1 part, C[:,2]
    # W21 has 2 rows (for 2x1 C). Map to W223:
    # C21[0,0] -> W223[4, :], C21[1,0] -> W223[5, :]
    W223[4, 7:] = W21[0, :]
    W223[5, 7:] = W21[1, :]

    # Now Kronecker product with M_{1,1,3}
    # M_{2,2,9} = M_{2,2,3} tensor M_{1,1,3}
    # U: (4, 11) tensor (1, 3) -> (4*1, 11*3) = (4, 33)
    # V: (6, 11) tensor (3, 3) -> (6*3, 11*3) = (18, 33)
    # W: (6, 11) tensor (3, 3) -> (6*3, 11*3) = (18, 33)

    U229 = np.zeros((4, 33), dtype=float)
    V229 = np.zeros((18, 33), dtype=float)
    W229 = np.zeros((18, 33), dtype=float)

    for r1 in range(11):
        for r2 in range(3):
            r = r1 * 3 + r2
            U229[:, r] = U223[:, r1] * U113[0, r2]
            # V: V223 is (6, 11), V113 is (3, 3)
            # V229 has 18 rows. For each row (j, k) in V223 and (l) in V113:
            # V229[j*3 + l, r] = V223[j, r1] * V113[l, r2]
            for j in range(6):
                for l in range(3):
                    V229[j * 3 + l, r] = V223[j, r1] * V113[l, r2]
            # W: W223 is (6, 11), W113 is (3, 3)
            for j in range(6):
                for l in range(3):
                    W229[j * 3 + l, r] = W223[j, r1] * W113[l, r2]

    # Now combine M_{2,1,9} and M_{2,2,9}
    # M_{2,3,9} = M_{2,1,9} + M_{2,2,9}
    # U: (2, 18) and (4, 33) -> need to align
    # For M_{2,1,9}: A is 2x1, so U has 2 rows
    # For M_{2,2,9}: A is 2x2, so U has 4 rows
    # Total A is 2x3, so U should have 6 rows
    # The M_{2,1,9} part uses column 0 of A
    # The M_{2,2,9} part uses columns 1,2 of A

    U = np.zeros((6, 51), dtype=float)
    V = np.zeros((27, 51), dtype=float)
    W = np.zeros((18, 51), dtype=float)

    # M_{2,1,9} part: first 18 terms
    # U: 2 rows (for A[:,0])
    U[0, :18] = U1[0, :]
    U[1, :18] = U1[1, :]
    # V: 27 rows (for B, which is 3x9)
    # For M_{2,1,9}, B is 1x9, so V1 has 9 rows
    # Map to V: B[0,k] -> V[k, :]
    for k in range(9):
        V[k, :18] = V1[k, :]
    # W: 18 rows (for C, which is 2x9)
    # For M_{2,1,9}, C is 2x9, so W1 has 18 rows
    W[:, :18] = W1

    # M_{2,2,9} part: next 33 terms
    # U: 4 rows (for A[:,1:3])
    U[2, 18:51] = U229[0, :]
    U[3, 18:51] = U229[1, :]
    U[4, 18:51] = U229[2, :]
    U[5, 18:51] = U229[3, :]
    # V: 27 rows (for B, which is 3x9)
    # For M_{2,2,9}, B is 2x9, so V229 has 18 rows
    # Map to V: B[0,k] -> V[9+k, :], B[1,k] -> V[18+k, :]
    for k in range(9):
        V[9 + k, 18:51] = V229[k, :]
        V[18 + k, 18:51] = V229[9 + k, :]
    # W: 18 rows (for C, which is 2x9)
    # For M_{2,2,9}, C is 2x9, so W229 has 18 rows
    W[:, 18:51] = W229

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
