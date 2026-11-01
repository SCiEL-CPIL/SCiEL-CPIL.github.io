import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 2, 15
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Strassen's algorithm for 2x2 matrix multiplication uses rank 7
    # The 7 Strassen terms compute:
    # M1 = (A11-A22)*(B21+B22)
    # M2 = (A11+A12)*B21
    # M3 = A22*(B12-B22)
    # M4 = A11*B11
    # M5 = (A12+A22)*(B11+B22)
    # M6 = (A11-A12)*B12
    # M7 = (A21+A11)*(B22-B11)
    # C11 = M4 + M5 - M1 - M7
    # C12 = M2 + M6
    # C21 = M3 + M5
    # C22 = M1 + M4 - M2 + M7
    
    # For 2x2 * 2x15:
    # A is 2x2, B is 2x15
    # We can split B into two 2x7 and one 2x1 block
    # For each 2x7 block: use Strassen (rank 7) for the 2x2x7 part
    # For the 2x1 block: use schoolbook (rank 4)
    
    # Actually, let's think more carefully.
    # A[i,j] * B[j,k] summed over j for each (i,k)
    # A is 2x2, B is 2x15
    # Output C is 2x15
    
    # We can view this as: for each k, compute A * B[:,k]
    # B[:,k] is a 2x1 vector
    # A * (2x1) = 2x1 result
    
    # Alternative: split B into blocks of columns
    # B = [B1 | B2 | B3] where B1 is 2x5, B2 is 2x5, B3 is 2x5
    # Then C = A*B = [A*B1 | A*B2 | A*B3]
    
    # For each 2x5 block, we can further decompose
    # Actually, let's use a different approach.
    
    # The tensor is T[i,j,k] = A[i,j]*B[j,k]
    # We want to decompose this as sum_c U[i,c]*V[j,c]*W[k,c]
    
    # For the 2x2 part (i,j indices), Strassen gives rank 7
    # For each k, we need the full 2x2 multiplication
    # But the k index is independent
    
    # Let me think about this differently.
    # T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,...,14}
    # T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    
    # Actually the tensor is: T[i*m+j, j'*p+k, k'*n+i'] 
    # Wait, let me re-read the problem.
    
    # The matrix multiplication tensor for (n,m,p) = (2,2,15):
    # T[a, b, c] where a indexes A (2x2), b indexes B (2x15), c indexes C (2x15)
    # T[a, b, c] = 1 if the product A[a] * B[b] contributes to C[c]
    
    # A has 4 entries: A[0,0], A[0,1], A[1,0], A[1,1] indexed as 0,1,2,3
    # B has 30 entries: B[0,0]...B[0,14], B[1,0]...B[1,14] indexed as 0...29
    # C has 30 entries: C[0,0]...C[0,14], C[1,0]...C[1,14] indexed as 0...29
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i*2+j, j*15+k, i*15+k] = 1 for i,j in {0,1}, k in {0,...,14}
    
    # Let me verify: a = i*2+j (A index), b = j*15+k (B index), c = i*15+k (C index)
    # T[a,b,c] = 1 iff there exist i,j,k such that a=i*2+j, b=j*15+k, c=i*15+k
    
    # For Strassen on the 2x2x2 part:
    # The 7 Strassen terms for multiplying A (2x2) by B_block (2x2):
    # Let B_block have entries B[0,0], B[0,1], B[1,0], B[1,1]
    # Strassen computes C_block = A * B_block using 7 multiplications
    
    # For our case, B is 2x15, so we can split B into:
    # B = [B_0 | B_1 | ... | B_14] where each B_k is 2x1
    
    # Or better: group the 15 columns into blocks
    # Let's split into: 7 blocks of 2 columns + 1 block of 1 column
    # That gives 7 * (Strassen rank 7) + 1 * (schoolbook rank 4) = 49 + 4 = 53
    
    # Wait, Strassen for 2x2 * 2x2 has rank 7.
    # For 2x2 * 2x1, the schoolbook rank is 4 (each of 4 products is needed).
    # Can we do better for 2x2 * 2x1?
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is a 4-term sum, and I believe rank 4 is minimal for this.
    
    # So strategy: split B's 15 columns into 7 groups of 2 and 1 group of 1
    # For each group of 2 columns: use Strassen (rank 7)
    # For the group of 1 column: use schoolbook (rank 4)
    # Total rank: 7*7 + 4 = 49 + 4 = 53
    
    # Let's implement this.
    
    # Strassen terms for 2x2 * 2x2:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00-a11)*(b10+b11)
    # M2 = (a00+a01)*b10
    # M3 = a11*(b01-b11)
    # M4 = a00*b00
    # M5 = (a01+a11)*(b00+b11)
    # M6 = (a00-a01)*b01
    # M7 = (a10+a00)*(b11-b00)
    # C00 = M4 + M5 - M1 - M7
    # C01 = M2 + M6
    # C10 = M3 + M5
    # C11 = M1 + M4 - M2 + M7
    
    # Now let's build the decomposition.
    # U has shape (4, rank), V has shape (30, rank), W has shape (30, rank)
    
    # For Strassen block handling columns k1, k2 (two consecutive columns):
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices for column k1: 0*15+k1=k1, 1*15+k1=15+k1
    # B indices for column k2: 0*15+k2=k2, 1*15+k2=15+k2
    # C indices for column k1: 0*15+k1=k1, 1*15+k1=15+k1
    # C indices for column k2: 0*15+k2=k2, 1*15+k2=15+k2
    
    # Strassen term t=0: M1 = (a00-a11)*(b10+b11)
    # U contribution: U[0,t]=1, U[3,t]=-1 (a00 - a11)
    # V contribution: V[15+k1,t]=1, V[15+k2,t]=1 (b10 + b11 for the two columns)
    # Wait, I need to be more careful.
    
    # Let me redefine. For a Strassen block handling B columns at positions k1 and k2:
    # B[0,k1] is at index 0*15+k1 = k1
    # B[1,k1] is at index 1*15+k1 = 15+k1
    # B[0,k2] is at index 0*15+k2 = k2
    # B[1,k2] is at index 1*15+k2 = 15+k2
    
    # M1 = (A[0,0]-A[1,1]) * (B[1,k1]+B[1,k2])
    # This contributes to:
    # C[0,k1] gets +M1? No wait...
    # C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1]
    # C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2]
    # C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1]
    # C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2]
    
    # Strassen:
    # M1 = (A[0,0]-A[1,1])*(B[1,k1]+B[1,k2])
    # M2 = (A[0,0]+A[0,1])*B[1,k1]
    # M3 = A[1,1]*(B[0,k2]-B[1,k2])
    # M4 = A[0,0]*B[0,k1]
    # M5 = (A[0,1]+A[1,1])*(B[0,k1]+B[1,k1])
    # M6 = (A[0,0]-A[0,1])*B[0,k2]
    # M7 = (A[1,0]+A[0,0])*(B[1,k2]-B[0,k2])
    
    # C[0,k1] = M4 + M5 - M1 - M7
    # C[0,k2] = M2 + M6
    # C[1,k1] = M3 + M5
    # C[1,k2] = M1 + M4 - M2 + M7
    
    # Let me verify:
    # C[0,k1] should be A[0,0]*B[0,k1] + A[0,1]*B[1,k1]
    # M4 + M5 - M1 - M7
    # = A[0,0]*B[0,k1] + (A[0,1]+A[1,1])*(B[0,k1]+B[1,k1]) - (A[0,0]-A[1,1])*(B[1,k1]+B[1,k2]) - (A[1,0]+A[0,0])*(B[1,k2]-B[0,k2])
    # Let me expand:
    # = A00*B0k1 + (A01+A11)*B0k1 + (A01+A11)*B1k1 - (A00-A11)*B1k1 - (A00-A11)*B1k2 - (A10+A00)*B1k2 + (A10+A00)*B0k2
    # = A00*B0k1 + A01*B0k1 + A11*B0k1 + A01*B1k1 + A11*B1k1 - A00*B1k1 + A11*B1k1 - A00*B1k2 + A11*B1k2 - A10*B1k2 - A00*B1k2 + A10*B0k2 + A00*B0k2
    
    # This doesn't look right. Let me re-derive Strassen more carefully.
    
    # Standard Strassen for C = A*B where A,B are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a-d)*(g+h)
    # M2 = (a+b)*g
    # M3 = d*(f-h)
    # M4 = a*e
    # M5 = (b+d)*(e+h)
    # M6 = (a-b)*f
    # M7 = (c+a)*(h-e)
    # C11 = M4 + M5 - M1 - M7
    # C12 = M2 + M6
    # C21 = M3 + M5
    # C22 = M1 + M4 - M2 + M7
    
    # Verify C11 = a*e + b*g:
    # M4 + M5 - M1 - M7 = a*e + (b+d)*(e+h) - (a-d)*(g+h) - (c+a)*(h-e)
    # = ae + be + bh + de + dh - ag - ah + dg + dh - ch + ce - ah + ae
    # Hmm, this is getting messy. Let me just trust the standard Strassen formulas.
    
    # In our notation:
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,k1], f = B[0,k2], g = B[1,k1], h = B[1,k2]
    
    # M1 = (a-d)*(g+h) = (A00-A11)*(B1k1+B1k2)
    # M2 = (a+b)*g = (A00+A01)*B1k1
    # M3 = d*(f-h) = A11*(B0k2-B1k2)
    # M4 = a*e = A00*B0k1
    # M5 = (b+d)*(e+h) = (A01+A11)*(B0k1+B1k1)
    # M6 = (a-b)*f = (A00-A01)*B0k2
    # M7 = (c+a)*(h-e) = (A10+A00)*(B1k2-B0k2)
    
    # C[0,k1] = M4 + M5 - M1 - M7
    # C[0,k2] = M2 + M6
    # C[1,k1] = M3 + M5
    # C[1,k2] = M1 + M4 - M2 + M7
    
    # Now let's build the factor matrices.
    # For each Strassen block (handling columns k1, k2), we add 7 terms.
    # For the single-column block (column k3), we add 4 schoolbook terms.
    
    # Key insight: The tensor T[a,b,c] has support only where the indices are consistent.
    # a = i*2+j, b = j'*15+k, c = i'*15+k'
    # T[a,b,c] = 1 iff i==i', j==j', k==k'
    # So T[i*2+j, j*15+k, i*15+k] = 1
    
    # For Strassen to work on 2x2*2x2, we need the B and C indices to be "paired"
    # in the same way. The issue is that in our problem, B has 15 columns and C has 15 columns,
    # and they're coupled through the k index.
    
    # Better approach: Think of it as 15 independent 2x2*2x1 multiplications,
    # but that gives rank 15*4=60 (schoolbook).
    
    # Actually, we can group columns. For columns k1 and k2:
    # We compute C[:,k1:k2+1] = A * B[:,k1:k2+1]
    # This is a 2x2 * 2x2 multiplication, which Strassen does in rank 7.
    # The key: B[:,k1] and B[:,k2] are two columns, and C[:,k1] and C[:,k2] are two columns.
    # The Strassen decomposition for 2x2*2x2 gives us 7 terms that correctly compute
    # all 4 output entries.
    
    # Let me re-derive carefully.
    # A = [[A00, A01], [A10, A11]]
    # B_block = [[B0k1, B0k2], [B1k1, B1k2]]
    # C_block = [[C0k1, C0k2], [C1k1, C1k2]]
    
    # Strassen:
    # M1 = (A00-A11)*(B1k1+B1k2)
    # M2 = (A00+A01)*B1k1
    # M3 = A11*(B0k2-B1k2)
    # M4 = A00*B0k1
    # M5 = (A01+A11)*(B0k1+B1k1)
    # M6 = (A00-A01)*B0k2
    # M7 = (A10+A00)*(B1k2-B0k2)
    
    # C0k1 = M4 + M5 - M1 - M7
    # C0k2 = M2 + M6
    # C1k1 = M3 + M5
    # C1k2 = M1 + M4 - M2 + M7
    
    # Let me verify C0k1:
    # M4 + M5 - M1 - M7
    # = A00*B0k1 + (A01+A11)*(B0k1+B1k1) - (A00-A11)*(B1k1+B1k2) - (A10+A00)*(B1k2-B0k2)
    # = A00*B0k1 + A01*B0k1 + A01*B1k1 + A11*B0k1 + A11*B1k1 
    #   - A00*B1k1 - A00*B1k2 + A11*B1k1 + A11*B1k2
    #   - A10*B1k2 - A00*B1k2 + A10*B0k2 + A00*B0k2
    
    # This should equal A00*B0k1 + A01*B1k1.
    # The extra terms are: A01*B0k1 + A11*B0k1 + A11*B1k1 - A00*B1k1 - A00*B1k2 + A11*B1k1 + A11*B1k2 - A10*B1k2 - A00*B1k2 + A10*B0k2 + A00*B0k2
    # These don't cancel! So my Strassen application is wrong.
    
    # Wait, I think I'm confusing myself. Let me look up the standard Strassen again.
    # The standard Strassen for C = A*B:
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A11+A12)B21
    # M3 = A22(B12-B22)
    # M4 = A11B11
    # M5 = (A12+A22)(B11+B22)
    # M6 = (A11-A12)B12
    # M7 = (A21+A11)(B22-B11)
    # C11 = M4+M5-M1-M7
    # C12 = M2+M6
    # C21 = M3+M5
    # C22 = M1+M4-M2+M7
    
    # In my notation: A11=A00, A12=A01, A21=A10, A22=A11
    # B11=B0k1, B12=B0k2, B21=B1k1, B22=B1k2
    
    # So:
    # M1 = (A00-A11)*(B1k1+B1k2)  ✓
    # M2 = (A00+A01)*B1k1  ✓
    # M3 = A11*(B0k2-B1k2)  ✓
    # M4 = A00*B0k1  ✓
    # M5 = (A01+A11)*(B0k1+B1k2)  <-- I had B1k1, should be B1k2!
    # M6 = (A00-A01)*B0k2  ✓
    # M7 = (A10+A00)*(B1k2-B0k1)  <-- I had B1k2-B0k2, should be B1k2-B0k1!
    
    # Let me redo with correct M5 and M7:
    # M5 = (A01+A11)*(B0k1+B1k2)
    # M7 = (A10+A00)*(B1k2-B0k1)
    
    # Now verify C0k1 = M4 + M5 - M1 - M7:
    # = A00*B0k1 + (A01+A11)*(B0k1+B1k2) - (A00-A11)*(B1k1+B1k2) - (A10+A00)*(B1k2-B0k1)
    # = A00*B0k1 + A01*B0k1 + A01*B1k2 + A11*B0k1 + A11*B1k2 
    #   - A00*B1k1 - A00*B1k2 + A11*B1k1 + A11*B1k2
    #   - A10*B1k2 - A00*B1k2 + A10*B0k1 + A00*B0k1
    # = A00*B0k1 + A01*B0k1 + A01*B1k2 + A11*B0k1 + A11*B1k2 
    #   - A00*B1k1 - A00*B1k2 + A11*B1k1 + A11*B1k2
    #   - A10*B1k2 - A00*B1k2 + A10*B0k1 + A00*B0k1
    # This still doesn't simplify to A00*B0k1 + A01*B1k1...
    
    # I think the issue is that Strassen's algorithm is for computing the full 2x2 product,
    # and the intermediate sums/differences are shared. Let me just trust the standard formulas
    # and verify numerically.
    
    # Let me take a completely different, simpler approach:
    # Use the fact that rank is additive over direct sums.
    # The 2x2x15 multiplication can be seen as 15 copies of 2x2x1, but that's rank 60.
    # OR we can pair up columns and use Strassen for each pair.
    
    # For a pair of columns (k1, k2), we're computing:
    # [C0k1, C0k2] = A * [B0k1, B0k2; B1k1, B1k2]
    # This is a 2x2 * 2x2 matrix multiplication, which has rank 7 (Strassen).
    # The tensor for this sub-problem is supported on 4*4*4 = 64 entries (out of 4*4*4).
    # When we embed this into the full 4x30x30 tensor, the supports don't overlap
    # between different column pairs, so the ranks add.
    
    # Total: 7 pairs * rank 7 + 1 single * rank 4 = 49 + 4 = 53
    
    # Let me implement this correctly by directly using the Strassen formulas
    # and verifying numerically.
    
    rank = 53
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(30)]
    
    col = 0
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
    
    for k1, k2 in pairs:
        # Strassen 7 terms for columns k1, k2
        # a=A[0,0]=idx0, b=A[0,1]=idx1, c=A[1,0]=idx2, d=A[1,1]=idx3
        # e=B[0,k1]=k1, f=B[0,k2]=k2, g=B[1,k1]=15+k1, h=B[1,k2]=15+k2
        # C[0,k1]=k1, C[0,k2]=k2, C[1,k1]=15+k1, C[1,k2]=15+k2
        
        # M1 = (a-d)*(g+h)
        U[0][col] += 1; U[3][col] += -1
        V[15+k1][col] += 1; V[15+k2][col] += 1
        W[k1][col] += -1; W[15+k2][col] += 1
        col += 1
        
        # M2 = (a+b)*g
        U[0][col] += 1; U[1][col] += 1
        V[15+k1][col] += 1
        W[k2][col] += 1; W[15+k2][col] += -1
        col += 1
        
        # M3 = d*(f-h)
        U[3][col] += 1
        V[k2][col] += 1; V[15+k2][col] += -1
        W[15+k1][col] += 1
        col += 1
        
        # M4 = a*e
        U[0][col] += 1
        V[k1][col] += 1
        W[k1][col] += 1; W[15+k2][col] += 1
        col += 1
        
        # M5 = (b+d)*(e+h)
        U[1][col] += 1; U[3][col] += 1
        V[k1][col] += 1; V[15+k2][col] += 1
        W[k1][col] += 1; W[15+k1][col] += 1
        col += 1
        
        # M6 = (a-b)*f
        U[0][col] += 1; U[1][col] += -1
        V[k2][col] += 1
        W[k2][col] += 1
        col += 1
        
        # M7 = (c+a)*(h-e)
        U[2][col] += 1; U[0][col] += 1
        V[15+k2][col] += 1; V[k1][col] += -1
        W[k1][col] += -1; W[15+k2][col] += 1
        col += 1
    
    # Schoolbook for column 14
    k = 14
    # A00*B0k -> C0k
    U[0][col] += 1; V[k][col] += 1; W[k][col] += 1
    col += 1
    # A01*B1k -> C0k
    U[1][col] += 1; V[15+k][col] += 1; W[k][col] += 1
    col += 1
    # A10*B0k -> C1k
    U[2][col] += 1; V[k][col] += 1; W[15+k][col] += 1
    col += 1
    # A11*B1k -> C1k
    U[3][col] += 1; V[15+k][col] += 1; W[15+k][col] += 1
    col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    # Verify numerically
    if time.time() < deadline - 5:
        T = np.zeros((4, 30, 30), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    T[i*2+j, j*15+k, i*15+k] = 1
        
        # Reconstruct
        T_recon = np.zeros((4, 30, 30), dtype=float)
        for r in range(rank):
            ua = [U[a][r] for a in range(4)]
            vb = [V[b][r] for b in range(30)]
            wc = [W[c][r] for c in range(30)]
            T_recon += np.einsum('a,b,c->abc', ua, vb, wc)
        
        max_err = np.max(np.abs(T - T_recon))
        if max_err > 1e-10:
            # Fallback to schoolbook
            rank = 60
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(30)]
            W = [[0]*rank for _ in range(30)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(15):
                        U[i*2+j][col] = 1
                        V[j*15+k][col] = 1
                        W[k*2+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
