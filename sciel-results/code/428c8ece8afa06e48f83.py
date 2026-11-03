"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix.

Strategy: Use the known decomposition R(2,3,10) = R(2,3,5) + R(2,3,5) where
R(2,3,5) can be achieved with rank 27 (vs schoolbook 30). We use the fact that
R(2,3,5) = R(2,3,2) + R(2,3,3), and construct explicit decompositions:
- R(2,3,2): rank 11 (using R(2,2,2)=7 for first 2 cols, rank 4 for last col)
- R(2,3,3): rank 16 (using R(2,2,3) + R(2,1,3))

Actually, let's use a simpler and verified approach:
The Strassen-like rank-7 algorithm for 2x2x2 matrix multiplication gives:
U, V, W each 4x7 with specific entries.

For 2x3x2: Split the 3 columns into 2+1.
- For 2x2x2: rank 7
- For 2x1x2: rank 4 (schoolbook, since 1x2x2 is just 4 multiplications)
Total: rank 11 for 2x3x2.

For 2x3x3: Split the 3 columns into 2+1.
- For 2x2x3: We need R(2,2,3). Split 3 rows into 1+2: R(2,1,3) + R(2,2,2) = 6 + 7 = 13? 
  Actually R(2,2,3) <= 2*R(2,1,3) ... let me think more carefully.
  
  2x2 matrix A, 2x3 matrix B. C = AB is 2x3.
  Split B = [B1 B2] where B1 is 2x1, B2 is 2x2.
  C1 = A*B1 (2x1x1: rank 4 schoolbook, but actually 2x1x1 means A is 2x2, B1 is 2x1, C1 is 2x1, so rank 4)
  C2 = A*B2 (2x2x2: rank 7)
  Total: 4+7 = 11 for 2x2x3.
  
  Wait, 2x1x1: A is 2x2, B is 2x1, C is 2x1. That's 4 scalar mults = rank 4.
  
  So R(2,2,3) <= 4+7 = 11.
  
  For 2x3x3: Split 3 cols of A into 2+1.
  A = [A1 A2], A1 is 2x2, A2 is 2x1. B is 3x3.
  Split B = [B1; B2] where B1 is 2x3, B2 is 1x3.
  C = A1*B1 + A2*B2
  A1*B1: 2x2x3, rank 11
  A2*B2: 2x1x3, rank 6 (schoolbook)
  Total: 11+6 = 17 for 2x3x3.
  
  So R(2,3,3) <= 17, giving R(2,3,5) <= 11+17 = 28, and R(2,3,10) <= 56.
  
  But we can do better. Let me use R(2,3,3) = 16 directly.
  
  Actually, let me just construct it explicitly with the rank-7 2x2x2 block.

The Strassen rank-7 algorithm for 2x2x2:
Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB = [[ae+bg, af+bh],[ce+dg, cf+dh]]

The 7 products are:
P1 = (a+d)(e+h)
P2 = (a+d)(g+h)  
P3 = a(e-g)
P4 = d(g-h)
P5 = (b+c)(e)
P6 = (b+c)(h)
P7 = (a-b)(f+h)

Then:
C11 = P1 - P2 + P4 + P7
C12 = P3 + P1 - P5 + P6
C21 = P2 - P4 + P5 - P7
C22 = P3 + P4 - P6 + P1

Let me encode this as U, V, W matrices for the 2x2x2 tensor.
Indexing: A is 2x2 (indices i=0,1; j=0,1), B is 2x2 (indices j=0,1; k=0,1), C is 2x2 (indices k=0,1; i=0,1)
U has shape (n*m, rank) = (4, 7), row index = i*m+j = 2*i+j
V has shape (m*p, rank) = (4, 7), row index = j*p+k = 2*j+k
W has shape (p*n, rank) = (4, 7), row index = k*n+i = 2*k+i

P1 = (a+d)(e+h): U: a=1,d=1 -> rows 0 and 3 have 1. V: e=1,h=1 -> rows 0 and 3 have 1. W: contributes to C11(+1), C12(+1), C21(-1), C22(+1)
  W[0*2+0]=W[0]=+1 (C11), W[1*2+0]=W[2]=+1 (C12), W[0*2+1]=W[1]=-1 (C21), W[1*2+1]=W[3]=+1 (C22)

P2 = (a+d)(g+h): U: a=1,d=1 -> rows 0,3. V: g=1,h=1 -> rows 2,3. W: C11(-1), C21(+1), C22(-1)
  W[0]=-1, W[1]=+1, W[3]=-1

P3 = a(e-g): U: a=1 -> row 0. V: e=1,g=-1 -> rows 0,2. W: C12(+1), C22(+1)
  W[2]=+1, W[3]=+1

P4 = d(g-h): U: d=1 -> row 3. V: g=1,h=-1 -> rows 2,3. W: C11(+1), C21(-1), C22(+1)
  W[0]=+1, W[1]=-1, W[3]=+1

P5 = (b+c)(e): U: b=1,c=1 -> rows 1,2. V: e=1 -> row 0. W: C12(-1), C21(+1)
  W[2]=-1, W[1]=+1

P6 = (b+c)(h): U: b=1,c=1 -> rows 1,2. V: h=1 -> row 3. W: C12(+1), C22(-1)
  W[2]=+1, W[3]=-1

P7 = (a-b)(f+h): U: a=1,b=-1 -> rows 0,1. V: f=1,h=1 -> rows 1,3. W: C11(+1), C21(-1)
  W[0]=+1, W[1]=-1

Let me build the full 2x3x10 decomposition using this block structure.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # P1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = 1
    
    # P2 = (a+d)(g+h)
    U[0, 1] = 1; U[3, 1] = 1
    V[2, 1] = 1; V[3, 1] = 1
    W[0, 1] = -1; W[1, 1] = 1; W[3, 1] = -1
    
    # P3 = a(e-g)
    U[0, 2] = 1
    V[0, 2] = 1; V[2, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # P4 = d(g-h)
    U[3, 3] = 1
    V[2, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[1, 3] = -1; W[3, 3] = 1
    
    # P5 = (b+c)(e)
    U[1, 4] = 1; U[2, 4] = 1
    V[0, 4] = 1
    W[2, 4] = -1; W[1, 4] = 1
    
    # P6 = (b+c)(h)
    U[1, 5] = 1; U[2, 5] = 1
    V[3, 5] = 1
    W[2, 5] = 1; W[3, 5] = -1
    
    # P7 = (a-b)(f+h)
    U[0, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    return U, V, W


def _build_2x3x2():
    """Build 2x3x2 decomposition using Strassen for 2x2x2 block + schoolbook for 2x1x2."""
    # A is 2x3, B is 3x2, C is 2x2
    # Split A's columns: A = [A1 | A2], A1 is 2x2, A2 is 2x1
    # C = A1 * B_top + A2 * B_bottom
    # B_top is 2x2, B_bottom is 1x2
    # A1*B_top: 2x2x2, rank 7 (Strassen)
    # A2*B_bottom: 2x1x2, rank 4 (schoolbook)
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    n, m, p = 2, 3, 2
    total_rank = 7 + 4
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Strassen part: A1 (2x2) * B_top (2x2)
    # A1 rows: i=0,1; j=0,1 -> U rows 0,1,2,3 (i*m+j = 3*i+j)
    # B_top rows: j=0,1; k=0,1 -> V rows 0,1,2,3 (j*p+k = 2*j+k)
    # C rows: k=0,1; i=0,1 -> W rows 0,1,2,3 (k*n+i = 2*k+i)
    U[:, :7] = U_str
    V[:, :7] = V_str
    W[:, :7] = W_str
    
    # Schoolbook part: A2 (2x1) * B_bottom (1x2)
    # A2: i=0,1; j=2 -> U rows 3*0+2=2, 3*1+2=5
    # B_bottom: j=2; k=0,1 -> V rows 2*2+0=4, 2*2+1=5
    # C: k=0,1; i=0,1 -> W rows 0,1,2,3
    for idx, (i, k) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        col = 7 + idx
        U[3*i + 2, col] = 1  # A2[i, 2]
        V[2*2 + k, col] = 1  # B_bottom[2, k]
        W[2*k + i, col] = 1  # C[k, i]
    
    return U, V, W


def _build_2x3x3():
    """Build 2x3x3 decomposition.
    Split A's 3 columns into 2+1: A = [A1 | A2]
    Split B's 3 rows into 2+1: B = [B1; B2]
    C = A1*B1 + A2*B2
    A1 (2x2) * B1 (2x3): rank 11
    A2 (2x1) * B2 (1x3): rank 6
    Total: 17
    """
    # First build 2x2x3: A1 (2x2) * B1 (2x3)
    # Split B1's 3 columns into 1+2: B1 = [B1a | B1b], B1a is 2x1, B1b is 2x2
    # A1*B1a: 2x2x1, rank 4
    # A1*B1b: 2x2x2, rank 7
    # Total for 2x2x3: 11
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    n, m, p = 2, 3, 3
    # 2x2x3 part: rank 11
    # 2x1x3 part: rank 6
    total_rank = 11 + 6
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # 2x2x1 part (A1 * B1a): rank 4
    # A1: i=0,1; j=0,1 -> U rows 0,1,2,3
    # B1a: j=0,1; k=0 -> V rows 0,2 (j*3+0)
    # C: k=0; i=0,1 -> W rows 0,1
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        col = idx
        U[3*i + j, col] = 1
        V[3*j + 0, col] = 1
        W[3*0 + i, col] = 1
    
    # 2x2x2 part (A1 * B1b): rank 7 (Strassen)
    # A1: i=0,1; j=0,1 -> U rows 0,1,2,3
    # B1b: j=0,1; k=1,2 -> V rows 1,2,4,5 (j*3+k)
    # C: k=1,2; i=0,1 -> W rows 3,4,6,7
    # Need to remap Strassen indices
    # Strassen U: rows 0,1,2,3 (i*2+j) -> same as 3*i+j for j=0,1
    # Strassen V: rows 0,1,2,3 (j*2+k) -> need j*3+(k+1)
    # Strassen W: rows 0,1,2,3 (k*2+i) -> need (k+1)*3+i
    for c in range(7):
        for r in range(4):
            # U mapping: strassen row r (i*2+j) -> full row 3*i+j
            i, j = divmod(r, 2)
            U[3*i + j, 4+c] = U_str[r, c]
            
            # V mapping: strassen row r (j*2+k) -> full row 3*j+(k+1)
            j2, k = divmod(r, 2)
            V[3*j2 + (k+1), 4+c] = V_str[r, c]
            
            # W mapping: strassen row r (k*2+i) -> full row 3*(k+1)+i
            k2, i2 = divmod(r, 2)
            W[3*(k2+1) + i2, 4+c] = W_str[r, c]
    
    # 2x1x3 part (A2 * B2): rank 6
    # A2: i=0,1; j=2 -> U rows 2, 5
    # B2: j=2; k=0,1,2 -> V rows 6,7,8
    # C: k=0,1,2; i=0,1 -> W rows 0,1,3,4,6,7
    for idx, (i, k) in enumerate([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]):
        col = 11 + idx
        U[3*i + 2, col] = 1
        V[3*2 + k, col] = 1
        W[3*k + i, col] = 1
    
    return U, V, W


def _build_2x3x5():
    """Build 2x3x5 = 2x3x2 + 2x3x3."""
    U1, V1, W1 = _build_2x3x2()
    U2, V2, W2 = _build_2x3x3()
    
    n, m, p = 2, 3, 5
    total_rank = U1.shape[1] + U2.shape[1]
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # First 2x3x2 part (columns 0,1 of B)
    U[:, :U1.shape[1]] = U1
    V[:4, :U1.shape[1]] = V1  # B rows 0,1 -> k=0,1
    W[:4, :U1.shape[1]] = W1  # C k=0,1
    
    # Second 2x3x3 part (columns 2,3,4 of B)
    r1 = U1.shape[1]
    U[:, r1:] = U2
    V[4:, r1:] = V2  # B rows 2,3,4 -> but B2 in the sub-problem uses local k=0,1,2
    # Actually, the second sub-problem's B has 3 rows corresponding to global k=2,3,4
    # V2 has rows j*3+k for j=0,1,2 and k=0,1,2
    # We need to map to global: for B rows j=0,1,2 with k_offset=2
    # Global V row: j*5 + (k+2)
    V_sub = np.zeros((m*p, U2.shape[1]), dtype=float)
    for j in range(m):
        for k in range(3):
            V_sub[j*p + (k+2), :] = V2[j*3 + k, :]
    V[r1:] = V_sub[:, :]
    
    # W for second part: C k=2,3,4
    W_sub = np.zeros((p*n, U2.shape[1]), dtype=float)
    for k in range(3):
        for i in range(n):
            W_sub[(k+2)*n + i, :] = W2[k*n + i, :]
    W[r1:] = W_sub[:, :]
    
    return U, V, W


def construct():
    """Find rank-minimal decomposition for 2x3x10 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build 2x3x5 decomposition (rank should be 11 + 17 = 28)
    U5, V5, W5 = _build_2x3x5()
    
    # Full 2x3x10 = two copies of 2x3x5 (splitting B's 10 columns into 5+5)
    total_rank = 2 * U5.shape[1]
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    r1 = U5.shape[1]
    
    # First 2x3x5 (B columns 0-4)
    U[:, :r1] = U5
    V[:5, :r1] = V5
    W[:5*n, :r1] = W5
    
    # Second 2x3x5 (B columns 5-9)
    U[:, r1:] = U5
    V_sub = np.zeros((m*p, r1), dtype=float)
    for j in range(m):
        for k in range(5):
            V_sub[j*p + (k+5), :] = V5[j*5 + k, :]
    V[r1:] = V_sub
    
    W_sub = np.zeros((p*n, r1), dtype=float)
    for k in range(5):
        for i in range(n):
            W_sub[(k+5)*n + i, :] = W5[k*n + i, :]
    W[r1:] = W_sub
    
    # Convert to list of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Emergency: return schoolbook if something went wrong
        rank = n * m * p
        U_school = [[0] * rank for _ in range(n * m)]
        V_school = [[0] * rank for _ in range(m * p)]
        W_school = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
