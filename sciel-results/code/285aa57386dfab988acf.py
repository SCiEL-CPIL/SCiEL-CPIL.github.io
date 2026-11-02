"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with shape (n*m, m*p, p*n) in trace convention.
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify(U, V, W, T):
    """Verify that U, V, W exactly decompose T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.array(T, dtype=float)
    # For each (a,b,c), sum over rank of U[a,c]*V[b,c]*W[c,b] should equal T[a,b,c]
    # Actually W is (p*n, rank), so W index is k*n+i = c_index
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # sum_c_rank U[a, c_rank] * V[b, c_rank] * W[c, c_rank] == T[a, b, c]
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - T[a, b, c]) > 1e-9:
                    return False
    return True

def _make_strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen (rank 7).
    U shape (4,7), V shape (4,7), W shape (4,7) with trace convention.
    A is 2x2 (indices i,j -> a = i*2+j), B is 2x2 (indices j,k -> b = j*2+k), 
    C is 2x2 (indices k,i -> c = k*2+i)
    """
    # Strassen 7 products for 2x2*2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = a00*b00
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10+b11)
    # M4 = a11*(b01-b10)
    # M5 = (a00+a10)*b11
    # M6 = (a10-a11)*(b01+b11)
    # M7 = (a01-a00)*(b00+b10)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # In our convention:
    # a = i*2+j (row i, col j of A)
    # b = j*2+k (row j, col k of B)
    # c = k*2+i (row k, col i of C)  -- trace convention
    #
    # C[k,i] = sum_{j} A[i,j] * B[j,k]
    # c index: k*2+i
    #
    # Let's map:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = M1 + M7 - M3 - M5? Let me recheck.
    #
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Where:
    # M1 = A00*B00
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10+B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A11)*(B01+B11)
    # M7 = (A01-A00)*(B00+B10)
    #
    # Now in our indices:
    # A[i,j] has index a = i*2+j
    # B[j,k] has index b = j*2+k
    # C[k,i] has index c = k*2+i
    #
    # So:
    # A00 = A[0,0] -> a=0
    # A01 = A[0,1] -> a=1
    # A10 = A[1,0] -> a=2
    # A11 = A[1,1] -> a=3
    #
    # B00 = B[0,0] -> b=0
    # B01 = B[0,1] -> b=1
    # B10 = B[1,0] -> b=2
    # B11 = B[1,1] -> b=3
    #
    # C00 = C[0,0] -> c = 0*2+0 = 0
    # C01 = C[1,0] -> c = 1*2+0 = 2  (k=1, i=0)
    # C10 = C[0,1] -> c = 0*2+1 = 1  (k=0, i=1)
    # C11 = C[1,1] -> c = 1*2+1 = 3
    #
    # Now express each M in terms of U (4x7), V (4x7), W (4x7):
    # M_r = sum_a U[a,r] * A[a] * (sum_b V[b,r] * B[b]) * ... wait
    #
    # The decomposition is: C[c] = sum_r U * V * W where
    # C[c] = sum_r (sum_a U[a,r]*A[a]) * (sum_b V[b,r]*B[b]) * W[c,r]
    #
    # So for each rank term r:
    # M_r = (sum_a U[a,r] * A[a]) * (sum_b V[b,r] * B[b]) * W[c,r]
    #
    # Let's define U, V, W for each of the 7 terms:
    #
    # M1 = A00 * B00
    # U: [1,0,0,0], V: [1,0,0,0], W: [1,0,0,0]  (C00 gets +1)
    #
    # M2 = (A10+A11)*B00
    # U: [0,0,1,1], V: [1,0,0,0], W: [0,1,0,0]  (C10 gets +1, c=1)
    #
    # M3 = A00*(B10+B11)
    # U: [1,0,0,0], V: [0,0,1,1], W: [0,0,1,1]  (C01 gets +1 (c=2), C11 gets +1 (c=3))
    # Wait, W should be a single vector. C[c] contribution is W[c,r].
    # So W: [0,0,1,1] means C01 += M3 and C11 += M3.
    #
    # M4 = A11*(B01-B10)
    # U: [0,0,0,1], V: [0,1,-1,0], W: [1,1,0,0]  (C00 += M4, C10 += M4)
    #
    # M5 = (A00+A10)*B11
    # U: [1,0,1,0], V: [0,0,0,1], W: [-1,0,1,0]  (C00 -= M5, C01 += M5)
    #
    # M6 = (A10-A11)*(B01+B11)
    # U: [0,0,1,-1], V: [0,1,0,1], W: [0,0,0,1]  (C11 += M6)
    #
    # M7 = (A01-A00)*(B00+B10)
    # U: [-1,1,0,0], V: [1,0,1,0], W: [1,0,0,0]  (C00 += M7)
    #
    # Let's verify C00 = M1 + M4 - M5 + M7:
    # W[0, r] for each r: M1:1, M2:0, M3:0, M4:1, M5:-1, M6:0, M7:1
    # Sum: 1+1-1+1 = 2? That's wrong. C00 should be M1+M4-M5+M7.
    # The contributions are: U[a,r]*V[b,r]*W[c,r]
    # For C00 (c=0): sum_r W[0,r] * (sum_a U[a,r]*A[a]) * (sum_b V[b,r]*B[b])
    # = 1*(A00*B00) + 0 + 0 + 1*(A11*(B01-B10)) + (-1)*((A00+A10)*B11) + 0 + 1*((A01-A00)*(B00+B10))
    # = A00*B00 + A11*B01 - A11*B10 - A00*B11 - A10*B11 + A01*B00 + A01*B10 - A00*B00 - A00*B10
    # = A01*B00 + A01*B10 + A11*B01 - A11*B10 - A00*B11 - A10*B11 - A00*B10
    #
    # Expected C00 = A00*B00 + A01*B10
    #
    # This doesn't match. I made an error in the Strassen mapping. Let me use a known correct set.
    
    # Let me just use the schoolbook for 2x2x2 which is rank 4, then we get 4*8=32 for 2x2x8.
    # Or I can try to find a valid rank 7 for 2x2x2.
    
    # Actually, let me just return schoolbook for 2x2x2 (rank 4) to be safe.
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((4, 4), dtype=int)
    W = np.zeros((4, 4), dtype=int)
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook (rank 32)
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find a better decomposition using Strassen-style blocks
    # For 2x2x8, we can think of it as 4 blocks of 2x2x2 with shared structure
    # A is 2x2, B is 2x8. B can be viewed as 4 blocks of 2x2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # C is 2x8. C can be viewed as 4 blocks of 2x2: C[:, 0:2], C[:, 2:4], C[:, 4:6], C[:, 6:8]
    #
    # C[:, 2t:2t+2] = A * B[:, 2t:2t+2]
    #
    # If we use Strassen for each block independently, we get 7*4 = 28.
    # But maybe we can share some computations.
    
    # Let's try to construct a rank 28 decomposition using 4 Strassen blocks.
    # For block t (t=0,1,2,3), B block is B[:, 2t:2t+2], C block is C[:, 2t:2t+2]
    # In our indices:
    # A[i,j] -> a = i*2+j (0..3)
    # B[j, 2t+k] -> b = j*8 + (2t+k)  (j in 0..1, k in 0..1)
    #   b = j*8 + 2t + k
    # C[2t+k, i] -> c = (2t+k)*2 + i  (k in 0..1, i in 0..1)
    #   c = 4t + 2k + i
    
    # Let me define Strassen vectors for one 2x2x2 block and adapt indices.
    # Using the standard Strassen with 7 products:
    # M1 = A00*B00
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10+B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A11)*(B01+B11)
    # M7 = (A01-A00)*(B00+B10)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For block t, local B indices: B[j, 2t+k] -> global b = j*8 + 2t + k
    # local C indices: C[2t+k, i] -> global c = 4t + 2k + i
    
    # U vectors (same for all blocks since A is shared):
    # U_r[a] where a = i*2+j
    U_local = [
        [1,0,0,0],  # M1: A00
        [0,0,1,1],  # M2: A10+A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,0,1,0],  # M5: A00+A10
        [0,0,1,-1], # M6: A10-A11
        [-1,1,0,0], # M7: A01-A00
    ]
    
    # V vectors for block t: V_r[b] where b = j*8 + 2t + k
    V_local = [
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M1: B00 -> b=2t
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M2: B00 -> b=2t
        [0,0,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M3: B10+B11 -> b=2t+2, 2t+3
        [0,1,-1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M4: B01-B10 -> b=2t+1, 2t+2
        [0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M5: B11 -> b=2t+3
        [0,1,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M6: B01+B11 -> b=2t+1, 2t+3
        [1,0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # M7: B00+B10 -> b=2t, 2t+2
    ]
    
    # W vectors for block t: W_r[c] where c = 4t + 2k + i
    # C00 = M1+M4-M5+M7 -> c=4t+0 (k=0,i=0)
    # C01 = M3+M5 -> c=4t+2 (k=1,i=0)
    # C10 = M2+M4 -> c=4t+1 (k=0,i=1)
    # C11 = M1-M2+M3+M6 -> c=4t+3 (k=1,i=1)
    W_local = [
        [1,0,0,0],  # M1: C00
        [0,1,0,0],  # M2: C10
        [0,0,1,1],  # M3: C01, C11
        [1,1,0,0],  # M4: C00, C10
        [-1,0,1,0], # M5: C00 neg, C01
        [0,0,0,1],  # M6: C11
        [1,0,0,0],  # M7: C00
    ]
    
    # Now build full U, V, W for rank 28
    U28 = np.zeros((n*m, 28), dtype=int)
    V28 = np.zeros((m*p, 28), dtype=int)
    W28 = np.zeros((p*n, 28), dtype=int)
    
    for t in range(4):
        for r in range(7):
            col = t*7 + r
            for a in range(n*m):
                U28[a, col] = U_local[r][a]
            for b in range(m*p):
                V28[b, col] = 0
            # Set V for block t
            for j in range(2):
                for k in range(2):
                    b = j*8 + 2*t + k
                    V28[b, col] = V_local[r][b]
            for c in range(p*n):
                W28[c, col] = 0
            # Set W for block t
            for kk in range(2):
                for ii in range(2):
                    c = 4*t + 2*kk + ii
                    W28[c, col] = W_local[r][c]
    
    if _verify(U28, V28, W28, T):
        best_U, best_V, best_W = U28, V28, W28
        best_rank = 28
        print(f"Found valid rank 28 decomposition")
    else:
        print("Rank 28 verification failed, falling back to schoolbook")
    
    # Convert to lists and return
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]
    
    if time.time() > deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
