"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Find a low-rank exact decomposition for 2x5x5 matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # We'll build the decomposition recursively.
    # The decomposition is represented as a list of tuples (U_cols, V_cols, W_cols)
    # where each tuple represents a rank-1 term with the relevant non-zero entries.
    # 
    # Actually, it's easier to just build the full matrices directly.
    # Let's use a recursive approach that returns (U, V, W) for a given (n, m, p).
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
        U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # Strassen's 7 products for 2x2x2:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U rows: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
        # V rows: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3
        # W rows: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 (W[k*2+i])
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # a=A[0,0]=U[0], d=A[1,1]=U[3]
        # e=B[0,0]=V[0], h=B[1,1]=V[3]
        # Contribution to C: M1 appears in C11 and C22
        # C11 = W[0*2+0] = W[0], C22 = W[1*2+1] = W[3]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        # c=A[1,0]=U[2], d=A[1,1]=U[3]
        # e=B[0,0]=V[0]
        # Contribution: C21 = W[0*2+1] = W[1]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = a(f-h)
        # a=A[0,0]=U[0]
        # f=B[0,1]=V[1], h=B[1,1]=V[3]
        # Contribution: C12 = W[1*2+0] = W[2], C22 = W[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        # d=A[1,1]=U[3]
        # g=B[1,0]=V[2], e=B[0,0]=V[0]
        # Contribution: C11 = W[0], C21 = W[1]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a+b)h
        # a=A[0,0]=U[0], b=A[0,1]=U[1]
        # h=B[1,1]=V[3]
        # Contribution: C11 = W[0] (with -1), C12 = W[2]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (c-a)(e+f)
        # c=A[1,0]=U[2], a=A[0,0]=U[0]
        # e=B[0,0]=V[0], f=B[0,1]=V[1]
        # Contribution: C22 = W[3]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (b-d)(g+h)
        # b=A[0,1]=U[1], d=A[1,1]=U[3]
        # g=B[1,0]=V[2], h=B[1,1]=V[3]
        # Contribution: C11 = W[0]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def build_decomposition(n, m, p):
        """Recursively build decomposition for n x m x p matrix multiplication.
        Returns (U, V, W) where U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        """
        # Base case: if any dimension is 1, use schoolbook
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            return U, V, W
        
        if n == 1 or m == 1 or p == 1:
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            return U, V, W
        
        # If n == 2, m == 2, p == 2: use Strassen
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # General case: split the largest dimension
        # We'll split p (the output columns) since it's often the largest
        # Actually, let's split based on which dimension is largest
        # For 2x5x5, we can split n: 2 = 1+1, or m: 5 = 2+3, or p: 5 = 2+3
        
        # Let's try splitting m first if m > 2
        if m > 2:
            # Split m into m1 + m2
            # Choose split that minimizes total rank
            best_rank = float('inf')
            best_split = None
            for m1 in range(1, m):
                m2 = m - m1
                # Recursively solve (n, m1, p) and (n, m2, p)
                if time.time() > deadline:
                    break
                U1, V1, W1 = build_decomposition(n, m1, p)
                U2, V2, W2 = build_decomposition(n, m2, p)
                r1 = U1.shape[1]
                r2 = U2.shape[1]
                total_rank = r1 + r2
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_split = (m1, m2, U1, V1, W1, U2, V2, W2)
            
            if best_split is not None:
                m1, m2, U1, V1, W1, U2, V2, W2 = best_split
                # Combine: U has rows for (n, m1) and (n, m2)
                U = np.zeros((n * m, best_rank))
                V = np.zeros((m * p, best_rank))
                W = np.zeros((p * n, best_rank))
                
                # First part: (n, m1, p)
                # U1: (n*m1, r1), V1: (m1*p, r1), W1: (p*n, r1)
                U[:n*m1, :r1] = U1
                # V1 rows: j in [0, m1), k in [0, p) -> index j*p+k
                V[:m1*p, :r1] = V1
                W[:, :r1] = W1
                
                # Second part: (n, m2, p)
                # U2: (n*m2, r2), V2: (m2*p, r2), W2: (p*n, r2)
                U[n*m1:, r1:] = U2
                # V2 rows: j in [m1, m), k in [0, p) -> index (j-m1)*p+k in V2, j*p+k in V
                V[m1*p:, r1:] = V2
                W[:, r1:] = W2
                
                return U, V, W
        
        # If p > 2, split p
        if p > 2:
            best_rank = float('inf')
            best_split = None
            for p1 in range(1, p):
                p2 = p - p1
                if time.time() > deadline:
                    break
                U1, V1, W1 = build_decomposition(n, m, p1)
                U2, V2, W2 = build_decomposition(n, m, p2)
                r1 = U1.shape[1]
                r2 = U2.shape[1]
                total_rank = r1 + r2
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_split = (p1, p2, U1, V1, W1, U2, V2, W2)
            
            if best_split is not None:
                p1, p2, U1, V1, W1, U2, V2, W2 = best_split
                U = np.zeros((n * m, best_rank))
                V = np.zeros((m * p, best_rank))
                W = np.zeros((p * n, best_rank))
                
                # First part: (n, m, p1)
                # U1: (n*m, r1), V1: (m*p1, r1), W1: (p1*n, r1)
                U[:, :r1] = U1
                V[:, :r1] = V1
                # W1 rows: k in [0, p1), i in [0, n) -> index k*n+i
                W[:p1*n, :r1] = W1
                
                # Second part: (n, m, p2)
                # U2: (n*m, r2), V2: (m*p2, r2), W2: (p2*n, r2)
                U[:, r1:] = U2
                V[:, r1:] = V2
                # W2 rows: k in [p1, p), i in [0, n) -> index (k-p1)*n+i in W2, k*n+i in W
                W[p1*n:, r1:] = W2
                
                return U, V, W
        
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Build the decomposition
    U, V, W = build_decomposition(n, m, p)
    
    # Convert to lists of lists for the output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
