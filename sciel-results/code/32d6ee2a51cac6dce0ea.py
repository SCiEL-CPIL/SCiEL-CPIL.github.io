import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # We'll use a Strassen-like recursive construction.
    # The key idea: 3x3 * 3x4 can be embedded in 4x4 * 4x4.
    # The rank of 4x4x4 matrix multiplication is 23 (Strassen's algorithm).
    # We project the 4x4x4 factors onto the 3x3x4 subspace.
    
    # First, let's build the rank-23 decomposition for 4x4x4 matrix multiplication.
    # This uses Strassen's algorithm recursively.
    
    # Strassen's 2x2x2 decomposition (rank 7)
    def strassen_2x2():
        # Returns (U, V, W) each of shape (4, 7) for 2x2 * 2x2
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        # Strassen's 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Row indices: U: 0=A11, 1=A12, 2=A21, 3=A22
        #             V: 0=B11, 1=B12, 2=B21, 3=B22
        #             W: 0=C11, 1=C12, 2=C21, 3=C22
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A11(B21-B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B12-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Now build 4x4x4 using Strassen's recursive algorithm.
    # 4x4 = 2x2 blocks, each 2x2.
    # This gives rank 7^2 = 49 for 4x4x4, which is not optimal.
    # The optimal rank for 4x4x4 is 23 (Alon-Boppana or similar).
    
    # Actually, let me use a different approach. Let me directly construct
    # a rank-22 or rank-23 decomposition for 3x3x4.
    
    # Alternative: Use the fact that 3x3 * 3x4 can be viewed as computing
    # 12 outputs, each requiring 3 multiplications (schoolbook = 36).
    # We can use Strassen-like sharing to reduce this.
    
    # Let me try a direct construction using the (2,3) Strassen algorithm.
    # This computes a 2x3 * 3x4 product using fewer than 24 multiplications.
    
    # Actually, let me try a more systematic approach.
    # I'll use the known result that the tensor rank of (3,3,4) is at most 22.
    # Let me construct it explicitly.
    
    # One approach: use the "border rank" or "approximate rank" results,
    # but we need exact rank.
    
    # Let me try to find a rank-22 decomposition by solving the system
    # with integer constraints.
    
    # Actually, the most reliable approach for small dimensions is to use
    # a known explicit construction. Let me use the following strategy:
    # 
    # For 3x3 * 3x4, we can use the following block decomposition:
    # Split the 3x4 into a 2x4 and a 1x4 row.
    # Compute 3x3 * 2x4 (the top two rows of B) and 3x3 * 1x4 (bottom row).
    
    # Hmm, this is getting complex. Let me just use a computational approach
    # with a better initialization.
    
    # Key insight: use the schoolbook decomposition as a starting point
    # and try to merge terms using algebraic identities.
    
    # Let me try a different strategy: use the known rank-23 decomposition
    # for 4x4x4 and project it down.
    
    # Actually, let me just implement a proper search with the constraint
    # that entries are integers or half-integers.
    
    # For now, let me return a known valid decomposition.
    # The schoolbook decomposition has rank 36.
    # Let me try to find something better.
    
    # I'll use a hybrid approach: start with a good initial guess based on
    # Strassen's algorithm and refine.
    
    # Let me construct a rank-24 decomposition using a block Strassen approach.
    
    # 3x3 * 3x4:
    # A is 3x3, B is 3x4, C is 3x4.
    # 
    # Split A into [A1; A2] where A1 is 2x3 and A2 is 1x3.
    # Split B into [B1; B2] where B1 is 2x4 and B2 is 1x4.
    # C1 = A1 * B1 (2x2 * 2x4, but A1 is 2x3, B1 is 3x4... wait)
    
    # Let me be more careful.
    # A: 3x3, B: 3x4, C: 3x4
    # 
    # Split A into rows: A = [A_top; A_bottom] where A_top is 2x3, A_bottom is 1x3
    # Split B into columns: B = [B_left | B_right] where B_left is 3x2, B_right is 3x2
    # 
    # C = [C_left | C_right] where C_left is 3x2, C_right is 3x2
    # 
    # C_left = A * B_left (3x3 * 3x2 = 3x2)
    # C_right = A * B_right (3x3 * 3x2 = 3x2)
    
    # For each 3x3 * 3x2 multiplication, the schoolbook rank is 18.
    # Can we do better?
    
    # Actually, let me just try to find a valid low-rank decomposition
    # using a randomized approach with integer rounding.
    
    def verify(U, V, W, rank):
        """Verify that U, V, W form a valid decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        # Also check that inactive triples give 0
        # Active triples are (i*m+j, j*p+k, k*n+i) for valid i,j,k
        # Inactive triples are all other (a,b,c) where a in [0,n*m), b in [0,m*p), c in [0,p*n)
        # but not of the form (i*m+j, j*p+k, k*n+i)
        
        active_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    active_set.add((i*m+j, j*p+k, k*n+i))
        
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in active_set:
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        max_err = max(max_err, abs(val))
        
        return max_err < 1e-6
    
    # Let me try a direct construction.
    # I'll use the following approach: start with the schoolbook decomposition
    # and try to find linear dependencies to reduce the rank.
    
    # Schoolbook decomposition:
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Now, let me try to find a lower rank decomposition.
    # I'll use a greedy approach: try to express one schoolbook term as a
    # combination of others.
    
    # Actually, let me just try a randomized search with a time limit.
    
    best_result = None
    best_rank = rank_school
    
    # Try ranks from 22 to best_rank - 1
    for rank in range(22, min(best_rank, 30)):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
            
            # Check if this is a valid decomposition
            if verify(U, V, W, rank):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: schoolbook decomposition
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return best_result
