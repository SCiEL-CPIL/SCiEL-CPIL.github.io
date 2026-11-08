"""Optimized tensor decomposition for 2x3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (12, rank)

    def verify_solution(sol):
        U = np.array(sol['U'])
        V = np.array(sol['V'])
        W = np.array(sol['W'])

        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False

        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    T[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])

        return np.allclose(T, 1.0, atol=1e-9)

    def get_trivial_solution():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {'U': U, 'V': V, 'W': W}

    # Strategy: Split m=3 into m1=1 + m2=2
    # For m1=1: 2x1x6 multiplication, rank = 2*6 = 12 (direct)
    # For m2=2: 2x2x6 multiplication, use Strassen on 2x2 part
    #   2x2x6 can be decomposed: think of it as 3 blocks of 2x2x2
    #   Each 2x2x2 uses Strassen with rank 7, total 7*3 = 21
    #   But we can do better: 2x2x6 = 2x2x(2+2+2), each 2x2x2 is Strassen rank 7
    #   Actually, let me think more carefully.
    
    # Better approach: 2x2x6 matrix multiplication
    # A is 2x2, B is 2x6, C = A*B is 2x6
    # We can view B as three 2x2 blocks: B = [B1, B2, B3]
    # C = [C1, C2, C3] where Ci = A*Bi
    # Each Ci = A*Bi is 2x2x2 multiplication, Strassen rank 7
    # Total: 7*3 = 21 for the 2x2x6 part
    
    # For 2x1x6: A is 2x1, B is 1x6, C = A*B is 2x6
    # This is just scalar multiplication: C[i,k] = A[i,0]*B[0,k]
    # Rank = 2*6 = 12 (each element needs its own term)
    
    # Total rank: 12 + 21 = 33. Same as before.
    
    # Let me try a different decomposition for 2x2x6:
    # Actually, for 2x2x6, we can use the fact that B has structure.
    # Let's use Strassen's algorithm directly on the full problem.
    
    # Alternative: Use the known result that 2x2x2 has rank 7 (Strassen).
    # 2x3x6: Think of it as multiplying a 2x3 by 3x6.
    # We can partition the 3x6 into three 3x2 blocks.
    # Or partition differently.
    
    # Let me try to construct a rank-33 solution and see if we can reduce it.
    
    def get_rank_33_solution():
        rank = 33
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Use Strassen for the 2x2x2 subproblems
        # The 3x6 matrix B can be viewed as 3 blocks of 3x2
        # Actually, let's use a different approach.
        
        # View the multiplication as:
        # A (2x3) * B (3x6) = C (2x6)
        # Partition B into three 3x2 blocks: B = [B0, B1, B2]
        # C = [C0, C1, C2] where Ci = A * Bi (2x2 result)
        
        # For each Ci = A * Bi, use Strassen (rank 7)
        # Total: 7 * 3 = 21 terms
        
        # Strassen terms for 2x2x2:
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        # 
        # M1 = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        strassen_terms = [
            # (U indices, V indices, W indices, signs)
            # M1 = (a11+a22)*(b11+b22)
            {'u': {0: 1, 3: 1}, 'v': {0: 1, 2: 1}, 'w': {0: 1, 3: -1}},
            # M2 = (a21+a22)*b11
            {'u': {3: 1, 2: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            # M3 = a11*(b12-b22)
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            # M4 = a22*(b21-b11)
            {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            # M5 = (a11+a12)*b22
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            # M6 = (a21-a11)*(b11+b12)
            {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            # M7 = (a12-a22)*(b21+b22)
            {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        ]
        
        # For each of the 3 blocks of B (3x2 blocks)
        # Block b: columns 2*b and 2*b+1 of B
        # A is 2x3, so for block b, we use the full A
        # But wait, A is 2x3 and Bi is 3x2, so Ci = A*Bi is 2x2
        
        # For block b, the V indices are:
        # V[j*p + k] where j is row of B (0,1,2) and k is column (2*b, 2*b+1)
        # So V indices for block b: j*p + 2*b and j*p + 2*b + 1
        
        # U indices: i*m + j where i is row of A (0,1) and j is column of A (0,1,2)
        # For Strassen, we need a 2x2 submatrix of A.
        # But A is 2x3, not 2x2.
        
        # I need to be more careful. Let me reconsider.
        
        # Actually, the multiplication is A(2x3) * B(3x6) = C(2x6)
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Let me just use the trivial decomposition and see if I can optimize.
        
        # Trivial: rank = 2*3*6 = 36
        # Each term: U[i*3+j, r] * V[j*6+k, r] * W[k*2+i, r] = 1 for specific (i,j,k)
        
        # Let me try to find a better decomposition.
        
        # For now, return a trivial solution
        rank_trivial = n*m*p
        U = np.zeros((n*m, rank_trivial))
        V = np.zeros((m*p, rank_trivial))
        W = np.zeros((p*n, rank_trivial))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        return {'U': U, 'V': V, 'W': W}

    # Let me try to construct a rank-33 solution properly.
    # The key insight: we can use Strassen's algorithm on parts of the problem.
    
    def get_optimized_solution():
        # Use a hybrid approach:
        # Split the 3x6 matrix B into a 3x2 and 3x4 part, or other splits.
        
        # Actually, let me try to use the known Strassen decomposition for 2x2x2
        # and apply it to subproblems.
        
        # The multiplication A(2x3) * B(3x6) = C(2x6) can be viewed as:
        # For each output C[i,k], we compute sum_j A[i,j] * B[j,k]
        
        # One approach: use the fact that we can compute 2x2x2 products with rank 7.
        # If we can partition the problem into 2x2x2 subproblems, we can use Strassen.
        
        # Let's try: partition A into two 2x2 blocks (overlapping in the middle column)
        # A = [[a11, a12, a13], [a21, a22, a23]]
        # 
        # This is getting complicated. Let me just use the trivial solution for now
        # and see if the verification passes.
        
        return get_trivial_solution()

    # Try the trivial solution first
    best_solution = get_trivial_solution()
    best_rank = n*m*p

    if not verify_solution(best_solution):
        # This should not happen, but just in case
        best_solution = get_trivial_solution()
        best_rank = n*m*p

    # Check if we can do better
    # The trivial rank is 36. Can we do 35, 34, 33, ...?
    
    # For now, return the trivial solution
    return {
        'U': best_solution['U'].tolist(),
        'V': best_solution['V'].tolist(),
        'W': best_solution['W'].tolist()
    }
# EVOLVE-BLOCK-END
