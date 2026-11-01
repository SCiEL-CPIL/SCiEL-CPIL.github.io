"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.
Strategy:
1. Start with the schoolbook rank 60 decomposition.
2. Attempt to construct a lower-rank decomposition using a structured approach:
   - View the 2x3 * 3x10 multiplication as combining a 2x1 * 1x3 and a 3x10 structure.
   - Specifically, try to find a decomposition that exploits the small dimension n=2.
   - The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,...,9}.
   - We can try to use a rank-2 decomposition for the i-dimension (since n=2).
   - For each j,k, the contribution is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
   - If we can write the "i-part" as a rank-2 tensor, we might reduce the total rank.
   
   Known result: The rank of matrix multiplication for 2x3 * 3x10 is not trivially known.
   However, we can try a heuristic:
   - Use the fact that for n=2, the W factor has shape (20, r) with W[k*2+i, c].
   - We can try to set up a system of linear equations for a target rank r < 60.
   - Since the tensor is bilinear in A and B, and linear in C, we can fix U and V and solve for W,
     or use iterative methods.

   Given the complexity, a robust approach is:
   1. Try rank 30 (the trivial lower bound max(n*m, m*p, p*n) = max(6, 30, 20) = 30).
   2. If rank 30 is not achievable with a simple construction, try slightly higher ranks.
   3. Use a randomized local search or algebraic construction if feasible.

   However, for 2x3 * 3x10, the schoolbook is 60. 
   A known decomposition for 2x3 * 3x10 might use rank 54 or similar?
   Actually, let's look at the structure.
   C = A B. A is 2x3, B is 3x10.
   C[i,k] = sum_j A[i,j] B[j,k].
   
   We can try to use a "split" method.
   Let's try to find a decomposition with rank < 60 using a simple iterative refinement on the schoolbook.
   Or, use the fact that rank(MM_{n,m,p}) <= rank(MM_{n,m,p'}) * (p/p') for some p'? No.
   
   Let's try a specific construction:
   We can decompose the multiplication into two parts:
   Part 1: 2x3 * 3x10.
   
   Let's try to use a rank-1 approximation for the "i" index? No, it must be exact.
   
   Alternative: Use the fact that 2x3 * 3x10 is a sub-problem of 2x3 * 3x12? No.
   
   Let's try to search for a valid decomposition with rank < 60 using a simple greedy algorithm:
   1. Initialize U, V, W with schoolbook.
   2. Try to remove one column (reduce rank) by solving for the remaining columns to compensate.
   3. This is a linear system problem: sum_c U_a,c V_b,c W_c,c = 1.
   
   Actually, a simpler heuristic:
   Try to use a decomposition based on the identity:
   (A00 A01 A02)   (B00 ... B09)
   (A10 A11 A12)   (B10 ... B19)
   (A20 A21 A22)   (B20 ... B29)
   
   We can try to combine the 3 intermediate dimensions.
   
   Given the time constraint and the difficulty of finding a provably lower rank without heavy computation,
   and the fact that previous attempts (random search) failed, we will stick to the schoolbook rank 60
   UNLESS we can quickly verify a lower rank.
   
   However, there is a known result that rank(MM_{2,3,10}) is likely 60 or close.
   Wait, the lower bound is 30. Can we do 30?
   If we can find U (6x30), V (30x30), W (20x30) such that the product is 1.
   
   Let's try a specific construction for rank 30.
   If we set r=30, we have 30 columns.
   For each (i,j,k), we need sum_c U[3i+j, c] V[3j+k, c] W[10k+i, c] = 1.
   
   This is a system of 60 equations (2*3*10) with 30 * (6+30+20) = 30*56 = 1680 variables? No, the variables are the entries of U, V, W.
   Total variables: 6*30 + 30*30 + 20*30 = 180 + 900 + 600 = 1680.
   Equations: 60.
   It is likely solvable.
   
   Let's try to find a solution for rank 30 using least squares (numpy.linalg.lstsq) and then round to integers/half-integers?
   The problem requires integer/half-integer entries.
   If we solve in reals, we might get non-integer values.
   However, if a rational solution exists, we might find it.
   
   Let's try to solve for rank 30 with integer constraints? That's an NP-hard problem.
   But maybe a simple solution exists.
   
   Let's try rank 30 with a specific structure.
   Suppose we use the same column index for all (i,j,k) in a block?
   
   Actually, let's just return the schoolbook for now, as it is guaranteed to work.
   But we can try to verify if rank 30 is possible with a quick linear solve and check if the solution is "nice" (integers).
   If not, return schoolbook.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, leave 5s margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60
    
    # 1. Construct Schoolbook (Guaranteed)
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()
    
    # 2. Attempt to find a lower rank decomposition (e.g., rank 30)
    # We will try to solve for U, V, W with rank r.
    # This is a multi-linear problem.
    # We can fix U and V to random (or structured) values and solve for W, then iterate.
    # Or, we can use the fact that for n=2, we can separate the i-index.
    
    # Let's try rank 30.
    r = 30
    # Initialize U, V with random integers in {-1, 0, 1}
    np.random.seed(42)
    U = np.random.randint(-1, 2, (n * m, r))
    V = np.random.randint(-1, 2, (m * p, r))
    
    # Solve for W
    # For each (i, k), we need W[10k+i, :] to satisfy:
    # sum_j (sum_c U[3i+j, c] V[3j+k, c] W[10k+i, c]) = 1
    # This is a linear system for W[10k+i, :] for each (i,k).
    # Let A_ik be a matrix of size (m, r) where A_ik[j, c] = U[3i+j, c] * V[3j+k, c].
    # Then we need A_ik @ W_ik = 1 (vector of ones of length m).
    
    W_sol = np.zeros((p * n, r), dtype=float)
    valid = True
    for i in range(n):
        for k in range(p):
            A_ik = np.zeros((m, r))
            for j in range(m):
                A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
            # Solve A_ik @ W_vec = ones(m)
            try:
                W_vec, residuals, rank_A, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                W_sol[k * n + i, :] = W_vec
            except:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        # Check if the solution is close to integers or half-integers
        # And verify the full decomposition
        # Scale W to see if it's nice?
        # For now, check if we can round U, V, W to half-integers and still satisfy the condition.
        # This is a strong constraint.
        # Let's just check the error with the real-valued solution first.
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W_sol[k * n + i, c]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        if max_err < 1e-9:
            # We have a real-valued solution.
            # Can we convert to integers?
            # If U, V are integers, and W is rational, we can scale.
            # But the problem requires integer/half-integer entries.
            # If W has entries like 0.5, 1.5, etc., it's valid.
            # Let's check if W entries are close to half-integers.
            W_half = np.round(W_sol * 2) / 2.0
            U_half = np.round(U * 2) / 2.0 # U is already integer, so this is identity
            V_half = np.round(V * 2) / 2.0
            
            # Verify with half-integer values
            max_err_half = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_half[i * m + j, c] * V_half[j * p + k, c] * W_half[k * n + i, c]
                        if abs(s - 1.0) > max_err_half:
                            max_err_half = abs(s - 1.0)
            
            if max_err_half < 1e-9:
                # Success! Return the half-integer solution
                return {
                    "U": to_list(U_half),
                    "V": to_list(V_half),
                    "W": to_list(W_half)
                }
    
    # Fallback to Schoolbook
    return {
        "U": to_list(U_school),
        "V": to_list(V_school),
        "W": to_list(W_school)
    }
