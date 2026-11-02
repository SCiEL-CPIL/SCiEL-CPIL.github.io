"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build the tensor from decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    The 7 products are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U factors: for each of 7 components, a 2x2 matrix
    # V factors: for each of 7 components, a 2x2 matrix
    # W factors: for each of 7 components, a 2x2 matrix (output)
    
    U = []
    V = []
    W = []
    
    # Component 1: (A11+A22)(B11+B22)
    # A part: [[1,0],[0,1]]  (A11 + A22)
    # B part: [[1,1],[0,1]]  (B11 + B22) -- wait, B is 2x2, indices B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # So B11+B22 means [[1,0],[0,1]]
    # C part: C11 gets +1, C22 gets +1
    U.append([1, 0, 0, 1])   # [[1,0],[0,1]]
    V.append([1, 0, 0, 1])   # [[1,0],[0,1]]
    W.append([1, 0, 0, 1])   # [[1,0],[0,1]]
    
    # Component 2: (A21+A22)B11
    # A part: [[0,0],[1,1]]  (A21 + A22)
    # B part: [[1,0],[0,0]]  (B11)
    # C part: C21 gets +1
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, 0, 1, 0])
    
    # Component 3: A11(B21-B22)
    # A part: [[1,0],[0,0]]  (A11)
    # B part: [[0,0],[1,-1]]  (B21 - B22)
    # C part: C12 gets +1
    U.append([1, 0, 0, 0])
    V.append([0, 0, 1, -1])
    W.append([0, 1, 0, 0])
    
    # Component 4: A22(B12-B11)
    # A part: [[0,0],[0,1]]  (A22)
    # B part: [[-1,1],[0,0]]  (B12 - B11)
    # C part: C11 gets +1, C21 gets +1
    U.append([0, 0, 0, 1])
    V.append([-1, 1, 0, 0])
    W.append([1, 0, 1, 0])
    
    # Component 5: (A11+A12)B22
    # A part: [[1,1],[0,0]]  (A11 + A12)
    # B part: [[0,0],[0,1]]  (B22)
    # C part: C11 gets -1, C12 gets +1
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([-1, 1, 0, 0])
    
    # Component 6: (A21-A11)(B11+B12)
    # A part: [[-1,0],[1,0]]  (A21 - A11)
    # B part: [[1,1],[0,0]]  (B11 + B12)
    # C part: C22 gets +1
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([0, 0, 0, 1])
    
    # Component 7: (A12-A22)(B21+B22)
    # A part: [[0,1],[-1,0]]  (A12 - A22)
    # B part: [[0,0],[1,1]]  (B21 + B22)
    # C part: C11 gets +1
    U.append([0, 1, -1, 0])
    V.append([0, 0, 1, 1])
    W.append([1, 0, 0, 0])
    
    return U, V, W

def _kronecker_sum_decomposition(n, m, p, time_limit):
    """
    Try to use Kronecker sum / block structure to find a lower-rank decomposition.
    
    For 3x5x6:
    - Split the 5-dimensional middle into 3+2
    - This gives two sub-problems: 3x3x6 and 3x2x6
    
    For 3x3x6:
    - Split the 6-dimensional output into 3+3
    - This gives 3x3x3 and 3x3x3
    
    For 3x2x6:
    - Split the 6-dimensional output into 3+3
    - This gives 3x2x3 and 3x2x3
    
    Known ranks:
    - 3x3x3: upper bound 23 (but we'll use schoolbook 27 for simplicity, or try to find better)
    - 2x2x2: rank 7 (Strassen)
    - 3x2x3: not well-known, use schoolbook 18 or try to find better
    
    Actually, let's think more carefully.
    
    The tensor for matmul is T[i,j,k] = delta_{i,k} for the identity, but really it's:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor is T[i,j,k] = 1 for all i,j,k (in the sense that each triple contributes).
    
    Wait, no. The matmul tensor has T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    contribute to C[i,k]. So actually T[i,j,k] = 1 for all valid i,j,k.
    
    Yes, that's correct. The tensor is all ones.
    
    So we need to decompose the all-ones tensor of shape (3,5,6) into a sum of rank-1 tensors.
    
    A rank-1 tensor is an outer product u ⊗ v ⊗ w where u is 3-dim, v is 5-dim, w is 6-dim.
    
    The schoolbook decomposition has rank 3*5*6 = 90, with each rank-1 term being a single (i,j,k) entry.
    
    Can we do better? Yes, by using linearity.
    
    For example, if we have two terms with the same u and v but different w, we can combine them.
    
    Let's try a different approach: use the fact that the tensor is separable in a certain way.
    
    Actually, the tensor T[i,j,k] = 1 is not separable, but it has low rank in some sense.
    
    Let's try to use the following strategy:
    - For each j (middle index), we have a 3x6 sub-tensor that is all ones.
    - The 3x6 all-ones matrix can be written as a sum of 3 rank-1 terms (since rank of all-ones 3x6 is 1,
      but we need to express it as a sum of rank-1 tensors in the 3-way sense).
    
    Wait, I'm confusing myself. Let me think again.
    
    The matmul tensor T is a 3-way tensor of shape (3,5,6) with T[i,j,k] = 1 for all i,j,k.
    
    We want to write T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c where u_c is in R^3, v_c in R^5, w_c in R^6.
    
    This is equivalent to finding a low-rank decomposition of the 3-way tensor.
    
    The "rank" here is the tensor rank (or multilinear rank), not the matrix rank.
    
    For the all-ones tensor of shape (n,m,p), what is the minimum rank?
    
    It's known that the rank of the all-ones tensor of shape (n,m,p) is at most n*m + n*p + m*p - n - m - p + 1 or something like that,
    but I'm not sure of the exact formula.
    
    Actually, I recall that the rank of the all-ones tensor of shape (n,n,n) is related to the rank of matrix multiplication,
    but I'm not sure of the exact relationship.
    
    Let me try a different approach: use a greedy algorithm or local search to find a low-rank decomposition.
    
    Given the time constraints, let's try a simple local search approach:
    1. Start with the schoolbook decomposition (rank 90).
    2. Try to merge pairs of rank-1 terms by checking if they can be combined.
    3. Use a local search to optimize the decomposition.
    
    Actually, let's try a more systematic approach.
    
    For the all-ones tensor of shape (3,5,6), we can try to find a decomposition of the form:
    T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    
    This is equivalent to finding matrices U (3 x r), V (5 x r), W (6 x r) such that:
    U @ V.T @ W.T = ones(3,5,6) ... no, that's not right.
    
    Actually, the condition is:
    sum_c U[i,c] * V[j,c] * W[k,c] = 1 for all i,j,k.
    
    This is a system of 3*5*6 = 90 equations in 3*r + 5*r + 6*r = 14*r unknowns.
    
    For r = 90, we have 1260 unknowns and 90 equations, so there are many solutions.
    
    We want to find the smallest r such that a solution exists.
    
    This is a non-convex optimization problem, so it's hard to solve exactly.
    
    Let's try a randomized approach:
    1. Pick a random r (starting from a lower bound).
    2. Generate random U, V, W.
    3. Check if the equation is satisfied.
    4. If not, try to adjust U, V, W to satisfy the equation.
    
    The lower bound on r is at least max(3,5,6) = 6, but probably much higher.
    
    Actually, I think the rank of the all-ones tensor of shape (n,m,p) is at least max(n,m,p),
    but I'm not sure of the exact lower bound.
    
    Let's try a different approach: use the fact that the all-ones tensor can be written as:
    T[i,j,k] = sum_{a=1}^{n} sum_{b=1}^{m} sum_{c=1}^{p} delta_{i,a} * delta_{j,b} * delta_{k,c}
    
    This is the schoolbook decomposition with rank n*m*p.
    
    Can we do better? Yes, by using the linearity of the sum.
    
    For example, if we fix j, we have a 3x6 tensor that is all ones.
    This can be written as a sum of 3 rank-1 terms (since the 3x6 all-ones matrix has rank 1,
    but in the 3-way sense, we need to be careful).
    
    Actually, let's just try a local search approach and see if we can find a better decomposition.
    
    Given the time constraints, let's implement a simple local search:
    1. Start with the schoolbook decomposition.
    2. For each pair of rank-1 terms, check if they can be merged.
    3. Use a local search to optimize the decomposition.
    
    Actually, let's try a simpler approach: use a randomized local search starting from a random initialization.
    """
    start_time = time.time()
    n, m, p = 3, 5, 6
    
    # Try ranks from a reasonable lower bound up to the schoolbook rank
    # Lower bound: at least max(n, m, p) = 6, but probably higher
    # Let's try from 20 up to 90
    
    best_rank = n * m * p
    best_solution = None
    
    for target_rank in range(20, n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            result = _local_search_rank(target_rank, n, m, p, attempt,
                                        time_limit - (time.time() - start_time))
            if result is not None:
                best_rank = target_rank
                best_solution = result
                break
        
        if best_solution is not None and best_rank == target_rank:
            # Found a solution, try to go lower
            continue
    
    return best_solution, best_rank

def _local_search_rank(rank, n, m, p, seed, time_limit):
    """
    Local search for a rank-r decomposition of the all-ones tensor.
    
    Start with a random initialization and iteratively improve.
    """
    start_time = time.time()
    np.random.seed(seed * 1000 + 42)
    
    # Initialize with random values
    U = np.random.randn(n, rank)
    V = np.random.randn(m, rank)
    W = np.random.randn(p, rank)
    
    # Target: all ones
    target = np.ones((n, m, p))
    
    # Compute the current tensor
    def compute_tensor(U, V, W):
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += U[i, c] * V[j, c] * W[k, c]
        return tensor
    
    # Compute initial error
    def compute_error(U, V, W):
        tensor = compute_tensor(U, V, W)
        return np.linalg.norm(tensor - target)
    
    current_error = compute_error(U, V, W)
    
    # Iterative improvement
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for c in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            # Try to optimize component c
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                
                # Save current state
                U_old = U[:, c].copy()
                V_old = V[:, c].copy()
                W_old = W[:, c].copy()
                
                # Perturb
                U[:, c] += 0.1 * np.random.randn(n)
                V[:, c] += 0.1 * np.random.randn(m)
                W[:, c] += 0.1 * np.random.randn(p)
                
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                else:
                    # Revert
                    U[:, c] = U_old
                    V[:, c] = V_old
                    W[:, c] = W_old
        
        if not improved:
            break
        
        # Check if we've found an exact solution
        if current_error < 1e-9:
            # Verify with exact check
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition(
                [[U_list[i][c] for c in range(rank)] for i in range(n*m)],
                [[V_list[i][c] for c in range(rank)] for i in range(m*p)],
                [[W_list[i][c] for c in range(rank)] for i in range(p*n)],
                n, m, p
            ):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # First, try to find a better decomposition using local search
    best_solution, best_rank = _kronecker_sum_decomposition(n, m, p, 60.0)
    
    if best_solution is not None and best_rank < n * m * p:
        # Convert the solution to the required format
        U, V, W = best_solution
        # U is (n, rank), V is (m, rank), W is (p, rank)
        # We need U to be (n*m, rank), V to be (m*p, rank), W to be (p*n, rank)
        
        # For U: U[i, c] should be placed at position (i*m + j, c) for all j
        # But wait, the decomposition is T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # The required format is:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So we need to map:
        # U[i, c] -> U[i*m+j, c] for all j? No, that doesn't make sense.
        
        # Actually, the tensor decomposition is:
        # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
        
        # And the required format is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So we need:
        # U[i*m+j, c] = u_c[i] for all j? That would mean U[i*m+j, c] depends only on i, not j.
        # Similarly, V[j*p+k, c] = v_c[j] for all k, and W[k*n+i, c] = w_c[k] for all i.
        
        # Let's check: if U[i*m+j, c] = u_c[i], V[j*p+k, c] = v_c[j], W[k*n+i, c] = w_c[k],
        # then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # = sum_c u_c[i] * v_c[j] * w_c[k]
        # = T[i,j,k]
        
        # Yes, that works!
        
        U_new = [[0] * best_rank for _ in range(n * m)]
        V_new = [[0] * best_rank for _ in range(m * p)]
        W_new = [[0] * best_rank for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(best_rank):
                    U_new[i * m + j][c] = U[i][c]
        
        for j in range(m):
            for k in range(p):
                for c in range(best_rank):
                    V_new[j * p + k][c] = V[j][c]
        
        for k in range(p):
            for i in range(n):
                for c in range(best_rank):
                    W_new[k * n + i][c] = W[k][c]
        
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
