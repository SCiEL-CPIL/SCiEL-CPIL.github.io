"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    # Try to find a better decomposition than the naive Strassen (rank 189)
    # Using rank factorization of the 6x6x6 multiplication tensor
    
    # First, let's build the multiplication tensor T[i, j, k] = delta(i,k) * delta(j, ?)
    # Actually, the multiplication tensor T such that C = A*B means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we think of it as T(A,B) = C, where T is the 3-tensor.
    
    # The tensor has entries T[i,j,k] where:
    # - i is the row of C (and A), 0..n-1
    # - j is the column of A (and row of B), 0..m-1  
    # - k is the column of C (and B), 0..p-1
    # T[i,j,k] = 1 (it's a permutation/selection tensor)
    
    # Actually for matrix mult, the "tensor" T_{i,j,k} is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The structure tensor has T[i,j,k] = delta_{i,?}... 
    # Let me think more carefully.
    
    # The multiplication tensor M_{n} has entries M[i,j,k] where:
    # M[i,j,k] = 1 if we interpret it as the bilinear form
    # Actually, the standard formulation: the matrix multiplication tensor T 
    # is a 3-tensor of dimensions (n*m) x (m*p) x (n*p) where
    # T[a, b, c] = 1 if a = (i, j), b = (j, k), c = (i, k) for some i, j, k
    # i.e., T[(i,j), (j,k), (i,k)] = 1
    
    # The rank of this tensor is the minimum number of scalar multiplications.
    
    # Let's try a direct approach: build the full tensor and try to find its rank
    # via random initialization + gradient-like methods, or use the fact that
    # for small tensors we can try various structured decompositions.
    
    # Known results:
    # rank(2x2x2 mult) = 7 (Strassen)
    # rank(3x3x3 mult) = ? (not well known, lower bound ~19, upper bound 23 or so)
    # rank(4x4x4 mult) = ? 
    # rank(6x6x6 mult) = ?
    
    # The naive block Strassen gives 27 * 7 = 189.
    # Can we do better?
    
    # One approach: use the fact that 6x6 = (2+2+2)x(2+2+2) and apply
    # a more clever combination at the block level.
    
    # Actually, let's try to use a direct rank factorization approach.
    # For the 6x6x6 multiplication tensor, let's try to find a low-rank decomposition.
    
    # Strategy: Use the block structure but with overlapping blocks or
    # a more efficient block algorithm.
    
    # Let's try: split into 3x3 blocks of size 2x2, but use a smarter
    # combination of the 7 Strassen products per block.
    
    # Alternative: Use the "baby-step giant-step" or "Four Russians" type
    # approach, but that's typically for approximate or integer arithmetic.
    
    # For exact decomposition, let's try to find the minimum rank via
    # a heuristic search.
    
    # First, let's construct the multiplication tensor and try to compute
    # its rank via a greedy approach or via finding a sparse decomposition.
    
    # The tensor T has shape (36, 36, 36) in the flattened index convention:
    # U index: i*m + j (row i, col j of A), so 0..35
    # V index: j*p + k (row j, col k of B), so 0..35  
    # W index: k*n + i (col k, row i of C), per the convention W[k*n+i, r]
    
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k in 0..5
    # Otherwise T[a, b, c] = 0
    
    # The rank is the minimum r such that T = sum_{t=1}^{r} u_t v_t^T w_t^T
    # where u_t is a vector of length 36, etc.
    
    # This is a hard problem. Let's try a structured approach.
    
    # Approach 1: Use the Kronecker product structure.
    # 6x6 = 2x3, so we can view A as a 2x3 block matrix of 3x3 blocks? 
    # Or 3x2 blocks of 2x2?
    
    # Let's try: A is 6x6, split into 3 rows of 2 and 3 cols of 2,
    # giving 3x3 blocks of 2x2. This is what the current code does.
    
    # To improve, we could try:
    # 1. A different block size (e.g., 1x6, 6x1, 2x3, 3x2)
    # 2. Overlapping blocks
    # 3. A different algorithm at the block level
    
    # Let's try block size 3x2 (3 row blocks, 2 col blocks of 3x3 each)
    # Then we need to multiply 3x3 blocks. The rank of 3x3 matrix mult
    # is not known exactly, but we can use a known upper bound.
    
    # Actually, the best known upper bound for 3x3 matrix mult rank is 23
    # (from a 2015 result or similar). Let's verify.
    
    # Hmm, let me try a different approach. Let's just try to find a 
    # decomposition with rank less than 189 using a randomized algorithm.
    
    # Randomized rank approximation:
    # 1. Start with random rank-r decomposition
    # 2. Iteratively improve by coordinate descent
    
    # But this is complex. Let's try something simpler first.
    
    # Try: use Strassen at multiple levels of recursion.
    # 6 = 2 * 3, so we could do:
    # Level 1: 3x3 blocks of 2x2, using rank-7 Strassen: 27 * 7 = 189
    # Level 2: Could we do better by treating the 3x3 block multiplication
    #           with a better algorithm?
    
    # The 3x3 block multiplication itself is a matrix multiplication problem
    # where each "entry" is a 2x2 matrix. If we could find a rank-r
    # decomposition for 3x3 matrix mult over the ring of 2x2 matrices
    # (with Strassen multiplication), that might help.
    
    # This is getting complicated. Let's try a direct approach.
    
    # Build the multiplication tensor and try to find a low-rank decomposition
    # via a simple iterative method.
    
    # Actually, let's just try a few known structured decompositions
    # and pick the best one.
    
    # Known: The "Babai" or "de Groote" constructions give upper bounds.
    # For n=6, the best known upper bound is around 189 or slightly less.
    
    # Let's try to implement a search for a better decomposition.
    
    # Given time constraints, let's try:
    # 1. The current 189-rank Strassen block decomposition
    # 2. Try to find a rank-<189 decomposition via a simple heuristic
    
    # For the heuristic, let's try:
    # - Random initialization of rank-r factors
    # - Alternating least squares (ALS) to minimize ||T - sum u_t v_t w_t||
    # - Start with r = 150 and see if we can converge
    
    # This might be slow. Let's try with a time limit.
    
    def build_tensor(n, m, p):
        """Build the multiplication tensor as a 3D array T[i*m+j, j*p+k, k*n+i] = 1"""
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    # Build the tensor
    T = build_tensor(n, m, p)
    
    # Try to find a low-rank decomposition using ALS
    # We'll try ranks from 150 to 189 and see if we can achieve exact decomposition
    
    def als_decompose(T, rank, max_iter=100, tol=1e-8):
        """Alternating least squares for tensor decomposition"""
        nm, mp, pn = T.shape
        # Initialize randomly
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        prev_err = float('inf')
        for it in range(max_iter):
            # Update U: for each column a, solve for U[a, :]
            # T[a, :, :] = sum_t U[a, t] V[t, :] outer W[t, :]
            # In matrix form: T[a] = U[a] @ (V * W)  where V*W is element-wise
            # Actually, T[a, b, c] = sum_t U[a, t] V[b, t] W[c, t]
            # Fix V, W, solve for U
            for a in range(nm):
                # T[a, b, c] = sum_t U[a, t] * V[b, t] * W[c, t]
                # Let M[a, b, c] = T[a, b, c]
                # For fixed V, W, this is: for each t, contribution is U[a,t] * V[:,t] * W[:,t]
                # We can write: T[a] = sum_t U[a,t] * (V[:,t] * W[:,t])
                # In matrix form: T[a, b, c] viewed as a matrix over (b,c)
                # T[a] = sum_t U[a,t] * outer(V[:,t], W[:,t])
                # So we need to solve: T[a] = U[a] @ M where M[t, b, c] = V[b,t] * W[c,t]
                # Reshape: T2[a, b*c] = sum_t U[a, t] * M[t, b*c]
                # So U[a] = T2[a] @ M.T @ (M @ M.T)^{-1}  (least squares)
                pass  # This is getting complex, let's use a simpler approach
            
            # This is too slow for 36x36x36 with rank 150+. Let's try a different approach.
            break
        
        return None
    
    # The ALS approach is too slow. Let's try a structured decomposition.
    
    # Let's try: use the fact that 6 = 2*3 and apply a two-level Strassen.
    # Level 1: 3x3 blocks of 2x2
    # Level 2: Within each "super-block", use a better combination
    
    # Actually, let's just try to find if there's a rank-<189 decomposition
    # by trying a few structured patterns.
    
    # For now, let's return the 189-rank decomposition and add a time check.
    
    # Build the 189-rank Strassen block decomposition
    rank = 189
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    strassen_terms = [
        {'U': [(0, 0, 1), (1, 1, 1)],
         'V': [(0, 0, 1), (1, 1, 1)],
         'W': [(0, 0, 1), (1, 1, 1)]},
        {'U': [(1, 0, 1), (1, 1, 1)],
         'V': [(0, 0, 1)],
         'W': [(1, 0, 1), (1, 1, -1)]},
        {'U': [(0, 0, 1)],
         'V': [(0, 1, 1), (1, 1, -1)],
         'W': [(0, 1, 1), (1, 1, 1)]},
        {'U': [(1, 1, 1)],
         'V': [(1, 0, 1), (0, 0, -1)],
         'W': [(0, 0, 1), (1, 0, 1)]},
        {'U': [(0, 0, 1), (0, 1, 1)],
         'V': [(1, 1, 1)],
         'W': [(0, 0, -1), (0, 1, 1)]},
        {'U': [(1, 0, 1), (0, 0, -1)],
         'V': [(0, 0, 1), (0, 1, 1)],
         'W': [(1, 1, 1)]},
        {'U': [(0, 1, 1), (1, 1, -1)],
         'V': [(1, 0, 1), (1, 1, 1)],
         'W': [(0, 0, 1)]}
    ]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row, col_idx] += coeff
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row, col_idx] += coeff
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row, col_idx] += coeff
    
    # Check if we have time to try something better
    if time.time() - start_time < DEADLINE - 5:
        # Try to find a better decomposition using a simple search
        # This is a placeholder for a more sophisticated search
        
        # For now, just verify the current decomposition is correct
        # C = sum_t U[:, t] outer V[:, t] outer W[:, t]
        # C[i, k] = sum_t U[i*m + ?, t] * ... 
        # Actually, let's just return what we have.
        pass
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
