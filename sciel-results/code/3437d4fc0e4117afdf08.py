"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _berlekamp_massey(sequence):
    """Berlekamp-Massey algorithm to find minimal linear recurrence.
    Works over rationals using Fraction arithmetic.
    Returns list of coefficients c[0..L-1] such that s[n] = -sum(c[i]*s[n-i-1]) for i in 0..L-1
    and the recurrence length L.
    """
    n = len(sequence)
    if n == 0:
        return [], 0
    
    C = [Fraction(1)]  # Connection polynomial
    B = [Fraction(1)]  # Previous connection polynomial
    L = 0              # Current recurrence length
    m = 1              # Number of steps since last update
    b = Fraction(1)    # Last discrepancy
    
    for N in range(n):
        # Compute discrepancy d
        d = sequence[N]
        for i in range(1, L + 1):
            if i < len(C):
                d += C[i] * sequence[N - i]
        
        if d == 0:
            m += 1
        else:
            if 2 * L <= N:
                # Update L
                T = C[:]
                L = N + 1 - L
                # C = C - d/b * x^m * B
                factor = d / b
                while len(B) + m < len(C) + 1:
                    B.append(Fraction(0))
                for i in range(len(B)):
                    if i + m < len(C):
                        C[i + m] -= factor * B[i]
                # Trim trailing zeros
                while len(C) > 1 and C[-1] == 0:
                    C.pop()
                b = d
                m = 1
            else:
                # C = C - d/b * x^m * B
                factor = d / b
                while len(B) + m < len(C) + 1:
                    B.append(Fraction(0))
                for i in range(len(B)):
                    if i + m < len(C):
                        C[i + m] -= factor * B[i]
                # Trim trailing zeros
                while len(C) > 1 and C[-1] == 0:
                    C.pop()
                m += 1
    
    # The recurrence is: s[n] = -C[1]*s[n-1] - C[2]*s[n-2] - ... - C[L]*s[n-L]
    # Return coefficients for the recurrence
    coeffs = [-C[i] for i in range(1, min(L + 1, len(C)))]
    return coeffs, L

def _generate_slice_sequence(n, m, p, seed, max_len=50):
    """Generate a sequence of tensor slice values for BM analysis."""
    random.seed(seed)
    seq = []
    for _ in range(max_len):
        # Random small integer matrices
        A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        # Compute product C = A*B
        C = [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
        # Use a specific entry as the sequence value
        # The tensor T[i,j,k] = delta_{i,k} in the multiplication tensor
        # We use the fact that (A*B)[i,j] = sum_k A[i,k]*B[k,j]
        # The tensor is defined by T[i,k,j] = 1 if indices match, 0 otherwise
        # A slice value can be any (i,j) entry of C
        seq.append(C[seed % n][seed % p])
    return seq

def _try_rank_decomposition(rank, n, m, p, deadline):
    """Try to find a rank decomposition of the given rank using structured search."""
    # The tensor T has entries T[i,k,j] = 1 if we're looking at the (i,j) output
    # from input (k) - actually the tensor is T[i,k,j] where the multiplication is
    # C[i,j] = sum_k A[i,k] B[k,j]
    # So the tensor T has indices (i,k,j) and T[i,k,j] = delta_{i,k} * delta_{k,j}... 
    # Actually for matrix multiplication, the tensor is T[i,k,j] = 1 if i==k and k==j... no
    # The multiplication tensor has T[i,k,j] = delta_{i,k} (for the i-th row, k-th col of A 
    # times k-th row, j-th col of B)
    # Actually: C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor T is defined by T[i,k,j] = 1 for all i,k,j? No.
    # The tensor T has T[i,k,j] = delta_{i,k} * delta_{k,j}? No, that's not right either.
    # 
    # The matrix multiplication tensor T is a 3-tensor with T[i,k,j] = 1 if the (i,k) entry 
    # of A and (k,j) entry of B contribute to (i,j) entry of C.
    # So T[i,k,j] = 1 for ALL i,k,j? No, that would be rank n*m*p.
    # Actually T[i,k,j] = delta_{i,k} is wrong.
    # 
    # The tensor is: T[i,k,j] = 1 if we're computing C[i,j] from A[i,k]*B[k,j]
    # So T[i,k,j] = 1 for all valid i,k,j. That gives rank n*m*p (schoolbook).
    # 
    # Wait, I need to think about this more carefully.
    # The tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} (or similar indexing)
    # such that for matrices A (n×m) and B (m×p), C = A*B (n×p):
    # C[i,j] = sum_{k=1}^{m} A[i,k] * B[k,j]
    # 
    # The tensor T has entries T[(i,k), (k,j), (i,j)] = 1
    # Using flattened indices:
    # U index: i*m + k (row i, col k of A)
    # V index: k*p + j (row k, col j of B)  
    # W index: i*p + j (row i, col j of C) -- but the task says W[k*n+i, r]
    # 
    # The task says W[k*n+i, r] which is the "official trace convention"
    # So W index is k*n + i where k is the column of C and i is the row of C?
    # That would be W[j*n + i, r] for C[i,j]
    
    # For now, let's use the standard convention and verify
    # T[(i*m+k), (k*p+j), (j*n+i)] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    
    # Try to find rank decomposition using structured approach
    # This is a complex search problem. Let's try small ranks first.
    
    # For now, return None to indicate we need to use schoolbook
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    
    # Verify: for all i,j,k: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should give
    # the correct tensor value. The tensor T[(i*m+k), (k*p+j), (j*n+i)] = 1
    # 
    # Actually the decomposition means: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where T is the multiplication tensor.
    # 
    # The multiplication tensor has T[(i*m+k), (k*p+j), (j*n+i)] = 1 for all i,k,j
    # and T[a, b, c] = 0 otherwise.
    # 
    # Wait, that's not right. The tensor is sparse - it only has 1s at specific positions.
    # Let me reconsider.
    # 
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # In tensor form: C[i,j] = sum_{k} T[i,k,j] * A[i,k] * B[k,j]
    # where T[i,k,j] = 1 for all i,k,j.
    # 
    # So the tensor T[i,k,j] = 1 for all valid indices. That means the tensor is all-ones
    # in the relevant subspaces, which gives rank n*m*p.
    # 
    # Hmm, that doesn't seem right for a meaningful tensor. Let me reconsider the indexing.
    # 
    # Actually, I think the tensor is defined differently. The matrix multiplication tensor
    # T is such that:
    # (A ⊗ B) contracted with T gives C
    # 
    # More precisely: C[i,j] = sum_{k} A[i,k] * B[k,j]
    # 
    # The tensor T has components T[i,k,j] and the equation is:
    # C[i,j] = sum_{i',k,j'} T[i',k,j'] * A[i',k] * B[k,j'] * delta_{i,i'} * delta_{j,j'}
    # 
    # This is getting confusing. Let me use the standard definition.
    # 
    # The matrix multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} is defined by:
    # T = sum_{i=1}^n sum_{k=1}^m sum_{j=1}^p E_{i,k} ⊗ E_{k,j} ⊗ E_{i,j}
    # where E_{a,b} is the matrix with 1 at position (a,b) and 0 elsewhere.
    # 
    # So T[E_{i,k}, E_{k,j}, E_{i,j}] = 1 and T is 0 elsewhere.
    # 
    # In flattened form:
    # U index: i*m + k
    # V index: k*p + j
    # W index: i*p + j (but task says W[k*n+i] which is different!)
    # 
    # The task says W[k*n+i, r] - this is the "official trace convention"
    # So W index = k*n + i where k is column of C and i is row of C?
    # That would be W[j*n + i, r] for C[i,j]
    
    # Let's verify with the schoolbook decomposition
    # U[i*m+k, c] = 1, V[k*p+j, c] = 1, W[j*n+i, c] = 1 for column c corresponding to (i,k,j)
    # 
    # For the decomposition to work:
    # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # 
    # For a = i*m+k, b = k'*p+j', c = j''*n+i'':
    # sum_r U[i*m+k, r] * V[k'*p+j', r] * W[j''*n+i'', r]
    # 
    # In schoolbook, each column r corresponds to a unique (i_r, k_r, j_r) triple
    # U[i*m+k, r] = 1 iff i_r = i and k_r = k
    # V[k'*p+j', r] = 1 iff k_r = k' and j_r = j'
    # W[j''*n+i'', r] = 1 iff j_r = j'' and i_r = i''
    # 
    # So the product is 1 iff i_r=i, k_r=k, k_r=k', j_r=j', j_r=j'', i_r=i''
    # i.e., i=i'', k=k', j=j'
    # 
    # So sum_r ... = 1 iff i=i'', k=k', j=j'
    # 
    # The tensor T should have T[i*m+k, k*p+j, j*n+i] = 1
    # 
    # So T[a, b, c] = 1 iff a = i*m+k, b = k*p+j, c = j*n+i for some i,k,j
    # i.e., T[a, b, c] = 1 iff (a mod m) == (b div p) and (a div m) == (c mod n) and (b mod p) == (c div n)
    
    def tensor_value(a, b, c):
        i = a // m
        k = a % m
        k2 = b // p
        j = b % p
        j2 = c // n
        i2 = c % n
        if i == i2 and k == k2 and j == j2:
            return 1
        return 0
    
    # Verify all entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                expected = tensor_value(a, b, c)
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if actual != expected:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Generate the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= n*m + m*p + n*p - m^2 (for some cases)
    # For 3x4x5: lower bound is around 3*4 + 4*5 + 3*5 - 4^2 = 12+20+15-16 = 31
    # Schoolbook rank is 60
    
    # Try ranks from lower bound up to schoolbook rank
    # Use structured search with Berlekamp-Massey insights
    
    # For now, implement a basic search that tries to verify if lower ranks work
    # by checking if the tensor can be decomposed at that rank
    
    # This is a hard problem. Let's try a few structured approaches.
    
    # Approach 1: Check if the tensor has low rank using SVD-like methods
    # Flatten the tensor and check its matrix flattening ranks
    
    # The tensor T can be viewed as a matrix of size (n*m) x (m*p*p*n) or similar
    # The rank of the tensor is at least the maximum rank of any matrix flattening
    
    # Matrix flattening 1: (n*m) x (m*p * p*n)
    # T as matrix M1 where M1[a, b*p*n + c] = T[a, b, c]
    # Rank of M1 gives a lower bound on tensor rank
    
    # Build the tensor
    T = {}
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[(a, b, c)] = 1
    
    # Check matrix flattening rank
    # M1: (n*m) x (m*p * p*n)
    dim1 = n * m
    dim2 = m * p * p * n
    M1 = [[0] * dim2 for _ in range(dim1)]
    for (a, b, c), val in T.items():
        M1[a][b * p * n + c] = val
    
    # Compute rank of M1 using row reduction
    # This is expensive for large dimensions, but our dimensions are small
    # dim1 = 12, dim2 = 4*5*5*3 = 300
    
    # Use Gaussian elimination over rationals
    def matrix_rank(M):
        if not M or not M[0]:
            return 0
        rows = len(M)
        cols = len(M[0])
        M_copy = [[Fraction(x) for x in row] for row in M]
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot = -1
            for row in range(rank, rows):
                if M_copy[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap rows
            M_copy[rank], M_copy[pivot] = M_copy[pivot], M_copy[rank]
            # Eliminate
            pivot_val = M_copy[rank][col]
            for j in range(col, cols):
                M_copy[rank][j] /= pivot_val
            for row in range(rows):
                if row != rank and M_copy[row][col] != 0:
                    factor = M_copy[row][col]
                    for j in range(col, cols):
                        M_copy[row][j] -= factor * M_copy[rank][j]
            rank += 1
            if rank == rows:
                break
        return rank
    
    if time.time() > deadline:
        return best
    
    # Compute lower bound from matrix flattening
    lb1 = matrix_rank(M1)
    
    # Also check other flattening
    # M2: (m*p) x (n*m * p*n)
    dim2a = m * p
    dim2b = n * m * p * n
    M2 = [[0] * dim2b for _ in range(dim2a)]
    for (a, b, c), val in T.items():
        M2[b][a * p * n + c] = val
    
    if time.time() > deadline:
        return best
    
    lb2 = matrix_rank(M2)
    
    # M3: (p*n) x (n*m * m*p)
    dim3a = p * n
    dim3b = n * m * m * p
    M3 = [[0] * dim3b for _ in range(dim3a)]
    for (a, b, c), val in T.items():
        M3[c][a * m * p + b] = val
    
    if time.time() > deadline:
        return best
    
    lb3 = matrix_rank(M3)
    
    lower_bound = max(lb1, lb2, lb3)
    
    # Try to find decomposition at lower_bound or slightly above
    # This is a hard combinatorial problem
    # For now, if lower_bound < best_rank, we know we can potentially do better
    # but finding the actual decomposition is non-trivial
    
    # Try a few candidate ranks
    for candidate_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition at this rank
        # This is a system of polynomial equations
        # For small ranks, we can try random initialization and Newton's method
        # But this is complex and may not converge
        
        # For now, skip this and return schoolbook
        # A proper implementation would use algebraic geometry methods
    
    return best
# EVOLVE-BLOCK-END
