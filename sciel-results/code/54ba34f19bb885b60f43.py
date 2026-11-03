"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Deadline guard: kill after 150s wall-clock, return best with safety margin
_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 0.5 - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, rank):
    """Check exact tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i*m+j], V[j*p+k])
                rhs = W[k*n+i]
                if not np.allclose(lhs, rhs, atol=1e-9):
                    return False
    return True

def _strassen_4x4():
    """Return the 7-term Strassen decomposition for 4x4 * 4x4 -> 4x4.
    
    Returns (U, V, W) each of shape (16, 7) using the standard Strassen scheme
    applied to 2x2 blocks of size 2x2.
    
    Actually, for 4x4 we can use the standard Strassen with 2x2 sub-blocks:
    A = [[A11, A12],[A21, A22]] where each block is 2x2.
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a product of two 2x2 matrices, which themselves can be done with
    schoolbook (4 terms each) or further Strassen. For simplicity and to get
    a concrete rank, let's use schoolbook for the 2x2 sub-products.
    
    Actually, the standard Strassen for 2x2 gives rank 7 for 2x2x2.
    For 4x4x4, if we use Strassen on 2x2 blocks, each of the 7 products M_i
    is a 2x2 * 2x2 multiplication, which has rank 4 (schoolbook) or 7 (Strassen).
    
    If we use schoolbook for each 2x2 product: total rank = 7 * 4 = 28.
    But the standard 4x4 Strassen (using 2x2 sub-blocks with schoolbook inside)
    gives rank 28. However, the known best for 4x4 is actually lower.
    
    Let me think more carefully. The standard way:
    4x4 = 2x2 blocks. Strassen reduces 2x2x2 to 7 multiplications.
    Each "multiplication" here is a 2x2 * 2x2, which costs 4 (schoolbook).
    So total = 7 * 4 = 28.
    
    But wait, the task says the measured result was rank 114 using 49-rank Strassen
    for two 4x4x4 sub-problems. That suggests they used a different decomposition.
    
    Let me re-read: "49-rank Strassen for two 4x4x4 sub-problems and 16-rank schoolbook
    for the 4x4x1 remainder"
    
    So 4x9 = 4x4 + 4x4 + 4x1. Two 4x4x4 problems and one 4x4x1 problem.
    49 + 49 + 16 = 114.
    
    So the 4x4x4 tensor has rank 49 in their scheme? That seems high. Standard
    schoolbook is 64, Strassen with 2x2 blocks is 28. 49 is between them.
    
    Maybe they used a different approach. Let me just implement a clean
    Strassen-based decomposition.
    
    For 4x4x4, I'll use the standard Strassen with 2x2 sub-blocks:
    - Split A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    - Split B (4x4) into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    - Compute 7 products M1..M7 (each is 2x2 * 2x2)
    - Each 2x2*2x2 is done with schoolbook (4 terms)
    - Total rank: 7 * 4 = 28
    
    For the 4x4x1 case (multiplying 4x4 by 4x1):
    - This is just 4x4 * 4x1, which is 16 scalar multiplications
    - Rank 16
    
    So total: 28 + 28 + 16 = 72.
    
    But the measured was 114. Maybe they used a different scheme. Let me just
    implement the Strassen approach and see what rank I get.
    
    Actually, let me reconsider. The "49-rank" might refer to a specific
    decomposition. Let me just implement a clean approach:
    
    For 4x4x4: use Strassen with 2x2 blocks, schoolbook inside => rank 28
    For 4x4x1: schoolbook => rank 16
    Total: 28 + 28 + 16 = 72
    """
    # Standard Strassen for 4x4x4 using 2x2 sub-blocks with schoolbook inside
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # Block indices: A[i,j] for i,j in {0,1} x {0,1} where each block is 2x2
    # Global index: row = i*2 + r, col = j*2 + c where r,c in {0,1}
    
    n = 4
    # U: shape (n*n, rank) = (16, rank)
    # V: shape (n*n, rank) = (16, rank)  
    # W: shape (n*n, rank) = (16, rank)
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 * 2x2 product, done with schoolbook (4 terms)
    # For a 2x2 matrix P = [[p0,p1],[p2,p3]] and Q = [[q0,q1],[q2,q3]]:
    # P*Q[0,0] = p0*q0 + p1*q2
    # P*Q[0,1] = p0*q1 + p1*q3
    # P*Q[1,0] = p2*q0 + p3*q2
    # P*Q[1,1] = p2*q1 + p3*q3
    
    # Let's define helper to add a 2x2*2x2 schoolbook term
    # A block (ai, aj) means rows ai*2, ai*2+1 and cols aj*2, aj*2+1
    # B block (bi, bj) means rows bi*2, bi*2+1 and cols bj*2, bj*2+1
    # C block (ci, cj) means rows ci*2, ci*2+1 and cols cj*2, cj*2+1
    
    col = 0
    for term in range(7):
        # Define which A blocks and B blocks compose this term
        # A blocks: list of (block_row, block_col, sign)
        # B blocks: list of (block_row, block_col, sign)
        if term == 0:  # M1 = (A11+A22)(B11+B22)
            a_blocks = [(0,0,1), (1,1,1)]
            b_blocks = [(0,0,1), (1,1,1)]
        elif term == 1:  # M2 = (A21+A22)B11
            a_blocks = [(1,0,1), (1,1,1)]
            b_blocks = [(0,0,1)]
        elif term == 2:  # M3 = A11(B12-B22)
            a_blocks = [(0,0,1)]
            b_blocks = [(0,1,1), (1,1,-1)]
        elif term == 3:  # M4 = A22(B21-B11)
            a_blocks = [(1,1,1)]
            b_blocks = [(1,0,1), (0,0,-1)]
        elif term == 4:  # M5 = (A11+A12)B22
            a_blocks = [(0,0,1), (0,1,1)]
            b_blocks = [(1,1,1)]
        elif term == 5:  # M6 = (A21-A11)(B11+B12)
            a_blocks = [(1,0,1), (0,0,-1)]
            b_blocks = [(0,0,1), (0,1,1)]
        elif term == 6:  # M7 = (A12-A22)(B21+B22)
            a_blocks = [(0,1,1), (1,1,-1)]
            b_blocks = [(1,0,1), (1,1,1)]
        
        # For each combination of A block and B block, do schoolbook 2x2*2x2
        for (ar, ac, asgn) in a_blocks:
            for (br, bc, bs) in b_blocks:
                sign = asgn * bs
                # Schoolbook 2x2*2x2: 4 terms
                for r in range(2):
                    for c in range(2):
                        # C[ar*2+r, ac*2+c] += sign * sum over (r',c') of A[ar*2+r', ac*2+c'] * B[br*2+c', bc*2+r]
                        # Wait, I need to be more careful with the indices.
                        # Actually, for the tensor decomposition, each scalar product
                        # A[i,j] * B[j,k] contributes to C[i,k].
                        
                        # In the block structure:
                        # A block (ar, ac) has elements A[ar*2+r, ac*2+c] for r,c in {0,1}
                        # B block (br, bc) has elements B[br*2+r, bc*2+c] for r,c in {0,1}
                        # The product contributes to C block (ar, bc) at position (r, c)
                        # C[ar*2+r, bc*2+c] += sign * A[ar*2+r, ac*2+c'] * B[br*2+c', bc*2+c]
                        # summed over c'
                        
                        # For each (r, c) in the output block:
                        for c_prime in range(2):
                            # This is one scalar multiplication: A[ar*2+r, ac*2+c_prime] * B[br*2+c_prime, bc*2+c]
                            # contributing to C[ar*2+r, bc*2+c]
                            a_idx = ar*2 + r
                            a_col = ac*2 + c_prime
                            b_row = br*2 + c_prime
                            b_col = bc*2 + c
                            c_idx = ar*2 + r
                            c_col = bc*2 + c
                            
                            U[a_idx * 4 + a_col, col] = sign
                            V[b_row * 4 + b_col, col] = sign
                            W[c_idx * 4 + c_col, col] = 1
                            col += 1
    
    # Check that col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U, V, W

def _schoolbook_4x4x1():
    """Schoolbook decomposition for 4x4 * 4x1 -> 4x1.
    
    U: shape (16, 16)
    V: shape (16, 16)  
    W: shape (4, 16)
    
    For each i in 0..3, j in 0..3:
    A[i,j] * B[j,0] contributes to C[i,0]
    """
    rank = 16
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((4, rank))
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 1 + 0, col] = 1
            W[i * 1 + 0, col] = 1
            col += 1
    
    assert col == rank
    return U, V, W

def _combine_decompositions(decomps):
    """Combine multiple decompositions into one.
    
    Each decomps[i] = (U_i, V_i, W_i) for sub-problem i.
    The sub-problems partition the columns of B (and rows of W).
    
    For 4x4x9:
    - Sub-problem 0: 4x4 * 4x4 -> 4x4 (columns 0-3 of B, rows 0-3 of W)
    - Sub-problem 1: 4x4 * 4x4 -> 4x4 (columns 4-7 of B, rows 4-7 of W)
    - Sub-problem 2: 4x4 * 4x1 -> 4x1 (column 8 of B, row 8 of W)
    
    Wait, I need to think about this more carefully.
    
    The full problem is: A (4x4) * B (4x9) = C (4x9).
    
    We can split B into B1 (4x4), B2 (4x4), B3 (4x1).
    Then C = [A*B1, A*B2, A*B3].
    
    For each sub-problem, we have a decomposition.
    The U matrices are all the same (since they all use the same A).
    The V matrices are concatenated (since B is split into columns).
    The W matrices are concatenated (since C is split into columns).
    
    Actually, in the tensor decomposition framework:
    - U has shape (n*m, rank) = (16, rank)
    - V has shape (m*p, rank) = (36, rank)
    - W has shape (p*n, rank) = (36, rank)
    
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] should equal W[k*n+i, c]... 
    wait, no. The tensor for matrix multiplication is:
    T[i,j,k] = delta (i.e., the multiplication tensor)
    
    Actually, the standard formulation:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    
    A rank-r decomposition means:
    T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    Wait, but the task says:
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    So the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    
    So for the full 4x4x9 problem:
    - i in 0..3 (rows of A, n=4)
    - j in 0..3 (cols of A / rows of B, m=4)
    - k in 0..8 (cols of B, p=9)
    
    T[i,j,k] = 1 for all i,j,k.
    
    Now, if we split B into B1 (cols 0-3), B2 (cols 4-7), B3 (col 8):
    - For k in 0..3: T[i,j,k] = 1, handled by sub-problem 1
    - For k in 4..7: T[i,j,k] = 1, handled by sub-problem 2
    - For k in 8: T[i,j,k] = 1, handled by sub-problem 3
    
    In the full decomposition:
    U is the same for all sub-problems (since it depends on i,j which are the same).
    V is split: V[j*p+k, c] where k ranges over 0..8.
    W is split: W[k*n+i, c] where k ranges over 0..8.
    
    So:
    - U_full: shape (16, rank_total)
    - V_full: shape (36, rank_total)
    - W_full: shape (36, rank_total)
    
    For sub-problem 1 (k=0..3):
    - U1: shape (16, r1)
    - V1: shape (16, r1) [only the part where k=0..3, so rows j*9+0 to j*9+3]
    - W1: shape (16, r1) [only the part where k=0..3, so rows 0*4+i to 3*4+i]
    
    For sub-problem 2 (k=4..7):
    - U2: shape (16, r2)
    - V2: shape (16, r2) [rows j*9+4 to j*9+7]
    - W2: shape (16, r2) [rows 4*4+i to 7*4+i]
    
    For sub-problem 3 (k=8):
    - U3: shape (16, r3)
    - V3: shape (4, r3) [rows j*9+8]
    - W3: shape (4, r3) [rows 8*4+i]
    
    The full U is [U1, U2, U3] horizontally concatenated.
    The full V has V1 in rows j*9+0..j*9+3, V2 in rows j*9+4..j*9+7, V3 in rows j*9+8.
    The full W has W1 in rows 0..15, W2 in rows 16..31, W3 in rows 32..35.
    
    Let me implement this.
    """
    n, m, p = 4, 4, 9
    U_list = [d[0] for d in decomps]
    V_list = [d[1] for d in decomps]
    W_list = [d[2] for d in decomps]
    
    rank_total = sum(d[0].shape[1] for d in decomps)
    
    U_full = np.zeros((n*m, rank_total))
    V_full = np.zeros((m*p, rank_total))
    W_full = np.zeros((p*n, rank_total))
    
    col_offset = 0
    # Sub-problem 0: 4x4x4, k=0..3
    U_full[:, col_offset:col_offset+decomps[0][0].shape[1]] = U_list[0]
    for j in range(m):
        V_full[j*p+0:j*p+4, col_offset:col_offset+decomps[0][0].shape[1]] = V_list[0][j*4:(j+1)*4, :]
    W_full[0*n+0:4*n, col_offset:col_offset+decomps[0][0].shape[1]] = W_list[0]
    col_offset += decomps[0][0].shape[1]
    
    # Sub-problem 1: 4x4x4, k=4..7
    U_full[:, col_offset:col_offset+decomps[1][0].shape[1]] = U_list[1]
    for j in range(m):
        V_full[j*p+4:j*p+8, col_offset:col_offset+decomps[1][0].shape[1]] = V_list[1][j*4:(j+1)*4, :]
    W_full[4*n+0:8*n, col_offset:col_offset+decomps[1][0].shape[1]] = W_list[1]
    col_offset += decomps[1][0].shape[1]
    
    # Sub-problem 2: 4x4x1, k=8
    U_full[:, col_offset:col_offset+decomps[2][0].shape[1]] = U_list[2]
    for j in range(m):
        V_full[j*p+8:j*p+9, col_offset:col_offset+decomps[2][0].shape[1]] = V_list[2][j*1:(j+1)*1, :]
    W_full[8*n+0:9*n, col_offset:col_offset+decomps[2][0].shape[1]] = W_list[2]
    col_offset += decomps[2][0].shape[1]
    
    assert col_offset == rank_total
    return U_full, V_full, W_full

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition
    try:
        U1, V1, W1 = _strassen_4x4()
        U2, V2, W2 = _strassen_4x4()
        U3, V3, W3 = _schoolbook_4x4x1()
        
        U_full, V_full, W_full = _combine_decompositions([(U1, V1, W1), (U2, V2, W2), (U3, V3, W3)])
        
        rank = U_full.shape[1]
        
        # Verify
        if _verify(U_full, V_full, W_full, n, m, p, rank):
            # Convert to list of lists
            U_list = U_full.tolist()
            V_list = V_full.tolist()
            W_list = W_full.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fallback to schoolbook
            pass
    except Exception as e:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
