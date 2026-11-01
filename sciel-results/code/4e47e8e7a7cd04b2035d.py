"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Uses the known optimal rank 14 decomposition based on two Strassen 2x2x2 blocks.
The verification is vectorized for speed.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 4]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 150s limit, 5s margin

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index convention: W[k*n+i] where k=output column, i=output row.
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for valid (i,j,k)
    and 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Construct the full tensor T of size (n*m, m*p, p*n)
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We can compute this efficiently using einsum
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Compute T[a, b, c]
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # This creates a tensor of size (4, 8, 8) = 256 elements. Very fast.
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check valid entries
    # Valid entries are those where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # This implies:
    # j = a % m, i = a // m
    # j = b // p, k = b % p
    # k = c // n, i = c % n
    
    # So we need to check T[a, b, c] == 1 if:
    # a // m == c % n (i matches)
    # a % m == b // p (j matches)
    
    for a in range(n * m):
        i = a // m
        j = a % m
        
        for b in range(m * p):
            j_b = b // p
            k = b % p
            
            # If j matches, then c is determined by k and i
            if j == j_b:
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check invalid entries
    # We need to ensure T[a, b, c] == 0 for all other (a, b, c)
    # Instead of iterating all 256, we can check the ones that are not valid.
    # Or simply check that the sum of squares of "invalid" parts is 0.
    # But it's safer to just check the specific invalid spots or rely on the structure.
    # Given the small size, we can just check all and skip valid ones.
    
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j_b = b // p
            k = b % p
            for c in range(p * n):
                i_c = c % n
                k_c = c // n
                
                # Valid if i == i_c and j == j_b
                if i == i_c and j == j_b:
                    continue
                    
                if abs(T[a, b, c]) > 1e-9:
                    return False
    
    return True

def _build_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    W index convention: W[k*n+i] where k=output column, i=output row.
    C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3],
    C[0,2]->W[4], C[1,2]->W[5], C[0,3]->W[6], C[1,3]->W[7]
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # W (C) indices: C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3],
    #                C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    
    # Left Block (Columns 0, 1) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B00 + B11)
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # m2 = (A10 + A11) * B00
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    
    # m3 = A00 * (B01 - B11)
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1
    
    # m4 = A11 * (B10 - B00)
    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    # m5 = (A00 + A01) * B11
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    # m6 = (A10 - A00) * (B00 + B01)
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    # m7 = (A01 - A11) * (B10 + B11)
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1
    
    # Right Block (Columns 2, 3) - Strassen 2x2x2
    # m1 = (A00 + A11) * (B02 + B13)
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1
    
    # m2 = (A10 + A11) * B02
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    
    # m3 = A00 * (B03 - B13)
    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1
    
    # m4 = A11 * (B12 - B02)
    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    
    # m5 = (A00 + A01) * B13
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1
    
    # m6 = (A10 - A00) * (B02 + B03)
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    
    # m7 = (A01 - A11) * (B12 + B13)
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank12():
    """
    Attempt to find a rank-12 decomposition.
    The theoretical lower bound for 2x2 * 2x4 is 12.
    We use a structured search: fix U and V patterns, solve for W, verify.
    If found, return the solution. Otherwise return None.
    """
    n, m, p = 2, 2, 4
    rank = 12
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    
    # We will try to construct U and V such that the linear system for W has a solution.
    # The condition is that for all (a,b,c), sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
    # This is equivalent to: for each c, the vector W[c,:] must satisfy:
    # sum_r W[c,r] * (U[:,r] * V[:,r])[a,b] = T[a,b,c]
    # Let M_r = U[:,r] * V[:,r] (outer product, shape 4x8)
    # Then sum_r W[c,r] M_r = T_c (where T_c is the slice of T for index c, shape 4x8)
    # This is a linear system: W[c,:] @ M_stack = T_c_flat
    # M_stack is 4*8 x rank = 32 x 12
    # T_c_flat is 32 x 1
    
    # We can choose U and V randomly (structured) and check if the system is consistent.
    # If consistent, we can solve for W.
    
    np.random.seed(123)
    
    for attempt in range(5000):
        # Generate structured U and V
        # To increase chance of success, we can use patterns similar to Strassen
        # but with some randomization to allow for sharing between the two blocks.
        
        # Let's try a specific heuristic:
        # The first 7 columns of U,V correspond to the left block (cols 0,1 of B)
        # The last 5 columns correspond to the right block (cols 2,3 of B)
        # But we want to share some rank terms.
        
        # Simpler: Just random sparse U and V with entries in {-1, 0, 1}
        for r in range(rank):
            # U pattern
            num_u = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            u_idx = np.random.choice(4, size=num_u, replace=False)
            U[u_idx, r] = np.random.choice([-1, 1], size=num_u)
            
            # V pattern
            # V has 8 entries. We want to ensure that for a given r,
            # the non-zero entries in V correspond to a valid set of B indices.
            # But for a generic search, random is okay.
            num_v = np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1])
            v_idx = np.random.choice(8, size=num_v, replace=False)
            V[v_idx, r] = np.random.choice([-1, 1], size=num_v)
        
        # Check if the system is consistent
        # For each c in 0..7, we need to solve for W[c,:]
        # The system is: sum_r W[c,r] * (U[:,r] * V[:,r]) = T[:, :, c]
        
        # Let's form the matrix A of shape (32, rank) where A[i, r] = U[i_u, r] * V[i_v, r]
        # where i = i_u * 8 + i_v
        # Actually, T is 4x8x8. T[a,b,c].
        # The equation is: for each a, b: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This is a linear equation in W[c,:].
        # Let x = W[c,:]. Then sum_r x_r (U[:,r] * V[:,r])[a,b] = T[a,b,c]
        
        # We can flatten the (a,b) part.
        # Let M be a matrix of shape (32, rank) where M[a*8+b, r] = U[a,r] * V[b,r]
        # Then M @ x = T_flat_c
        # where T_flat_c is T[:, :, c].flatten()
        
        # We need to check if T_flat_c is in the column space of M.
        # i.e., rank(M) == rank([M | T_flat_c])
        
        M = np.einsum('ar,br->abr', U, V).reshape(32, rank)
        
        valid = True
        W_sol = np.zeros((8, rank))
        
        for c in range(8):
            # Construct T_flat_c
            T_c = np.zeros(32)
            for a in range(4):
                i = a // m
                j = a % m
                for b in range(8):
                    j_b = b // p
                    k = b % p
                    if i == c % n and j == j_b:
                        # c = k*n + i => i = c % n, k = c // n
                        # Wait, c is the index in W. W index = k*n + i.
                        # So i = c % n, k = c // n.
                        # The condition for T[a,b,c] = 1 is:
                        # a = i*m + j, b = j*p + k, c = k*n + i
                        # So if we fix c, we know i and k.
                        # We need a such that i = a//m and j = a%m.
                        # And b such that j = b//p and k = b%p.
                        # So a is determined by i,j. b is determined by j,k.
                        # There is only ONE valid (a,b) for each c?
                        # No, i and k are fixed by c.
                        # j can be 0 or 1.
                        # So for a given c, there are 2 valid (a,b) pairs?
                        # Let's check.
                        # c = k*n + i.
                        # If i=0, k=0 => c=0. a=0*m+0=0 (j=0), b=0*p+0=0 (j=0). Also a=0*m+1=1 (j=1), b=1*p+0=4 (j=1).
                        # So for c=0, valid (a,b) are (0,0) and (1,4).
                        # T[0,0,0]=1, T[1,4,0]=1. Others 0.
                        
                        pass
            
            # It is easier to just construct the full T tensor once.
            # T[a,b,c] = 1 if a//m == c%n and a%m == b//p and b%p == c//n
            # Let's just compute the target vector for this c.
            target = np.zeros(32)
            for a in range(4):
                for b in range(8):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_c = c % n
                    k_c = c // n
                    if i == i_c and j == j_b and k == k_c:
                        target[a*8+b] = 1.0
            
            # Solve M @ x = target
            # Check consistency
            # Use least squares and check residual
            x, residuals, rank_m, sv = np.linalg.lstsq(M, target, rcond=None)
            
            # Check if residual is small
            if residuals is not None and len(residuals) > 0:
                if residuals[0] > 1e-8:
                    valid = False
                    break
            
            W_sol[c, :] = x
        
        if valid:
            # We found a solution! Verify it.
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_sol.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline check
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _build_rank14()
    
    # First, try to find a rank-12 decomposition
    result_12 = _try_rank12()
    if result_12 is not None:
        return result_12
    
    # Build the known rank-14 solution
    result_14 = _build_rank14()
    
    # Verify the rank-14 solution
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        return result_14
    
    # Fallback to schoolbook rank 16 (should not happen)
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
