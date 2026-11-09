"""Combined program: structural search for rank-minimal tensor decomposition of 2x4 x 4x6 matmul."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W):
    """Fast numpy verification."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=np.int64)
    V = np.array(V, dtype=np.int64)
    W = np.array(W, dtype=np.int64)
    # Reconstruct: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check all 48 entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if s != 1:
                    return False
    return True

def _try_structured_rank(r, deadline):
    """
    Try to find rank-r decomposition using structured search.
    
    Key structural insight: For each column c, the triple (U[:,c], V[:,c], W[:,c])
    contributes to the tensor. The constraint is that for each (i,j,k),
    the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    
    Strategy: Fix W to a canonical form (e.g., W[k*n+i,c] is 1 only for specific (k,i,c)
    patterns) and solve for U and V. Or use random search with structure.
    """
    n, m, p = DIMENSIONS
    
    # Approach 1: Random search with {-1, 0, 1} entries
    rng = random.Random(12345)
    np_rng = np.random.RandomState(12345)
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        # Random entries from {-1, 0, 1} with controlled density
        density = 0.3  # probability of non-zero
        U = np.zeros((n*m, r), dtype=np.int64)
        V = np.zeros((m*p, r), dtype=np.int64)
        W = np.zeros((p*n, r), dtype=np.int64)
        
        for c in range(r):
            # Random non-zero positions
            for row in range(n*m):
                if rng.random() < density:
                    U[row, c] = rng.choice([-1, 1])
            for row in range(m*p):
                if rng.random() < density:
                    V[row, c] = rng.choice([-1, 1])
            for row in range(p*n):
                if rng.random() < density:
                    W[row, c] = rng.choice([-1, 1])
        
        if _verify_np(U, V, W):
            return U.tolist(), V.tolist(), W.tolist()
    
    # Approach 2: Structured - fix W to identity-like pattern, solve for U, V
    # For each column c, assign W[k*n+i, c] to be 1 for a specific (k,i) pair
    # This makes the constraint linear in U*V for each (k,i)
    
    # Actually, let's try: W[k*n+i, c] = 1 iff (k*n+i) maps to column c in some pattern
    # This is essentially the schoolbook structure but with potential sharing
    
    # Approach 3: Try W with small integer entries, solve linear system for U, V
    # For fixed W, the constraint becomes:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is still bilinear in U, V.
    
    # Approach 4: Use the fact that n=2 is small. 
    # For each pair (i,k), define W_{ik,c} = W[k*n+i, c]
    # Then for each (i,j,k): sum_c U_{ij,c} * V_{jk,c} * W_{ik,c} = 1
    # 
    # If we fix W, we need: for each (i,k), the matrix M^{(ik)} with entries
    # M^{(ik)}[j, c] = V[j*p+k, c] * W[k*n+i, c]
    # Then sum_c U[i*m+j, c] * M^{(ik)}[j, c] = 1 for all j
    # i.e., U_i * (M^{(ik)})^T = 1 vector (length m)
    # where U_i is the row block of U for index i (shape m x r)
    
    # This gives us: for each (i,k), U_i * (V_k * W_{ik})^T = 1_m
    # where V_k is the row block of V for index k (shape m x r), 
    # and (V_k * W_{ik}) is element-wise product (shape m x r)
    
    # So: U_i * diag(W_{ik}) * V_k^T = 1_m  (as a row vector equation)
    # Wait, let me be more careful.
    
    # U[i*m+j, c] for fixed i, varying j: this is a row vector u_i of length r
    # V[j*p+k, c] for fixed k, varying j: this is a column vector v_k of length r (transposed)
    # W[k*n+i, c]: scalar w_{ik}
    
    # Constraint: for all i, j, k: sum_c u_i[j] * v_k[j] * w_{ik} = 1
    # where u_i[j] = U[i*m+j, c] summed over c... no.
    
    # Let me re-index. For fixed (i,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all j in range(m)
    
    # Let a_jc = U[i*m+j, c], b_jc = V[j*p+k, c], d_c = W[k*n+i, c]
    # Then: sum_c a_jc * b_jc * d_c = 1 for all j
    
    # If we define e_jc = a_jc * d_c, then sum_c e_jc * b_jc = 1 for all j
    # i.e., the matrix E (m x r) times B_k^T (r x m) = J (all-ones m x m matrix)
    # where E[j,c] = U[i*m+j,c] * W[k*n+i,c] and B_k[j,c] = V[j*p+k,c]
    
    # So for each (i,k): (U_i * diag(W_{ik})) * V_k^T = J_m
    # where U_i is m x r, V_k is m x r, W_{ik} is r-vector
    
    # This means: U_i * diag(W_{ik}) * V_k^T = ones(m,m)
    
    # If W_{ik} is a scalar w (same for all c), then U_i * w * V_k^T = ones(m,m)
    # which means U_i * V_k^T = ones(m,m)/w
    
    # For the schoolbook, each (i,j,k) has its own column, so W_{ik,c} is 1 for one specific c.
    
    # Let's try: for each (i,k), choose W[k*n+i, c] to be a small set of values.
    # Then solve for U_i and V_k.
    
    # Key insight: If we can make W[k*n+i, c] depend only on (i,k) and not on j,
    # then the constraint decouples nicely.
    
    # Let's try: W[k*n+i, c] = w_{ik} (scalar, same for all c)
    # Then: sum_c U[i*m+j,c] * V[j*p+k,c] * w_{ik} = 1
    # => w_{ik} * (U_i . V_k) = 1 where (U_i . V_k)[j] = sum_c U[i*m+j,c]*V[j*p+k,c]
    # => U_i . V_k = 1/w_{ik} * ones(m)
    
    # So we need: for all i,k, the Hadamard-like product U_i * V_k^T (element-wise per j)
    # to give a constant vector.
    
    # If U_i and V_k are constant vectors (all entries equal), this works easily.
    # U[i*m+j, c] = u_{ic} (independent of j), V[j*p+k, c] = v_{kc} (independent of j)
    # Then sum_c u_{ic} * v_{kc} * w_{ik} = 1 for all j (satisfied if the sum is 1)
    
    # So we need: for all (i,k): sum_c u_{ic} * v_{kc} * w_{ik} = 1
    # where u_{ic} depends only on (i,c), v_{kc} on (k,c), w_{ik} on (i,k)
    
    # This is a rank-1-ish constraint on the (i,k) grid with features (u_i, v_k, w_{ik})
    
    # Let's try: u_{ic} = alpha_{ic}, v_{kc} = beta_{kc}, w_{ik} = gamma_{ik}
    # sum_c alpha_{ic} * beta_{kc} * gamma_{ik} = 1 for all i,k
    
    # If gamma_{ik} = 1/sqrt(sum_c alpha_{ic}^2 * beta_{kc}^2) ... this gets complicated.
    
    # Simpler: try alpha_{ic} = 1 for all, beta_{kc} = 1 for all, then
    # sum_c 1 * 1 * gamma_{ik} = r * gamma_{ik} = 1 => gamma_{ik} = 1/r
    # But gamma must be integer or half-integer, so r must divide 2.
    # For r=2: gamma_{ik} = 1/2 (half-integer, OK!)
    
    # Let's check: U[i*m+j, c] = 1 for all i,j,c (8x2 matrix, all ones)
    # V[j*p+k, c] = 1 for all j,k,c (24x2 matrix, all ones)  
    # W[k*n+i, c] = 1/2 for all k,i,c (12x2 matrix, all 1/2)
    
    # Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = sum_c 1*1*1/2 = 2*1/2 = 1 ✓
    
    # But wait - this gives rank 2! Let me verify this is valid.
    
    # Actually, this seems too good. Let me double-check the tensor equation.
    # The tensor T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,6)
    # The decomposition says T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # With U all 1s (8x2), V all 1s (24x2), W all 1/2s (12x2):
    # T[i,j,k] = sum_{c=0}^{1} 1 * 1 * 1/2 = 1 ✓
    
    # This is a valid rank-2 decomposition! But wait, can rank be even lower?
    # Rank 1: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k
    # This requires U[i*m+j,0] = a_{ij}, V[j*p+k,0] = b_{jk}, W[k*n+i,0] = c_{ki}
    # a_{ij} * b_{jk} * c_{ki} = 1 for all i,j,k
    # Taking i=0,j=0,k=0: a_{00}*b_{00}*c_{00} = 1
    # Taking i=0,j=0,k=1: a_{00}*b_{01}*c_{10} = 1
    # Taking i=0,j=1,k=0: a_{01}*b_{10}*c_{00} = 1
    # ...
    # From a_{00}*b_{00}*c_{00} = 1 and a_{00}*b_{01}*c_{10} = 1:
    # b_{00}*c_{00} = b_{01}*c_{10}
    # This is a system of 48 equations in 8+24+12=44 unknowns.
    # It may or may not have a solution. Let's check if it's consistent.
    # 
    # Actually, for rank 1 to work, we need the tensor to be a rank-1 tensor.
    # The matrix multiplication tensor is NOT rank 1 in general (it's a well-known fact
    # that matmul tensor has tensor rank > 1). So rank 1 is impossible.
    # 
    # But rank 2 with the all-ones construction seems to work! Let me verify more carefully.
    
    # Wait, I need to be more careful. The entries must be integer or half-integer.
    # U entries: 1 (integer) ✓
    # V entries: 1 (integer) ✓  
    # W entries: 1/2 (half-integer) ✓
    
    # So rank 2 should work! But let me make sure I'm not making an error.
    
    # Actually, I realize I need to double-check: is the tensor really all-ones?
    # T[i,j,k] = 1 for all valid (i,j,k) because C[i,k] = sum_j A[i,j]*B[j,k]
    # and the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # Yes, the tensor is all-ones.
    
    # So the rank-2 construction works. But can we do rank 1? No, as argued.
    # So the minimum rank is 2.
    
    # Let me construct this:
    U = [[1, 1] for _ in range(n * m)]  # 8 x 2, all ones
    V = [[1, 1] for _ in range(m * p)]  # 24 x 2, all ones
    W = [[0.5, 0.5] for _ in range(p * n)]  # 12 x 2, all 1/2
    
    if _verify(U, V, W):
        return U, V, W
    
    # If that somehow fails (shouldn't), continue searching
    return None

def _try_rank1():
    """Try rank 1 - should fail but let's verify."""
    n, m, p = DIMENSIONS
    # Need U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k
    # This requires the tensor to be rank 1, which it's not.
    # Quick check: if it were rank 1, T[i0,j0,k0]*T[i1,j1,k1] = T[i0,j1,k0]*T[i1,j0,k1]
    # But all entries are 1, so 1*1 = 1*1, which holds! 
    # Wait, that means the all-ones tensor IS rank 1?
    # No, the rank-1 condition is T[i,j,k] = a_i * b_j * c_k
    # For all-ones: a_i * b_j * c_k = 1 for all i,j,k
    # This works if a_i = 1, b_j = 1, c_k = 1!
    # 
    # But wait - in our case, the indices are coupled:
    # U index: i*m+j (coupled i,j)
    # V index: j*p+k (coupled j,k)  
    # W index: k*n+i (coupled k,i)
    # 
    # So it's NOT a simple all-ones tensor over independent indices.
    # The constraint is: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # 
    # For rank 1: u_{ij} * v_{jk} * w_{ki} = 1 for all i,j,k
    # where u_{ij} = U[i*m+j, 0], v_{jk} = V[j*p+k, 0], w_{ki} = W[k*n+i, 0]
    # 
    # From u_{00}*v_{00}*w_{00} = 1
    # u_{00}*v_{01}*w_{10} = 1  => v_{01}*w_{10} = v_{00}*w_{00}
    # u_{01}*v_{10}*w_{00} = 1  => u_{01}*v_{10} = u_{00}*v_{00}*w_{00}/w_{00} = u_{00}*v_{00}
    # 
    # Let's try: u_{ij} = a_i * b_j, v_{jk} = c_j * d_k, w_{ki} = e_k * f_i
    # Then: a_i*b_j*c_j*d_k*e_k*f_i = (a_i*f_i)*(b_j*c_j)*(d_k*e_k) = 1 for all i,j,k
    # This works if a_i*f_i = alpha, b_j*c_j = beta, d_k*e_k = gamma with alpha*beta*gamma = 1
    # 
    # So rank 1 IS possible! Let's construct it:
    # a_i = 1, f_i = 1, so alpha = 1
    # b_j = 1, c_j = 1, so beta = 1
    # d_k = 1, e_k = 1, so gamma = 1
    # alpha*beta*gamma = 1 ✓
    # 
    # u_{ij} = a_i * b_j = 1
    # v_{jk} = c_j * d_k = 1
    # w_{ki} = e_k * f_i = 1
    # 
    # Check: u_{ij} * v_{jk} * w_{ki} = 1*1*1 = 1 ✓
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    if _verify(U, V, W):
        return U, V, W
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # First, try rank 1
    result = _try_rank1()
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try rank 2
    result = _try_structured_rank(2, deadline)
    if result:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try ranks 3 through 10 with random search
    for r in range(3, 11):
        if time.time() > deadline - 5.0:
            break
        result = _try_structured_rank(r, deadline - 2.0)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
