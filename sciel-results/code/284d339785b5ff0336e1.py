import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
DEADLINE = 145  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try structured decompositions using Strassen-like patterns
    # Key insight: 4 = 2*2, so we can use rank-7 Strassen on 2x2x2 cores
    
    # Strategy: Split m=4 into two blocks of size 2
    # For each (i,k) pair, C[i,k] = sum over j in block1 + sum over j in block2
    # Use Strassen to reduce the 2x2x2 core from rank 8 to rank 7
    
    # Actually, let's try a more direct approach:
    # The tensor T[i,j,k] = delta_{i,k} for matrix mult is NOT correct
    # T[i,j,k] = 1 for all (i,j,k) valid indices, and the decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    
    # Let's try rank values from 55 down, using structured initialization
    for target_rank in range(55, 44, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        result = _try_structured_strassen(n, m, p, target_rank)
        if result is not None:
            best_result = result
            break
    
    # If no improvement, try a more aggressive search with lower ranks
    if best_result == {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}:
        for target_rank in range(55, 40, -1):
            if time.time() - start_time > DEADLINE - 5:
                break
            result = _try_random_structured(n, m, p, target_rank)
            if result is not None:
                best_result = result
                break
    
    return best_result

def _try_structured_strassen(n, m, p, target_rank):
    """
    Try to use Strassen-like structure for 2x2x2 subproblems.
    Split m=4 into two blocks of 2. For each block, the sub-tensor is 3x2x5.
    We can further decompose using the fact that for fixed i,k, the j-sum 
    over 2 elements can be handled with rank-2 (schoolbook) or potentially less.
    
    Actually, the key is: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Group as: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    Each group is a 1x2x1 multiplication, which has rank 2 (schoolbook).
    But we can use Strassen if we group multiple (i,k) pairs.
    
    Let's try: group all i for a fixed k, or all k for a fixed i.
    """
    if n != 3 or m != 4 or p != 5:
        return None
    
    # Try a specific structured decomposition
    # Use the fact that we can share structure across i or k dimensions
    
    # Construction idea:
    # For the 2x2x2 core (j in {0,1}, and we need to handle i,k),
    # Strassen gives: 7 multiplications instead of 8
    
    # Let's build a decomposition with rank = 3*5*2 - 3*5 = 30? No, that's not right.
    
    # Actually, let me try a different approach: use the known result that
    # R(1,2,2) = 3 (Strassen), R(2,2,2) = 7 (Strassen)
    
    # For 3x4x5:
    # = (3x1x1) x (1x4x5) + ... no, that's not a valid decomposition
    
    # Let's try: view as sum over j of (n x 1 x 1) outer (1 x 1 x p)
    # That's the schoolbook with rank n*m*p = 60
    
    # To improve, we need to find a decomposition where some U,V,W entries 
    # are shared across different (i,j,k) tuples.
    
    # Try: use rank-7 Strassen on the j-dimension by grouping
    # Split j into {0,1} and {2,3}
    # For each (i,k), C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    
    # Define intermediate products using Strassen on the 2x2 block:
    # M1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])  -- but this mixes i and k
    # This doesn't work directly because Strassen requires 2x2 matrices.
    
    # Alternative: fix i, vary k. For fixed i, we have a 1x4x5 multiplication.
    # R(1,4,5) = ? We know R(1,2,2) = 3, so R(1,4,4) <= 3*3 = 9? 
    # Actually R(1,m,p) = m*p generally, but can be less with clever constructions.
    
    # For 1x4x5: C[k] = sum_j A[j] B[j,k] for k=0..4
    # This is a linear map from (A[0..3], B[0..4]) to C[0..4]
    # The tensor is 1x4x5 with T[0,j,k] = 1 for all j,k
    # Rank of this tensor: it's the "all-ones" tensor of size 1x4x5
    # The rank of the all-ones tensor of size a x b x c is min(a,b,c) if one dim is 1?
    # No, T[0,j,k] = 1 means it's a matrix of all 1s when flattened, which has rank 1!
    # Wait, that's not right for matrix multiplication tensor.
    
    # Let me re-examine: for matrix mult, T[i,j,k] is defined such that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # This is NOT the all-ones tensor in the sense of rank-1; it's a 3D tensor.
    
    # The rank of the all-ones tensor of size n x m x p is n*m*p (schoolbook)
    # unless there's a special structure.
    
    # I think for generic n,m,p, the rank is n*m*p, but for specific cases
    # like 2x2x2, Strassen shows rank 7 < 8.
    
    # For 3x4x5, I'm not sure of the exact minimum rank. Let's try to find
    # a decomposition with rank < 60 using a search.
    
    return None

def _try_random_structured(n, m, p, target_rank):
    """
    Try to find a decomposition with half-integer entries using structured random search.
    Use the constraint that the decomposition must be exact.
    """
    if time.time() - _start_time_global() > DEADLINE - 10:
        return None
    
    np.random.seed(42)
    
    # Use values in {0, 0.5, 1} with some probability of -0.5, -1
    values = [0, 0.5, 1, -0.5, -1]
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - _start_time_global() > DEADLINE - 10:
            break
        
        # Initialize with sparse random half-integers
        density = 0.3
        U = np.random.choice(values, size=(n*m, target_rank), p=[0.5, 0.2, 0.15, 0.1, 0.05])
        V = np.random.choice(values, size=(m*p, target_rank), p=[0.5, 0.2, 0.15, 0.1, 0.05])
        W = np.random.choice(values, size=(p*n, target_rank), p=[0.5, 0.2, 0.15, 0.1, 0.05])
        
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

_start_time_cache = None
def _start_time_global():
    global _start_time_cache
    if _start_time_cache is None:
        _start_time_cache = time.time()
    return _start_time_cache

def _verify_fast(U, V, W, n, m, p):
    """
    Fast verification: check that for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    """
    # This is O(n*m*p*rank) which for n=3,m=4,p=5,rank=50 is 3*4*5*50 = 3000 ops
    # Very fast
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify(U, V, W, n, m, p):
    """
    Full verification using random matrices.
    """
    np.random.seed(0)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for r in range(U.shape[1]):
                sum_U = 0.0
                for j in range(m):
                    sum_U += U[i * m + j, r] * A[i, j]
                sum_V = 0.0
                for j in range(m):
                    sum_V += V[j * p + k, r] * B[j, k]
                val += W[k * n + i, r] * sum_U * sum_V
            C_comp[i, k] = val
            
    if not np.allclose(C_true, C_comp, atol=1e-9):
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p):
    return _verify(U, V, W, n, m, p)
