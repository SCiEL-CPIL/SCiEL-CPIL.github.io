import numpy as np
import time
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x4 matrix.
    Returns a dict with keys 'U', 'V', 'W' representing factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 4
    dim_u = n * m  # 4
    dim_v = m * p  # 8
    dim_w = p * n  # 8
    
    # The multiplication tensor T has entries T[i, j, k] = 1 if output (i,k) depends on A[i,j] * B[j,k]
    # Here A is 2x2, B is 2x4.
    # A[i, j] for i in 0..1, j in 0..1
    # B[j, k] for j in 0..1, k in 0..3
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # U indices: i * m + j -> i in 0..1, j in 0..1
    # V indices: j * p + k -> j in 0..1, k in 0..3
    # W indices: k * n + i -> k in 0..3, i in 0..1
    
    # We know rank 14 works (Larsen). Let's try to find rank 13 or lower.
    # If we can't find one quickly, we return the known rank 14 solution.
    
    # First, let's define the known rank 14 solution to use as fallback
    def get_rank14_solution():
        rank = 14
        U = [[0.0]*rank for _ in range(dim_u)]
        V = [[0.0]*rank for _ in range(dim_v)]
        W = [[0.0]*rank for _ in range(dim_w)]
        
        def set_col(col, u_dict, v_dict, w_dict):
            for idx, val in u_dict.items():
                U[idx][col] = val
            for idx, val in v_dict.items():
                V[idx][col] = val
            for idx, val in w_dict.items():
                W[idx][col] = val
                
        set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
        set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
        set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
        set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
        set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
        set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
        set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
        
        set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
        set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
        set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
        set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
        set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
        set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
        set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
        
        return {"U": U, "V": V, "W": W}
    
    # Verify the rank 14 solution
    sol14 = get_rank14_solution()
    
    def verify_solution(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Build the tensor from the decomposition
        # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # Wait, the convention is:
        # U has shape (n*m, rank) -> U[i*m + j, r]
        # V has shape (m*p, rank) -> V[j*p + k, r]
        # W has shape (p*n, rank) -> W[k*n + i, r]
        
        # Let's compute C[i, k] for all i, k
        # C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal sum_j A[i, j] * B[j, k]
        # Which means the coefficient of A[i, j] * B[j, k] in C[i, k] should be 1,
        # and for all other (i, j, k) combinations, the coefficient should be 0.
        
        # Actually, let's think about this differently.
        # The tensor T is defined by T[i, j, k] = 1 if C[i, k] contains A[i, j] * B[j, k]
        # So for each (i, k), C[i, k] = sum_j A[i, j] * B[j, k]
        # The term A[i, j] * B[j, k] appears in C[i, k] with coefficient 1.
        # It does NOT appear in C[i', k'] for (i', k') != (i, k).
        # It does NOT appear in C[i, k] for j' != j.
        
        # So the condition is:
        # For all i, j, k:
        #   sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 if this is the correct term, else 0
        # But wait, for a given (i, k), there are m terms in the sum over j.
        # The decomposition gives a single scalar for each (i, j, k) triple.
        # Let's define S[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # We need S[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k], else 0.
        # Since C[i, k] = sum_j A[i, j] * B[j, k], the term A[i, j] * B[j, k] ALWAYS contributes to C[i, k].
        # So we need S[i, j, k] = 1 for ALL i, j, k?
        # No, that's not right either. Let's look at the structure again.
        
        # Actually, the standard way to verify is:
        # The bilinear map is: (A, B) -> C where C[i, k] = sum_j A[i, j] * B[j, k]
        # A rank-r decomposition is a set of r triples (u_r, v_r, w_r) where:
        # u_r is in R^(n*m), v_r is in R^(m*p), w_r is in R^(p*n)
        # And for all A, B:
        #   sum_r (u_r . vec(A)) * (v_r . vec(B)) * w_r = vec(C)
        # where vec(A) is the vectorization of A.
        
        # Let's use the index notation:
        # u_r[i*m + j] * v_r[j'*p + k] * w_r[k'*n + i']
        # Summed over r, this should give:
        # delta_{i, i'} * delta_{k, k'} * (sum_j delta_{j, j'} ... )
        
        # Actually, the simplest check is:
        # For each output element C[i, k], the coefficient of A[i, j] * B[j, k] should be 1.
        # The coefficient of A[i, j] * B[j', k'] in C[i, k] should be 0 if (j, k') != (j, k) or if we're looking at the wrong output.
        
        # Let's just compute the full tensor and compare.
        # T_decomposed[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # This should equal 1 for all valid (i, j, k) in the multiplication, and 0 otherwise.
        # But in matrix multiplication, EVERY (i, j, k) is valid in the sense that A[i, j] * B[j, k] is a term.
        # The issue is which output C[i, k] it goes to.
        
        # Let's re-read the problem statement: "output factor follows the official trace convention W[k*n+i,r]"
        # This confirms W is indexed by (k, i) where k is the column of B (and C), i is the row of A (and C).
        
        # So the decomposition is:
        # C[i, k] = sum_r (sum_j U[i*m + j, r] * A[i, j]) * (sum_k' V[j'*p + k, r] * B[j', k]) * W[k*n + i, r]
        # Wait, this doesn't look right. The standard form is:
        # C[i, k] = sum_r (u_r . A) * (v_r . B) * w_{r, i, k}
        # where u_r is a 2x2 matrix, v_r is a 2x4 matrix, w_{r, i, k} is a scalar.
        
        # In our vectorized form:
        # u_r is a vector of length n*m = 4, with u_r[i*m + j] being the (i, j) entry of matrix U_r.
        # v_r is a vector of length m*p = 8, with v_r[j*p + k] being the (j, k) entry of matrix V_r.
        # w_r is a vector of length p*n = 8, with w_r[k*n + i] being the coefficient for C[i, k].
        
        # So: C[i, k] = sum_r (sum_{i', j'} u_r[i'*m + j'] * A[i', j']) * (sum_{j', k'} v_r[j'*p + k'] * B[j', k']) * w_r[k*n + i]
        
        # We want this to equal sum_j A[i, j] * B[j, k].
        
        # Let's expand the product:
        # (sum_{i', j'} u_r[i'*m + j'] * A[i', j']) * (sum_{j', k'} v_r[j'*p + k'] * B[j', k'])
        # = sum_{i', j', j'', k'} u_r[i'*m + j'] * v_r[j''*p + k'] * A[i', j'] * B[j'', k']
        
        # So the coefficient of A[i', j'] * B[j'', k'] in C[i, k] is:
        # sum_r u_r[i'*m + j'] * v_r[j''*p + k'] * w_r[k*n + i]
        
        # We want this to be:
        # 1 if i' = i and k' = k and j' = j'' (i.e., the term A[i, j] * B[j, k] in C[i, k])
        # 0 otherwise
        
        # So the condition is:
        # For all i, i', k, k', j, j'':
        #   sum_r U[i'*m + j, r] * V[j''*p + k', r] * W[k*n + i, r] = 
        #     1 if i' = i and k' = k and j = j''
        #     0 otherwise
        
        # Let's implement this check.
        for i in range(n):
            for k in range(p):
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j in range(m):
                            for j_prime in range(m):
                                coeff = 0.0
                                for r in range(rank):
                                    coeff += U[i_prime*m + j][r] * V[j_prime*p + k_prime][r] * W[k*n + i][r]
                                expected = 1.0 if (i_prime == i and k_prime == k and j == j_prime) else 0.0
                                if abs(coeff - expected) > 1e-9:
                                    return False
        return True
    
    if verify_solution(sol14["U"], sol14["V"], sol14["W"]):
        # We have a valid rank 14 solution. Now let's try to find rank 13.
        # This is a hard problem. We'll use a simple random search with verification.
        # Given the time limit, we'll do a few random tries.
        
        # Try rank 13 with random initialization and verification
        import random
        random.seed(42)
        
        for rank in [13, 12, 11]:
            if time.time() > deadline - 10:
                break
            
            # Try a few random initializations
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random integer/half-integer entries
                U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(dim_u)]
                V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(dim_v)]
                W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(dim_w)]
                
                if verify_solution(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        # If we didn't find a lower rank, return the rank 14 solution
        return sol14
    else:
        # This shouldn't happen, but just in case
        raise ValueError("Rank 14 solution failed verification")

# Test the solution
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
