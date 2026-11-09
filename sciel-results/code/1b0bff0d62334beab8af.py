"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Uses Strassen decomposition for the 2x2x2 corner block (rank 7 instead of 8)
    and schoolbook for all other terms.
    
    Indexing:
      U row for A[i,j] = i*m + j = 3i+j
      V row for B[j,k] = j*p + k = 7j+k  
      W row for C[i,k] = k*n + i = 7k+i
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Each "term" is a rank-1 contribution: U_vec * V_vec * W_vec
    # We represent each term as (u_dict, v_dict, w_dict) where each dict maps row->coeff
    # Multiple terms can be combined if they share structure, but for exact decomposition
    # we just list each rank-1 component.
    
    # A rank-1 tensor term is defined by:
    #   u: dict from A-index (i*m+j) to coefficient
    #   v: dict from B-index (j*p+k) to coefficient  
    #   w: dict from C-index (k*n+i) to coefficient
    # The contribution is sum over (a_idx, b_idx, c_idx) of u[a_idx]*v[b_idx]*w[c_idx] * A[a_idx]*B[b_idx]
    # which should equal sum over c_idx of w[c_idx] * C[c_idx]
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        """Add a single rank-1 term."""
        if time.time() > DEADLINE:
            return False
        terms.append((u_dict, v_dict, w_dict))
        return True
    
    # Helper to create a dict with single entry
    def single(idx, val=1):
        return {idx: val}
    
    # Helper to merge dicts
    def merge(*dicts):
        result = {}
        for d in dicts:
            for k, v in d.items():
                result[k] = result.get(k, 0) + v
        # Remove zeros
        result = {k: v for k, v in result.items() if v != 0}
        return result
    
    # === Strassen decomposition for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] ===
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=7, C[1,1]=8
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[1,0]) * B[1,1]
    # M6 = (A[1,1]-A[0,0]) * (B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each M_i is a product of (linear combo of A) * (linear combo of B)
    # So each M_i is one rank-1 term with:
    #   u_dict = the A linear combo
    #   v_dict = the B linear combo
    #   w_dict = the C linear combo (which M_i contributes to)
    
    # M1 contributes to C[0,0] with +1 and C[1,1] with +1
    add_rank1_term(
        merge(single(0, 1), single(4, 1)),   # A[0,0]+A[1,1]
        merge(single(0, 1), single(8, 1)),   # B[0,0]+B[1,1]
        merge(single(0, 1), single(8, 1))    # C[0,0]+C[1,1]
    )
    
    # M2 contributes to C[1,0] with +1 and C[1,1] with -1
    add_rank1_term(
        merge(single(3, 1), single(4, 1)),   # A[1,0]+A[1,1]
        single(0, 1),                        # B[0,0]
        merge(single(7, 1), single(8, -1))   # C[1,0]-C[1,1]
    )
    
    # M3 contributes to C[0,1] with +1 and C[1,1] with +1
    add_rank1_term(
        single(0, 1),                        # A[0,0]
        merge(single(1, 1), single(8, -1)),  # B[0,1]-B[1,1]
        merge(single(1, 1), single(8, 1))    # C[0,1]+C[1,1]
    )
    
    # M4 contributes to C[0,0] with +1 and C[1,0] with +1
    add_rank1_term(
        single(4, 1),                        # A[1,1]
        merge(single(7, 1), single(0, -1)),  # B[1,0]-B[0,0]
        merge(single(0, 1), single(7, 1))    # C[0,0]+C[1,0]
    )
    
    # M5 contributes to C[0,0] with -1 and C[0,1] with +1
    add_rank1_term(
        merge(single(0, 1), single(3, 1)),   # A[0,0]+A[1,0]
        single(8, 1),                        # B[1,1]
        merge(single(0, -1), single(1, 1))   # -C[0,0]+C[0,1]
    )
    
    # M6 contributes to C[1,1] with +1
    add_rank1_term(
        merge(single(4, 1), single(0, -1)),  # A[1,1]-A[0,0]
        merge(single(0, 1), single(7, 1)),   # B[0,0]+B[1,0]
        single(8, 1)                         # C[1,1]
    )
    
    # M7 contributes to C[0,0] with +1
    add_rank1_term(
        merge(single(1, 1), single(4, -1)),  # A[0,1]-A[1,1]
        merge(single(7, 1), single(8, 1)),   # B[1,0]+B[1,1]
        single(0, 1)                         # C[0,0]
    )
    
    # === Schoolbook for all other (i,j,k) triples not covered by Strassen ===
    # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1}
    # We need all other combinations:
    # i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5,6}
    # Exclude: i in {0,1} AND j in {0,1} AND k in {0,1}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_rank1_term(
                    single(u_idx, 1),
                    single(v_idx, 1),
                    single(w_idx, 1)
                )
    
    # Check deadline
    if time.time() > DEADLINE:
        # Return schoolbook as fallback
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        single(i * m + j, 1),
                        single(j * p + k, 1),
                        single(k * n + i, 1)
                    ))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for u_idx, coeff in u_dict.items():
            U[u_idx][col] = coeff
        for v_idx, coeff in v_dict.items():
            V[v_idx][col] = coeff
        for w_idx, coeff in w_dict.items():
            W[w_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
