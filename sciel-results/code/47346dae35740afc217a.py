import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 2, 4]

def _generate_candidate_components():
    """
    Generate a set of candidate rank-1 components (u, v, w) for the 2x2x4 multiplication tensor.
    The tensor T has indices (i, j, k) with i in {0,1}, j in {0,1}, k in {0,1,2,3}.
    T[i, j, k] = 1 for all valid i, j, k.
    
    We look for components where u, v, w are vectors of length 4, 8, 8 respectively (flattened).
    Actually:
    U shape: (n*m, rank) -> (4, R). u_r is a vector of length 4.
    V shape: (m*p, rank) -> (8, R). v_r is a vector of length 8.
    W shape: (p*n, rank) -> (8, R). w_r is a vector of length 8.
    
    The condition is: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k.
    
    We generate candidates based on Strassen-like blocks for the two 2x2 sub-matrices of B.
    """
    n, m, p = 2, 2, 4
    
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    candidates = []
    
    # Helper to add a candidate
    def add_candidate(u_vals, v_vals, w_vals):
        # u_vals: dict or list of 4
        # v_vals: dict or list of 8
        # w_vals: dict or list of 8
        u = np.zeros(4)
        v = np.zeros(8)
        w = np.zeros(8)
        
        for idx, val in u_vals.items():
            u[idx] = val
        for idx, val in v_vals.items():
            v[idx] = val
        for idx, val in w_vals.items():
            w[idx] = val
            
        # Normalize to avoid scaling issues? 
        # The ILP will handle linear combination. 
        # However, to keep numbers small, we prefer integer/half-integer entries.
        
        candidates.append((u, v, w))
        
    # We need to cover the "diagonal" structure.
    # Let's generate candidates that are non-zero on specific subsets.
    
    # Strategy: Generate all combinations of "signs" for Strassen terms on the two blocks?
    # That's too many.
    
    # Better: Use the known rank-7 Strassen decomposition for 2x2x2.
    # For 2x2x4, we can view it as two 2x2x2 multiplications sharing the A matrix.
    # Let's construct candidates that are "local" to a specific k-block or span them.
    
    # Known rank-7 decomposition for 2x2x2 (indices i,j,k in {0,1}):
    # We need to map this to the 2x2x4 context.
    # Let's just generate a large set of random-ish but structured candidates and let the ILP pick.
    # Or, generate the specific Strassen terms for k=0,1 and k=2,3.
    
    # Strassen terms for a single 2x2 block (k in {k0, k0+1}):
    # Let's define the 7 Strassen patterns for a generic 2x2 block.
    # The 7 patterns for (i,j,k) in {0,1}^3 are:
    # M1: (A00+A11)(B00+B11) -> u: [1,0,0,1], v: [1,0,0,1], w: [1,0,0,1] (in 2x2 indexing)
    # M2: (A10+A11)B00       -> u: [0,0,1,1], v: [1,0,0,0], w: [0,0,0,1] (wait, w maps to C)
    # Let's be precise about the tensor indices.
    # T[i,j,k] = 1.
    # Rank 1 component: u[i,j] * v[j,k] * w[k,i].
    
    # Standard Strassen Rank 7 decomposition for 2x2x2:
    # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1]  (M1)
    # u2 = [0,0,1,1], v2 = [1,0,0,0], w2 = [0,0,0,1]  (M2) -> C10, C11?
    # Let's verify M2: u2[i,j] is 1 if i=1, j in {0,1}. v2[j,k] is 1 if j=0, k=0.
    # Product is 1 if i=1, j=0, k=0. w2[k,i] must be 1 if k=0, i=1.
    # So w2[0,1] = 1. In flattened (k*2+i), index is 0*2+1=1.
    # So w2 = [0,1,0,0].
    
    # Let's list the 7 standard Strassen components for 2x2x2 (flattened indices):
    # Indices: u: i*2+j, v: j*2+k, w: k*2+i
    
    s1_u = [1, 0, 0, 1] # A00, A11
    s1_v = [1, 0, 0, 1] # B00, B11
    s1_w = [1, 0, 0, 1] # C00, C11
    
    s2_u = [0, 0, 1, 1] # A10, A11
    s2_v = [1, 0, 0, 0] # B00
    s2_w = [0, 1, 0, 0] # C10 (k=0, i=1)
    
    s3_u = [1, 0, 0, 0] # A00
    s3_v = [0, 0, 1, -1] # B10, -B11 (j=1, k=0 and k=1) -> v[1*2+0]=1, v[1*2+1]=-1
    s3_w = [0, 0, 1, 0] # C01 (k=1, i=0)
    
    s4_u = [0, 0, 0, 1] # A11
    s4_v = [1, -1, 0, 0] # B01, -B00 (j=0, k=1 and k=0) -> v[0*2+1]=1, v[0*2+0]=-1
    s4_w = [1, 0, 0, 1] # C00, C11
    
    s5_u = [1, 1, 0, 0] # A00, A01
    s5_v = [0, 0, 0, 1] # B11
    s5_w = [0, 0, 0, 1] # C11
    
    s6_u = [0, -1, 1, 0] # A10, -A00
    s6_v = [1, 0, 1, 0] # B00, B10
    s6_w = [0, 1, 0, 0] # C10
    
    s7_u = [0, 1, 0, -1] # A01, -A11
    s7_v = [0, 1, 0, 1] # B01, B11
    s7_w = [0, 0, 1, 1] # C01, C11
    
    # Verify one: M1
    # sum u1[i,j]v1[j,k]w1[k,i]
    # i=0,j=0,k=0: u[0]=1, v[0]=1, w[0]=1 -> 1
    # i=0,j=0,k=1: u[0]=1, v[1]=0, w[1]=0 -> 0
    # i=0,j=1,k=0: u[1]=0, v[2]=0, w[0]=1 -> 0
    # i=0,j=1,k=1: u[1]=0, v[3]=1, w[1]=0 -> 0
    # i=1,j=0,k=0: u[2]=0, v[0]=1, w[2]=0 -> 0
    # i=1,j=0,k=1: u[2]=0, v[1]=0, w[3]=1 -> 0
    # i=1,j=1,k=0: u[3]=1, v[2]=0, w[2]=0 -> 0
    # i=1,j=1,k=1: u[3]=1, v[3]=1, w[3]=1 -> 1
    # Wait, M1 should contribute to C00 and C11.
    # C00 is i=0, k=0. Sum over j: j=0 (1*1*1=1), j=1 (0*0*1=0). Total 1. Correct.
    # C11 is i=1, k=1. Sum over j: j=0 (0*0*1=0), j=1 (1*1*1=1). Total 1. Correct.
    # Other C entries should be 0.
    # C01 (i=0, k=1): j=0 (1*0*0=0), j=1 (0*1*0=0). Total 0. Correct.
    # C10 (i=1, k=0): j=0 (0*1*0=0), j=1 (1*0*0=0). Total 0. Correct.
    
    # So the 7 Strassen components for a 2x2 block are valid.
    
    # Now, for 2x2x4, we have k in {0,1,2,3}.
    # We can apply these 7 patterns to the first block (k=0,1) and the second block (k=2,3).
    # This gives 14 candidates.
    # We can also mix them? The ILP will find the best combination.
    
    # Let's map the 2x2 patterns to the 2x4 context.
    # For block 0 (k=0,1):
    # v indices: j*4+k.
    # If j=0, k=0 -> 0. j=0, k=1 -> 1.
    # If j=1, k=0 -> 4. j=1, k=1 -> 5.
    # w indices: k*2+i.
    # If k=0, i=0 -> 0. k=0, i=1 -> 1.
    # If k=1, i=0 -> 2. k=1, i=1 -> 3.
    
    # For block 1 (k=2,3):
    # v indices: j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 6. j=1, k=3 -> 7.
    # w indices: k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.
    
    def expand_to_2x4(u_2x2, v_2x2, w_2x2, block_offset):
        u = np.zeros(4)
        v = np.zeros(8)
        w = np.zeros(8)
        
        for i in range(2):
            for j in range(2):
                u[i*2+j] = u_2x2[i*2+j]
                
        for j in range(2):
            for k_local in range(2):
                k_global = k_local + block_offset
                v[j*4+k_global] = v_2x2[j*2+k_local]
                
        for k_local in range(2):
            for i in range(2):
                k_global = k_local + block_offset
                w[k_global*2+i] = w_2x2[k_local*2+i]
                
        return u, v, w
    
    strassen_2x2 = [
        (s1_u, s1_v, s1_w),
        (s2_u, s2_v, s2_w),
        (s3_u, s3_v, s3_w),
        (s4_u, s4_v, s4_w),
        (s5_u, s5_v, s5_w),
        (s6_u, s6_v, s6_w),
        (s7_u, s7_v, s7_w)
    ]
    
    for offset in [0, 2]:
        for u, v, w in strassen_2x2:
            u_exp, v_exp, w_exp = expand_to_2x4(u, v, w, offset)
            candidates.append((u_exp, v_exp, w_exp))
            
    # Also add the trivial rank-8 components for each (i,j,k) to ensure feasibility.
    for i in range(2):
        for j in range(2):
            for k in range(4):
                u = np.zeros(4); u[i*2+j] = 1
                v = np.zeros(8); v[j*4+k] = 1
                w = np.zeros(8); w[k*2+i] = 1
                candidates.append((u, v, w))
                
    # Add some "cross" candidates? 
    # Maybe terms that span both blocks in V but are zero in W? No, W must match.
    # The ILP will combine them.
    
    return candidates

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Fallback: Trivial rank 16
    U_trivial = [[0] * (n*m*p) for _ in range(n*m)]
    V_trivial = [[0] * (n*m*p) for _ in range(m*p)]
    W_trivial = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j][col] = 1
                V_trivial[j*p+k][col] = 1
                W_trivial[k*n+i][col] = 1
                col += 1
                
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    best_rank = n*m*p
    
    try:
        candidates = _generate_candidate_components()
        num_candidates = len(candidates)
        
        # Build the linear system
        # Variables: x_c for each candidate c (binary 0 or 1)
        # Objective: min sum x_c
        # Constraints: For each (i,j,k), sum_c x_c * u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
        
        # There are n*m*p = 16 constraints.
        # There are num_candidates variables.
        
        A_matrix = np.zeros((n*m*p, num_candidates))
        b_vector = np.ones(n*m*p)
        
        for c_idx, (u, v, w) in enumerate(candidates):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        A_matrix[idx, c_idx] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                        
        # Solve ILP
        c_obj = np.ones(num_candidates)
        constraints = [LinearConstraint(A_matrix, b_vector, b_vector)]
        bounds = Bounds(0, 1)
        
        res = milp(c=c_obj, constraints=constraints, integrality=np.ones(num_candidates), bounds=bounds)
        
        if res.status == 0:
            selected = res.x.astype(int)
            rank_found = int(np.sum(selected))
            
            if rank_found < best_rank:
                U_new = np.zeros((n*m, rank_found))
                V_new = np.zeros((m*p, rank_found))
                W_new = np.zeros((p*n, rank_found))
                
                col = 0
                for c_idx in range(num_candidates):
                    if selected[c_idx] == 1:
                        u, v, w = candidates[c_idx]
                        U_new[:, col] = u
                        V_new[:, col] = v
                        W_new[:, col] = w
                        col += 1
                        
                # Verify
                # Reconstruct tensor
                tensor_recon = np.zeros((n, m, p))
                for r in range(rank_found):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                tensor_recon[i,j,k] += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                                
                if np.allclose(tensor_recon, 1.0):
                    best_rank = rank_found
                    # Convert to lists
                    best_result = {
                        "U": U_new.tolist(),
                        "V": V_new.tolist(),
                        "W": W_new.tolist()
                    }
                    
    except Exception as e:
        # If ILP fails, return trivial
        pass
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
