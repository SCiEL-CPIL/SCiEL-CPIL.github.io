import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Target Tensor T[i, j, k] = 1 if (i, j, k) is a valid matmul index, else 0.
    Indexing:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    """
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    
    # Compute tensor contraction
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    # We only care about the specific indices (i, j, k) that form the matmul tensor.
    # The matmul tensor has 1s at (i, j, k) for all i in [0,n), j in [0,m), k in [0,p).
    # And 0s everywhere else.
    
    # Optimization: Instead of computing the full tensor (which is huge), 
    # we check the specific constraints.
    # Constraint 1: For all (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Constraint 2: For all (a, b, c) NOT in the support, sum_r U[a, r] * V[b, r] * W[c, r] == 0
    
    # This is hard to check directly without constructing the tensor.
    # Let's construct the tensor T of shape (n*m, m*p, p*n).
    # Size: 16 * 36 * 36 = 20736. Small enough.
    
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    
    # Check support
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                if abs(T[idx_u, idx_v, idx_w] - 1.0) > 1e-9:
                    return False
    
    # Check zeros
    # Create a mask of valid indices
    valid_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                valid_mask[idx_u, idx_v, idx_w] = True
                
    # Zero out valid indices and check if rest is 0
    T_check = T.copy()
    T_check[valid_mask] = 0.0
    
    if np.max(np.abs(T_check)) > 1e-9:
        return False
        
    return True

def _generate_dictionary(n, m, p, max_sparsity=2):
    """
    Generates a dictionary of candidate rank-1 tensors.
    Each candidate is a tuple (u_pattern, v_pattern, w_pattern)
    where patterns are sparse vectors.
    We use integer values {-1, 1} to keep it simple, but allow 0.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    dictionary = []
    
    # To avoid explosion, we restrict patterns to be "localized" or specific structures.
    # A standard rank-1 term in matmul is u=e_i, v=e_j, w=e_k.
    # Strassen terms involve sums.
    # Let's generate all patterns with up to `max_sparsity` non-zeros from {-1, 1}.
    # This is still huge if dims are large.
    # dim_u=16, dim_v=36, dim_w=36.
    # Number of vectors with <=2 non-zeros in dim 16: 1 + 16*2 + C(16,2)*4 = 1 + 32 + 120*4 = 513.
    # Number of vectors with <=2 non-zeros in dim 36: 1 + 72 + C(36,2)*4 = 1 + 72 + 630*4 = 2593.
    # Total combinations: 513 * 2593 * 2593 ~ 3.4 billion. Too many.
    
    # Strategy: Use a "block" structure or random sampling.
    # Let's sample a manageable subset.
    # We will generate:
    # 1. All "standard" basis vectors (sparsity 1).
    # 2. A random set of "Strassen-like" patterns (sparsity 2-4).
    
    # 1. Basis vectors
    for i in range(dim_u):
        u = np.zeros(dim_u); u[i] = 1
        for j in range(dim_v):
            v = np.zeros(dim_v); v[j] = 1
            for k in range(dim_w):
                w = np.zeros(dim_w); w[k] = 1
                dictionary.append((u, v, w))
                
    # 2. Random sparse patterns
    # We need patterns that might cancel out or add up to form the matmul tensor.
    # Let's generate 5000 random patterns with sparsity 2-4.
    rng = np.random.default_rng(42)
    for _ in range(5000):
        sp_u = rng.integers(1, max_sparsity+1)
        sp_v = rng.integers(1, max_sparsity+1)
        sp_w = rng.integers(1, max_sparsity+1)
        
        u_idx = rng.choice(dim_u, size=sp_u, replace=False)
        v_idx = rng.choice(dim_v, size=sp_v, replace=False)
        w_idx = rng.choice(dim_w, size=sp_w, replace=False)
        
        u = np.zeros(dim_u)
        v = np.zeros(dim_v)
        w = np.zeros(dim_w)
        
        for idx in u_idx: u[idx] = rng.choice([-1, 1])
        for idx in v_idx: v[idx] = rng.choice([-1, 1])
        for idx in w_idx: w[idx] = rng.choice([-1, 1])
        
        dictionary.append((u, v, w))
        
    return dictionary

def _greedy_pursuit(target_tensor, dictionary, n, m, p, max_iter=1000, time_limit=100.0):
    """
    Greedy pursuit to find a linear combination of dictionary atoms that matches the target tensor.
    """
    start_time = time.time()
    
    # Precompute dictionary tensors? 
    # T_dict[i] = u_i * v_i * w_i (outer product)
    # This is expensive. Let's compute on the fly or store compactly.
    # Since we only check against the target tensor, and the target is sparse (mostly 0),
    # we can represent the dictionary atoms as their contribution to the target's support and the "error" elsewhere.
    
    # Actually, let's just store the full tensor for the dictionary?
    # 5000+16*36*36 atoms. Each is 16*36*36 floats. 
    # 3.4M atoms * 20736 floats * 8 bytes = ~500 GB. Impossible.
    
    # We must compute the inner product <atom, current_error> efficiently.
    # atom = u * v * w.
    # <atom, E> = sum_{a,b,c} u[a]v[b]w[c] E[a,b,c]
    # This can be computed as:
    # temp1 = E * u[:, None, None] -> (dim_u, dim_v, dim_w) * (dim_u, 1, 1)
    # This is still large.
    
    # Alternative: Keep E in a compact form?
    # E is the difference between Target and CurrentApprox.
    # Target is fixed. CurrentApprox is sum of selected atoms.
    
    # Let's restrict the dictionary to be much smaller and "smarter".
    # The standard matmul tensor is a sum of 144 rank-1 terms: e_i * e_j * e_k.
    # We want to replace groups of these with fewer terms.
    
    # Let's use a specific set of "Strassen-like" patterns for 4x4x4 blocks, 
    # and extend to 4x4x9.
    
    # This greedy approach on a huge random dictionary is likely to fail or be too slow.
    # Let's pivot the strategy slightly: 
    # **Compressed Sensing on the Coefficients of a Fixed Basis**
    
    # Instead of a huge dictionary, let's assume a structure.
    # We know Strassen works for 4x4x4.
    # Let's try to find a decomposition that uses the Strassen structure but allows 
    # the coefficients to be optimized (integer/half-integer) and potentially 
    # adds a few extra terms to handle the 9 columns efficiently.
    
    # Actually, the prompt asks for a DIFFERENT strategy.
    # Let's try **Integer Programming / Local Search on a Small Candidate Set**.
    
    # Candidate Set: 
    # 1. The 144 standard terms.
    # 2. The 7 Strassen terms for each 4x4x4 block (14 blocks total? No, 4x4x4 is one block).
    #    For 4x4x9, we have four 4x4x4 blocks and one 4x4x1 block.
    #    Block 0: cols 0-3
    #    Block 1: cols 4-7
    #    Block 2: cols 8 (4x4x1)
    
    # Let's construct a candidate pool of ~200 specific rank-1 tensors.
    # Then use a local search (hill climbing) to find a subset + coefficients that works.
    
    pass

def _build_candidate_pool(n, m, p):
    """
    Builds a pool of candidate rank-1 tensors.
    Includes standard basis and Strassen-like patterns for 4x4x4 blocks.
    """
    pool = []
    
    # 1. Standard Basis Terms (144)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                pool.append((u, v, w))
                
    # 2. Strassen Terms for 4x4x4 blocks
    # Strassen 4x4x4 has rank 7? No, standard Strassen is 7 for 2x2x2.
    # For 4x4x4, we can use 4 blocks of 2x2x2 Strassen? 
    # Rank would be 4 * 7 = 28 for the 4x4x4 part.
    # Plus 4x4x1 part.
    
    # Let's define the 2x2x2 Strassen patterns and embed them.
    # U_sums, V_sums, W_sums for 2x2x2 (indices 0..3 for 2x2)
    # Standard Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to map these to 4x4x4.
    # Let's just add a few random "dense" patterns to the pool to allow mixing.
    
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        
        # Sparse random
        for _ in range(rng.integers(1, 4)):
            idx = rng.integers(0, n*m)
            u[idx] = rng.choice([-1, 1])
        for _ in range(rng.integers(1, 4)):
            idx = rng.integers(0, m*p)
            v[idx] = rng.choice([-1, 1])
        for _ in range(rng.integers(1, 4)):
            idx = rng.integers(0, p*n)
            w[idx] = rng.choice([-1, 1])
            
        pool.append((u, v, w))
        
    return pool

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][c] = 1
                V_school[j * p + k][c] = 1
                W_school[k * n + i][c] = 1
                c += 1
                
    best_res = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Strategy: Local Search / Greedy Addition
    # Start with empty decomposition.
    # Add terms from a candidate pool to minimize error.
    # This is essentially finding a sparse solution to a linear system.
    
    # Target Tensor T
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Candidate Pool
    pool = _build_candidate_pool(n, m, p)
    
    # Precompute pool tensors? No, too big.
    # Compute contribution on the fly.
    
    current_U = []
    current_V = []
    current_W = []
    current_T = np.zeros_like(T_target)
    
    # Greedy: At each step, find the atom in the pool that best reduces the error.
    # Error E = T_target - current_T
    # We want to find atom A and coeff c (from {-1, 1}) such that ||E - c*A|| is minimized.
    # This is equivalent to maximizing |<E, A>|.
    
    max_steps = 500
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        best_atom_idx = -1
        best_dot = 0
        best_sign = 1
        
        # Sample a subset of the pool to speed up search?
        # Or iterate through all?
        # Pool size ~ 250. 
        # Computing <E, A> for each A.
        # E is 16x36x36. A is u*v*w.
        # <E, A> = sum E[a,b,c] u[a] v[b] w[c]
        # We can precompute E_u = sum_{b,c} E[:,b,c] v[b] w[c] ? No, v,w depend on A.
        
        # Just compute directly. 250 atoms * 20736 ops = 5M ops per step.
        # 500 steps * 5M = 2.5B ops. Might be slow in Python.
        # Let's vectorize the pool evaluation?
        
        # Reshape pool to arrays
        # U_pool: (N, dim_u)
        # V_pool: (N, dim_v)
        # W_pool: (N, dim_w)
        
        if step == 0:
            U_pool = np.array([a[0] for a in pool])
            V_pool = np.array([a[1] for a in pool])
            W_pool = np.array([a[2] for a in pool])
            
        # Compute dot products for all atoms at once?
        # dot[i] = sum_{a,b,c} E[a,b,c] U_pool[i,a] V_pool[i,b] W_pool[i,c]
        # This is a tensor contraction.
        # We can do:
        # 1. Contract E with W_pool: Temp1[a, b, i] = sum_c E[a,b,c] W_pool[i,c]
        #    E: (du, dv, dw), W_pool: (N, dw) -> Temp1: (du, dv, N)
        # 2. Contract Temp1 with V_pool: Temp2[a, i] = sum_b Temp1[a,b,i] V_pool[i,b]
        #    Temp1: (du, dv, N), V_pool: (N, dv) -> Temp2: (du, N)
        # 3. Contract Temp2 with U_pool: Dots[i] = sum_a Temp2[a,i] U_pool[i,a]
        #    Temp2: (du, N), U_pool: (N, du) -> Dots: (N)
        
        # This is efficient!
        
        # Step 1: E @ W_pool.T
        # E is (du, dv, dw). W_pool is (N, dw).
        # Temp1 = np.einsum('abc,ic->abi', E, W_pool)
        
        Temp1 = np.einsum('abc,ic->abi', current_T_error := (T_target - current_T), W_pool)
        
        # Step 2: Temp1 @ V_pool.T (careful with indices)
        # Temp1[a, b, i]. V_pool[i, b].
        # Temp2[a, i] = sum_b Temp1[a, b, i] V_pool[i, b]
        Temp2 = np.einsum('abi,ib->ai', Temp1, V_pool)
        
        # Step 3: Temp2 @ U_pool.T
        # Temp2[a, i]. U_pool[i, a].
        # Dots[i] = sum_a Temp2[a, i] U_pool[i, a]
        Dots = np.einsum('ai,ia->i', Temp2, U_pool)
        
        # Find max absolute dot
        max_idx = np.argmax(np.abs(Dots))
        max_val = Dots[max_idx]
        
        if abs(max_val) < 1e-9:
            break
            
        # Add this atom with sign matching the dot product to reduce error
        sign = 1 if max_val > 0 else -1
        
        current_U.append(pool[max_idx][0].tolist())
        current_V.append(pool[max_idx][1].tolist())
        current_W.append(pool[max_idx][2].tolist())
        
        # Update current_T
        # current_T += sign * outer(u, v, w)
        u = pool[max_idx][0]
        v = pool[max_idx][1]
        w = pool[max_idx][2]
        current_T += sign * np.einsum('a,b,c->abc', u, v, w)
        
        # Check if we are close to target
        if np.max(np.abs(T_target - current_T)) < 1e-9:
            # We found an exact solution!
            # Convert to list of lists
            U_final = np.array(current_U).T
            V_final = np.array(current_V).T
            W_final = np.array(current_W).T
            
            # Ensure integer/half-integer
            # The entries are sums of +/-1, so they are integers.
            
            res = {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
            
            if _verify(res["U"], res["V"], res["W"], n, m, p):
                new_rank = len(res["U"][0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_res = res
            break
            
    # If greedy didn't finish or didn't find exact, try a simpler fallback:
    # The Strassen-based construction from the seed is a good fallback.
    # But we want a DIFFERENT strategy. 
    # Let's stick with the result of the greedy search if it's valid, 
    # otherwise return the schoolbook (which is guaranteed valid).
    
    # Actually, the greedy search might find a rank < 144 solution if the pool is good enough.
    # Given the pool is small, it might not. 
    # But the structure is different.
    
    # To ensure we don't return an invalid result, we verify the best_res.
    if not _verify(best_res["U"], best_res["V"], best_res["W"], n, m, p):
        # Re-verify schoolbook
        if _verify(U_school, V_school, W_school, n, m, p):
            best_res = {"U": U_school, "V": V_school, "W": W_school}
            
    return best_res

# To make the greedy search more likely to succeed, we should include 
# "Strassen-like" terms in the pool more systematically.
# But the code above is a complete implementation of the new strategy.

# Let's refine the pool to include the 4x4x4 Strassen terms explicitly 
# to give the greedy search a better chance of finding a low-rank solution.

def _build_enhanced_pool(n, m, p):
    pool = _build_candidate_pool(n, m, p)
    
    # Add 4x4x4 Strassen terms for each 4x4x4 block
    # Strassen 4x4x4 can be built from 2x2x2 Strassen blocks.
    # We'll add the 7 terms for the top-left 2x2x2, top-right 2x2x2, etc.
    
    # Define 2x2x2 Strassen patterns
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B21-B22) -> C12 += M3
    # M4 = A22(B12-B11) -> C22 += M4 (Wait, C22 += M1 - M4? No, C22 = M1 - M4 + M5? 
    # Standard Strassen:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M7
    
    # Let's just add the 7 M terms as atoms.
    # For a 2x2x2 block at (i0, j0, k0):
    
    def add_strassen_2x2(i0, j0, k0):
        # Indices in 4x4x4
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        
        # M1: (A11+A22)(B11+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1; w[(k0+1)*n+(i0+1)] = 1
        pool.append((u, v, w))
        
        # M2: (A21+A22)B11
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m+j0] = 1; u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+k0] = 1
        w[(k0+1)*n+i0] = 1
        pool.append((u, v, w))
        
        # M3: A11(B21-B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1
        v[(j0+1)*p+k0] = 1; v[(j0+1)*p+(k0+1)] = -1
        w[k0*n+(i0+1)] = 1
        pool.append((u, v, w))
        
        # M4: A22(B12-B11)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m+(j0+1)] = 1
        v[j0*p+(k0+1)] = 1; v[j0*p+k0] = -1
        w[(k0+1)*n+(i0+1)] = 1
        pool.append((u, v, w))
        
        # M5: (A11+A12)B22
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+j0] = 1; u[i0*m+(j0+1)] = 1
        v[(j0+1)*p+(k0+1)] = 1
        w[(k0+1)*n+i0] = 1
        pool.append((u, v, w))
        
        # M6: (A21-A11)(B11+B12)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[(i0+1)*m+j0] = 1; u[i0*m+j0] = -1
        v[j0*p+k0] = 1; v[j0*p+(k0+1)] = 1
        w[(k0+1)*n+(i0+1)] = 1
        pool.append((u, v, w))
        
        # M7: (A12-A22)(B21+B22)
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        u[i0*m+(j0+1)] = 1; u[(i0+1)*m+(j0+1)] = -1
        v[(j0+1)*p+k0] = 1; v[(j0+1)*p+(k0+1)] = 1
        w[k0*n+i0] = 1
        pool.append((u, v, w))

    # Add Strassen terms for 2x2 blocks in 4x4x4
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                add_strassen_2x2(i0, j0, k0)
                
    # Also add for the 4x4x1 part? 
    # 4x4x1 is just matrix-vector. Rank 16 is schoolbook.
    # Strassen doesn't help much for 1 column unless we use 2x1 blocks.
    
    return pool

# Update construct to use enhanced pool
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][c] = 1
                V_school[j * p + k][c] = 1
                W_school[k * n + i][c] = 1
                c += 1
                
    best_res = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Target Tensor T
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Candidate Pool
    pool = _build_enhanced_pool(n, m, p)
    
    current_U = []
    current_V = []
    current_W = []
    current_T = np.zeros_like(T_target)
    
    U_pool = np.array([a[0] for a in pool])
    V_pool = np.array([a[1] for a in pool])
    W_pool = np.array([a[2] for a in pool])
    
    max_steps = 200
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        E = T_target - current_T
        
        # Compute dot products
        Temp1 = np.einsum('abc,ic->abi', E, W_pool)
        Temp2 = np.einsum('abi,ib->ai', Temp1, V_pool)
        Dots = np.einsum('ai,ia->i', Temp2, U_pool)
        
        max_idx = np.argmax(np.abs(Dots))
        max_val = Dots[max_idx]
        
        if abs(max_val) < 1e-9:
            break
            
        sign = 1 if max_val > 0 else -1
        
        current_U.append(pool[max_idx][0].tolist())
        current_V.append(pool[max_idx][1].tolist())
        current_W.append(pool[max_idx][2].tolist())
        
        u = pool[max_idx][0]
        v = pool[max_idx][1]
        w = pool[max_idx][2]
        current_T += sign * np.einsum('a,b,c->abc', u, v, w)
        
        if np.max(np.abs(T_target - current_T)) < 1e-9:
            U_final = np.array(current_U).T
            V_final = np.array(current_V).T
            W_final = np.array(current_W).T
            
            res = {
                "U": U_final.tolist(),
                "V": V_final.tolist(),
                "W": W_final.tolist()
            }
            
            if _verify(res["U"], res["V"], res["W"], n, m, p):
                new_rank = len(res["U"][0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_res = res
            break
            
    if not _verify(best_res["U"], best_res["V"], best_res["W"], n, m, p):
        if _verify(U_school, V_school, W_school, n, m, p):
            best_res = {"U": U_school, "V": V_school, "W": W_school}
            
    return best_res
