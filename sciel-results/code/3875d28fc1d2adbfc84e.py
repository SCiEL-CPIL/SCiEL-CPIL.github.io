<<<<<<< SEARCH
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p, r):
    """Verify exact decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k"""
    P = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    return np.allclose(P, 1.0, atol=1e-9)

def _construct_4x4x4_lander_bailey():
    """
    Lander-Bailey rank-19 decomposition for 4x4x4 matrix multiplication.
    Returns U (16,19), V (16,19), W (16,19) where:
      U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    such that sum_c U[i*4+j,c]*V[j*4+k,c]*W[k*4+i,c] = delta(i,j,k) for the
    matrix multiplication tensor (i.e., (AB)[i,k] = sum_j A[i,j]*B[j,k]).
    
    The matrix multiplication tensor T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j] B[j,k]
    Wait - let me reconsider. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    The tensor T has entries T[i,j,k] = 1 (all ones) because every triple (i,j,k) contributes.
    The decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    where W uses the trace convention W[k*n+i, c].
    """
    # For 4x4x4, n=m=p=4
    # The tensor T[i,j,k] = 1 for all i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1,2,3}
    # We need rank-19 tensors U, V, W with entries in {0,1,-1} (or half-integers)
    # such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Lander-Bailey scheme for 4x4x4:
    # Based on the known rank-19 scheme. Let me construct it using the standard approach.
    # 
    # The key insight: we can use a recursive/Strassen-like approach.
    # For 4x4x4, one known approach uses the 2x2x2 Strassen recursion.
    # 
    # Actually, let me use a direct construction. The rank-19 Lander-Bailey scheme
    # can be described as follows. We'll use the fact that 4x4x4 can be done
    # with 19 multiplications.
    #
    # Let me use a different approach: construct it via the known formulas.
    # 
    # For 4x4x4 matrix multiplication with rank 19:
    # The 19 terms are organized as follows (using 0-indexed):
    #
    # I'll use the approach of building from Strassen 2x2x2 blocks.
    # A 4x4 matrix can be partitioned into four 2x2 blocks.
    # Strassen's algorithm for 2x2x2 uses 7 multiplications.
    # For 4x4x4, a naive Strassen recursion would give 7^2 = 49, but
    # Lander-Bailey shows 19 is possible.
    #
    # Let me use an explicit known construction. I'll define the 19 rank-1 terms
    # directly.
    
    n = 4
    r = 19
    U = np.zeros((n*n, r), dtype=np.float64)
    V = np.zeros((n*n, r), dtype=np.float64)
    W = np.zeros((n*n, r), dtype=np.float64)
    
    # Lander-Bailey 4x4x4 rank-19 scheme
    # Using the standard enumeration:
    # The scheme uses the following pattern. Let me define it via the known
    # 19 multiplication terms.
    
    # I'll use a construction based on the following idea:
    # Partition indices into groups and use Strassen-like formulas.
    
    # Actually, let me use a well-known explicit rank-19 scheme.
    # Reference: Lander (1984), Bailey (1987)
    
    # The 19 terms for 4x4x4 matrix multiplication:
    # Term structure (U entries, V entries, W entries)
    # Using 0/1/-1 entries
    
    # I'll construct this using the known pattern. Let me use the following
    # approach based on the 2x2x2 Strassen base case extended.
    
    # Strassen 2x2x2: 7 terms
    # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)(e), M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)(h), M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # For 4x4x4, we can use a block structure. Let me use the known rank-19.
    # 
    # After research, the Lander-Bailey 4x4x4 rank-19 can be constructed as follows:
    # Use the 2x2x2 Strassen decomposition recursively but with shared terms.
    
    # Let me try a direct algebraic construction.
    # For 4x4x4, indices i,j,k in {0,1,2,3}.
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # This is equivalent to: for each (i,k), sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # which means (A*B)[i,k] = sum_j A[i,j] B[j,k] is computed correctly.
    
    # Let me use a known explicit scheme. I'll define the 19 terms.
    
    # Term 0-6: Strassen-like for the (0,0) block of the 4x4
    # Actually, let me use a different, more systematic approach.
    
    # I'll use the following known rank-19 scheme for 4x4x4:
    # (This is based on the work of Lander and Bailey)
    
    # Define helper: set U[i*4+j, c] = val
    def setU(i, j, c, val):
        U[i*4+j, c] = val
    def setV(j, k, c, val):
        V[j*4+k, c] = val
    def setW(k, i, c, val):
        W[k*4+i, c] = val
    
    # The 19 terms (using the standard Lander-Bailey enumeration):
    # I'll use a construction based on 2x2x2 Strassen blocks with sharing.
    
    # Block partition: indices {0,1} and {2,3}
    # 4x4 = [[A00, A01], [A10, A11]] where each is 2x2
    
    # For the 4x4x4 product, we have 16 output entries (i,k) for i,k in {0,1,2,3}
    # and 16 input pairs (i,j) and (j,k).
    
    # Let me use the following known rank-19 construction:
    
    # Terms 0-3: Diagonal-like terms
    for t in range(4):
        setU(t, t, t, 1)
        setV(t, t, t, 1)
        setW(t, t, t, 1)
    
    # This gives 4 terms for the diagonal.
    # Now we need 15 more terms for the off-diagonal.
    
    # Actually, this simple construction won't work. Let me use a proper scheme.
    
    # Let me use a different approach: I'll construct a valid rank-19 scheme
    # using the known formulas from the literature.
    
    # Reset
    U = np.zeros((16, 19), dtype=np.float64)
    V = np.zeros((16, 19), dtype=np.float64)
    W = np.zeros((16, 19), dtype=np.float64)
    
    # Lander-Bailey 4x4x4 rank-19 scheme (explicit):
    # Based on the following pattern (0-indexed, entries in {0, 1, -1}):
    
    # I'll use the construction from the known algorithm.
    # The 19 multiplication terms are:
    
    # Let me use a verified construction. I'll define it step by step.
    
    # Approach: Use the fact that 4x4x4 can be decomposed using
    # the 2x2x2 Strassen algorithm applied to blocks, with some sharing.
    
    # 4x4 matrix A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # where each Ai0, Ai1, etc. is 2x2.
    
    # C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    
    # Using Strassen on the 2x2 block structure:
    # M1 = (A00+A11)(B00+B11)  -- 7 mults per block pair, but we need to be careful
    # Actually for 4x4x4, the block multiplication gives:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each of these is a 2x2 = 2x2 product, which takes 7 mults via Strassen.
    # Total: 4 * 7 = 28, but with sharing we can do better.
    
    # Lander-Bailey shows 19 is possible. Let me use the explicit scheme.
    
    # After careful consideration, let me use a known explicit rank-19 scheme.
    # I'll define the 19 terms based on the following pattern:
    
    # The scheme uses the following 19 rank-1 tensors:
    # (U_term, V_term, W_term) where each is a 4x4x4 tensor with 0/1/-1 entries
    
    # I'll use the construction from:
    # "The minimum number of multiplications needed to multiply 4x4 matrices"
    
    # Let me use a different, more reliable approach.
    # I'll construct the scheme using the following known valid rank-19 decomposition.
    
    # Here's a known rank-19 scheme for 4x4x4 (verified):
    # The 19 terms are defined as follows (using 0-indexed rows/cols):
    
    # I'll use the following explicit construction based on Strassen's method
    # with optimal sharing:
    
    # Define the 19 terms. Each term c has:
    # U_c: 4x4 matrix (flattened to 16 entries)
    # V_c: 4x4 matrix (flattened to 16 entries)  
    # W_c: 4x4 matrix (flattened to 16 entries, with W[k*4+i, c] convention)
    
    # The product: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k
    
    # Let me use a verified construction. I'll use the following:
    
    # Based on the Lander-Bailey scheme, here are the 19 terms:
    # (I'll use a known valid set)
    
    # Term definitions (using the standard Lander-Bailey 4x4x4 rank-19):
    
    # I'll use the following approach: construct via the 2x2x2 Strassen
    # decomposition applied hierarchically with sharing.
    
    # For a 4x4x4 multiplication, partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    
    # Strassen for the block product:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_i is a 2x2 product, requiring 7 scalar multiplications.
    # Total: 7 * 7 = 49 without sharing.
    
    # With sharing between the 7 block-level terms, we can reduce to 19.
    
    # This is complex. Let me use a known explicit scheme instead.
    
    # I'll use the following verified rank-19 scheme:
    
    # After careful analysis, I'll use a construction based on the following:
    # The rank-19 scheme for 4x4x4 can be described using 19 multiplication
    # terms where each term is a rank-1 tensor.
    
    # Let me use a practical approach: I'll define the scheme using the
    # known formulas and verify it.
    
    # Using the Lander-Bailey scheme (one valid representation):
    
    # I'll define the 19 terms explicitly. Here's a known valid set:
    
    # (This is getting complex. Let me use a simpler but valid approach.)
    
    # Alternative: Use the fact that we can construct a rank-19 scheme
    # by combining Strassen 2x2x2 schemes with appropriate sharing.
    
    # For now, let me use a fallback: construct a valid decomposition
    # using a search approach, but seeded with a good initial guess.
    
    # Actually, let me use a known valid rank-19 scheme.
    # Here's one based on the literature:
    
    # I'll use the following 19 terms (verified to give rank 19 for 4x4x4):
    
    # Let me use a different strategy: I'll construct the 4x4x5 decomposition
    # directly using a search approach, but with a smarter initialization.
    
    # For the 4x4x5 case, the lower bound is at least ceil(4*4*5 / max(4,4,5)) = 16
    # and at most 4*4*5 = 80 (schoolbook).
    # The actual rank is likely around 40-50.
    
    # Let me use a more efficient search strategy.
    
    return None  # Fallback to search

def _search_decomposition(n, m, p, time_limit):
    """Search for a low-rank decomposition of the matrix multiplication tensor."""
    start_time = time.time()
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    # Decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    best_rank = n * m * p  # Schoolbook upper bound
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition as initial best
    def _schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, r
    
    U_s, V_s, W_s, r_s = _schoolbook()
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, r_s
    
    # Try to find better decompositions
    # Use a greedy/local search approach
    
    # Try ranks from a lower estimate up to the schoolbook rank
    # Lower bound: at least max(n*m, m*p, p*n) / something
    # For 4x4x5: lower bound is at least 4*5=20 (from p*n dimension)
    # Actually, the tensor rank lower bound for matrix multiplication M(n,m,p)
    # is at least max(n*m, m*p, p*n) / max(n,m,p) ... not quite.
    # A simple lower bound: rank >= max(n*m, m*p, p*n) / max(n,m,p)
    # For 4x4x5: max(16, 20, 20)/5 = 4. That's weak.
    # Better: rank >= (n*m*p) / (max entry count per term)
    # Each rank-1 term has at most n*m*p non-zero entries, but typically much fewer.
    # For {0,1,-1} entries, each term has at most n*m*p entries but many are zero.
    
    # Let's try ranks from 20 to 80
    for r in range(20, min(n*m*p + 1, 81)):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random restarts for this rank
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random {0, 1, -1} values
            U = np.random.choice([0, 1, -1], size=(n*m, r)).astype(np.float64)
            V = np.random.choice([0, 1, -1], size=(m*p, r)).astype(np.float64)
            W = np.random.choice([0, 1, -1], size=(p*n, r)).astype(np.float64)
            
            # Compute initial residual
            def _compute_residual(U, V, W):
                # P[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # Residual = 1 - P
                P = np.zeros((n, m, p), dtype=np.float64)
                for c in range(r):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    # P[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                P[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
                return 1.0 - P
            
            R = _compute_residual(U, V, W)
            current_norm = np.linalg.norm(R)
            
            if current_norm < 1e-9:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            
            # Local search: try to reduce residual
            for step in range(2000):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick which matrix to modify
                mat = np.random.randint(3)
                
                if mat == 0:
                    idx = (np.random.randint(n*m), np.random.randint(r))
                    old_val = U[idx]
                    new_val = np.random.choice([0, 1, -1])
                    if old_val == new_val:
                        continue
                    U[idx] = new_val
                    R_new = _compute_residual(U, V, W)
                    new_norm = np.linalg.norm(R_new)
                    if new_norm < current_norm - 1e-12:
                        current_norm = new_norm
                        R = R_new
                    else:
                        U[idx] = old_val
                elif mat == 1:
                    idx = (np.random.randint(m*p), np.random.randint(r))
                    old_val = V[idx]
                    new_val = np.random.choice([0, 1, -1])
                    if old_val == new_val:
                        continue
                    V[idx] = new_val
                    R_new = _compute_residual(U, V, W)
                    new_norm = np.linalg.norm(R_new)
                    if new_norm < current_norm - 1e-12:
                        current_norm = new_norm
                        R = R_new
                    else:
                        V[idx] = old_val
                else:
                    idx = (np.random.randint(p*n), np.random.randint(r))
                    old_val = W[idx]
                    new_val = np.random.choice([0, 1, -1])
                    if old_val == new_val:
                        continue
                    W[idx] = new_val
                    R_new = _compute_residual(U, V, W)
                    new_norm = np.linalg.norm(R_new)
                    if new_norm < current_norm - 1e-12:
                        current_norm = new_norm
                        R = R_new
                    else:
                        W[idx] = old_val
                
                if current_norm < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if best_rank == r:
                break
        
        if best_rank == r:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # 10s margin from 150s limit
    
    # Try to find a good decomposition
    U, V, W, rank = _search_decomposition(n, m, p, time_limit)
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
=======
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _compute_P(U, V, W, n, m, p, r):
    """Compute P[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    # Reshape for einsum
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)

def _schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, r

def _try_rank(n, m, p, r, max_restarts, time_limit, start_time):
    """Try to find a rank-r decomposition. Returns (U, V, W) or (None, None, None)."""
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            return None, None, None
        
        # Initialize with sparse random {0, 1, -1}
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Each entry is 0 with prob 0.5, else 1 or -1
        for _ in range(n*m*r):
            if np.random.random() < 0.5:
                U.flat[_] = np.random.choice([1.0, -1.0])
        for _ in range(m*p*r):
            if np.random.random() < 0.5:
                V.flat[_] = np.random.choice([1.0, -1.0])
        for _ in range(p*n*r):
            if np.random.random() < 0.5:
                W.flat[_] = np.random.choice([1.0, -1.0])
        
        P = _compute_P(U, V, W, n, m, p, r)
        R = 1.0 - P
        current_norm = np.linalg.norm(R)
        
        if current_norm < 1e-9:
            return U, V, W
        
        # Local search
        for step in range(3000):
            if time.time() - start_time > time_limit:
                return None, None, None
            
            mat = np.random.randint(3)
            
            if mat == 0:
                row = np.random.randint(n*m)
                col = np.random.randint(r)
                old = U[row, col]
                new = np.random.choice([0.0, 1.0, -1.0])
                if old == new:
                    continue
                U[row, col] = new
                P_new = _compute_P(U, V, W, n, m, p, r)
                R_new = 1.0 - P_new
                new_norm = np.linalg.norm(R_new)
                if new_norm < current_norm - 1e-12:
                    current_norm = new_norm
                    R = R_new
                    P = P_new
                else:
                    U[row, col] = old
            elif mat == 1:
                row = np.random.randint(m*p)
                col = np.random.randint(r)
                old = V[row, col]
                new = np.random.choice([0.0, 1.0, -1.0])
                if old == new:
                    continue
                V[row, col] = new
                P_new = _compute_P(U, V, W, n, m, p, r)
                R_new = 1.0 - P_new
                new_norm = np.linalg.norm(R_new)
                if new_norm < current_norm - 1e-12:
                    current_norm = new_norm
                    R = R_new
                    P = P_new
                else:
                    V[row, col] = old
            else:
                row = np.random.randint(p*n)
                col = np.random.randint(r)
                old = W[row, col]
                new = np.random.choice([0.0, 1.0, -1.0])
                if old == new:
                    continue
                W[row, col] = new
                P_new = _compute_P(U, V, W, n, m, p, r)
                R_new = 1.0 - P_new
                new_norm = np.linalg.norm(R_new)
                if new_norm < current_norm - 1e-12:
                    current_norm = new_norm
                    R = R_new
                    P = P_new
                else:
                    W[row, col] = old
            
            if current_norm < 1e-9:
                return U, V, W
    
    return None, None, None

def _try_rank_with_linear_solve(n, m, p, r, max_restarts, time_limit, start_time):
    """
    Try to find rank-r decomposition by fixing U and V, then solving for W linearly.
    For fixed U and V, the constraint is:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    This is linear in W. We can set up a linear system.
    """
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            return None, None, None
        
        # Initialize U and V with sparse random {0, 1, -1}
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        for _ in range(n*m*r):
            if np.random.random() < 0.4:
                U.flat[_] = np.random.choice([1.0, -1.0])
        for _ in range(m*p*r):
            if np.random.random() < 0.4:
                V.flat[_] = np.random.choice([1.0, -1.0])
        
        # For each (i,j,k), the equation is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is a system of n*m*p equations in p*n*r unknowns (W entries).
        # But W has structure: W[k*n+i, c] for k in 0..p-1, i in 0..n-1, c in 0..r-1.
        
        # Let's set up the linear system.
        # Variables: W[k*n+i, c] for all k, i, c -> total p*n*r variables
        # Equations: for each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Build the system A * w_flat = b
        # w_flat is a vector of length p*n*r
        # A has n*m*p rows and p*n*r columns
        
        # This might be too large. Let's check: n*m*p = 80, p*n*r = 20*r
        # For r=40: 80 x 800 system. That's manageable.
        
        # Build A
        num_eq = n * m * p
        num_var = p * n * r
        A = np.zeros((num_eq, num_var))
        b = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            # W[k*n+i, c] is at position (k*n+i)*r + c in w_flat
                            var_idx = (k*n+i)*r + c
                            A[eq_idx, var_idx] = coeff
                    eq_idx += 1
        
        # Solve A * w = b (least squares if overdetermined)
        try:
            w_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is close to integer/half-integer
        w_flat = np.round(w_flat * 2) / 2  # Round to nearest half-integer
        
        # Reconstruct W
        W = w_flat.reshape(p*n, r)
        
        # Verify
        P = _compute_P(U, V, W, n, m, p, r)
        if np.allclose(P, 1.0, atol=1e-9):
            return U, V, W
        
        # If not exact, try local search to fix
        R = 1.0 - P
        current_norm = np.linalg.norm(R)
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                return None, None, None
            
            # Only modify W to keep U and V fixed
            row = np.random.randint(p*n)
            col = np.random.randint(r)
            old = W[row, col]
            new = np.random.choice([0.0, 0.5, 1.0, -0.5, -1.0])
            if abs(old - new) < 1e-12:
                continue
            W[row, col] = new
            P_new = _compute_P(U, V, W, n, m, p, r)
            R_new = 1.0 - P_new
            new_norm = np.linalg.norm(R_new)
            if new_norm < current_norm - 1e-12:
                current_norm = new_norm
                R = R_new
                P = P_new
            else:
                W[row, col] = old
            
            if current_norm < 1e-9:
                return U, V, W
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # 10s margin from 150s limit
    
    # Start with schoolbook as fallback
    U_best, V_best, W_best, rank_best = _schoolbook(n, m, p)
    
    # Try to find better decompositions
    # Try ranks from 20 upward
    for r in range(20, rank_best + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try linear solve approach (more structured)
        U, V, W = _try_rank_with_linear_solve(n, m, p, r, max_restarts=20, 
                                                  time_limit=time_limit, 
                                                  start_time=start_time)
        if U is not None:
            U_best, V_best, W_best, rank_best = U, V, W, r
            break
        
        # Try random local search
        U, V, W = _try_rank(n, m, p, r, max_restarts=30, 
                              time_limit=time_limit, 
                              start_time=start_time)
        if U is not None:
            U_best, V_best, W_best, rank_best = U, V, W, r
            break
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
