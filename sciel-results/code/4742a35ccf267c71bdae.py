"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor using exact rational arithmetic."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1 if (i == ip and j == jp and k == kp) else 0
                            rhs = Fraction(0)
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if lhs != rhs:
                                return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification for screening."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor and compare
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = sum(U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r] for r in range(rank))
                            if abs(lhs - rhs) > tol:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 28
    
    # Build the tensor T[a, b, c] where:
    # a = i*m + j, b = jp*p + k, c = kp*n + ip
    # T[a, b, c] = 1 if i==ip and j==jp and k==kp else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                a = i*m + j
                                b = jp*p + k
                                c = kp*n + ip
                                T[a, b, c] = 1.0
    
    # Compute matricization ranks to get lower bounds
    T_mat1 = T.reshape(n*m, m*p * p*n)
    lb1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)
    
    T_mat2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    lb2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
    
    T_mat3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    lb3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)
    
    lower_bound = max(lb1, lb2, lb3)
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # STRATEGIC APPROACH: Use the structure of matrix multiplication
    # The tensor T[i,j; jp,kp; kp,ip] = delta(i,ip) * delta(j,jp) * delta(k,kp)
    # This is a product structure: T = T_2x2 ⊗ T_1x7 (in a sense)
    # 
    # For 2x2 * 2x7: we can think of it as 7 independent 2x2 * 1x1 multiplications
    # but with shared structure. Actually, the correct decomposition:
    # A (2x2) * B (2x7) = C (2x7)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor has rank at least max(4, 14, 14) from matricizations
    # But the actual minimum rank is what we need to find.
    
    # Key insight: Use Strassen-like decomposition for the 2x2 part
    # Strassen decomposes 2x2*2x2 into 7 rank-1 terms
    # For 2x2*2x7, we might be able to share these 7 terms across columns
    
    # Let's try a structured approach:
    # The multiplication A(2x2) * B(2x7) can be viewed as:
    # For each column k of B (which is a 2x1 vector), compute A * B[:,k]
    # This is 7 independent 2x2 * 2x1 multiplications
    # But they share the same A!
    
    # Actually, let's use the fact that the tensor is:
    # T[a,b,c] where a=(i,j), b=(jp,k), c=(kp,ip)
    # T = 1 iff i=ip, j=jp, k=kp
    # This factors as: T[i,j; jp,kp; kp,ip] = δ(i,ip)δ(j,jp)δ(k,kp)
    
    # The rank of this tensor: we need to find the minimum r such that
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # Let's try rank-1 peeling from the schoolbook decomposition
    # and also try structured decompositions
    
    np.random.seed(42)
    
    # Try to find a decomposition using rank-1 peeling
    # Start from schoolbook and try to merge terms
    
    # First, let's try a direct approach: assume a structure where
    # we use the same "Strassen-like" factors for the row/col mixing
    # and separate factors for the 7 columns
    
    # The 2x2 multiplication tensor has rank 7 (Strassen)
    # The 1x7 (or 2x7 with shared A) structure...
    
    # Let me try: can we do it with rank 7 * something?
    # Or can we share?
    
    # Actually, the correct lower bound analysis:
    # T_mat1 has shape (4, 14*14=196), rank = 4 (since for each (i,j) there are 14*14=196 entries but they're structured)
    # Wait, let me recompute. T[a,b,c] = 1 iff i=ip, j=jp, k=kp
    # For fixed a=(i,j), T[a,b,c] is nonzero only when b=(j,k) for some k, and c=(k,i) for some k
    # So for fixed a, the nonzero pattern is: b ranges over (j,0..6), c ranges over (0..6, i)
    # That's 7 * 7 = 49 nonzero entries, but they form a permutation-like structure
    
    # Let's just do the search properly with rank-1 peeling
    
    # Method: Start with the full tensor, repeatedly extract rank-1 components
    # and check if the remainder can be represented with fewer terms
    
    def try_rank_peel(T_tensor, max_rank, time_limit):
        """Try to decompose tensor into at most max_rank rank-1 terms via peeling."""
        T_work = T_tensor.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(max_rank):
            if time.time() > time_limit:
                return None
            
            # Find the best rank-1 approximation of T_work
            # Use SVD on matricizations to find candidate rank-1 terms
            
            # Matricize in different ways and use SVD
            best_score = -1
            best_u, best_v, best_w = None, None, None
            
            for mat_type in range(3):
                if mat_type == 0:
                    # Reshape to (n*m, m*p*p*n)
                    M = T_work.reshape(n*m, m*p*p*n)
                    try:
                        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
                    except:
                        continue
                    if S[0] < 1e-10:
                        continue
                    # First left singular vector
                    u = U_svd[:, 0]
                    # Corresponding right vector gives v*w combined
                    vw = Vt[0, :].reshape(m*p, p*n)
                    # Now find rank-1 approx of vw
                    try:
                        U2, S2, V2t = np.linalg.svd(vw, full_matrices=False)
                    except:
                        continue
                    if S2[0] < 1e-10:
                        continue
                    v = U2[:, 0]
                    w = V2t[0, :]
                    score = S[0] * S2[0]
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u, v, w
                        
                elif mat_type == 1:
                    # Reshape to (m*p, n*m*p*n)
                    M = T_work.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                    try:
                        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
                    except:
                        continue
                    if S[0] < 1e-10:
                        continue
                    v = U_svd[:, 0]
                    uw = Vt[0, :].reshape(n*m, p*n)
                    try:
                        U2, S2, V2t = np.linalg.svd(uw, full_matrices=False)
                    except:
                        continue
                    if S2[0] < 1e-10:
                        continue
                    u = U2[:, 0]
                    w = V2t[0, :]
                    score = S[0] * S2[0]
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u, v, w
                        
                else:
                    # Reshape to (p*n, n*m*m*p)
                    M = T_work.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                    try:
                        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
                    except:
                        continue
                    if S[0] < 1e-10:
                        continue
                    w = U_svd[:, 0]
                    uv = Vt[0, :].reshape(n*m, m*p)
                    try:
                        U2, S2, V2t = np.linalg.svd(uv, full_matrices=False)
                    except:
                        continue
                    if S2[0] < 1e-10:
                        continue
                    u = U2[:, 0]
                    v = V2t[0, :]
                    score = S[0] * S2[0]
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u, v, w
            
            if best_u is None:
                return None
            
            # Subtract this rank-1 term
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        T_work[a, b, c] -= best_u[a] * best_v[b] * best_w[c]
            
            U_list.append(best_u)
            V_list.append(best_v)
            W_list.append(best_w)
            
            # Check if remaining tensor is close to zero
            if np.max(np.abs(T_work)) < 1e-8:
                break
        
        # Build final matrices
        R = len(U_list)
        U_final = np.column_stack(U_list)
        V_final = np.column_stack(V_list)
        W_final = np.column_stack(W_list)
        
        # Verify
        if _verify_float(U_final, V_final, W_final, n, m, p):
            return U_final, V_final, W_final, R
        return None
    
    # Try rank-1 peeling for various target ranks
    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_peel(T, target_rank, deadline - 10)
        if result is not None:
            U_f, V_f, W_f, R = result
            if R < best_rank:
                best_U = U_f.tolist()
                best_V = V_f.tolist()
                best_W = W_f.tolist()
                best_rank = R
            break  # Found a valid decomposition, stop
    
    # Also try a structured approach: exploit the Kronecker product structure
    # The 2x2*2x7 multiplication can be seen as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j A[i,j] * B[j,k]
    # 
    # We can use Strassen's 7-term decomposition for the 2x2 part
    # and apply it to each column, but sharing the A-side factors
    
    # Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x7, B is 2x7, so B11, B12 etc. are 1x7 vectors
    # We can still use the same 7 intermediate products
    # But the "B" factors are now vectors, so the multiplication is just scaling
    
    # Let's construct this explicitly
    # A is 2x2: A[i,j] for i,j in {0,1}
    # B is 2x7: B[j,k] for j in {0,1}, k in {0..6}
    # C is 2x7: C[i,k] for i in {0,1}, k in {0..6}
    
    # Strassen intermediates (each is a 1x7 vector since B is 2x7):
    # M1 = (A00 + A11) * (B00 + B10)  -> 1x7 vector
    # M2 = (A10 + A11) * B00           -> 1x7 vector  
    # M3 = A00 * (B01 - B11)           -> 1x7 vector
    # M4 = A11 * (B10 - B00)           -> 1x7 vector
    # M5 = (A00 + A01) * B11           -> 1x7 vector
    # M6 = (A10 - A00) * (B00 + B01)   -> 1x7 vector
    # M7 = (A01 - A11) * (B10 + B11)   -> 1x7 vector
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mk is a rank-1 term: scalar(A-side) * vector(B-side)
    # The A-side scalar is linear in A entries
    # The B-side is a linear combination of B entries (specifically, one row of B scaled)
    
    # For the tensor decomposition:
    # Each Mk contributes: (A-coeff vector) ⊗ (B-coeff vector) ⊗ (C-coeff vector)
    # But since B is 2x7, the B-side for Mk is:
    #   For M1: (B00 + B10) means B-side has 1 at position (0,k) and 1 at position (1,k) for all k
    #   Actually, B[j,k] where j is row, k is column
    #   M1 = (A00+A11) * (B00 + B10)
    #   The B part: sum_k (B00[k] + B10[k]) * 1 = sum_k B[0,k] + sum_k B[1,k]
    #   Wait, B00 means B[0,0]... no, B is 2x7, so B[j,k] with j in {0,1}, k in {0..6}
    #   B00 in Strassen notation for 2x2*2x2 means top-left, but here B is 2x7
    #   Let me redefine: B has rows 0,1 and columns 0..6
    #   In Strassen for 2x2*2x7:
    #   M1 = (A[0,0] + A[1,1]) * (B[0,:] + B[1,:])  -- this is a 1x7 vector
    #   M2 = (A[1,0] + A[1,1]) * B[0,:]
    #   M3 = A[0,0] * (B[0,:] - B[1,:])
    #   M4 = A[1,1] * (B[1,:] - B[0,:])
    #   M5 = (A[0,0] + A[0,1]) * B[1,:]
    #   M6 = (A[1,0] - A[0,0]) * (B[0,:] + B[1,:])
    #   M7 = (A[0,1] - A[1,1]) * (B[1,:] + B[0,:])
    
    # C[0,:] = M1 + M4 - M5 + M7
    # C[1,:] = M2 + M4  ... wait, that's not right for 2x7
    
    # Actually for 2x2 * 2x7:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Using Strassen-like:
    # M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k]) for each k
    # M2 = (A[1,0]+A[1,1]) * B[0,k]
    # M3 = A[0,0] * (B[0,k]-B[1,k])
    # M4 = A[1,1] * (B[1,k]-B[0,k])
    # M5 = (A[0,0]+A[0,1]) * B[1,k]
    # M6 = (A[1,0]-A[0,0]) * (B[0,k]+B[1,k])
    # M7 = (A[0,1]-A[1,1]) * (B[1,k]+B[0,k])
    
    # C[0,k] = M1 + M4 - M5 + M7
    # C[1,k] = M2 + M4  ... let me verify
    # M1+M4-M5+M7 = (A00+A11)(B0k+B1k) + A11(B1k-B0k) - (A00+A01)B1k + (A01-A11)(B1k+B0k)
    # = A00*B0k + A00*B1k + A11*B0k + A11*B1k + A11*B1k - A11*B0k - A00*B1k - A01*B1k + A01*B1k + A01*B0k - A11*B1k - A11*B0k
    # = A00*B0k + A00*B1k + A11*B0k + A11*B1k + A11*B1k - A11*B0k - A00*B1k - A01*B1k + A01*B1k + A01*B0k - A11*B1k - A11*B0k
    # Let me collect:
    # A00*B0k: +1
    # A00*B1k: +1 -1 = 0
    # A11*B0k: +1 -1 -1 = -1
    # A11*B1k: +1 +1 -1 -1 = 0
    # A01*B1k: -1 +1 = 0
    # A01*B0k: +1
    # So we get: A00*B0k - A11*B0k + A01*B0k = (A00 - A11 + A01)*B0k ... that's not right
    
    # I made an error. Let me just verify the standard Strassen formulas work for this case.
    # The standard Strassen for 2x2*2x2 gives:
    # C00 = M1 + M4 - M5 + M7 where:
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11), M4=A11(B10-B00)
    # M5=(A00+A01)B11, M6=(A10-A00)(B00+B01), M7=(A01-A11)(B10+B11)
    # C00 = A00*B00 + A01*B10
    # Let me verify: M1+M4-M5+M7
    # = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10  ✓
    
    # For 2x2*2x7, each "element" of B is a 1x7 vector, and the arithmetic is the same
    # So the same 7-term decomposition works!
    # Each Mk is: (linear combo of A) * (linear combo of B-rows, applied element-wise to columns)
    # In tensor form, each Mk is a rank-1 term where:
    # - U part: coefficients on A[i,j]
    # - V part: coefficients on B[j,k] (which is 14-dimensional: j in {0,1}, k in {0..6})
    # - W part: coefficients on C[i,k] (which is 14-dimensional: i in {0,1}, k in {0..6})
    
    # For Mk = (sum_a alpha_a * A[a]) * (sum_b beta_b * B[b]) where B[b] is a row:
    # The V vector has beta_b at position (b, k) for all k
    # The W vector has 1 at position (i, k) for the output row i, for all k
    
    # Wait, but the output C[i,k] gets contributions from multiple Mk's
    # C[0,k] = M1[k] + M4[k] - M5[k] + M7[k]
    # So the W vector for each Mk has entries at (0,k) and/or (1,k)
    
    # Let me construct the 7 rank-1 terms:
    def make_strassen_based():
        """Construct decomposition using Strassen structure, rank = 7 * 7 = 49? No, 7 terms but each spans all 7 columns."""
        # Actually each Mk is ONE rank-1 term in the full tensor!
        # Because: Mk = (A-linear-combo) * (B-linear-combo) and it contributes to C as a fixed linear combo
        # The tensor entry T[a,b,c] for this term is: U[a] * V[b] * W[c]
        # where U is the A-coefficients, V is the B-coefficients (14-dim), W is the C-coefficients (14-dim)
        
        # For M1 = (A00+A11) * (B00+B10) contributing to C00 and C11:
        # U: A00=1, A11=1, others 0 -> [1,0,0,1] (for A[0,0],A[0,1],A[1,0],A[1,1])
        # V: B[0,k]=1 for all k, B[1,k]=1 for all k -> [1,1,1,1,1,1,1,1,1,1,1,1,1,1] (14 entries)
        # W: contributes to C[0,k] with +1 and C[1,k] with +1 -> [1,1,1,1,1,1,1,1,1,1,1,1,1,1] (14 entries)
        # But wait, M1 contributes +1 to C00 and +1 to C11 in the 2x2 case
        # In 2x2*2x7: M1 contributes to C[0,k] and C[1,k]
        # C[0,k] gets +M1[k], C[1,k] gets +M1[k]
        # So W has 1 at (0,k) for all k and 1 at (1,k) for all k
        
        # But the contribution is: U[a]*V[b]*W[c] summed over the term
        # For a=(0,0), b=(0,k), c=(0,k): U[0,0]*V[0,k]*W[0,k] = 1*1*1 = 1 ✓ (should be 1 for A00*B0k contributing to C0k)
        # For a=(0,0), b=(1,k), c=(0,k): U[0,0]*V[1,k]*W[0,k] = 1*1*1 = 1 ✓ (A00*B1k... but A00 should multiply B[0,k] for C[0,k], not B[1,k])
        
        # I see the issue. In Strassen, M1 = (A00+A11)(B00+B11) contributes to BOTH C00 and C11
        # C00 = A00*B00 + A01*B10, and M1 contributes A00*B00 + A00*B11 + A11*B00 + A11*B11
        # The extra terms cancel with other Mk's
        # So in the tensor, the rank-1 term for M1 is:
        # U[a] = coeff of A in (A00+A11): U[0,0]=1, U[1,1]=1
        # V[b] = coeff of B in (B00+B11): V[0,k]=1, V[1,k]=1 for all k
        # W[c] = coeff of C: W[0,k]=1, W[1,k]=1 for all k (since M1 adds to both C0 and C1)
        # But this gives T[a,b,c] += U[a]*V[b]*W[c] which for a=(0,0),b=(0,0),c=(0,0) gives 1*1*1=1
        # And for a=(0,0),b=(1,0),c=(0,0) gives 1*1*1=1, but the true tensor has T[(0,0),(1,0),(0,0)]=0
        # So we need the sum of all 7 terms to give the correct tensor, not each term individually
        
        # This is fine - we're summing 7 rank-1 terms to get the full tensor
        # Let me just construct all 7 and verify
        
        U_terms = []
        V_terms = []
        W_terms = []
        
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B indices: (j,k) -> j*7+k, so 0..6 = B[0,0..6], 7..13 = B[1,0..6]
        # C indices: (i,k) -> i*7+k, so 0..6 = C[0,0..6], 7..13 = C[1,0..6]
        
        # M1 = (A00+A11)*(B0+B1) -> contributes +1 to C0, +1 to C1
        U1 = [1, 0, 0, 1]
        V1 = [1]*7 + [1]*7
        W1 = [1]*7 + [1]*7
        U_terms.append(U1)
        V_terms.append(V1)
        W_terms.append(W1)
        
        # M2 = (A10+A11)*B0 -> contributes +1 to C1
        U2 = [0, 0, 1, 1]
        V2 = [1]*7 + [0]*7
        W2 = [0]*7 + [1]*7
        U_terms.append(U2)
        V_terms.append(V2)
        W_terms.append(W2)
        
        # M3 = A00*(B0-B1) -> contributes +1 to C0
        U3 = [1, 0, 0, 0]
        V3 = [1]*7 + [-1]*7
        W3 = [1]*7 + [0]*7
        U_terms.append(U3)
        V_terms.append(V3)
        W_terms.append(W3)
        
        # M4 = A11*(B1-B0) -> contributes +1 to C0, +1 to C1
        U4 = [0, 0, 0, 1]
        V4 = [-1]*7 + [1]*7
        W4 = [1]*7 + [1]*7
        U_terms.append(U4)
        V_terms.append(V4)
        W_terms.append(W4)
        
        # M5 = (A00+A01)*B1 -> contributes -1 to C0, +1 to C1
        U5 = [1, 1, 0, 0]
        V5 = [0]*7 + [1]*7
        W5 = [-1]*7 + [1]*7
        U_terms.append(U5)
        V_terms.append(V5)
        W_terms.append(W5)
        
        # M6 = (A10-A00)*(B0+B1) -> contributes +1 to C1
        U6 = [-1, 0, 1, 0]
        V6 = [1]*7 + [1]*7
        W6 = [0]*7 + [1]*7
        U_terms.append(U6)
        V_terms.append(V6)
        W_terms.append(W6)
        
        # M7 = (A01-A11)*(B1+B0) -> contributes +1 to C0
        U7 = [0, 1, 0, -1]
        V7 = [1]*7 + [1]*7
        W7 = [1]*7 + [0]*7
        U_terms.append(U7)
        V_terms.append(V7)
        W_terms.append(W7)
        
        U_mat = np.array(U_terms).T  # (4, 7)
        V_mat = np.array(V_terms).T  # (14, 7)
        W_mat = np.array(W_terms).T  # (14, 7)
        
        # Verify
        if _verify_float(U_mat, V_mat, W_mat, n, m, p):
            return U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), 7
        return None
    
    # Try Strassen-based decomposition
    strassen_result = make_strassen_based()
    if strassen_result is not None:
        U_s, V_s, W_s, R_s = strassen_result
        if R_s < best_rank:
            best_U, best_V, best_W, best_rank = U_s, V_s, W_s, R_s
    
    # Final verification
    if not _verify_float(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W, best_rank = make_schoolbook()
    
    return {"U": best_U, "V": best_V, "W": best_W}
