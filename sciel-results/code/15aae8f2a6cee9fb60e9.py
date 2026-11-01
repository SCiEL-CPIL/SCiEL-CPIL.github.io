"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: exploit 6 = 3*2 by splitting into three 2x3x2 sub-problems.
    Each 2x3x2 sub-problem has rank 11 (vs schoolbook 12), giving total rank 33.
    Use a structured Strassen-style construction for the 2x3x2 block, then tile.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Build the full multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback (rank 36)
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def check_exact(U, V, W, T, tol=1e-8):
        recon = np.zeros(T.shape)
        rank = U.shape[1]
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol
    
    # Construct rank-11 decomposition for 2x3x2 sub-problem
    # 2x3x2: A is 2x3, B is 3x2, output C is 2x2
    # Indices: i in {0,1} (row), j in {0,1,2} (inner), k in {0,1} (col)
    # Schoolbook: 12 terms. We seek rank 11.
    # 
    # Construction via Strassen-style reduction:
    # The 2x3x2 tensor can be seen as computing 4 dot products of length-3 vectors.
    # We use the following 11 bilinear forms (each is product of a linear form in A
    # and a linear form in B):
    
    def build_2x3x2_rank11():
        """
        Build rank-11 decomposition for 2x3x2.
        Returns U (6, 11), V (6, 11), W (4, 11) for the sub-problem.
        Sub-problem tensor: T_sub[i, j, k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1}
        Decomposition: T_sub[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        """
        # For 2x3x2, we use a known rank-11 construction.
        # The 12 schoolbook terms: (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1}
        # We merge terms using Strassen's idea on the 2x2 output structure.
        #
        # Let A[i,j] for i in {0,1}, j in {0,1,2}
        # Let B[j,k] for j in {0,1,2}, k in {0,1}
        # C[i,k] = sum_j A[i,j] B[j,k]
        #
        # Strassen for 2x2 (treating each "entry" as a vector of length 3 in inner dim):
        # Actually, we define 7 intermediate bilinear forms M_r, each = (linear in A) * (linear in B)
        # where "linear in A" means sum over (i,j) of coef * A[i,j] and similarly for B.
        # Each M_r is a scalar. Then C[i,k] = sum_r w[i,k,r] * M_r.
        #
        # For the 2x3x2 case, the "entries" A[i,j] and B[j,k] are scalars (inner dim is the
        # summation index). So this is just 2x3 times 3x2 = 2x2, a standard bilinear map.
        #
        # The rank of the 2x3x2 matrix multiplication tensor is 11 (known).
        # Construction: use Strassen's 7 products for the 2x2 "outer" structure,
        # but since inner dim is 3, each Strassen product involves sums over j.
        
        # Let's define the 11 bilinear forms explicitly.
        # A = [[a00, a01, a02],
        #      [a10, a11, a12]]
        # B = [[b00, b01],
        #      [b10, b11],
        #      [b20, b21]]
        # C = [[c00, c01],
        #      [c10, c11]]
        # c00 = a00*b00 + a01*b10 + a02*b20
        # c01 = a00*b01 + a01*b11 + a02*b21
        # c10 = a10*b00 + a11*b10 + a12*b20
        # c11 = a10*b01 + a11*b11 + a12*b21
        
        # Strassen-style 7 products (each is a scalar bilinear form):
        # M1 = (a00+a12)*(b00+b21)  -- but this mixes different j indices
        # This doesn't work directly because the inner index j must match.
        
        # Correct approach: the bilinear form is sum_j A[i,j] B[j,k].
        # Each term A[i,j]*B[j,k] is a rank-1 term. We need to find 11 rank-1 terms
        # whose sum equals the tensor.
        
        # Known rank-11 decomposition for 2x3x2:
        # We use the following construction based on the fact that 2x3x2 has rank 11.
        # The 12 schoolbook terms can be reduced by 1 using a linear dependency.
        
        # Let's use ALS to find it, with a structured initialization.
        # For 2x3x2: U is (6, rank), V is (6, rank), W is (4, rank)
        # T_sub is (6, 6, 4)
        
        T_sub = np.zeros((6, 6, 4), dtype=np.float64)
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    a = i * 3 + j  # 0..5
                    b = j * 2 + k  # 0..5
                    c = k * 2 + i  # 0..3
                    T_sub[a, b, c] = 1.0
        
        T_sub_vec = T_sub.reshape(-1)
        
        def khatri_rao(A, B, size_a, size_b):
            rank = A.shape[1]
            KR = np.zeros((size_a * size_b, rank))
            for r in range(rank):
                KR[:, r] = np.kron(A[:, r], B[:, r])
            return KR
        
        def als_2x3x2(rank, n_init=50, max_iter=200, seed_base=0):
            for trial in range(n_init):
                if time.time() > deadline:
                    break
                rng = np.random.RandomState(seed_base + trial)
                U = rng.randn(6, rank)
                V = rng.randn(6, rank)
                W = rng.randn(4, rank)
                
                # Normalize
                for r in range(rank):
                    Un = np.linalg.norm(U[:, r])
                    Vn = np.linalg.norm(V[:, r])
                    Wn = np.linalg.norm(W[:, r])
                    if Un > 1e-12: U[:, r] /= Un
                    if Vn > 1e-12: V[:, r] /= Vn
                    if Wn > 1e-12: W[:, r] /= Wn
                
                prev_err = np.inf
                for iteration in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    VkrW = khatri_rao(V, W, 6, 4)
                    try:
                        U_new, _, _, _ = np.linalg.lstsq(VkrW, T_sub_vec, rcond=None)
                    except:
                        break
                    U = U_new.reshape(6, rank)
                    
                    WkrU = khatri_rao(W, U, 4, 6)
                    try:
                        V_new, _, _, _ = np.linalg.lstsq(WkrU, T_sub_vec, rcond=None)
                    except:
                        break
                    V = V_new.reshape(6, rank)
                    
                    UkrV = khatri_rao(U, V, 6, 6)
                    try:
                        W_new, _, _, _ = np.linalg.lstsq(UkrV, T_sub_vec, rcond=None)
                    except:
                        break
                    W = W_new.reshape(4, rank)
                    
                    recon = np.zeros((6, 6, 4))
                    for r in range(rank):
                        recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    err = np.max(np.abs(recon - T_sub))
                    
                    if err < 1e-10:
                        return U, V, W, True
                    
                    if abs(prev_err - err) < 1e-14 and iteration > 20:
                        break
                    prev_err = err
            
            # Check last result
            if U is not None:
                recon = np.zeros((6, 6, 4))
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                if np.max(np.abs(recon - T_sub)) < 1e-8:
                    return U, V, W, True
            return None, None, None, False
        
        # Try rank 11 for 2x3x2
        U_sub, V_sub, W_sub, found = als_2x3x2(11, n_init=30, max_iter=150, seed_base=42)
        if found:
            # Round to half-integers if possible
            U_r = np.round(U_sub * 2) / 2
            V_r = np.round(V_sub * 2) / 2
            W_r = np.round(W_sub * 2) / 2
            recon = np.zeros((6, 6, 4))
            for r in range(11):
                recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            if np.max(np.abs(recon - T_sub)) < 1e-8:
                return U_r, V_r, W_r
            return U_sub, V_sub, W_sub
        return None, None, None
    
    # Try to build the rank-33 decomposition
    U_sub, V_sub, W_sub = build_2x3x2_rank11()
    
    if U_sub is not None:
        # Tile the 2x3x2 decomposition 3 times for the 3 blocks of 2 columns each
        # Block t (t=0,1,2) handles columns k in {2t, 2t+1}
        rank_total = 33
        U_full = np.zeros((6, rank_total))
        V_full = np.zeros((18, rank_total))
        W_full = np.zeros((12, rank_total))
        
        for t in range(3):
            # Columns 2t, 2t+1
            col_offset = t * 11
            for r in range(11):
                # U: same for all blocks (rows of A)
                U_full[:, col_offset + r] = U_sub[:, r]
                
                # V: for block t, the B part is rows j*2+(2t or 2t+1)
                # In the sub-problem, V index b = j*2 + k where k in {0,1}
                # In full problem, B index = j*p + k = j*6 + (2t + k_sub)
                for j in range(3):
                    for k_sub in range(2):
                        b_sub = j * 2 + k_sub
                        b_full = j * 6 + (2 * t + k_sub)
                        V_full[b_full, col_offset + r] = V_sub[b_sub, r]
                
                # W: for block t, output columns are 2t, 2t+1
                # In sub-problem, W index c = k*2 + i where k in {0,1}, i in {0,1}
                # In full problem, C index = k_full*n + i = (2t+k_sub)*2 + i
                for i in range(2):
                    for k_sub in range(2):
                        c_sub = k_sub * 2 + i
                        c_full = (2 * t + k_sub) * 2 + i
                        W_full[c_full, col_offset + r] = W_sub[c_sub, r]
        
        # Verify
        if check_exact(U_full, V_full, W_full, T):
            # Round to half-integers
            U_r = np.round(U_full * 2) / 2
            V_r = np.round(V_full * 2) / 2
            W_r = np.round(W_full * 2) / 2
            if check_exact(U_r, V_r, W_r, T):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook
    U, V, W, rank = schoolbook_decomp()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
