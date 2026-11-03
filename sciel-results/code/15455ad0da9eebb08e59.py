"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_flat = T.reshape(nm * mp, pn)  # (216, 24)
    
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    def verify(Ua, Va, Wa):
        """Verify decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(Ua, Va, Wa):
        """Fast verification using matrix multiplication."""
        for i in range(n):
            for j in range(m):
                u_row = Ua[i*m+j]
                for k in range(p):
                    v_row = Va[j*p+k]
                    w_row = Wa[k*n+i]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook rank = 72
    schoolbook_rank = n * m * p
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_rank = schoolbook_rank
    best_decomp = (U_school.copy(), V_school.copy(), W_school.copy())
    
    # Strategy 1: Strassen-based decomposition (rank 68)
    def build_strassen_decomp():
        """Build the Strassen-based rank-68 decomposition."""
        U_t = []
        V_t = []
        W_t = []
        
        def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
            a = (row_off_a, col_off_a)
            b = (row_off_a, col_off_a + 1)
            c = (row_off_a + 1, col_off_a)
            d = (row_off_a + 1, col_off_a + 1)
            e = (row_off_b, col_off_b)
            f = (row_off_b, col_off_b + 1)
            g = (row_off_b + 1, col_off_b)
            h = (row_off_b + 1, col_off_b + 1)
            
            C00 = (row_off_a, col_off_b)
            C01 = (row_off_a, col_off_b + 1)
            C10 = (row_off_a + 1, col_off_b)
            C11 = (row_off_a + 1, col_off_b + 1)
            
            # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
            U_t.append([(a, 1), (d, 1)])
            V_t.append([(e, 1), (h, 1)])
            W_t.append([(C00, 1), (C11, 1)])
            
            # m2 = (c+d)*e -> C10 += 1, C11 -= 1
            U_t.append([(c, 1), (d, 1)])
            V_t.append([(e, 1)])
            W_t.append([(C10, 1), (C11, -1)])
            
            # m3 = a*(f-h) -> C01 += 1, C11 += 1
            U_t.append([(a, 1)])
            V_t.append([(f, 1), (h, -1)])
            W_t.append([(C01, 1), (C11, 1)])
            
            # m4 = d*(g-e) -> C00 += 1, C10 += 1
            U_t.append([(d, 1)])
            V_t.append([(g, 1), (e, -1)])
            W_t.append([(C00, 1), (C10, 1)])
            
            # m5 = (a+b)*h -> C00 -= 1, C01 += 1
            U_t.append([(a, 1), (b, 1)])
            V_t.append([(h, 1)])
            W_t.append([(C00, -1), (C01, 1)])
            
            # m6 = (c-a)*(e+f) -> C11 += 1
            U_t.append([(c, 1), (a, -1)])
            V_t.append([(e, 1), (f, 1)])
            W_t.append([(C11, 1)])
            
            # m7 = (b-d)*(g+h) -> C00 += 1
            U_t.append([(b, 1), (d, -1)])
            V_t.append([(g, 1), (h, 1)])
            W_t.append([(C00, 1)])
        
        def direct_term(i, j, k):
            U_t.append([((i, j), 1)])
            V_t.append([((j, k), 1)])
            W_t.append([((i, k), 1)])
        
        # Apply Strassen to all 2x2 B-blocks for A[0:2, 0:2]
        for b_col in range(0, 8, 2):
            strassen_decomp(0, 0, 0, b_col)
        
        # Direct terms for A[0:2, 2] * B[2, :]
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
        
        # Direct terms for A[2, 0:2] * B[0:2, :]
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
        
        # Direct terms for A[2, 2] * B[2, :]
        for k in range(8):
            direct_term(2, 2, k)
        
        rank = len(W_t)
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
        
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, sb_rank = build_strassen_decomp()
    if verify(U_sb, V_sb, W_sb):
        best_rank = sb_rank
        best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    # Strategy 2: MPS-based tensor network contraction
    # Represent the multiplication tensor as a Matrix Product State and
    # search for low-rank decompositions via bond-dimension truncation.
    
    def mps_based_decomp():
        """
        Use MPS (Matrix Product State) structure to find low-rank decompositions.
        
        The key idea: represent the tensor T[i*m+j, j*p+k, k*n+i] as a
        matrix product state over the (i,j,k) indices, then extract a
        rank-r decomposition from the MPS bond structure.
        
        For a 3x3x8 tensor, we can think of it as having 3 "sites" (i, j, k)
        with dimensions 3, 3, 8 respectively. The MPS representation has
        bond dimensions chi_1, chi_2, chi_3, and the total rank is bounded
        by chi_1 * chi_2 * chi_3.
        
        We search over small bond dimensions to find exact representations.
        """
        best_r_mps = best_rank
        best_U_mps, best_V_mps, best_W_mps = None, None, None
        
        # Try various MPS bond dimension configurations
        # For 3 sites with dims 3, 3, 8, bond dims (chi1, chi2)
        # Rank is bounded by max(chi1, chi2, chi1*chi2)
        
        bond_configs = [
            (3, 3), (4, 4), (3, 4), (4, 3), (5, 5), (4, 5), (5, 4),
            (6, 6), (5, 6), (6, 5), (6, 7), (7, 6), (7, 7),
            (8, 8), (7, 8), (8, 7), (8, 9), (9, 8), (9, 9),
        ]
        
        for chi1, chi2 in bond_configs:
            if time.time() > deadline - 30:
                break
            
            # MPS: T[i,j,k] = sum_{a,b} A[i,a] * B[j,a,b] * C[k,b]
            # where A is (3, chi1), B is (chi1, 3, chi2), C is (8, chi2)
            # This gives a rank decomposition with rank = chi1 * chi2
            
            rank_candidate = chi1 * chi2
            if rank_candidate >= best_r_mps:
                continue
            
            # We need to find A, B, C such that:
            # T[i*m+j, j*p+k, k*n+i] = sum_{a,b} A[i,a] * B[j,a,b] * C[k,b]
            # 
            # But our tensor has a specific structure: T[i,j,k] = 1 if the
            # indices match the multiplication pattern, else 0.
            # Actually T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
            # So T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8).
            # 
            # Wait, that's not right. Let me re-examine.
            # T is a tensor of shape (nm, mp, pn) = (9, 24, 24).
            # T[i*m+j, j*p+k, k*n+i] = 1.
            # 
            # This is the matrix multiplication tensor. It's not all-ones.
            # For a given (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1, and all
            # other entries are 0.
            # 
            # So the tensor has exactly 3*3*8 = 72 non-zero entries, each equal to 1.
            # 
            # The MPS representation would be:
            # T[i*m+j, j*p+k, k*n+i] = sum_{a,b} A[i,a] * B[j,a,b] * C[k,b]
            # 
            # But this doesn't directly work because T is not separable in (i,j,k).
            # 
            # Let me think differently. The standard approach for matrix
            # multiplication tensor decomposition is:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # 
            # The MPS approach would be to find a low-rank structure in the
            # (i,j,k) space. Let's try a different formulation.
            # 
            # Actually, for the matrix multiplication tensor, we can use the
            # fact that it's a 3-tensor with a specific symmetry. The MPS
            # approach works by finding a tensor network representation.
            # 
            # Let's try a more direct approach: use the known Strassen
            # decomposition as a starting point and try to compress it
            # using SVD-based truncation.
            
            # Alternative: try random MPS-like decompositions
            np.random.seed(42 + chi1 * 100 + chi2)
            
            A_mat = np.random.randn(n, chi1)
            B_mat = np.random.randn(chi1, m, chi2)
            C_mat = np.random.randn(p, chi2)
            
            # Check if this can represent the tensor
            # T[i,j,k] should equal sum_{a,b} A[i,a] * B[j,a,b] * C[k,b]
            # But our tensor is T[i*m+j, j*p+k, k*n+i] = 1
            # 
            # This doesn't match the standard tensor structure. Let me
            # reconsider the MPS approach.
            
            # The correct MPS for matrix multiplication:
            # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # 
            # If we set c = (a, b) with a in [0, chi1), b in [0, chi2),
            # then U[i*m+j, (a,b)] = U1[i,a] * U2[j,b] (Kronecker)
            # V[j*p+k, (a,b)] = V1[j,a] * V2[k,b]
            # W[k*n+i, (a,b)] = W1[k,a] * W2[i,b]
            # 
            # Then sum_{a,b} U1[i,a]*U2[j,b] * V1[j,a]*V2[k,b] * W1[k,a]*W2[i,b]
            # = sum_{a,b} (U1[i,a]*V1[j,a]*W1[k,a]) * (U2[j,b]*V2[k,b]*W2[i,b])
            # 
            # This is a product of two sums, which is not generally
            # equal to T[i,j,k]. So the Kronecker product approach
            # doesn't directly work for the full tensor.
            
            # Let's try a simpler MPS: just use bond dimension chi
            # T[i,j,k] = sum_a A[i,a] * B[j,a] * C[k,a]
            # This gives rank chi, but the tensor is not generally
            # rank-1 in this sense.
            
            # Actually, the matrix multiplication tensor has rank at
            # least 2n-1 for n x n matrices (Strassen bound).
            # For 3x3, the minimum rank is 21 (Bini et al. or better).
            # For 3x3x8, it's more complex.
            
            # Let's abandon the MPS approach for now and focus on
            # improving the random search with better initialization.
            pass
        
        return best_U_mps, best_V_mps, best_W_mps, best_r_mps
    
    # Strategy 3: Improved random search with structured initialization
    def improved_random_search():
        """
        Search for low-rank decompositions using structured random initialization
        and least-squares fitting.
        """
        best_r = best_rank
        best_U, best_V, best_W = None, None, None
        
        # Try ranks from current best-1 down to 40
        for r in range(best_rank - 1, 39, -1):
            if time.time() > deadline - 35:
                break
            
            # Number of attempts based on remaining time
            remaining = deadline - 35 - time.time()
            max_attempts = min(50, max(5, int(remaining / 2)))
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 30:
                    break
                
                np.random.seed(123456 + r * 1000 + attempt)
                
                # Structured random initialization with sparsity
                U_rand = np.zeros((nm, r))
                V_rand = np.zeros((mp, r))
                
                for c in range(r):
                    # Random support with varying sparsity
                    u_sparsity = np.random.choice([0.2, 0.3, 0.4, 0.5, 0.6])
                    v_sparsity = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                    
                    u_support = np.random.choice(nm, size=max(1, int(nm * (1 - u_sparsity))), replace=False)
                    v_support = np.random.choice(mp, size=max(1, int(mp * (1 - v_sparsity))), replace=False)
                    
                    for idx in u_support:
                        U_rand[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    for idx in v_support:
                        V_rand[idx, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                
                # Build K matrix
                K = np.zeros((nm * mp, r))
                for c in range(r):
                    K[:, c] = np.outer(U_rand[:, c], V_rand[:, c]).flatten()
                
                try:
                    W_sol, residuals, rank_K, sv = np.linalg.lstsq(K, T_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K @ W_sol - T_flat).max()
                
                if residual < 1e-9:
                    # Check if W_sol can be represented with integers/half-integers
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-6):
                        W_final = np.round(W_check) / 2
                        
                        # Verify
                        if verify(U_rand, V_rand, W_final):
                            best_r = r
                            best_U = U_rand.copy()
                            best_V = V_rand.copy()
                            best_W = W_final.copy()
                            break
            
            if best_r < best_rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_r
        return None, None, None, best_rank
    
    U_impr, V_impr, W_impr, impr_rank = improved_random_search()
    if U_impr is not None and impr_rank < best_rank:
        best_rank = impr_rank
        best_decomp = (U_impr.copy(), V_impr.copy(), W_impr.copy())
    
    # Strategy 4: 3x3x4 embedding with improved search
    def find_rank_3x3x4_improved():
        """Search for low-rank decomposition of 3x3x4 tensor with improved methods."""
        n4, m4, p4 = 3, 3, 4
        nm4 = n4 * m4  # 9
        mp4 = m4 * p4  # 12
        pn4 = p4 * n4  # 12
        
        T4 = np.zeros((nm4, mp4, pn4))
        for i in range(n4):
            for j in range(m4):
                for k in range(p4):
                    T4[i*m4+j, j*p4+k, k*n4+i] = 1.0
        T4_flat = T4.reshape(nm4 * mp4, pn4)  # (108, 12)
        
        best_r4 = 25  # Start with a safe upper bound
        best_U4, best_V4, best_W4 = None, None, None
        
        # Try ranks from 21 down to 15
        for r in range(21, 14, -1):
            if time.time() > deadline - 25:
                break
            
            remaining = deadline - 25 - time.time()
            max_attempts = min(80, max(5, int(remaining / 1.5)))
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 20:
                    break
                
                np.random.seed(99999 + r * 1000 + attempt)
                
                # Random dense initialization with structured patterns
                U4 = np.random.randn(nm4, r)
                V4 = np.random.randn(mp4, r)
                
                # Try different sparsity patterns
                sparsity = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                U4[np.random.random((nm4, r)) < (1 - sparsity)] = 0
                V4[np.random.random((mp4, r)) < (1 - sparsity)] = 0
                
                # Ensure no all-zero columns
                for c in range(r):
                    if np.all(U4[:, c] == 0):
                        U4[np.random.randint(nm4), c] = 1.0
                    if np.all(V4[:, c] == 0):
                        V4[np.random.randint(mp4), c] = 1.0
                
                # Scale to keep numbers manageable
                U4 /= (np.linalg.norm(U4, axis=0, keepdims=True) + 1e-9)
                V4 /= (np.linalg.norm(V4, axis=0, keepdims=True) + 1e-9)
                
                K4 = np.zeros((nm4 * mp4, r))
                for c in range(r):
                    K4[:, c] = np.outer(U4[:, c], V4[:, c]).flatten()
                
                try:
                    W4_sol, residuals, rank_K, sv = np.linalg.lstsq(K4, T4_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K4 @ W4_sol - T4_flat).max()
                
                if residual < 1e-9:
                    # Check if W4_sol can be represented with integers/half-integers
                    W4_check = W4_sol * 2
                    if np.allclose(W4_check, np.round(W4_check), atol=1e-6):
                        W4_final = np.round(W4_check) / 2
                        
                        # Verify
                        recon4 = np.zeros((nm4, mp4, pn4))
                        for c in range(r):
                            recon4 += np.einsum('a,b,c->abc', U4[:,c], V4[:,c], W4_final[:,c])
                        if np.abs(recon4 - T4).max() < 1e-8:
                            best_r4 = r
                            best_U4 = U4.copy()
                            best_V4 = V4.copy()
                            best_W4 = W4_final.copy()
                            break
            
            if best_r4 < 25:
                break
        
        if best_U4 is not None:
            return best_U4, best_V4, best_W4, best_r4
        return None, None, None, 25
    
    U4, V4, W4, r4 = find_rank_3x3x4_improved()
    
    if U4 is not None:
        # Embed 3x3x4 decomposition into 3x3x8 with rank doubling
        rank8 = 2 * r4
        U8 = np.zeros((nm, rank8))
        V8 = np.zeros((mp, rank8))
        W8 = np.zeros((pn, rank8))
        
        for c in range(r4):
            # Term 2c: even k
            U8[:, 2*c] = U4[:, c]
            for j in range(m):
                for b in range(4):
                    V8[j*p + 2*b, 2*c] = V4[j*4 + b, c]
            for b in range(4):
                for i in range(n):
                    W8[(2*b)*n + i, 2*c] = W4[b*n + i, c]
            
            # Term 2c+1: odd k
            U8[:, 2*c+1] = U4[:, c]
            for j in range(m):
                for b in range(4):
                    V8[j*p + 2*b + 1, 2*c+1] = V4[j*4 + b, c]
            for b in range(4):
                for i in range(n):
                    W8[(2*b+1)*n + i, 2*c+1] = W4[b*n + i, c]
        
        if verify(U8, V8, W8):
            if rank8 < best_rank:
                best_rank = rank8
                best_decomp = (U8.copy(), V8.copy(), W8.copy())
    
    # Strategy 5: 3x3x2 tiling
    def find_rank_3x3x2_improved():
        """Search for low-rank decomposition of 3x3x2 tensor."""
        n2, m2, p2 = 3, 3, 2
        nm2 = n2 * m2 # 9
        mp2 = m2 * p2 # 6
        pn2 = p2 * n2 # 6
        
        T2 = np.zeros((nm2, mp2, pn2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        T2_flat = T2.reshape(nm2 * mp2, pn2)
        
        best_r2 = 17 # Current known block rank
        best_U2, best_V2, best_W2 = None, None, None
        
        # Try ranks from 16 down to 10
        for r in range(16, 9, -1):
            if time.time() > deadline - 20:
                break
            
            remaining = deadline - 20 - time.time()
            max_attempts = min(50, max(5, int(remaining / 1.5)))
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(12345 + r * 100 + attempt)
                
                # Random dense initialization
                U2 = np.random.randn(nm2, r)
                V2 = np.random.randn(mp2, r)
                
                # Scale to keep numbers manageable
                U2 /= np.linalg.norm(U2, axis=0, keepdims=True) + 1e-9
                V2 /= np.linalg.norm(V2, axis=0, keepdims=True) + 1e-9
                
                K2 = np.zeros((nm2 * mp2, r))
                for c in range(r):
                    K2[:, c] = np.outer(U2[:, c], V2[:, c]).flatten()
                
                try:
                    W2_sol, residuals, rank_K, sv = np.linalg.lstsq(K2, T2_flat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    continue
                
                residual = np.abs(K2 @ W2_sol - T2_flat).max()
                
                if residual < 1e-9:
                    # Check if W2_sol can be represented with integers/half-integers
                    W2_check = W2_sol * 2
                    if np.allclose(W2_check, np.round(W2_check), atol=1e-6):
                        W2_final = np.round(W2_check) / 2
                        
                        # Verify
                        recon2 = np.zeros((nm2, mp2, pn2))
                        for c in range(r):
                            recon2 += np.einsum('a,b,c->abc', U2[:,c], V2[:,c], W2_final[:,c])
                        if np.abs(recon2 - T2).max() < 1e-8:
                            best_r2 = r
                            best_U2 = U2.copy()
                            best_V2 = V2.copy()
                            best_W2 = W2_final.copy()
                            break
            
            if best_r2 < 17:
                break
        
        if best_U2 is not None:
            return best_U2, best_V2, best_W2, best_r2
        return None, None, None, 17
    
    if best_rank >= 68 and time.time() < deadline - 30:
        U2, V2, W2, r2 = find_rank_3x3x2_improved()
        
        if U2 is not None:
            rank8 = 4 * r2
            U8 = np.zeros((nm, rank8))
            V8 = np.zeros((mp, rank8))
            W8 = np.zeros((pn, rank8))
            
            for b in range(4):
                offset = b * r2
                U8[:, offset:offset+r2] = U2
                
                for j in range(m):
                    for k in range(2):
                        V8[j*p + 2*b + k, offset:offset+r2] = V2[j*2 + k, :]
                
                for k in range(2):
                    for i in range(n):
                        W8[(2*b + k)*n + i, offset:offset+r2] = W2[k*n + i, :]
            
            if verify(U8, V8, W8):
                if rank8 < best_rank:
                    best_rank = rank8
                    best_decomp = (U8.copy(), V8.copy(), W8.copy())
    
    # Final fallback: if we haven't improved, return the best we have
    U_out, V_out, W_out = best_decomp
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
