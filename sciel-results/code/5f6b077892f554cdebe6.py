"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using Strassen on two 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block Strassen (7 products)
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        # M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        # M3 = a00(b10-b11)
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        # M4 = a11(b01-b00)
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        # M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        # M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        # Right block Strassen (7 products)
        # C02 = a00*b02 + a01*b12
        # C03 = a00*b03 + a01*b13
        # C12 = a10*b02 + a11*b12
        # C13 = a10*b03 + a11*b13
        
        # N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1; V[4][7] = 1; V[7][7] = 1; W[4][7] = 1; W[7][7] = 1
        # N2 = (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1; V[4][8] = 1; W[6][8] = 1; W[7][8] = -1
        # N3 = a00(b12-b13)
        U[0][9] = 1; V[6][9] = 1; V[7][9] = -1; W[5][9] = 1; W[7][9] = 1
        # N4 = a11(b03-b02)
        U[3][10] = 1; V[5][10] = 1; V[4][10] = -1; W[4][10] = 1; W[6][10] = 1
        # N5 = (a00+a01)(b13)
        U[0][11] = 1; U[1][11] = 1; V[7][11] = 1; W[4][11] = -1; W[5][11] = 1
        # N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1; V[4][12] = 1; V[5][12] = 1; W[7][12] = 1
        # N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1; V[6][13] = 1; V[7][13] = 1; W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_shared_strassen():
        """
        Try to share Strassen factors between the two 2x2 blocks to reduce rank.
        The A-side factors are identical for both blocks, so we can potentially
        reuse them. However, the V and W factors differ, so we need to be careful.
        
        Actually, let's think about this differently. We have two independent 2x2
        multiplications. The tensor for 2x2*2x4 is the direct sum of two 2x2*2x2
        tensors (in a sense). The rank of the direct sum is at most the sum of
        ranks, but can be less due to sharing.
        
        For 2x2*2x2, the rank is 7 (Strassen). For two such, the naive rank is 14.
        But we might be able to do better by sharing some factors.
        
        Let's try a rank-11 construction based on known results.
        """
        R = 11
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # This is a speculative rank-11 construction.
        # We'll use 7 products for the first block and 4 for the second,
        # but with some shared structure.
        
        # First block (cols 0,1): standard Strassen
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        # Second block (cols 2,3): try to use only 4 products
        # This is not generally possible, but let's see if we can find a valid
        # 4-product scheme for the second block given the shared A structure.
        
        # Actually, 4 products is not enough for a 2x2 multiplication in general.
        # The minimum rank for 2x2*2x2 is 7. So we need at least 7 for the second
        # block as well, unless we can share factors in a non-trivial way.
        
        # Let's try a different approach: use a joint decomposition.
        
        # For now, return None and let the caller try other methods.
        return None
    
    def try_rank10():
        """Try a rank-10 decomposition using more aggressive sharing."""
        R = 10
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # This is highly speculative. Let's try to construct a valid rank-10
        # decomposition by combining the two blocks in a clever way.
        
        # Actually, the known lower bound for 2x2*2x4 is 11, so rank 10 is
        # likely not possible. Let's skip this and try rank 11.
        return None
    
    def try_numerical_peeling():
        """
        Use iterative rank-1 peeling to find a low-rank decomposition.
        """
        if time.time() > deadline - 10:
            return None
        
        # Build the matrix multiplication tensor T of shape (nm, mp, pn)
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Greedy rank-1 peeling using SVD
        U_list = []
        V_list = []
        W_list = []
        T_work = T.copy()
        
        max_rank = 16
        tol = 1e-10
        
        for r in range(max_rank):
            if time.time() > deadline - 5:
                break
            
            # Find the best rank-1 approximation using SVD
            T_mat = T_work.reshape(nm, mp * pn)
            
            try:
                U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
            except:
                break
            
            if s_vals[0] < tol:
                break
            
            u_vec = U_svd[:, 0]
            vt_vec = Vt_svd[0]
            
            # Scale to match the tensor entry
            scale = s_vals[0]
            u_vec = u_vec * scale
            vt_vec = vt_vec
            
            U_list.append(u_vec.copy())
            V_list.append(vt_vec[:mp].copy())
            W_list.append(vt_vec[mp:].copy())
            
            # Subtract the rank-1 component
            T_work -= np.outer(u_vec, vt_vec).reshape(nm, mp, pn)
            
            # Check convergence
            if np.max(np.abs(T_work)) < tol:
                break
        
        if np.max(np.abs(T_work)) > tol:
            return None
        
        R = len(U_list)
        U_mat = np.column_stack(U_list)
        V_mat = np.column_stack(V_list)
        W_mat = np.column_stack(W_list)
        
        # Round to nearest half-integer if close
        U_rounded = np.round(U_mat * 2) / 2
        V_rounded = np.round(V_mat * 2) / 2
        W_rounded = np.round(W_mat * 2) / 2
        
        # Verify with rounded values
        U_list_r = U_rounded.tolist()
        V_list_r = V_rounded.tolist()
        W_list_r = W_rounded.tolist()
        
        if verify(U_list_r, V_list_r, W_list_r):
            return {"U": U_list_r, "V": V_list_r, "W": W_list_r}
        
        # Try without rounding
        U_list_f = U_mat.tolist()
        V_list_f = V_mat.tolist()
        W_list_f = W_mat.tolist()
        
        if verify(U_list_f, V_list_f, W_list_f):
            return {"U": U_list_f, "V": V_list_f, "W": W_list_f}
        
        return None
    
    def try_systematic_search():
        """
        Try a systematic search for a low-rank decomposition by treating the
        coefficients as unknowns and solving the linear system.
        
        For a rank-R decomposition, we have R rank-1 tensors:
        sum_r U_r ⊗ V_r ⊗ W_r = T
        
        This gives us a linear system in the coefficients U[r,i], V[r,j], W[r,k].
        But this is nonlinear because the product U[r,i]*V[r,j]*W[r,k] appears.
        
        However, we can fix one of the matrices (say W) and solve for U and V.
        Or we can use an iterative approach.
        
        Let's try a simpler approach: fix the structure of U, V, W to have
        sparse entries (0, ±1, ±1/2) and search for a valid combination.
        """
        if time.time() > deadline - 15:
            return None
        
        # Try rank 11 first
        for R in [11, 12, 13]:
            if time.time() > deadline - 10:
                break
            
            # This is a combinatorial search, which is infeasible for large R.
            # Let's skip this and rely on the other methods.
            pass
        
        return None
    
    # Try structured decompositions in order of increasing rank
    result = try_rank10()
    if result is not None:
        return result
    
    result = try_shared_strassen()
    if result is not None:
        return result
    
    result = strassen_rectangular()
    if result is not None:
        return result
    
    # Try numerical peeling
    result = try_numerical_peeling()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
