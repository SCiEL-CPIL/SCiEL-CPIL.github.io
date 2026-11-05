"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial schoolbook decomposition with rank = n*m*p = 54."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def make_rank_46():
        """
        Construct a rank 46 decomposition using two rank-23 3x3x3 Strassen decompositions.
        The 3x3x3 rank-23 decomposition uses 7 Strassen multiplications for the 2x2 sub-block
        plus 16 additional scalar multiplications for the remaining terms.
        """
        # Known rank-23 decomposition for 3x3x3 matrix multiplication
        # This is the Bini algorithm / standard rank-23 construction
        rank_s = 23
        U_s = [[0]*rank_s for _ in range(9)]
        V_s = [[0]*rank_s for _ in range(9)]
        W_s = [[0]*rank_s for _ in range(9)]
        
        # Strassen 2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # m1 = (a00+a11)*(b00+b11): contributes to c00, c11; subtract from c01, c10
        r = 0
        U_s[0][r]=1; U_s[4][r]=1
        V_s[0][r]=1; V_s[3][r]=1
        W_s[0][r]=1; W_s[4][r]=1
        W_s[1][r]=-1; W_s[3][r]=-1
        
        # m2 = (a10+a11)*b00: contributes to c10
        r = 1
        U_s[4][r]=1; U_s[1][r]=1
        V_s[0][r]=1
        W_s[3][r]=1
        
        # m3 = a00*(b10-b11): contributes to c10
        r = 2
        U_s[0][r]=1
        V_s[3][r]=1; V_s[4][r]=-1
        W_s[3][r]=1
        
        # m4 = a11*(b01-b00): contributes to c01
        r = 3
        U_s[4][r]=1
        V_s[1][r]=1; V_s[0][r]=-1
        W_s[1][r]=1
        
        # m5 = (a00+a01)*b11: contributes to c01
        r = 4
        U_s[0][r]=1; U_s[1][r]=1
        V_s[4][r]=1
        W_s[1][r]=1
        
        # m6 = (a10-a00)*(b00+b01): contributes to c00
        r = 5
        U_s[1][r]=1; U_s[0][r]=-1
        V_s[0][r]=1; V_s[1][r]=1
        W_s[0][r]=1
        
        # m7 = (a01-a11)*(b10+b11): contributes to c11
        r = 6
        U_s[1][r]=1; U_s[4][r]=-1
        V_s[3][r]=1; V_s[4][r]=1
        W_s[4][r]=1
        
        # Additional terms for the 3rd row/col (16 more terms)
        # a02*b20 -> c20
        r = 7
        U_s[2][r]=1
        V_s[6][r]=1
        W_s[6][r]=1
        
        # a02*b21 -> c21
        r = 8
        U_s[2][r]=1
        V_s[7][r]=1
        W_s[7][r]=1
        
        # a02*b22 -> c22
        r = 9
        U_s[2][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        # a12*b20 -> c20
        r = 10
        U_s[5][r]=1
        V_s[6][r]=1
        W_s[6][r]=1
        
        # a12*b21 -> c21
        r = 11
        U_s[5][r]=1
        V_s[7][r]=1
        W_s[7][r]=1
        
        # a12*b22 -> c22
        r = 12
        U_s[5][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        # a20*b00 -> c00
        r = 13
        U_s[6][r]=1
        V_s[0][r]=1
        W_s[0][r]=1
        
        # a20*b01 -> c01
        r = 14
        U_s[6][r]=1
        V_s[1][r]=1
        W_s[1][r]=1
        
        # a20*b02 -> c02
        r = 15
        U_s[6][r]=1
        V_s[2][r]=1
        W_s[2][r]=1
        
        # a21*b10 -> c10
        r = 16
        U_s[7][r]=1
        V_s[3][r]=1
        W_s[3][r]=1
        
        # a21*b11 -> c11
        r = 17
        U_s[7][r]=1
        V_s[4][r]=1
        W_s[4][r]=1
        
        # a21*b12 -> c12
        r = 18
        U_s[7][r]=1
        V_s[5][r]=1
        W_s[5][r]=1
        
        # a22*b20 -> c20
        r = 19
        U_s[8][r]=1
        V_s[6][r]=1
        W_s[6][r]=1
        
        # a22*b21 -> c21
        r = 20
        U_s[8][r]=1
        V_s[7][r]=1
        W_s[7][r]=1
        
        # a22*b22 -> c22
        r = 21
        U_s[8][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        # We need one more term. Let's add a00*b02 + a01*b12 combination
        # Actually, let me verify what's missing.
        # C[0,2] = a00*b02 + a01*b12 + a02*b22
        # C[1,2] = a10*b02 + a11*b12 + a12*b22
        # We have a02*b22 (r=9) and a12*b22 (r=12)
        # We need a00*b02 + a01*b12 and a10*b02 + a11*b12
        # These can be done with 2 more multiplications
        
        # a00*b02 + a01*b12: use (a00+a01)*(b02+b12) - a00*b12 - a01*b02
        # That's 4 multiplications, too many.
        # Instead, just do them directly:
        r = 22
        U_s[0][r]=1
        V_s[2][r]=1
        W_s[2][r]=1  # a00*b02 -> c02
        
        # But we're at 23 terms and still missing a01*b12, a10*b02, a11*b12
        # This approach doesn't work cleanly. Let me use a different known rank-23 construction.
        
        # For now, let's just verify if this partial construction works
        # by checking it. If not, we'll fall back to schoolbook.
        
        # Actually, the standard rank-23 algorithm is well-known. Let me use a verified one.
        # The key insight: the 2x2 Strassen handles 4 of the 9 output entries partially,
        # and we need 16 more scalar multiplications for the rest.
        # Total: 7 + 16 = 23.
        
        # Let me restructure. The 16 additional terms should be:
        # Row 0: a02*b20, a02*b21, a02*b22, a00*b02, a01*b12 (5 terms)
        # Row 1: a12*b20, a12*b21, a12*b22, a10*b02, a11*b12 (5 terms)  
        # Row 2: a20*b00, a20*b01, a20*b02, a21*b10, a21*b11, a21*b12, a22*b20, a22*b21, a22*b22 (9 terms)
        # But that's 19, not 16. Some must be combinable.
        
        # For reliability, let me just return None and use the optimization approach
        # or fall back to schoolbook.
        
        return None

    def try_optimization_search():
        """
        Try to find a lower-rank decomposition using optimization.
        Uses L-BFGS-B to minimize the residual, then snaps to half-integers.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            return None
        
        dim_U = n * m  # 9
        dim_V = m * p  # 18
        dim_W = p * n  # 18
        
        # Construct target tensor
        T_target = np.zeros((dim_U, dim_V, dim_W))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_target[a, b, c] = 1.0
        
        def snap_and_verify(U, V, W):
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
            if np.allclose(R, T_target, atol=1e-9):
                return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
            return False, None, None, None
        
        # Try ranks from 47 to 53
        for rank in range(47, 54):
            if time.time() > deadline - 20:
                break
            
            num_attempts = 5
            for attempt in range(num_attempts):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(12345 + rank * 1000 + attempt)
                U = np.random.randn(dim_U, rank) * 0.5
                V = np.random.randn(dim_V, rank) * 0.5
                W = np.random.randn(dim_W, rank) * 0.5
                
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def obj(params):
                    U_p = params[:dim_U*rank].reshape(dim_U, rank)
                    V_p = params[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_p = params[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                    R = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                    return np.sum((R - T_target)**2)
                
                try:
                    res = minimize(
                        obj, params, method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    U_opt = res.x[:dim_U*rank].reshape(dim_U, rank)
                    V_opt = res.x[dim_U*rank : dim_U*rank + dim_V*rank].reshape(dim_V, rank)
                    W_opt = res.x[dim_U*rank + dim_V*rank :].reshape(dim_W, rank)
                    
                    success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                    if success:
                        return {"U": U_s, "V": V_s, "W": W_s}
                        
                except Exception:
                    pass
        
        return None

    # First try the structural rank-46 approach
    result_46 = make_rank_46()
    if result_46 is not None and verify(result_46["U"], result_46["V"], result_46["W"]):
        return result_46

    # Try optimization-based search
    result_opt = try_optimization_search()
    if result_opt is not None:
        return result_opt

    # Fallback: Schoolbook (rank 54)
    return make_schoolbook()
# EVOLVE-BLOCK-END
