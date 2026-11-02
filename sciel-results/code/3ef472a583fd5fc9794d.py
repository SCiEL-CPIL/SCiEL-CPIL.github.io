"""Program for 4x4x4 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication using
    the tensor square of Strassen's 2x2x2 algorithm.
    
    The 4x4x4 multiplication tensor is the tensor product of two 2x2x2
    multiplication tensors. Since Strassen's algorithm gives a rank-7
    decomposition for 2x2x2, the tensor square gives rank 7*7 = 49 for 4x4x4.
    
    We then attempt to find a linear dependence among the 49 terms to reduce
    to rank 48, but if none exists we return the valid rank-49 decomposition.
    """
    n, m, p = 4, 4, 4
    
    # Strassen's 7 rank-1 terms for 2x2x2 matrix multiplication.
    # Each term is (U, V, W) where U, V, W are 2x2 matrices.
    # The identity: for all 2x2 A, B:
    #   (AB)[i,k] = sum_r U[r][i,j] * A[i,j] * V[r][j,k] * B[j,k] * W[r][i,k]
    # summed appropriately. More precisely, the bilinear form is:
    #   C[i,k] = sum_r W[r][i,k] * (sum_{i',j'} U[r][i',j'] * A[i',j']) * (sum_{j',k'} V[r][j',k'] * B[j',k'])
    
    def strassen_2x2():
        """Return list of 7 (U, V, W) tuples, each a 2x2 numpy array."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        U = np.array([[1,0],[0,1]], dtype=float); V = np.array([[1,0],[0,1]], dtype=float); W = np.array([[1,0],[0,1]], dtype=float)
        terms.append((U,V,W))
        # M2 = (A10+A11)B00
        U = np.array([[0,0],[1,1]], dtype=float); V = np.array([[1,0],[0,0]], dtype=float); W = np.array([[0,0],[1,0]], dtype=float)
        terms.append((U,V,W))
        # M3 = A00(B01-B11)
        U = np.array([[1,0],[0,0]], dtype=float); V = np.array([[0,1],[0,-1]], dtype=float); W = np.array([[0,1],[0,0]], dtype=float)
        terms.append((U,V,W))
        # M4 = A11(B10-B00)
        U = np.array([[0,0],[0,1]], dtype=float); V = np.array([[-1,0],[1,0]], dtype=float); W = np.array([[1,0],[0,1]], dtype=float)
        terms.append((U,V,W))
        # M5 = (A00+A10)B11
        U = np.array([[1,0],[1,0]], dtype=float); V = np.array([[0,0],[0,1]], dtype=float); W = np.array([[-1,0],[0,-1]], dtype=float)
        terms.append((U,V,W))
        # M6 = (A01-A10)(B00+B10)
        U = np.array([[0,1],[-1,0]], dtype=float); V = np.array([[1,0],[1,0]], dtype=float); W = np.array([[0,0],[0,1]], dtype=float)
        terms.append((U,V,W))
        # M7 = (A01-A11)(B11+B01)
        U = np.array([[0,1],[0,-1]], dtype=float); V = np.array([[0,1],[1,1]], dtype=float); W = np.array([[1,0],[0,0]], dtype=float)
        terms.append((U,V,W))
        return terms
    
    s2x2 = strassen_2x2()
    num_terms = len(s2x2)  # 7
    
    # Construct the 49-term decomposition for 4x4x4.
    # Index convention: 
    #   A is 4x4, indexed as A[i,j] where i,j in {0,1,2,3}
    #   Block decomposition: i = 2*i_hi + i_lo, j = 2*j_hi + j_lo
    #   where i_hi, j_hi in {0,1} select the 2x2 block, i_lo, j_lo in {0,1} select within block.
    #
    # The 4x4x4 multiplication tensor is the tensor product of two 2x2x2 tensors.
    # For term index (r1, r2) where r1, r2 in {0..6}:
    #   U_4x4[2*i_hi + i_lo, 2*j_hi + j_lo] = U_s[r1][i_hi, j_hi] * U_s[r2][i_lo, j_lo]
    #   V_4x4[2*j_hi + j_lo, 2*k_hi + k_lo] = V_s[r1][j_hi, k_hi] * V_s[r2][j_lo, k_lo]
    #   W_4x4[2*k_hi + k_lo, 2*i_hi + i_lo] = W_s[r1][k_hi, i_hi] * W_s[r2][k_lo, i_lo]
    
    U49 = []
    V49 = []
    W49 = []
    
    for r1 in range(num_terms):
        for r2 in range(num_terms):
            U_mat = np.zeros((4, 4), dtype=float)
            V_mat = np.zeros((4, 4), dtype=float)
            W_mat = np.zeros((4, 4), dtype=float)
            
            for i_hi in range(2):
                for j_hi in range(2):
                    for k_hi in range(2):
                        for i_lo in range(2):
                            for j_lo in range(2):
                                for k_lo in range(2):
                                    i = 2 * i_hi + i_lo
                                    j = 2 * j_hi + j_lo
                                    k = 2 * k_hi + k_lo
                                    
                                    U_mat[i, j] = s2x2[r1][0][i_hi, j_hi] * s2x2[r2][0][i_lo, j_lo]
                                    V_mat[j, k] = s2x2[r1][1][j_hi, k_hi] * s2x2[r2][1][j_lo, k_lo]
                                    W_mat[k, i] = s2x2[r1][2][k_hi, i_hi] * s2x2[r2][2][k_lo, i_lo]
            
            U49.append(U_mat)
            V49.append(V_mat)
            W49.append(W_mat)
    
    # Convert to flat row-major format as required:
    # U: shape (n*m, rank) = (16, 49), U[i*m+j, r]
    # V: shape (m*p, rank) = (16, 49), V[j*p+k, r]
    # W: shape (p*n, rank) = (16, 49), W[k*n+i, r]
    
    rank = len(U49)
    U_flat = np.zeros((16, rank), dtype=float)
    V_flat = np.zeros((16, rank), dtype=float)
    W_flat = np.zeros((16, rank), dtype=float)
    
    for r in range(rank):
        for i in range(4):
            for j in range(4):
                U_flat[i * 4 + j, r] = U49[r][i, j]
                V_flat[j * 4 + i, r] = V49[r][j, i]  # Wait, V is indexed by [j,k]
                W_flat[i * 4 + j, r] = W49[r][i, j]  # Wait, W is indexed by [k,i]
    
    # Let me redo this more carefully.
    # U_flat[i*4+j, r] = U49[r][i, j] where U49[r] is a 4x4 matrix with U49[r][i,j]
    # V_flat[j*4+k, r] = V49[r][j, k] where V49[r] is a 4x4 matrix with V49[r][j,k]
    # W_flat[k*4+i, r] = W49[r][k, i] where W49[r] is a 4x4 matrix with W49[r][k,i]
    
    U_flat = np.zeros((16, rank), dtype=float)
    V_flat = np.zeros((16, rank), dtype=float)
    W_flat = np.zeros((16, rank), dtype=float)
    
    for r in range(rank):
        for i in range(4):
            for j in range(4):
                U_flat[i * 4 + j, r] = U49[r][i, j]
        for j in range(4):
            for k in range(4):
                V_flat[j * 4 + k, r] = V49[r][j, k]
        for k in range(4):
            for i in range(4):
                W_flat[k * 4 + i, r] = W49[r][k, i]
    
    # Verify the decomposition
    def verify(U_f, V_f, W_f):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = U_f.shape[1]
        for i in range(4):
            for ip in range(4):
                for k in range(4):
                    for kp in range(4):
                        for j in range(4):
                            # Coefficient of A[ip,j] * B[j,kp] in C[i,k]
                            lhs = 0.0
                            for r in range(rank):
                                lhs += W_f[k * 4 + i, r] * U_f[ip * 4 + j, r] * V_f[j * 4 + kp, r]
                            rhs = 1.0 if (i == ip and k == kp) else 0.0
                            if abs(lhs - rhs) > 1e-10:
                                return False
        return True
    
    # Verify the rank-49 decomposition
    assert verify(U_flat, V_flat, W_flat), "Rank-49 verification failed"
    
    # Now try to find a linear dependence among the 49 terms.
    # If term t can be expressed as a linear combination of the other 48 terms,
    # we can eliminate it to get rank 48.
    #
    # For each term t, we check if there exist coefficients c_0, ..., c_{48} (excluding c_t)
    # such that for all (i, ip, k, kp, j):
    #   U[t][ip*4+j] * V[t][j*4+kp] * W[t][k*4+i] = sum_{r != t} c_r * U[r][ip*4+j] * V[r][j*4+kp] * W[r][k*4+i]
    
    # This is a linear system. Let's set it up for each t.
    # The number of equations is 4*4*4*4*4 = 1024 (for all i, ip, k, kp, j).
    # The number of unknowns is 48.
    
    for t in range(rank):
        # Build the linear system A * c = b
        # A has shape (1024, 48), b has shape (1024,)
        A_sys = np.zeros((1024, rank - 1), dtype=float)
        b_sys = np.zeros(1024, dtype=float)
        
        idx = 0
        for i in range(4):
            for ip in range(4):
                for k in range(4):
                    for kp in range(4):
                        for j in range(4):
                            # b[idx] = U[t][ip*4+j] * V[t][j*4+kp] * W[t][k*4+i]
                            b_sys[idx] = U_flat[ip * 4 + j, t] * V_flat[j * 4 + kp, t] * W_flat[k * 4 + i, t]
                            
                            col = 0
                            for r in range(rank):
                                if r == t:
                                    continue
                                A_sys[idx, col] = U_flat[ip * 4 + j, r] * V_flat[j * 4 + kp, r] * W_flat[k * 4 + i, r]
                                col += 1
                            idx += 1
        
        # Solve the least squares problem
        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
        
        # Check if the solution is exact (residual is close to zero)
        residual = np.linalg.norm(A_sys @ sol - b_sys)
        
        if residual < 1e-8:
            # We found a valid linear dependence! Build the rank-48 decomposition.
            U48 = np.zeros((16, rank - 1), dtype=float)
            V48 = np.zeros((16, rank - 1), dtype=float)
            W48 = np.zeros((16, rank - 1), dtype=float)
            
            col = 0
            for r in range(rank):
                if r == t:
                    continue
                U48[:, col] = U_flat[:, r]
                V48[:, col] = V_flat[:, r]
                W48[:, col] = W_flat[:, r]
                col += 1
            
            # Verify the rank-48 decomposition
            if verify(U48, V48, W48):
                # Convert to list format
                U_list = [[U48[i, r] for r in range(rank - 1)] for i in range(16)]
                V_list = [[V48[i, r] for r in range(rank - 1)] for i in range(16)]
                W_list = [[W48[i, r] for r in range(rank - 1)] for i in range(16)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # No linear dependence found; return the rank-49 decomposition.
    U_list = [[U_flat[i, r] for r in range(rank)] for i in range(16)]
    V_list = [[V_flat[i, r] for r in range(rank)] for i in range(16)]
    W_list = [[W_flat[i, r] for r in range(rank)] for i in range(16)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
