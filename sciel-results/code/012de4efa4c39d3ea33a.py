"""Rewritten program for 2x4x4 matrix multiplication tensor rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    U: shape (n*m, rank) = (8, rank)
    V: shape (m*p, rank) = (16, rank)
    W: shape (p*n, rank) = (8, rank), indexed W[k*n+i, r]
    
    The decomposition satisfies: for all i,j,k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster vectorized verification."""
        r = U.shape[1]
        # Build all 32 (i,j,k) triples and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def clean_to_half_int(U, V, W):
        """Round entries to nearest integer or half-integer, verify still works."""
        def round_half(x):
            return round(x * 2) / 2.0
        
        U_clean = np.round(U * 2) / 2.0
        V_clean = np.round(V * 2) / 2.0
        W_clean = np.round(W * 2) / 2.0
        
        if verify_fast(U_clean, V_clean, W_clean, tol=1e-6):
            return U_clean, V_clean, W_clean
        return U, V, W
    
    def to_lists(U, V, W):
        """Convert numpy arrays to lists of lists."""
        U_list = [[float(U[i, c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[float(V[i, c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[float(W[i, c]) for c in range(W.shape[1])] for i in range(W.shape[0])]
        return U_list, V_list, W_list
    
    # ============================================================
    # Strategy: Build Strassen-based rank-28 decomposition, then
    # attempt to find a rank-27 decomposition via targeted search.
    # ============================================================
    
    # First: construct a valid rank-28 decomposition using Strassen on 2x2 blocks.
    # 2x4 x 4x4: split A into two 2x2 blocks (rows), B into four 2x2 blocks (cols).
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split: A = [A00, A01, A10, A11] where Aab is 2x2, row-block a, col-block b
    #         B = [B00, B10; B01, B11] where Bab is 2x2, row-block a, col-block b
    # C = [A00*B00 + A01*B10, A00*B01 + A01*B11; A10*B00 + A11*B10, A10*B01 + A11*B11]
    # Each 2x2 product uses Strassen rank 7.
    # C00 needs A00*B00 + A01*B10: 2 Strassen products = 14 terms
    # C01 needs A00*B01 + A01*B11: 2 Strassen products = 14 terms
    # C10 needs A10*B00 + A11*B10: 2 Strassen products = 14 terms
    # C11 needs A10*B01 + A11*B11: 2 Strassen products = 14 terms
    # But we can share: total = 4 * 2 * 7 = 56? No, that's too high.
    
    # Actually, let's think differently. The 2x4x4 tensor has 32 entries.
    # Schoolbook rank = 32. Strassen-like gives 28.
    # Let's use a direct approach: find a rank-28 decomposition first, then try 27.
    
    # Build the tensor T as a 3D array T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,4)
    # We want: T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c]
    
    # Approach: Use the known fact that rank(2x4x4) <= 28 via block Strassen.
    # Let's construct it explicitly.
    
    # For 2x2x2 Strassen:
    # M1 = (A11-A21)(B12-B22)
    # M2 = (A11+A21)B22
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M5+M4-M3-M7
    # C12 = M1+M7
    # C21 = M2+M4
    # C22 = M5+M1-M2-M6
    
    # For 2x4 x 4x4, we can view it as:
    # A is 2x4 = two 2x2 blocks side by side: A = [A_left, A_right]
    # B is 4x4 = four 2x2 blocks: B = [[B_tl, B_tr], [B_bl, B_br]]
    # C is 2x4 = two 2x2 blocks side by side: C = [C_left, C_right]
    # C_left = A_left * B_tl + A_right * B_bl  (2x2 * 2x2 + 2x2 * 2x2)
    # C_right = A_left * B_tr + A_right * B_br
    
    # Each 2x2*2x2 product needs rank 7 (Strassen).
    # C_left = product(A_left, B_tl) + product(A_right, B_bl): 7 + 7 = 14 terms
    # C_right = product(A_left, B_tr) + product(A_right, B_br): 7 + 7 = 14 terms
    # Total: 28 terms. This is the rank-28 construction.
    
    # Let's build this explicitly.
    # U has shape (8, 28): U[i*4+j, c]
    # V has shape (16, 28): V[j*4+k, c]
    # W has shape (8, 28): W[k*2+i, c]
    
    # For Strassen on 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # The 7 bilinear forms are:
    # M1 = (d-c)(f-g): U factor: d-c on A, f-g on B
    # M2 = (a+c)h: U: a+c, V: h
    # M3 = a(e-g): U: a, V: e-g
    # M4 = d(h-f): U: d, V: h-f
    # M5 = (a+d)(e+h): U: a+d, V: e+h
    # M6 = (c-a)(e+f): U: c-a, V: e+f
    # M7 = (b-d)(g+h): U: b-d, V: g+h
    # C entries:
    # C[0,0] = M5+M4-M3-M7
    # C[0,1] = M1+M7
    # C[1,0] = M2+M4
    # C[1,1] = M5+M1-M2-M6
    
    # For our 2x4x4 case:
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    #   A_left = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   A_right = A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # B is 4x4: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3}
    #   B_tl = B[0:2, 0:2], B_tr = B[0:2, 2:4]
    #   B_bl = B[2:4, 0:2], B_br = B[2:4, 2:4]
    # C is 2x4: C[i, k] where i in {0,1}, k in {0,1,2,3}
    #   C_left = C[:, 0:2], C_right = C[:, 2:4]
    
    # C_left[i, k] = sum_j A[i, j] * B[j, k] for j in 0..3, k in 0..1
    # = sum_j in 0..1 A[i,j]*B[j,k] + sum_j in 2..3 A[i,j]*B[j,k]
    # = (A_left * B_tl)[i, k] + (A_right * B_bl)[i, k]
    
    # So C_left = Strassen(A_left, B_tl) + Strassen(A_right, B_bl)
    # C_right = Strassen(A_left, B_tr) + Strassen(A_right, B_br)
    
    # Let's build the 28-term decomposition.
    # Term indices 0-13: for C_left (7 from A_left*B_tl, 7 from A_right*B_bl)
    # Term indices 14-27: for C_right (7 from A_left*B_tr, 7 from A_right*B_br)
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    
    # Strassen coefficients for 2x2:
    # For product P = X * Y where X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x11-x10)(y01-y11)
    # M2 = (x00+x10)*y11
    # M3 = x00*(y00-y10)
    # M4 = x11*(y11-y01)
    # M5 = (x00+x11)*(y00+y11)
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # P[0,0] = M5+M4-M3-M7
    # P[0,1] = M1+M7
    # P[1,0] = M2+M4
    # P[1,1] = M5+M1-M2-M6
    
    # X indices in A (2x4): x00=A[0,0], x01=A[0,1], x10=A[1,0], x11=A[1,1]
    #   For A_left: col offset 0 -> A[i, j] at U[i*4+j]
    #   For A_right: col offset 2 -> A[i, j] at U[i*4+(j+2)]
    # Y indices in B (4x4): y00=B[0,0], y01=B[0,1], y10=B[1,0], y11=B[1,1]
    #   For B_tl: row offset 0, col offset 0 -> B[j, k] at V[j*4+k]
    #   For B_tr: row offset 0, col offset 2 -> B[j, k] at V[j*4+(k+2)]
    #   For B_bl: row offset 2, col offset 0 -> B[j, k] at V[(j+2)*4+k]
    #   For B_br: row offset 2, col offset 2 -> B[j, k] at V[(j+2)*4+(k+2)]
    # C indices: C[i, k] at W[k*2+i]
    #   For C_left: col offset 0 -> C[i, k] at W[k*2+i]
    #   For C_right: col offset 2 -> C[i, k] at W[(k+2)*2+i]
    
    def add_strassen_block(U, V, W, term_start, a_col_off, b_row_off, b_col_off, c_col_off):
        """Add 7 Strassen terms for a 2x2*2x2 product block."""
        # X = A[:, a_col_off:a_col_off+2], Y = B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
        # X[i,j] -> U[i*4 + a_col_off + j]
        # Y[i,j] -> V[(b_row_off+i)*4 + b_col_off + j]
        # C[i,j] -> W[(c_col_off+j)*2 + i]
        
        # M1 = (x11-x10)(y01-y11)
        c = term_start + 0
        U[1*4 + a_col_off + 1, c] += 1  # x11
        U[1*4 + a_col_off + 0, c] += -1  # -x10
        V[(b_row_off+0)*4 + b_col_off + 1, c] += 1  # y01
        V[(b_row_off+1)*4 + b_col_off + 1, c] += -1  # -y11
        # C contributions: C[0,1] += M1, C[1,1] += M1
        W[(c_col_off+1)*2 + 0, c] += 1
        W[(c_col_off+1)*2 + 1, c] += 1
        
        # M2 = (x00+x10)*y11
        c = term_start + 1
        U[0*4 + a_col_off + 0, c] += 1  # x00
        U[1*4 + a_col_off + 0, c] += 1  # x10
        V[(b_row_off+1)*4 + b_col_off + 1, c] += 1  # y11
        # C contributions: C[1,0] += M2, C[1,1] -= M2
        W[(c_col_off+0)*2 + 1, c] += 1
        W[(c_col_off+1)*2 + 1, c] += -1
        
        # M3 = x00*(y00-y10)
        c = term_start + 2
        U[0*4 + a_col_off + 0, c] += 1  # x00
        V[(b_row_off+0)*4 + b_col_off + 0, c] += 1  # y00
        V[(b_row_off+1)*4 + b_col_off + 0, c] += -1  # -y10
        # C contributions: C[0,0] -= M3
        W[(c_col_off+0)*2 + 0, c] += -1
        
        # M4 = x11*(y11-y01)
        c = term_start + 3
        U[1*4 + a_col_off + 1, c] += 1  # x11
        V[(b_row_off+1)*4 + b_col_off + 1, c] += 1  # y11
        V[(b_row_off+0)*4 + b_col_off + 1, c] += -1  # -y01
        # C contributions: C[0,0] += M4, C[1,0] += M4
        W[(c_col_off+0)*2 + 0, c] += 1
        W[(c_col_off+0)*2 + 1, c] += 1
        
        # M5 = (x00+x11)*(y00+y11)
        c = term_start + 4
        U[0*4 + a_col_off + 0, c] += 1  # x00
        U[1*4 + a_col_off + 1, c] += 1  # x11
        V[(b_row_off+0)*4 + b_col_off + 0, c] += 1  # y00
        V[(b_row_off+1)*4 + b_col_off + 1, c] += 1  # y11
        # C contributions: C[0,0] += M5, C[1,1] += M5
        W[(c_col_off+0)*2 + 0, c] += 1
        W[(c_col_off+1)*2 + 1, c] += 1
        
        # M6 = (x10-x00)*(y00+y01)
        c = term_start + 5
        U[1*4 + a_col_off + 0, c] += 1  # x10
        U[0*4 + a_col_off + 0, c] += -1  # -x00
        V[(b_row_off+0)*4 + b_col_off + 0, c] += 1  # y00
        V[(b_row_off+0)*4 + b_col_off + 1, c] += 1  # y01
        # C contributions: C[1,1] -= M6
        W[(c_col_off+1)*2 + 1, c] += -1
        
        # M7 = (x01-x11)*(y10+y11)
        c = term_start + 6
        U[0*4 + a_col_off + 1, c] += 1  # x01
        U[1*4 + a_col_off + 1, c] += -1  # -x11
        V[(b_row_off+1)*4 + b_col_off + 0, c] += 1  # y10
        V[(b_row_off+1)*4 + b_col_off + 1, c] += 1  # y11
        # C contributions: C[0,0] -= M7, C[0,1] += M7
        W[(c_col_off+0)*2 + 0, c] += -1
        W[(c_col_off+1)*2 + 0, c] += 1
    
    # C_left = Strassen(A_left, B_tl) + Strassen(A_right, B_bl)
    add_strassen_block(U, V, W, 0, 0, 0, 0, 0)   # A_left * B_tl -> C_left
    add_strassen_block(U, V, W, 7, 2, 2, 0, 0)   # A_right * B_bl -> C_left
    
    # C_right = Strassen(A_left, B_tr) + Strassen(A_right, B_br)
    add_strassen_block(U, V, W, 14, 0, 0, 2, 2)  # A_left * B_tr -> C_right
    add_strassen_block(U, V, W, 21, 2, 2, 2, 2)  # A_right * B_br -> C_right
    
    # Verify rank-28 decomposition
    assert verify_fast(U, V, W), "Rank-28 Strassen decomposition failed verification!"
    
    best_rank = 28
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # ============================================================
    # Now try to find a rank-27 decomposition.
    # Strategy: Try to eliminate one term by finding a linear dependency
    # or use optimization to find a 27-term decomposition.
    # ============================================================
    
    if time.time() < deadline - 15.0:
        # Approach 1: Try to find a rank-27 decomposition by optimization
        # We'll use a structured random search + gradient-based refinement
        
        np.random.seed(12345)
        
        # Try multiple random initializations for rank 27
        for restart in range(30):
            if time.time() > deadline - 10.0:
                break
            
            U27 = np.random.randn(8, 27) * 0.3
            V27 = np.random.randn(16, 27) * 0.3
            W27 = np.random.randn(8, 27) * 0.3
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_err = float('inf')
            stuck = 0
            
            for step in range(2000):
                if time.time() > deadline - 5.0:
                    break
                
                # Compute error and gradients
                err = 0.0
                dU = np.zeros_like(U27)
                dV = np.zeros_like(V27)
                dW = np.zeros_like(W27)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U27[u_idx, :], V27[v_idx, :] * W27[w_idx, :])
                            e = val - 1.0
                            err += e * e
                            dU[u_idx, :] += 2 * e * V27[v_idx, :] * W27[w_idx, :]
                            dV[v_idx, :] += 2 * e * U27[u_idx, :] * W27[w_idx, :]
                            dW[w_idx, :] += 2 * e * U27[u_idx, :] * V27[v_idx, :]
                
                # Update
                U27 -= lr * dU
                V27 -= lr * dV
                W27 -= lr * dW
                
                if err < best_err:
                    best_err = err
                    stuck = 0
                else:
                    stuck += 1
                    # Reduce learning rate if stuck
                    if stuck > 50:
                        lr *= 0.5
                        stuck = 0
                
                if err < 1e-10:
                    break
            
            if best_err < 1e-8:
                # Check if we can round to half-integers
                U27_clean, V27_clean, W27_clean = clean_to_half_int(U27, V27, W27)
                if verify_fast(U27_clean, V27_clean, W27_clean, tol=1e-6):
                    best_rank = 27
                    best_U, best_V, best_W = U27_clean, V27_clean, W27_clean
                    break
                elif verify_fast(U27, V27, W27, tol=1e-6):
                    best_rank = 27
                    best_U, best_V, best_W = U27, V27, W27
                    break
    
    # ============================================================
    # If rank-27 not found, try a different structural approach:
    # Try to find rank-27 by modifying the Strassen construction.
    # The idea: in the 28-term construction, check if any term is
    # linearly dependent on others.
    # ============================================================
    
    if best_rank == 28 and time.time() < deadline - 10.0:
        # Try to merge two terms in the Strassen construction
        # by finding a linear combination that preserves the tensor.
        # 
        # Alternative: try a different block decomposition.
        # What if we split A differently?
        # A is 2x4. Instead of [A_left(2x2), A_right(2x2)],
        # we could try other splits or a more clever construction.
        
        # Try: use rank-7 for a 2x2x2 and handle the remaining
        # structure more efficiently.
        
        # Actually, let's try a different approach:
        # The 2x4x4 tensor can be seen as two 2x2x4 tensors (one for each row of A).
        # Or as four 2x4x1 tensors (one for each column of B).
        
        # Let's try to find a rank-27 decomposition by perturbing
        # the rank-28 and checking if one column becomes redundant.
        
        # For each pair of columns (c1, c2), check if column c2 can be
        # expressed as a linear combination involving column c1 and others.
        # This is complex, so let's try a simpler approach:
        # Randomly perturb the rank-28 and see if we can drop one term.
        
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Start from rank-28 and try to zero out one column
            # by adjusting others
            c_target = attempt % 28
            
            # Try to find coefficients alpha such that
            # column c_target = sum_{c != c_target} alpha_c * column_c
            # in the tensor sense. This is a system of 32 equations.
            
            # Build the system: for each (i,j,k):
            # U[i*m+j, c_target] * V[j*p+k, c_target] * W[k*n+i, c_target]
            # = sum_{c != c_target} alpha_c * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # This is nonlinear in alpha, so let's try a different approach.
            # Instead, let's just try random rank-27 initializations
            # seeded near the rank-28 solution.
            
            U27 = U[:, :27].copy()
            V27 = V[:, :27].copy()
            W27 = W[:, :27].copy()
            
            # Add small perturbation
            U27 += np.random.randn(8, 27) * 0.1
            V27 += np.random.randn(16, 27) * 0.1
            W27 += np.random.randn(8, 27) * 0.1
            
            # Optimize
            lr = 0.02
            for step in range(1000):
                if time.time() > deadline - 3.0:
                    break
                
                err = 0.0
                dU = np.zeros_like(U27)
                dV = np.zeros_like(V27)
                dW = np.zeros_like(W27)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U27[u_idx, :], V27[v_idx, :] * W27[w_idx, :])
                            e = val - 1.0
                            err += e * e
                            dU[u_idx, :] += 2 * e * V27[v_idx, :] * W27[w_idx, :]
                            dV[v_idx, :] += 2 * e * U27[u_idx, :] * W27[w_idx, :]
                            dW[w_idx, :] += 2 * e * U27[u_idx, :] * V27[v_idx, :]
                
                U27 -= lr * dU
                V27 -= lr * dV
                W27 -= lr * dW
                
                if err < 1e-10:
                    break
            
            if err < 1e-8:
                U27c, V27c, W27c = clean_to_half_int(U27, V27, W27)
                if verify_fast(U27c, V27c, W27c, tol=1e-6):
                    best_rank = 27
                    best_U, best_V, best_W = U27c, V27c, W27c
                    break
                elif verify_fast(U27, V27, W27, tol=1e-6):
                    best_rank = 27
                    best_U, best_V, best_W = U27, V27, W27
                    break
    
    # ============================================================
    # Final: try to find even lower rank (26, 25, ...) if time permits
    # ============================================================
    
    if best_rank == 28 and time.time() < deadline - 20.0:
        # Try rank 26
        np.random.seed(999)
        for restart in range(10):
            if time.time() > deadline - 10.0:
                break
            
            U26 = np.random.randn(8, 26) * 0.4
            V26 = np.random.randn(16, 26) * 0.4
            W26 = np.random.randn(8, 26) * 0.4
            
            lr = 0.03
            for step in range(1500):
                if time.time() > deadline - 5.0:
                    break
                
                err = 0.0
                dU = np.zeros_like(U26)
                dV = np.zeros_like(V26)
                dW = np.zeros_like(W26)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U26[u_idx, :], V26[v_idx, :] * W26[w_idx, :])
                            e = val - 1.0
                            err += e * e
                            dU[u_idx, :] += 2 * e * V26[v_idx, :] * W26[w_idx, :]
                            dV[v_idx, :] += 2 * e * U26[u_idx, :] * W26[w_idx, :]
                            dW[w_idx, :] += 2 * e * U26[u_idx, :] * V26[v_idx, :]
                
                U26 -= lr * dU
                V26 -= lr * dV
                W26 -= lr * dW
                
                if err < 1e-10:
                    break
            
            if err < 1e-8:
                U26c, V26c, W26c = clean_to_half_int(U26, V26, W26)
                if verify_fast(U26c, V26c, W26c, tol=1e-6):
                    best_rank = 26
                    best_U, best_V, best_W = U26c, V26c, W26c
                    break
                elif verify_fast(U26, V26, W26, tol=1e-6):
                    best_rank = 26
                    best_U, best_V, best_W = U26, V26, W26
                    break
    
    # Convert to lists and return
    U_list, V_list, W_list = to_lists(best_U, best_V, best_W)
    
    # Final rounding to half-integers
    for lst in [U_list, V_list, W_list]:
        for i in range(len(lst)):
            for c in range(len(lst[i])):
                lst[i][c] = round(lst[i][c] * 2) / 2.0
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
