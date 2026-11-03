"""
Rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.

Strategy:
1. Start with the known rank-28 Strassen-based decomposition (split 4x4 into four 2x2 blocks,
   apply Strassen's 7-multiplication algorithm to each, yielding 4*7 = 28 terms).
2. Try to reduce the rank further (27, 26, ...) using a numerical search:
   - Randomly initialize U, V, W with rank r.
   - Use gradient descent / Newton-like iteration on the 32 tensor equations:
       sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1  for all (i,k,j)
   - If we find a solution with residual < 1e-10, return it.
3. If no lower rank is found, return the best known (rank-28).

The key insight is that the problem is highly underdetermined for r < 32:
  - 32 equations, 32*r unknowns.
  - For r=28: 896 unknowns, 32 equations.
  - We can use random restarts and gradient methods to find solutions.
"""
import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 115  # safety margin before 120s limit

    # Helper: verify a decomposition
    def verify(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        r = U.shape[1]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Helper: convert to list-of-lists format
    def to_lists(mat):
        return [[float(x) for x in row] for row in mat]

    # Build the rank-28 Strassen decomposition
    def build_strassen_r28():
        """
        Split A (2x4) into [A00 A01] where A00, A01 are 2x2.
        Split B (4x4) into [[B00 B01], [B10 B11]] where each is 2x2.
        C = A * B = [A00*B00 + A01*B10, A00*B01 + A01*B11]
        
        For each 2x2*2x2 product, use Strassen's 7 multiplications.
        This gives 4 * 7 = 28 terms.
        
        Strassen for 2x2:
        Given X (2x2) * Y (2x2):
        m1 = (X00+X11)*(Y00+Y11)
        m2 = (X10+X11)*Y00
        m3 = X00*(Y01-Y11)
        m4 = X11*(Y10-Y00)
        m5 = (X00+X01)*Y11
        m6 = (X10-X00)*(Y00+Y01)
        m7 = (X01-X11)*(Y10+Y11)
        
        Result:
        Z00 = m1 + m4 - m5 + m7
        Z01 = m3 + m5
        Z10 = m2 + m4
        Z11 = m1 - m2 + m3 + m6
        
        Each mi is a product of a linear combination of X entries and a linear combination of Y entries.
        In tensor decomposition terms, each mi contributes one rank-1 term.
        
        For the full 2x4 * 4x4:
        A is indexed as A[i, k] where i in {0,1}, k in {0,1,2,3}
        B is indexed as B[k, j] where k in {0,1,2,3}, j in {0,1,2,3}
        
        Block structure:
        A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
        B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
        B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
        
        C00 = A00*B00 + A01*B10  (2x2 result, contributes to C[:, 0:2])
        C01 = A00*B01 + A01*B11  (2x2 result, contributes to C[:, 2:4])
        
        For each 2x2*2x2 product, we have 7 Strassen terms.
        Total: 4 * 7 = 28 terms.
        
        Let me build this carefully.
        
        For a 2x2 * 2x2 product X*Y:
        X[i, a] where i in {0,1}, a in {0,1}
        Y[a, b] where a in {0,1}, b in {0,1}
        Z[i, b] where i in {0,1}, b in {0,1}
        
        The 7 Strassen products:
        m1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
        m2 = (X[1,0]+X[1,1]) * Y[0,0]
        m3 = X[0,0] * (Y[0,1]-Y[1,1])
        m4 = X[1,1] * (Y[1,0]-Y[0,0])
        m5 = (X[0,0]+X[0,1]) * Y[1,1]
        m6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
        m7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
        
        Z[0,0] = m1 + m4 - m5 + m7
        Z[0,1] = m3 + m5
        Z[1,0] = m2 + m4
        Z[1,1] = m1 - m2 + m3 + m6
        
        In the tensor decomposition, each rank-1 term s contributes:
        U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s]
        
        For a Strassen term in block (X, Y) contributing to output block (i, b):
        The U part involves the X indices (which are A indices)
        The V part involves the Y indices (which are B indices)
        The W part involves the output index (i, b) mapped to (j, i) in the W convention.
        
        Let me define the blocks more carefully.
        
        Block 0: X = A00, Y = B00, output = C[:, 0:2] (j=0,1), rows i=0,1
        Block 1: X = A01, Y = B10, output = C[:, 0:2] (j=0,1), rows i=0,1
        Block 2: X = A00, Y = B01, output = C[:, 2:4] (j=2,3), rows i=0,1
        Block 3: X = A01, Y = B11, output = C[:, 2:4] (j=2,3), rows i=0,1
        
        For block b (0..3), with X = A[rows, cols_x], Y = B[rows_y, cols_y]:
        X indices: i in {0,1}, k = k_offset + a where a in {0,1}
        Y indices: k = k_offset_y + a where a in {0,1}, j = j_offset + b where b in {0,1}
        Output: i in {0,1}, j = j_offset + b where b in {0,1}
        
        Block 0: X = A[0:2, 0:2], so k_offset=0; Y = B[0:2, 0:2], so k_offset_y=0, j_offset=0
        Block 1: X = A[0:2, 2:4], so k_offset=2; Y = B[2:4, 0:2], so k_offset_y=2, j_offset=0
        Block 2: X = A[0:2, 0:2], so k_offset=0; Y = B[0:2, 2:4], so k_offset_y=0, j_offset=2
        Block 3: X = A[0:2, 2:4], so k_offset=2; Y = B[2:4, 2:4], so k_offset_y=2, j_offset=2
        
        For each block, the 7 Strassen terms give 7 rank-1 contributions.
        Total 28 terms.
        """
        r = 28
        U = np.zeros((n*m, r))  # (8, 28)
        V = np.zeros((m*p, r))  # (16, 28)
        W = np.zeros((p*n, r))  # (8, 28)
        
        # Block definitions: (k_offset_x, k_offset_y, j_offset)
        blocks = [
            (0, 0, 0),  # A00 * B00 -> C[:, 0:2]
            (2, 2, 0),  # A01 * B10 -> C[:, 0:2]
            (0, 0, 2),  # A00 * B01 -> C[:, 2:4]
            (2, 2, 2),  # A01 * B11 -> C[:, 2:4]
        ]
        
        # Strassen coefficients for each of the 7 products
        # For product mi, we have:
        #   X coefficients: for each (i, a) in {0,1}x{0,1}, coeff_x[i,a]
        #   Y coefficients: for each (a, b) in {0,1}x{0,1}, coeff_y[a,b]
        #   Output: for each (i, b) in {0,1}x{0,1}, coeff_z[i,b] (contribution to Z[i,b])
        
        # m1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
        #   Z[0,0] += 1, Z[1,1] += 1
        strassen = [
            # (coeff_x, coeff_y, coeff_z)
            # coeff_x: dict mapping (i,a) -> coeff
            # coeff_y: dict mapping (a,b) -> coeff
            # coeff_z: dict mapping (i,b) -> coeff
            {
                'x': {(0,0): 1, (1,1): 1},
                'y': {(0,0): 1, (1,1): 1},
                'z': {(0,0): 1, (1,1): 1}
            },
            {
                'x': {(1,0): 1, (1,1): 1},
                'y': {(0,0): 1},
                'z': {(1,0): 1, (1,1): -1}
            },
            {
                'x': {(0,0): 1},
                'y': {(0,1): 1, (1,1): -1},
                'z': {(0,1): 1, (1,1): 1}
            },
            {
                'x': {(1,1): 1},
                'y': {(1,0): 1, (0,0): -1},
                'z': {(0,0): 1, (1,0): 1}
            },
            {
                'x': {(0,0): 1, (0,1): 1},
                'y': {(1,1): 1},
                'z': {(0,0): -1, (0,1): 1}
            },
            {
                'x': {(1,0): 1, (0,0): -1},
                'y': {(0,0): 1, (0,1): 1},
                'z': {(1,1): 1}
            },
            {
                'x': {(0,1): 1, (1,1): -1},
                'y': {(1,0): 1, (1,1): 1},
                'z': {(0,0): 1}
            },
        ]
        
        for block_idx, (k_off_x, k_off_y, j_off) in enumerate(blocks):
            for s_local, term in enumerate(strassen):
                s = block_idx * 7 + s_local
                # Fill U: for each (i, a) in x coeffs
                for (i, a), cx in term['x'].items():
                    k = k_off_x + a
                    U[i*m + k, s] += cx
                
                # Fill V: for each (a, b) in y coeffs
                for (a, b), cy in term['y'].items():
                    k = k_off_y + a
                    j = j_off + b
                    V[k*p + j, s] += cy
                
                # Fill W: for each (i, b) in z coeffs
                for (i, b), cz in term['z'].items():
                    j = j_off + b
                    W[j*n + i, s] += cz
        
        return U, V, W

    # First, build the rank-28 solution
    U28, V28, W28 = build_strassen_r28()
    
    # Verify it
    if not verify(U28, V28, W28, tol=1e-10):
        # Fallback to rank-32
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = i*(m*p) + k*p + j
                    U[i*m+k, s] = 1
                    V[k*p+j, s] = 1
                    W[j*n+i, s] = 1
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
    
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
    best_rank = 28
    
    # Try to find lower rank decompositions
    def try_reduce_rank(r, max_restarts=20, max_iter=3000, seed_base=0):
        """Try to find a rank-r decomposition."""
        nonlocal deadline, best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return False
        if time.time() > deadline - 5:
            return False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = seed_base + restart
            rng = np.random.RandomState(seed)
            
            # Initialize
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_res = np.inf
            best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual and gradient
                R = np.zeros((n, m, p))
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            R[i,k,j] = np.dot(U[i*m+k, :], V[k*p+j, :] * W[j*n+i, :]) - 1.0
                            grad = V[k*p+j, :] * W[j*n+i, :]
                            dU[i*m+k, :] += 2 * R[i,k,j] * grad
                            dV[k*p+j, :] += 2 * R[i,k,j] * (U[i*m+k, :] * W[j*n+i, :])
                            dW[j*n+i, :] += 2 * R[i,k,j] * (U[i*m+k, :] * V[k*p+j, :])
                
                res_sq = np.sum(R**2)
                if res_sq < best_res:
                    best_res = res_sq
                    best_U_r, best_V_r, best_W_r = U.copy(), V.copy(), W.copy()
                
                if best_res < 1e-12:
                    break
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 0:
                    step = lr * res_sq / (grad_norm + 1e-10)
                    step = min(step, 1.0)  # cap step size
                    U -= step * dU / grad_norm
                    V -= step * dV / grad_norm
                    W -= step * dW / grad_norm
                else:
                    break
                
                # Decay learning rate
                if it % 500 == 0 and it > 0:
                    lr *= 0.8
            
            if best_res < 1e-10:
                best_U, best_V, best_W = best_U_r, best_V_r, best_W_r
                best_rank = r
                return True
        
        return False
    
    # Try ranks 27, 26, 25, ...
    for r in range(27, 20, -1):
        if time.time() > deadline - 10:
            break
        if try_reduce_rank(r, max_restarts=15, max_iter=2000, seed_base=r*100):
            break
    
    # Also try a different approach: start from rank-28 and try to zero out one term
    if best_rank == 28 and time.time() < deadline - 15:
        # Try to find a rank-27 by starting near rank-28 and forcing one column to be small
        for force_col in range(28):
            if time.time() > deadline - 5:
                break
            
            U = U28.copy()
            V = V28.copy()
            W = W28.copy()
            
            # Perturb and try to eliminate column force_col
            rng = np.random.RandomState(1000 + force_col)
            U[:, force_col] += rng.randn(n*m) * 0.1
            V[:, force_col] += rng.randn(m*p) * 0.1
            W[:, force_col] += rng.randn(p*n) * 0.1
            
            # Now do gradient descent trying to make column force_col go to zero
            # while maintaining the tensor identity
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                
                R = np.zeros((n, m, p))
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            R[i,k,j] = np.dot(U[i*m+k, :], V[k*p+j, :] * W[j*n+i, :]) - 1.0
                            grad = V[k*p+j, :] * W[j*n+i, :]
                            dU[i*m+k, :] += 2 * R[i,k,j] * grad
                            dV[k*p+j, :] += 2 * R[i,k,j] * (U[i*m+k, :] * W[j*n+i, :])
                            dW[j*n+i, :] += 2 * R[i,k,j] * (U[i*m+k, :] * V[k*p+j, :])
                
                # Add penalty for the forced column
                penalty = 0.1
                dU[:, force_col] += penalty * U[:, force_col]
                dV[:, force_col] += penalty * V[:, force_col]
                dW[:, force_col] += penalty * W[:, force_col]
                
                res_sq = np.sum(R**2) + penalty * (np.sum(U[:, force_col]**2) + 
                                                    np.sum(V[:, force_col]**2) + 
                                                    np.sum(W[:, force_col]**2))
                
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 0:
                    step = min(0.01, res_sq / (grad_norm + 1e-10))
                    U -= step * dU / grad_norm
                    V -= step * dV / grad_norm
                    W -= step * dW / grad_norm
                
                # Check if we can drop the forced column
                col_norm = np.sqrt(np.sum(U[:, force_col]**2) + 
                                   np.sum(V[:, force_col]**2) + 
                                   np.sum(W[:, force_col]**2))
                if col_norm < 0.01:
                    # Drop the column
                    U_new = np.delete(U, force_col, axis=1)
                    V_new = np.delete(V, force_col, axis=1)
                    W_new = np.delete(W, force_col, axis=1)
                    if verify(U_new, V_new, W_new, tol=1e-8):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = 27
                        break
            
            if best_rank == 27:
                break
    
    return {"U": to_lists(best_U), "V": to_lists(best_V), "W": to_lists(best_W)}
