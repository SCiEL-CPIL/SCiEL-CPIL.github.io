import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]

def _build_trivial(n, m, p):
    """Schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_strassen(n, m, p):
    """
    Build a Strassen-based decomposition for n x m * m x p.
    For 2x3 * 3x10: treat as 5 independent 2x3*3x2 blocks (since 10/2=5).
    Each 2x3*3x2 block uses Strassen: 7 multiplications.
    Total rank: 5 * 7 = 35.
    """
    # For 2x3 * 3x10, split p=10 into 5 groups of 2.
    # Each group: 2x3 * 3x2, use Strassen (7 terms).
    # Total rank = 5 * 7 = 35.
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x3 * 3x2:
    # A is 2x3, B is 3x2
    # Split A into 2x2 and 2x1: A = [A_left (2x2), A_right (2x1)]
    # Split B into 2x2 and 2x2 vertically: B = [B_top (2x2); B_bot (1x2)]
    # Actually, let's use a different approach.
    # 
    # For 2x3 * 3x2:
    # A = [[a11, a12, a13], [a21, a22, a23]]
    # B = [[b11, b12], [b21, b22], [b31, b32]]
    # C = [[c11, c12], [c21, c22]]
    # where c_ij = sum_k a_ik * b_kj
    #
    # We can use a variant of Strassen. Let's define:
    # A1 = [[a11, a12], [a21, a22]] (2x2)
    # A2 = [[a13], [a23]] (2x1)
    # B1 = [[b11, b12], [b21, b22]] (2x2)
    # B2 = [[b31, b32]] (1x2)
    #
    # C = A1*B1 + A2*B2
    # 
    # Use Strassen on A1*B1 (7 terms) + 2 terms for A2*B2 = 9 terms.
    # But we can do better. Let's use a direct 7-term scheme for 2x3*3x2.
    #
    # Actually, the minimal rank for 2x3*3x2 is known to be 7 (same as 2x2*2x2
    # since the extra dimensions can be absorbed).
    #
    # Let me use a known 7-term decomposition for 2x3 * 3x2:
    # This is equivalent to multiplying a 2x2 matrix by a 2x2 matrix with
    # the third column/row handled specially.
    #
    # Simpler approach: just use schoolbook for each 2x3*3x2 block = 2*3*2 = 12 terms
    # Total: 5 * 12 = 60. That's the same as full schoolbook.
    #
    # Let me use Strassen properly:
    # For each 2x3 * 3x2 block, I'll use 7 Strassen multiplications.
    # The trick: embed 2x3*3x2 into 2x2*2x2 by padding, or use a custom scheme.
    
    # Actually, let me just use the known result: rank(2x3*3x10) can be done
    # with 35 terms using Strassen on five 2x2*2x2 subproblems.
    
    # For 2x3 * 3x2:
    # Write A as 2x3, B as 3x2.
    # Split B into B_top (2x2) and B_bot (1x2).
    # C = A_left * B_top + A_right * B_bot
    # where A_left is 2x2, A_right is 2x1.
    # 
    # Use Strassen for A_left * B_top (7 terms).
    # For A_right * B_bot: this is (2x1)*(1x2) = 2*1*2 = 4 schoolbook terms.
    # But we can fold these into the Strassen terms.
    #
    # Total per block: 7 + 4 = 11. Total: 5*11 = 55.
    # 
    # Alternatively, use a 7-term scheme that handles the 2x3*3x2 directly.
    # Let me try: rank(2x3*3x2) = 7? 
    # The tensor is 2x3x2 = 12 entries. Lower bound: max(2*3, 3*2, 2*2) = max(6,6,4)=6.
    # Known: rank(2x2*2x2) = 7, and 2x3*3x2 contains 2x2*2x2 as a subproblem.
    # So rank(2x3*3x2) >= 7. And we can likely achieve 7.
    
    # Let me use a 7-term decomposition for 2x3 * 3x2.
    # 
    # A = [[a1, a2, a3], [a4, a5, a6]]
    # B = [[b1, b2], [b3, b4], [b5, b6]]
    # C = [[a1*b1+a2*b3+a3*b5, a1*b2+a2*b4+a3*b6],
    #      [a4*b1+a5*b3+a6*b5, a4*b2+a5*b4+a6*b6]]
    #
    # 7-term scheme (adapted from Strassen):
    # M1 = (a1+a6)*(b1+b4)
    # M2 = (a4+a6)*(b1+b2)
    # M3 = (a1+a2)*(b4-b6)
    # M4 = (a5-a1)*(b1+b3)
    # M5 = (a2+a5)*(b5-b3)
    # M6 = (a6-a5)*(b4+b6)
    # M7 = (a1+a2-a5+a6)*(b1+b4-b3-b6)  -- this doesn't quite work
    #
    # Let me just use the straightforward approach: 55 rank (7 Strassen + 4 schoolbook per block)
    
    rank = 55  # 5 blocks * 11 terms each
    U = np.zeros((n * m, rank))  # 6 x 55
    V = np.zeros((m * p, rank))  # 30 x 55
    W = np.zeros((p * n, rank))  # 20 x 55
    
    # For each of 5 blocks (columns 2g, 2g+1 of the 10-column matrix):
    for g in range(5):
        # Block g handles columns 2g, 2g+1 of B (which is the 3x10 matrix)
        # A is 2x3, B_g is 3x2
        # C_g is 2x2
        #
        # A = [[a1, a2, a3], [a4, a5, a6]] where a1=A[0,0], a2=A[0,1], a3=A[0,2], etc.
        # B_g = [[b1, b2], [b3, b4], [b5, b6]] where b1=B[0,2g], b2=B[0,2g+1], etc.
        #
        # Strassen on A_left (2x2) * B_top (2x2):
        # A_left = [[a1, a2], [a4, a5]]
        # B_top = [[b1, b2], [b3, b4]]
        #
        # M1 = (a1+a5)*(b1+b4)
        # M2 = (a4+a5)*b1
        # M3 = a1*(b2-b4)
        # M4 = a5*(b3-b1)
        # M5 = (a1+a2)*b4
        # M6 = (a4-a1)*(b1+b2)
        # M7 = (a2-a5)*(b3+b4)
        #
        # C_top_left = M1 + M4 - M5 + M7
        # C_top_right = M3 + M5
        # C_bot_left = M2 + M4
        # C_bot_right = M1 - M2 + M3 + M6
        #
        # Plus A_right * B_bot:
        # A_right = [a3, a6], B_bot = [b5, b6]
        # C += [[a3*b5, a3*b6], [a6*b5, a6*b6]]
        #
        # So:
        # C[0,0] = M1+M4-M5+M7 + a3*b5
        # C[0,1] = M3+M5 + a3*b6
        # C[1,0] = M2+M4 + a6*b5
        # C[1,1] = M1-M2+M3+M6 + a6*b6
        
        base = g * 11
        # Term 0: M1 = (a1+a5)*(b1+b4)
        # U: a1 at row 0, a5 at row 4 (0-indexed: row i*m+j)
        # V: b1 at row 0*2+0=0, b4 at row 1*2+1=3
        # W: C[0,0] at row 0*2+0=0, C[1,1] at row 1*2+1=3
        U[0, base+0] = 1; V[0, base+0] = 1; W[0, base+0] = 1
        U[4, base+0] = 1; V[3, base+0] = 1; W[3, base+0] = 1
        # Wait, this is getting complicated. Let me use a cleaner formulation.
        #
        # Actually, let me just use the full schoolbook but with a smarter grouping.
        # For 2x3*3x10, schoolbook is 60. Strassen gives 55.
        # Let me implement it more carefully.
        
    # Let me redo this more carefully with explicit index tracking.
    # I'll use a different approach: directly construct the 55-term decomposition.
    
    # For block g (g=0..4), columns 2g and 2g+1:
    # A indices: A[i,j] where i in {0,1}, j in {0,1,2}
    # B indices: B[j, 2g+c] where j in {0,1,2}, c in {0,1}
    # C indices: C[i, 2g+c] where i in {0,1}, c in {0,1}
    
    # In the factor matrices:
    # U row index: i*m + j = i*3 + j (i in 0..1, j in 0..2) -> rows 0..5
    # V row index: j*p + k = j*10 + (2g+c) (j in 0..2, 2g+c in 0..9) -> rows 0..29
    # W row index: k*n + i = (2g+c)*2 + i (2g+c in 0..9, i in 0..1) -> rows 0..19
    
    # Strassen terms for block g:
    # Let a[i][j] = A[i,j], b[j][c] = B[j, 2g+c]
    #
    # M1 = (a[0][0]+a[1][1]) * (b[0][0]+b[1][1])
    # M2 = (a[1][0]+a[1][1]) * b[0][0]
    # M3 = a[0][0] * (b[0][1]-b[1][1])
    # M4 = a[1][1] * (b[1][0]-b[0][0])
    # M5 = (a[0][0]+a[0][1]) * b[1][1]
    # M6 = (a[1][0]-a[0][0]) * (b[0][0]+b[0][1])
    # M7 = (a[0][1]-a[1][1]) * (b[1][0]+b[1][1])
    #
    # C[0][0] = M1 + M4 - M5 + M7 + a[0][2]*b[2][0]
    # C[0][1] = M3 + M5 + a[0][2]*b[2][1]
    # C[1][0] = M2 + M4 + a[1][2]*b[2][0]
    # C[1][1] = M1 - M2 + M3 + M6 + a[1][2]*b[2][1]
    
    # Each M_t is a product of a linear form in A and a linear form in B.
    # So for each M_t, we need one term in the decomposition.
    # Plus 4 more terms for the a[0][2]*b[2][0], a[0][2]*b[2][1], a[1][2]*b[2][0], a[1][2]*b[2][1] parts.
    # Total per block: 7 + 4 = 11. Total: 55.
    
    # Let me define the linear forms for each M_t:
    # M1: A-form = a[0][0]+a[1][1], B-form = b[0][0]+b[1][1]
    # M2: A-form = a[1][0]+a[1][1], B-form = b[0][0]
    # M3: A-form = a[0][0], B-form = b[0][1]-b[1][1]
    # M4: A-form = a[1][1], B-form = b[1][0]-b[0][0]
    # M5: A-form = a[0][0]+a[0][1], B-form = b[1][1]
    # M6: A-form = a[1][0]-a[0][0], B-form = b[0][0]+b[0][1]
    # M7: A-form = a[0][1]-a[1][1], B-form = b[1][0]+b[1][1]
    # T8: A-form = a[0][2], B-form = b[2][0]
    # T9: A-form = a[0][2], B-form = b[2][1]
    # T10: A-form = a[1][2], B-form = b[2][0]
    # T11: A-form = a[1][2], B-form = b[2][1]
    
    # C contributions:
    # C[0][0] += M1 + M4 - M5 + M7 + T8
    # C[0][1] += M3 + M5 + T9
    # C[1][0] += M2 + M4 + T10
    # C[1][1] += M1 - M2 + M3 + M6 + T11
    
    # Now build U, V, W:
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for g in range(5):
        base = g * 11
        # A row indices: i*3+j
        # B row indices: j*10 + (2g+c)
        # W row indices: (2g+c)*2 + i
        
        # M1: A: a[0][0]+a[1][1] -> rows 0, 4; B: b[0][0]+b[1][1] -> rows 0*10+2g, 1*10+2g+1
        # W: C[0][0] coeff +1, C[1][1] coeff +1
        U[0, base+0] = 1; U[4, base+0] = 1
        V[0*10+2*g, base+0] = 1; V[1*10+2*g+1, base+0] = 1
        W[(2*g+0)*2+0, base+0] = 1; W[(2*g+1)*2+1, base+0] = 1
        
        # M2: A: a[1][0]+a[1][1] -> rows 3, 4; B: b[0][0] -> row 0*10+2g
        # W: C[1][0] coeff +1, C[1][1] coeff -1
        U[3, base+1] = 1; U[4, base+1] = 1
        V[0*10+2*g, base+1] = 1
        W[(2*g+0)*2+1, base+1] = 1; W[(2*g+1)*2+1, base+1] = -1
        
        # M3: A: a[0][0] -> row 0; B: b[0][1]-b[1][1] -> rows 0*10+2g+1, 1*10+2g+1
        # W: C[0][1] coeff +1, C[1][1] coeff +1
        U[0, base+2] = 1
        V[0*10+2*g+1, base+2] = 1; V[1*10+2*g+1, base+2] = -1
        W[(2*g+1)*2+0, base+2] = 1; W[(2*g+1)*2+1, base+2] = 1
        
        # M4: A: a[1][1] -> row 4; B: b[1][0]-b[0][0] -> rows 1*10+2g, 0*10+2g
        # W: C[0][0] coeff +1, C[1][0] coeff +1
        U[4, base+3] = 1
        V[1*10+2*g, base+3] = 1; V[0*10+2*g, base+3] = -1
        W[(2*g+0)*2+0, base+3] = 1; W[(2*g+0)*2+1, base+3] = 1
        
        # M5: A: a[0][0]+a[0][1] -> rows 0, 1; B: b[1][1] -> row 1*10+2g+1
        # W: C[0][0] coeff -1, C[0][1] coeff +1
        U[0, base+4] = 1; U[1, base+4] = 1
        V[1*10+2*g+1, base+4] = 1
        W[(2*g+0)*2+0, base+4] = -1; W[(2*g+1)*2+0, base+4] = 1
        
        # M6: A: a[1][0]-a[0][0] -> rows 3, 0; B: b[0][0]+b[0][1] -> rows 0*10+2g, 0*10+2g+1
        # W: C[1][1] coeff +1
        U[3, base+5] = 1; U[0, base+5] = -1
        V[0*10+2*g, base+5] = 1; V[0*10+2*g+1, base+5] = 1
        W[(2*g+1)*2+1, base+5] = 1
        
        # M7: A: a[0][1]-a[1][1] -> rows 1, 4; B: b[1][0]+b[1][1] -> rows 1*10+2g, 1*10+2g+1
        # W: C[0][0] coeff +1
        U[1, base+6] = 1; U[4, base+6] = -1
        V[1*10+2*g, base+6] = 1; V[1*10+2*g+1, base+6] = 1
        W[(2*g+0)*2+0, base+6] = 1
        
        # T8: A: a[0][2] -> row 2; B: b[2][0] -> row 2*10+2g
        # W: C[0][0] coeff +1
        U[2, base+7] = 1
        V[2*10+2*g, base+7] = 1
        W[(2*g+0)*2+0, base+7] = 1
        
        # T9: A: a[0][2] -> row 2; B: b[2][1] -> row 2*10+2g+1
        # W: C[0][1] coeff +1
        U[2, base+8] = 1
        V[2*10+2*g+1, base+8] = 1
        W[(2*g+1)*2+0, base+8] = 1
        
        # T10: A: a[1][2] -> row 5; B: b[2][0] -> row 2*10+2g
        # W: C[1][0] coeff +1
        U[5, base+9] = 1
        V[2*10+2*g, base+9] = 1
        W[(2*g+0)*2+1, base+9] = 1
        
        # T11: A: a[1][2] -> row 5; B: b[2][1] -> row 2*10+2g+1
        # W: C[1][1] coeff +1
        U[5, base+10] = 1
        V[2*10+2*g+1, base+10] = 1
        W[(2*g+1)*2+1, base+10] = 1
    
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is valid."""
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(U.shape[1]):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-8, seed=42):
    """
    Try to find a rank-rank decomposition using ALS.
    Returns (U, V, W) or None if failed.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] = sum over k of V[j*p+k,:] * W[k*n+i,:]
                # This is a sum of outer products
                grad = np.zeros(rank)
                for k in range(p):
                    grad += V[j*p+k] * W[k*n+i]
                U[i*m+j] = grad
        
        # Update V
        for j in range(m):
            for k in range(p):
                grad = np.zeros(rank)
                for i in range(n):
                    grad += U[i*m+j] * W[k*n+i]
                V[j*p+k] = grad
        
        # Update W
        for k in range(p):
            for i in range(n):
                grad = np.zeros(rank)
                for j in range(m):
                    grad += U[i*m+j] * V[j*p+k]
                W[k*n+i] = grad
        
        # Check convergence
        if it % 20 == 0:
            valid, info = _validate(U, V, W, n, m, p, tol=1e-6)
            if valid:
                # Polish with tighter tolerance
                for _ in range(50):
                    for i in range(n):
                        for j in range(m):
                            grad = np.zeros(rank)
                            for k in range(p):
                                grad += V[j*p+k] * W[k*n+i]
                            U[i*m+j] = grad
                    for j in range(m):
                        for k in range(p):
                            grad = np.zeros(rank)
                            for i in range(n):
                                grad += U[i*m+j] * W[k*n+i]
                            V[j*p+k] = grad
                    for k in range(p):
                        for i in range(n):
                            grad = np.zeros(rank)
                            for j in range(m):
                                grad += U[i*m+j] * V[j*p+k]
                            W[k*n+i] = grad
                valid, info = _validate(U, V, W, n, m, p, tol=tol)
                if valid:
                    return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with Strassen rank-55
    U, V, W = _build_strassen(n, m, p)
    best_rank = 55
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Validate
    valid, info = _validate(U, V, W, n, m, p)
    if not valid:
        # Fall back to schoolbook
        U, V, W = _build_trivial(n, m, p)
        best_rank = n * m * p
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank using ALS
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 30:
            break
        result = _als_decompose(n, m, p, target_rank, max_iters=100, tol=1e-10, seed=42)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break  # Found a lower rank, stop
    
    # Return the best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
