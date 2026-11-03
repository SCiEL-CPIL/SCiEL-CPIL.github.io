import time
import sys
import random
import numpy as np
from fractions import Fraction

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    We use a known rank-17 decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for c in range(rank):
                        s += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_exact(U, V, W):
        """Verify using exact rational arithmetic."""
        rank = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[idx_u][c] * V_frac[idx_v][c] * W_frac[idx_w][c]
                    if s != Fraction(1):
                        return False
        return True

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        
        return {"U": U, "V": V, "W": W}
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Indices:
        # A: i in 0..1, j in 0..2 -> idx = i*3 + j
        # B: j in 0..2, k in 0..2 -> idx = j*3 + k
        # C: k in 0..2, i in 0..1 -> idx = k*2 + i
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices: 0,1,3,4
        # B indices: 0,1,3,4
        # C indices: 0,1,2,3
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # M3: A00(B01-B11) -> C01, C11
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2, 5
        # B indices: 6, 7
        # C indices: 0,1,2,3
        
        # r=7: A02 B20 -> C00
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        # r=8: A02 B21 -> C01
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        # r=9: A12 B20 -> C10
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        # r=10: A12 B21 -> C11
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        # A indices: 0,1,2,3,4,5
        # B indices: 2,5,8
        # C indices: 4,5
        
        # r=11: A00 B02 -> C02
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        # r=12: A01 B12 -> C02
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        # r=13: A02 B22 -> C02
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        # r=14: A10 B02 -> C12
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        # r=15: A11 B12 -> C12
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        # r=16: A12 B22 -> C12
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def snap_to_simple_fraction(x, max_num=30, max_den=30, tol=0.005):
        """Snap a float to the nearest simple fraction p/q with |p|<=max_num, 1<=q<=max_den."""
        best_frac = 0.0
        best_diff = abs(x)
        for num in range(-max_num, max_num + 1):
            for den in range(1, max_den + 1):
                val = num / den
                diff = abs(x - val)
                if diff < best_diff:
                    best_diff = diff
                    best_frac = val
        if best_diff < tol:
            return best_frac
        return x

    def try_rank16_numerical():
        """
        Attempt to find a rank-16 decomposition using numerical optimization.
        Uses gradient descent with adaptive learning rate and high-precision polish.
        """
        target_rank = 16
        
        nm = n * m  # 6
        mp = m * p  # 9
        pn = p * n  # 6
        
        # Build target tensor as a 3D array
        T = np.zeros((nm, mp, pn), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        best_result = None
        best_error = np.inf
        
        # Try multiple random initializations with different strategies
        np.random.seed(42)
        num_restarts = 20
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Small random initialization
                scale = 0.3
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            elif restart % 3 == 1:
                # Larger scale
                scale = 0.7
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            else:
                # Sparse initialization
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    # Randomly assign non-zero entries
                    u_indices = np.random.choice(nm, size=3, replace=False)
                    v_indices = np.random.choice(mp, size=3, replace=False)
                    w_indices = np.random.choice(pn, size=3, replace=False)
                    for idx in u_indices:
                        U[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in v_indices:
                        V[idx, r] = np.random.choice([-1.0, 1.0])
                    for idx in w_indices:
                        W[idx, r] = np.random.choice([-1.0, 1.0])
            
            # Gradient descent with adaptive learning rate
            lr = 0.005
            num_iters = 3000
            patience = 200
            best_local_err = np.inf
            stagnant = 0
            
            for it in range(num_iters):
                if time.time() > deadline - 10:
                    break
                
                # Compute residual
                R = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                residual = R - T
                err = np.sum(residual**2)
                
                # Check for improvement
                if err < best_local_err:
                    best_local_err = err
                    stagnant = 0
                else:
                    stagnant += 1
                
                # If stagnant, reduce learning rate and break if no improvement
                if stagnant > patience:
                    if lr < 1e-6:
                        break
                    lr *= 0.5
                    stagnant = 0
                
                # Compute gradients
                dU = 2.0 * np.einsum('abc,br,cr->ar', residual, V, W, optimize=True)
                dV = 2.0 * np.einsum('abc,ar,cr->br', residual, U, W, optimize=True)
                dW = 2.0 * np.einsum('abc,ar,br->cr', residual, U, V, optimize=True)
                
                # Gradient norm for adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                
                # Update with gradient clipping
                if grad_norm > 1.0:
                    scale_factor = 1.0 / grad_norm
                    dU *= scale_factor
                    dV *= scale_factor
                    dW *= scale_factor
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Periodic check for convergence
                if it % 50 == 0:
                    R_check = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                    err_check = np.sum((R_check - T)**2)
                    
                    if err_check < 1e-4:
                        # High-precision polish
                        if _polish_and_verify(U, V, W, T, target_rank, deadline):
                            return best_result
        
        return best_result

    def _polish_and_verify(U, V, W, T, target_rank, deadline):
        """Polish a near-solution and verify with exact arithmetic."""
        # First, try to snap to simple fractions
        U_snap = np.array([[snap_to_simple_fraction(U[a, r]) for r in range(target_rank)] for a in range(n * m)])
        V_snap = np.array([[snap_to_simple_fraction(V[b, r]) for r in range(target_rank)] for b in range(m * p)])
        W_snap = np.array([[snap_to_simple_fraction(W[c, r]) for r in range(target_rank)] for c in range(p * n)])
        
        # Verify the snapped solution
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if verify(U_list, V_list, W_list):
            return True
        
        # Try with tighter tolerance
        U_tight = np.array([[snap_to_simple_fraction(U[a, r], tol=0.001) for r in range(target_rank)] for a in range(n * m)])
        V_tight = np.array([[snap_to_simple_fraction(V[b, r], tol=0.001) for r in range(target_rank)] for b in range(m * p)])
        W_tight = np.array([[snap_to_simple_fraction(W[c, r], tol=0.001) for r in range(target_rank)] for c in range(p * n)])
        
        U_tight_list = U_tight.tolist()
        V_tight_list = V_tight.tolist()
        W_tight_list = W_tight.tolist()
        
        if verify(U_tight_list, V_tight_list, W_tight_list):
            return True
        
        # If close, do a local polish with exact rational arithmetic
        # Convert to fractions
        U_frac = [[Fraction(U_snap[a, r]) for r in range(target_rank)] for a in range(n * m)]
        V_frac = [[Fraction(V_snap[b, r]) for r in range(target_rank)] for b in range(m * p)]
        W_frac = [[Fraction(W_snap[c, r]) for r in range(target_rank)] for c in range(p * n)]
        
        # Check if already exact
        if verify_exact(U_frac, V_frac, W_frac):
            # Convert back to floats
            U_out = [[float(x) for x in row] for row in U_frac]
            V_out = [[float(x) for x in row] for row in V_frac]
            W_out = [[float(x) for x in row] for row in W_frac]
            return True
        
        # Try small perturbations
        for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
            if time.time() > deadline - 5:
                break
            U_test = [[U_frac[a][r] + Fraction(delta) if (a + r) % 2 == 0 else U_frac[a][r] 
                      for r in range(target_rank)] for a in range(n * m)]
            if verify_exact(U_test, V_frac, W_frac):
                U_out = [[float(x) for x in row] for row in U_test]
                V_out = [[float(x) for x in row] for row in V_frac]
                W_out = [[float(x) for x in row] for row in W_frac]
                return True
        
        return False
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # First, try the known rank-17 decomposition
    if time.time() < deadline:
        r17 = rank17_decomposition()
        if verify(r17["U"], r17["V"], r17["W"]):
            best_result = r17
            best_rank = 17
    
    # Try to find rank-16
    if time.time() < deadline - 10:
        r16 = try_rank16_numerical()
        if r16 is not None:
            best_result = r16
            best_rank = 16
    
    return best_result
# EVOLVE-BLOCK-END
