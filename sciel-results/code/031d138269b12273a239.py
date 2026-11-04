"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check exactness with tight tolerance
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_rank_als(r, n, m, p, T_true, max_iters=2000, seed=42, time_budget=30):
    """Try to find rank-r decomposition using ALS with half-integer rounding."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    best_solution = None
    best_err = float('inf')
    
    for init_seed in range(3):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(seed + init_seed)
        
        # Initialize with random values
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            R = T_true - T_recon
            
            # Check if we're close enough
            err = np.max(np.abs(R))
            if err < 1e-6:
                # Try rounding to half-integers
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
                
                # If rounding didn't work, continue with original
                if err < 1e-8:
                    # Keep original values if they work exactly
                    best_solution = (U, V, W)
                    best_err = err
            
            # ALS updates
            # Update U: for each a, solve (V*W^T) U[a,:]^T = T_true[a,:,:] weighted
            # Simpler: gradient-based update
            eta = 0.05
            
            grad_U = np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = np.einsum('abc,ar,br->cr', R, U, V)
            
            U += eta * grad_U
            V += eta * grad_V
            W += eta * grad_W
            
            # Periodically check and try rounding
            if it % 50 == 0 and err < 0.1:
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                
                T_recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.max(np.abs(T_recon_r - T_true))
                
                if err_r < 1e-8:
                    best_solution = (U_r, V_r, W_r)
                    best_err = err_r
                    return best_solution
    
    return best_solution

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _kron_decomposition():
    """
    Build a Kronecker-product-based decomposition:
      M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    where:
      M_{3,2,4}: schoolbook rank 24
      M_{1,2,2}: rank 4 (schoolbook)
    Total rank = 24 * 4 = 96 (same as schoolbook, but let's try to do better)
    
    Actually, let's try a different factorization.
    n=3, m=4, p=8
    
    Factor m=4 as 2*2, p=8 as 4*2, n=3 as 3*1
    So M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    M_{3,2,4}: rank 24
    M_{1,2,2}: rank 4
    Total: 96
    
    Alternative: try to use Strassen for some parts.
    M_{2,2,2} has Strassen rank 7.
    
    Let's try: M_{3,4,8} as combination.
    Split n=3 into 2+1.
    
    Actually, let's just build a structured decomposition using block matrices.
    We can use the fact that 4 = 2*2 and 8 = 2*2*2.
    
    Let's try a rank 88 decomposition using:
    M_{3,4,8} = M_{3,2,2} ⊗ M_{1,2,2} ⊗ M_{1,1,2}
    M_{3,2,2}: rank 11 (Strassen for 2x2x2 + schoolbook for 1x2x2)
    M_{1,2,2}: rank 4
    M_{1,1,2}: rank 2
    Total: 11 * 4 * 2 = 88
    """
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n  # 12, 32, 24
    
    # M_{3,2,2}: n1=3, m1=2, p1=2, rank r1=11
    # M_{1,2,2}: n2=1, m2=2, p2=2, rank r2=4
    # M_{1,1,2}: n3=1, m3=1, p3=2, rank r3=2
    # Total rank: 11 * 4 * 2 = 88
    
    n1, m1, p1 = 3, 2, 2
    n2, m2, p2 = 1, 2, 2
    n3, m3, p3 = 1, 1, 2
    
    r1, r2, r3 = 11, 4, 2
    r_total = r1 * r2 * r3  # 88
    
    # Build U1, V1, W1 for M_{3,2,2} (rank 11)
    # Strassen for i=0,1 (2x2x2, rank 7) + schoolbook for i=2 (1x2x2, rank 4)
    # Strassen 2x2x2 factors:
    U_str = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # row 0
        [0, 0, 1, 0, 0, 0, 0],  # row 1
    ])
    V_str = np.array([
        [0, 1, 0, 0, 0, 0, 0],  # col 0
        [0, 1, 1, 0, 0, 0, 0],  # col 1
    ])
    W_str = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # row 0
        [0, 0, 0, 1, 0, 0, 0],  # row 1
    ])
    
    # Wait, let me use the standard Strassen formulas.
    # For 2x2 matrix multiplication C = A*B:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor indices: U has shape (n*m, rank) = (4, 7) for 2x2x2
    # V has shape (m*p, rank) = (4, 7)
    # W has shape (p*n, rank) = (4, 7)
    
    # For 2x2x2: n=2, m=2, p=2
    # U indices: i*m + j = i*2 + j, i in {0,1}, j in {0,1}
    # V indices: j*p + k = j*2 + k, j in {0,1}, k in {0,1}
    # W indices: k*n + i = k*2 + i, k in {0,1}, i in {0,1}
    
    # Strassen factors (U is 4x7, V is 4x7, W is 4x7):
    # Column 0: M1 = (a+d)(e+h)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # (a+d) -> U: rows 0 (i=0,j=0) and 2 (i=1,j=0) both get 1
    # (e+h) -> V: cols 0 (j=0,k=0) and 2 (j=1,k=0) both get 1
    # Wait, V index is j*p+k. For (e+h): e is B[0,0] so j=0,k=0 -> idx 0. h is B[1,1] so j=1,k=1 -> idx 3.
    # So V[0,0]=1, V[3,0]=1
    # W index is k*n+i. For M1=(a+d)(e+h): this contributes to C[0,0] and C[1,1] etc.
    # Actually W for M1: it multiplies with U and V to give the product term.
    # The output C[i,k] gets contribution from M1 when appropriate.
    # C[0,0] = M1 + M4 - M5 + M7: W[0*2+0, 0] = 1 (k=0,i=0)
    # C[1,1] = M1 - M2 + M3 + M6: W[1*2+1, 0] = 1 (k=1,i=1)
    
    # This is getting complex. Let me just use the known Strassen rank-7 factors.
    # I'll hardcode the standard Strassen factors.
    
    # Standard Strassen 2x2x2 rank-7 decomposition:
    # U (4x7): rows are (i,j) pairs, cols are the 7 Strassen terms
    U_str = np.zeros((4, 7))
    V_str = np.zeros((4, 7))
    W_str = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U_str[0, 0] = 1  # a (i=0,j=0)
    U_str[2, 0] = 1  # d (i=1,j=0)  -- wait, d is A[1,1] so i=1,j=1 -> idx 3
    # Let me redo: A[i,j], i in {0,1}, j in {0,1}
    # a=A[0,0] -> U idx 0
    # b=A[0,1] -> U idx 1
    # c=A[1,0] -> U idx 2
    # d=A[1,1] -> U idx 3
    # e=B[0,0] -> V idx 0
    # f=B[0,1] -> V idx 1
    # g=B[1,0] -> V idx 2
    # h=B[1,1] -> V idx 3
    # C[i,k], i in {0,1}, k in {0,1}
    # C[0,0] -> W idx 0
    # C[0,1] -> W idx 1
    # C[1,0] -> W idx 2
    # C[1,1] -> W idx 3
    
    # M1 = (a+d)(e+h): U: a(0),d(3); V: e(0),h(3); W: C00(0),C11(3)
    U_str[0, 0] = 1; U_str[3, 0] = 1
    V_str[0, 0] = 1; V_str[3, 0] = 1
    W_str[0, 0] = 1; W_str[3, 0] = 1
    
    # M2 = (c+d)e: U: c(2),d(3); V: e(0); W: C21(2),C22(3)
    U_str[2, 1] = 1; U_str[3, 1] = 1
    V_str[0, 1] = 1
    W_str[2, 1] = 1; W_str[3, 1] = 1
    
    # M3 = a(f-h): U: a(0); V: f(1),h(3) with h negative; W: C12(1),C22(3)
    U_str[0, 2] = 1
    V_str[1, 2] = 1; V_str[3, 2] = -1
    W_str[1, 2] = 1; W_str[3, 2] = 1
    
    # M4 = d(g-e): U: d(3); V: g(2),e(0) with e negative; W: C11(3),C21(2)
    U_str[3, 3] = 1
    V_str[2, 3] = 1; V_str[0, 3] = -1
    W_str[3, 3] = 1; W_str[2, 3] = 1
    
    # M5 = (a+b)h: U: a(0),b(1); V: h(3); W: C11(3) negative, C12(1)
    U_str[0, 4] = 1; U_str[1, 4] = 1
    V_str[3, 4] = 1
    W_str[3, 4] = -1; W_str[1, 4] = 1
    
    # M6 = (c-d)(f+g): U: c(2),d(3) with d negative; V: f(1),g(2); W: C22(3)
    U_str[2, 5] = 1; U_str[3, 5] = -1
    V_str[1, 5] = 1; V_str[2, 5] = 1
    W_str[3, 5] = 1
    
    # M7 = (b-d)(e+h): U: b(1),d(3) with d negative; V: e(0),h(3); W: C11(3)
    U_str[1, 6] = 1; U_str[3, 6] = -1
    V_str[0, 6] = 1; V_str[3, 6] = 1
    W_str[3, 6] = 1
    
    # Build U1 (3x2x2, rank 11): n1=3, m1=2, p1=2
    # U1 shape: (n1*m1, r1) = (6, 11)
    # V1 shape: (m1*p1, r1) = (4, 11)
    # W1 shape: (p1*n1, r1) = (6, 11)
    
    U1 = np.zeros((6, 11))
    V1 = np.zeros((4, 11))
    W1 = np.zeros((6, 11))
    
    # Strassen part: i=0,1 (2 rows of A)
    # U for Strassen: rows 0-3 (i=0,1; j=0,1) -> U1 rows 0-3
    U1[0:4, 0:7] = U_str
    V1[0:4, 0:7] = V_str
    # W for Strassen: C[i,k], i in {0,1}, k in {0,1}
    # W1 index: k*n1 + i = k*3 + i
    # C[0,0] -> W1[0*3+0, :] = W1[0, 0:7]
    # C[0,1] -> W1[1*3+0, :] = W1[3, 0:7]
    # C[1,0] -> W1[0*3+1, :] = W1[1, 0:7]
    # C[1,1] -> W1[1*3+1, :] = W1[4, 0:7]
    W1[0, 0:7] = W_str[0, :]  # C[0,0]
    W1[3, 0:7] = W_str[1, :]  # C[0,1]
    W1[1, 0:7] = W_str[2, :]  # C[1,0]
    W1[4, 0:7] = W_str[3, :]  # C[1,1]
    
    # Schoolbook part: i=2 (1 row of A)
    # A[2,0], A[2,1] -> U1 rows 4,5
    # B[0,0], B[0,1], B[1,0], B[1,1] -> V1 rows 0-3
    # C[2,0], C[2,1] -> W1 rows 0*3+2=2, 1*3+2=5
    for j in range(2):
        for k in range(2):
            col = 7 + j*2 + k
            U1[4 + j, col] = 1  # A[2,j]
            V1[j*2 + k, col] = 1  # B[j,k]
            W1[k*3 + 2, col] = 1  # C[2,k]
    
    # M_{1,2,2}: n2=1, m2=2, p2=2, rank 4 (schoolbook)
    # U2 shape: (1*2, 4) = (2, 4)
    # V2 shape: (2*2, 4) = (4, 4)
    # W2 shape: (2*1, 4) = (2, 4)
    U2 = np.zeros((2, 4))
    V2 = np.zeros((4, 4))
    W2 = np.zeros((2, 4))
    for j in range(2):
        for k in range(2):
            col = j*2 + k
            U2[j, col] = 1
            V2[j*2 + k, col] = 1
            W2[k, col] = 1
    
    # M_{1,1,2}: n3=1, m3=1, p3=2, rank 2
    # U3 shape: (1*1, 2) = (1, 2)
    # V3 shape: (1*2, 2) = (2, 2)
    # W3 shape: (2*1, 2) = (2, 2)
    U3 = np.zeros((1, 2))
    V3 = np.zeros((2, 2))
    W3 = np.zeros((2, 2))
    for k in range(2):
        U3[0, k] = 1
        V3[k, k] = 1
        W3[k, k] = 1
    
    # Now combine using Kronecker product
    # Total: n=3, m=4, p=8
    # U_total shape: (n*m, r_total) = (12, 88)
    # V_total shape: (m*p, r_total) = (32, 88)
    # W_total shape: (p*n, r_total) = (24, 88)
    
    # Index mapping:
    # i = i1 * (n2*n3) + i2 * n3 + i3
    # j = j1 * (m2*m3) + j2 * m3 + j3
    # k = k1 * (p2*p3) + k2 * p3 + k3
    # r = r1 * (r2*r3) + r2 * r3 + r3
    
    n2n3 = n2 * n3  # 1
    n3_val = n3  # 1
    m2m3 = m2 * m3  # 2
    m3_val = m3  # 1
    p2p3 = p2 * p3  # 4
    p3_val = p3  # 2
    r2r3 = r2 * r3  # 8
    r3_val = r3  # 2
    
    U_total = np.zeros((nm, r_total))
    V_total = np.zeros((mp, r_total))
    W_total = np.zeros((pn, r_total))
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            for i3 in range(n3):
                                for j3 in range(m3):
                                    for k3 in range(p3):
                                        i = i1 * n2n3 + i2 * n3_val + i3
                                        j = j1 * m2m3 + j2 * m3_val + j3
                                        k = k1 * p2p3 + k2 * p3_val + k3
                                        
                                        for r1 in range(r1_val if False else 11):
                                            # Only contribute if U1[i1*m1+j1, r1] != 0
                                            pass
    
    # Let me just do it more directly using the Kronecker product structure
    # For each r1, r2, r3:
    r1_val, r2_val, r3_val = 11, 4, 2
    
    for r1 in range(r1_val):
        for r2 in range(r2_val):
            for r3 in range(r3_val):
                r = r1 * r2r3 + r2 * r3_val + r3
                for i1 in range(n1):
                    for j1 in range(m1):
                        u1_val = U1[i1*m1 + j1, r1]
                        if u1_val == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_val = U2[i2*m2 + j2, r2]
                                if u2_val == 0:
                                    continue
                                for i3 in range(n3):
                                    for j3 in range(m3):
                                        u3_val = U3[i3*m3 + j3, r3]
                                        if u3_val == 0:
                                            continue
                                        i = i1 * n2n3 + i2 * n3_val + i3
                                        j = j1 * m2m3 + j2 * m3_val + j3
                                        U_total[i*m + j, r] += u1_val * u2_val * u3_val
    
    for r1 in range(r1_val):
        for r2 in range(r2_val):
            for r3 in range(r3_val):
                r = r1 * r2r3 + r2 * r3_val + r3
                for j1 in range(m1):
                    for k1 in range(p1):
                        v1_val = V1[j1*p1 + k1, r1]
                        if v1_val == 0:
                            continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2_val = V2[j2*p2 + k2, r2]
                                if v2_val == 0:
                                    continue
                                for j3 in range(m3):
                                    for k3 in range(p3):
                                        v3_val = V3[j3*p3 + k3, r3]
                                        if v3_val == 0:
                                            continue
                                        j = j1 * m2m3 + j2 * m3_val + j3
                                        k = k1 * p2p3 + k2 * p3_val + k3
                                        V_total[j*p + k, r] += v1_val * v2_val * v3_val
    
    for r1 in range(r1_val):
        for r2 in range(r2_val):
            for r3 in range(r3_val):
                r = r1 * r2r3 + r2 * r3_val + r3
                for k1 in range(p1):
                    for i1 in range(n1):
                        w1_val = W1[k1*n1 + i1, r1]
                        if w1_val == 0:
                            continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                w2_val = W2[k2*n2 + i2, r2]
                                if w2_val == 0:
                                    continue
                                for k3 in range(p3):
                                    for i3 in range(n3):
                                        w3_val = W3[k3*n3 + i3, r3]
                                        if w3_val == 0:
                                            continue
                                        k = k1 * p2p3 + k2 * p3_val + k3
                                        i = i1 * n2n3 + i2 * n3_val + i3
                                        W_total[k*n + i, r] += w1_val * w2_val * w3_val
    
    # Verify
    T_true = _build_true_tensor(3, 4, 8)
    T_recon = np.einsum('ar,br,cr->abc', U_total, V_total, W_total)
    if not np.allclose(T_recon, T_true, atol=1e-8):
        # Fallback to schoolbook
        return _schoolbook(3, 4, 8)
    
    # Round to half-integers
    U_total = _round_to_half_integer(U_total)
    V_total = _round_to_half_integer(V_total)
    W_total = _round_to_half_integer(W_total)
    
    T_recon2 = np.einsum('ar,br,cr->abc', U_total, V_total, W_total)
    if not np.allclose(T_recon2, T_true, atol=1e-8):
        # Fallback to schoolbook
        return _schoolbook(3, 4, 8)
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    T_true = _build_true_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 96
    
    # Try Kronecker decomposition (rank 88)
    kron_result = _kron_decomposition()
    if kron_result is not None:
        U_k = np.array(kron_result["U"], dtype=float)
        V_k = np.array(kron_result["V"], dtype=float)
        W_k = np.array(kron_result["W"], dtype=float)
        T_recon_k = np.einsum('ar,br,cr->abc', U_k, V_k, W_k)
        if np.allclose(T_recon_k, T_true, atol=1e-8):
            best_result = kron_result
            best_rank = len(kron_result["U"][0]) if kron_result["U"] else 0
    
    # Try to find even lower rank with ALS
    lower_bound = max(nm, mp, pn)  # 32
    
    search_start = time.time()
    max_search_time = 100.0
    
    # Try a few candidate ranks below best_rank
    candidate_ranks = list(range(max(lower_bound, best_rank - 20), best_rank))
    
    for r in candidate_ranks:
        if time.time() - search_start > max_search_time:
            break
        
        if r >= best_rank:
            break
        
        result = _try_rank_als(r, n, m, p, T_true, 
                               max_iters=1000, seed=42, 
                               time_budget=10)
        
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = r
    
    return best_result
# EVOLVE-BLOCK-END
