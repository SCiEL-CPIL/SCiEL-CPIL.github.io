import time
import numpy as np
import itertools

DIMENSIONS = [2, 2, 4]


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition satisfies sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_2x2x2(u, v, w):
    """Verify rank-7 decomposition of 2x2x2 tensor: sum_r u[r][i,j]*v[r][j,k]*w[r][k,i] = 1 for all i,j,k in {0,1}.
    u, v, w are lists of 7 vectors, each of length 4 (flattened 2x2).
    """
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += u[r][i * 2 + j] * v[r][j * 2 + k] * w[r][k * 2 + i]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _find_rank7_2x2x2(deadline):
    """
    Find a rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    Uses a structured search: fix the U matrix to a known basis-like structure
    and solve for V and W, or use a randomized search over small integer coefficients.
    
    Strategy: The 2x2x2 tensor T[i,j,k] = 1 for all i,j,k.
    We need 7 triples (u_r, v_r, w_r) with u_r, v_r, w_r in Q^4 (flattened 2x2)
    such that sum_r u_r ⊗ v_r ⊗ w_r = T.
    
    We'll use a randomized approach: generate random u_r, v_r, w_r with entries
    from {-1, 0, 1} or {-2,-1,0,1,2} and check if they sum to the tensor.
    But 84 variables is too many for pure random. Instead, we use a known
    algebraic structure.
    
    Better approach: Use the fact that the 2x2x2 tensor has a known rank-7
    decomposition. We'll hardcode a CORRECT one after verifying it computationally
    by trying a small set of candidate decompositions.
    
    Actually, let me use a smarter approach: solve the linear system.
    Fix 7 u_r vectors (randomly), then the system sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    is bilinear in (v, w). This is hard.
    
    Alternative: Use the known Strassen decomposition but verify it computationally
    before using it. Let me try several candidate decompositions and verify.
    """
    # Candidate 1: The standard Strassen decomposition
    # Let me define it carefully and verify.
    # 
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)(B00)
    # P3 = (A00)(B10-B11)
    # P4 = (A11)(B01-B00)
    # P5 = (A00+A01)(B11)
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    #
    # C00 = P1 - P5 + P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    #
    # Let me verify this with a random matrix:
    import random
    random.seed(42)
    
    # Define the 7 products as (u_vec, v_vec, w_vec) where:
    # u_vec[i*2+j] = coefficient of A[i,j]
    # v_vec[j*2+k] = coefficient of B[j,k]  
    # w_vec[k*2+i] = coefficient of C[k,i] (note: output index is C[i,j] = sum_k A[i,k]B[k,j])
    # Wait, the tensor is T[i,j,k] where i=A-row, j=A-col/B-row, k=B-col.
    # T[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k].
    # So u_r[i,j] * v_r[j,k] * w_r[k,i] should sum to 1 for each (i,j,k).
    # Actually, the standard tensor decomposition is:
    # T = sum_r u_r ⊗ v_r ⊗ w_r where T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[i,k]
    # But the problem says W[k*n+i, r] so w_r[k,i] and the product is u[i,j]*v[j,k]*w[k,i].
    # This means w_r is indexed by (k,i) and the output is C[i,k] = sum_j A[i,j]*B[j,k].
    # So w_r[k,i] is the coefficient for output C[i,k] from product r.
    
    # Let me redefine: for each product r,
    # u_r[i*2+j] = coeff of A[i,j]
    # v_r[j*2+k] = coeff of B[j,k]
    # w_r[k*2+i] = coeff of output C[i,k]
    
    # P1 = (A00+A11)(B00+B11)
    # u1: A00=1, A11=1 -> [1,0,0,1]
    # v1: B00=1, B11=1 -> [1,0,0,1]
    # w1: C00=1, C11=1 -> w[k*2+i]: C00 is i=0,k=0 -> w[0]=1; C11 is i=1,k=1 -> w[3]=1
    # w1 = [1,0,0,1]
    
    # P2 = (A10+A11)(B00)
    # u2: A10=1, A11=1 -> [0,0,1,1]
    # v2: B00=1 -> [1,0,0,0]
    # w2: C10=1 -> i=1,k=0 -> w[0*2+1]=w[1]=1
    # w2 = [0,1,0,0]
    
    # P3 = (A00)(B10-B11)
    # u3: A00=1 -> [1,0,0,0]
    # v3: B10=1, B11=-1 -> [0,0,1,-1]
    # w3: C01=1, C11=1 -> C01: i=0,k=1 -> w[1*2+0]=w[2]=1; C11: i=1,k=1 -> w[1*2+1]=w[3]=1
    # w3 = [0,0,1,1]
    
    # P4 = (A11)(B01-B00)
    # u4: A11=1 -> [0,0,0,1]
    # v4: B01=1, B00=-1 -> [-1,1,0,0]
    # w4: C00=1, C10=1 -> C00: i=0,k=0 -> w[0]=1; C10: i=1,k=0 -> w[1]=1
    # w4 = [1,1,0,0]
    
    # P5 = (A00+A01)(B11)
    # u5: A00=1, A01=1 -> [1,1,0,0]
    # v5: B11=1 -> [0,0,0,1]
    # w5: C00=-1, C01=1 -> C00: i=0,k=0 -> w[0]=-1; C01: i=0,k=1 -> w[2]=1
    # w5 = [-1,0,1,0]
    
    # P6 = (A10-A00)(B00+B10)
    # u6: A10=1, A00=-1 -> [-1,0,1,0]
    # v6: B00=1, B10=1 -> [1,0,1,0]
    # w6: Not used in output formulas above. Let me check if this is correct.
    # If P6 is not used, then it shouldn't be in the decomposition. But Strassen has 7 products.
    # Let me check: maybe P6 IS used and I have the wrong output formula.
    
    # P7 = (A01-A11)(B01+B11)
    # u7: A01=1, A11=-1 -> [0,1,0,-1]
    # v7: B01=1, B11=1 -> [0,1,0,1]
    # w7: C00=1, C11=1 -> w[0]=1, w[3]=1
    # w7 = [1,0,0,1]
    
    # Now let me verify: for each (i,j,k), sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] should = 1.
    # Let me test (i=0, j=0, k=0):
    # r=0 (P1): u[0,0]=1, v[0,0]=1, w[0,0]=1 -> 1
    # r=1 (P2): u[0,0]=0, v[0,0]=1, w[0,1]=1 -> 0
    # r=2 (P3): u[0,0]=1, v[0,0]=0, w[0,0]=0 -> 0
    # r=3 (P4): u[0,0]=0, v[0,0]=-1, w[0,0]=1 -> 0
    # r=4 (P5): u[0,0]=1, v[0,0]=0, w[0,0]=-1 -> 0
    # r=5 (P6): u[0,0]=-1, v[0,0]=1, w[0,0]=? (w6 not defined)
    # r=6 (P7): u[0,0]=0, v[0,0]=0, w[0,0]=1 -> 0
    # Sum so far: 1. If w6[0,0]=0, sum=1. Good.
    
    # Let me test (i=0, j=1, k=0):
    # r=0 (P1): u[0,1]=0, v[1,0]=0, w[0,0]=1 -> 0
    # r=1 (P2): u[0,1]=0, v[1,0]=0, w[0,1]=1 -> 0
    # r=2 (P3): u[0,1]=0, v[1,0]=1, w[0,0]=0 -> 0
    # r=3 (P4): u[0,1]=0, v[1,0]=0, w[0,0]=1 -> 0
    # r=4 (P5): u[0,1]=1, v[1,0]=0, w[0,0]=-1 -> 0
    # r=5 (P6): u[0,1]=0, v[1,0]=1, w[0,0]=? -> 0 (if u[0,1]=0)
    # r=6 (P7): u[0,1]=1, v[1,0]=0, w[0,0]=1 -> 0
    # Sum: 0. But should be 1! ERROR.
    
    # So my decomposition is wrong. The issue is that I'm not accounting for all terms correctly.
    # Let me try a different approach: use a computational search.
    
    # Since pure random search over 84 variables is infeasible, let me use a
    # structured approach: I'll try to find a decomposition by solving a system
    # of equations using a greedy or local search method.
    
    # Actually, let me just try a few known candidate decompositions and verify.
    # If none work, I'll fall back to rank-16.
    
    candidates = []
    
    # Candidate: Try a decomposition based on the "Kronecker" structure
    # where we use the 8 trivial terms and then merge 2 pairs.
    
    # For now, let me try a randomized local search approach:
    # Start with random u, v, w and try to minimize the error.
    
    np.random.seed(42)
    best_error = float('inf')
    best_sol = None
    
    # Try a small number of random initializations
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Generate random u, v, w with entries in {-1, 0, 1}
        u = [[np.random.randint(-1, 2) for _ in range(4)] for _ in range(7)]
        v = [[np.random.randint(-1, 2) for _ in range(4)] for _ in range(7)]
        w = [[np.random.randint(-1, 2) for _ in range(4)] for _ in range(7)]
        
        # Check if this is a valid decomposition
        if _verify_2x2x2(u, v, w):
            return u, v, w
        
        # Compute error
        error = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(7):
                        s += u[r][i*2+j] * v[r][j*2+k] * w[r][k*2+i]
                    error += abs(s - 1.0)
        
        if error < best_error:
            best_error = error
            best_sol = (u, v, w)
    
    # If we found a solution with small error, try to refine it
    if best_sol is not None and best_error < 0.1:
        u, v, w = best_sol
        # Try local search: perturb one entry at a time
        for step in range(1000):
            if time.time() > deadline:
                break
            if _verify_2x2x2(u, v, w):
                return u, v, w
            
            # Try perturbing a random entry
            mat = np.random.choice([u, v, w])
            r = np.random.randint(0, 7)
            idx = np.random.randint(0, 4)
            old_val = mat[r][idx]
            new_val = old_val + np.random.choice([-1, 1])
            mat[r][idx] = new_val
            
            if _verify_2x2x2(u, v, w):
                return u, v, w
            
            # If error increased, revert
            error_new = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        s = 0.0
                        for r in range(7):
                            s += u[r][i*2+j] * v[r][j*2+k] * w[r][k*2+i]
                        error_new += abs(s - 1.0)
            
            if error_new > best_error:
                mat[r][idx] = old_val
            else:
                best_error = error_new
                best_sol = (u, v, w)
    
    return None


def _trivial_rank16():
    """Return the trivial rank-16 decomposition."""
    n, m, p = 2, 2, 4
    rank = n * m * p  # 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _build_rank14_from_rank7(u7, v7, w7):
    """
    Build a rank-14 decomposition of 2x2x2x4 from a rank-7 decomposition of 2x2x2x2.
    The 2x2x2x4 tensor is the direct sum of two 2x2x2x2 tensors (B split into cols 0-1 and 2-3).
    
    U (4x14): For each of the 7 products in block 0 and block 1, U is the same (A is shared).
    V (8x14): For block 0 (k=0,1), V[j*4+k] = v7[r][j*2+k_local]. For block 1 (k=2,3), V[j*4+k] = v7[r][j*2+k_local].
    W (8x14): For block 0, W[k*2+i] = w7[r][k*2+i]. For block 1, W[k*2+i] = w7[r][k*2+i].
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0.0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 14
    
    for block in range(2):
        r_offset = block * 7
        k_offset = block * 2
        
        for r_local in range(7):
            r = r_offset + r_local
            
            # U is the same for both blocks (A is shared)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][r] = float(u7[r_local][i * 2 + j])
            
            # V: for block b, k_global = k_offset + k_local
            for j in range(2):
                for k_local in range(2):
                    k_global = k_offset + k_local
                    V[j * p + k_global][r] = float(v7[r_local][j * 2 + k_local])
            
            # W: for block b, k_global = k_offset + k_local, output is C[i, k_global]
            for k_local in range(2):
                for i in range(2):
                    k_global = k_offset + k_local
                    W[k_global * n + i][r] = float(w7[r_local][k_local * 2 + i])
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # Try to find a rank-7 decomposition of 2x2x2x2
    result = _find_rank7_2x2x2(deadline)
    if result is not None:
        u7, v7, w7 = result
        U, V, W = _build_rank14_from_rank7(u7, v7, w7)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: trivial rank-16
    U, V, W = _trivial_rank16()
    return {"U": U, "V": V, "W": W}
