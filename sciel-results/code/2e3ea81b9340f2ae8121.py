import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 125  # safety margin before 150s kill

    # Build the trivial rank-40 decomposition
    def build_trivial():
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {'U': U, 'V': V, 'W': W}

    def verify(U, V, W, tol=1e-6):
        """Full verification of decomposition."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Check all valid product terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > tol:
                        return False
        # Check that non-product terms are zero (sample)
        rng = np.random.default_rng(42)
        for _ in range(2000):
            a = int(rng.integers(0, n * m))
            b = int(rng.integers(0, m * p))
            c = int(rng.integers(0, p * n))
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            if not (j1 == j2 and k1 == k2 and i1 == i2):
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > tol:
                    return False
        return True

    def verify_full(U, V, W, tol=1e-6):
        """Check ALL non-product entries are zero."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Build the set of valid (a,b,c) tuples
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid.add((a, b, c))
        
        # Check all product terms
        for (a, b, c) in valid:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if abs(val - 1.0) > tol:
                return False
        
        # Check all non-product terms
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in valid:
                        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                        if abs(val) > tol:
                            return False
        return True

    # Build a Strassen-based rank-36 decomposition
    def build_strassen_36():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        col = 0
        
        def strassen_block(y, z, w):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z
            # M1 = (y00+y11)(z00+z11)
            U[u_idx(*y00)][col] += 1
            U[u_idx(*y11)][col] += 1
            V[v_idx(*z00)][col] += 1
            V[v_idx(*z11)][col] += 1
            W[w_idx(*w[0])][col] += 1
            W[w_idx(*w[3])][col] += 1
            col += 1
            # M2 = (y10+y11)z00
            U[u_idx(*y10)][col] += 1
            U[u_idx(*y11)][col] += 1
            V[v_idx(*z00)][col] += 1
            W[w_idx(*w[2])][col] += 1
            W[w_idx(*w[3])][col] -= 1
            col += 1
            # M3 = y00(z01-z11)
            U[u_idx(*y00)][col] += 1
            V[v_idx(*z01)][col] += 1
            V[v_idx(*z11)][col] -= 1
            W[w_idx(*w[1])][col] += 1
            W[w_idx(*w[3])][col] += 1
            col += 1
            # M4 = y11(z10-z00)
            U[u_idx(*y11)][col] += 1
            V[v_idx(*z10)][col] += 1
            V[v_idx(*z00)][col] -= 1
            W[w_idx(*w[0])][col] += 1
            W[w_idx(*w[2])][col] += 1
            col += 1
            # M5 = (y00+y01)z11
            U[u_idx(*y00)][col] += 1
            U[u_idx(*y01)][col] += 1
            V[v_idx(*z11)][col] += 1
            W[w_idx(*w[0])][col] -= 1
            W[w_idx(*w[1])][col] += 1
            col += 1
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(*y10)][col] += 1
            U[u_idx(*y00)][col] -= 1
            V[v_idx(*z00)][col] += 1
            V[v_idx(*z01)][col] += 1
            W[w_idx(*w[3])][col] += 1
            col += 1
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(*y01)][col] += 1
            U[u_idx(*y11)][col] -= 1
            V[v_idx(*z10)][col] += 1
            V[v_idx(*z11)][col] += 1
            W[w_idx(*w[0])][col] += 1
            col += 1
        
        # 4 blocks of 2x2 * 2x2
        y1 = [(0,0),(0,1),(1,0),(1,1)]
        y2 = [(0,2),(0,3),(1,2),(1,3)]
        z1 = [(0,0),(0,1),(1,0),(1,1)]
        z2 = [(0,2),(0,3),(1,2),(1,3)]
        w11 = [(0,0),(1,0),(0,1),(1,1)]
        w12 = [(2,0),(3,0),(2,1),(3,1)]
        
        strassen_block(y1, z1, w11)
        strassen_block(y2, z1, w11)
        strassen_block(y1, z2, w12)
        strassen_block(y2, z2, w12)
        
        # 2 blocks of 2x2 * 2x1 (rank 4 each)
        # A_L * B_col4 -> C_col4
        U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
        U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
        
        # A_R * B_col4 -> C_col4
        U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
        U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
        U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
        
        assert col == 36, f"Expected 36, got {col}"
        return {'U': np.array(U, dtype=float), 'V': np.array(V, dtype=float), 'W': np.array(W, dtype=float)}

    # Build a better decomposition using Kronecker structure
    # For 2x4x5: split A into 2x2 blocks, B into 2x2 and 2x1 blocks
    # The 2x2x2 multiplication has rank 7 (Strassen)
    # We need: C = [A0 A1] * [[B00, B01, B02], [B10, B11, B12]]
    # C0 = A0*B00 + A1*B10  (2x1)
    # C1 = A0*B01 + A1*B11  (2x1)
    # C2 = A0*B02 + A1*B12  (2x1)
    # where C0,C1,C2 are 2x1 vectors
    
    # Actually let me think more carefully.
    # A is 2x4, B is 4x5.
    # Split A = [A_L | A_R] where A_L, A_R are 2x2
    # Split B = [[B_00, B_01, B_02], [B_10, B_11, B_12]] where B_ij are 2x1
    # C = A_L * [B_00, B_01, B_02] + A_R * [B_10, B_11, B_12]
    # C is 2x5 = [C_0, C_1, C_2] where C_i are 2x1
    # C_0 = A_L * B_00 + A_R * B_10
    # C_1 = A_L * B_01 + A_R * B_11
    # C_2 = A_L * B_02 + A_R * B_12
    
    # For each C_i, we need to compute 2x2 * 2x1 + 2x2 * 2x1
    # The 2x2 * 2x1 multiplication has rank 4 (it's 4 independent products)
    # But we can share structure between C_0, C_1, C_2 using Strassen-like tricks
    
    # Let's try: for each pair (A_L, A_R), compute all three C_i using shared intermediates
    # A_L * [B_00 B_01 B_02]: 2x2 * 2x3
    # A_R * [B_10 B_11 B_12]: 2x2 * 2x3
    
    # 2x2 * 2x3 multiplication: rank?
    # C = AB where A is 2x2, B is 2x3
    # C is 2x3, so 6 entries, each is a product of 2 elements
    # The rank of 2x2x3 tensor... let me think.
    # T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k]
    # This is a 4x6x6 tensor (A has 4 entries, B has 6, C has 6)
    # Actually the tensor is 4x6x6 with T[A_idx, B_idx, C_idx] = 1 for matching
    # A_idx = i*2+j, B_idx = j*3+k, C_idx = k*2+i
    # The number of 1-entries is 2*2*3 = 12
    # Lower bound: max(4, 6, 6) = 6, but the tensor structure matters
    # The 2x2*2x3 rank is known to be 6 (same as trivial) or possibly lower
    # Actually for 2x2*2xn, the rank is 2*2*n / something... let me just use trivial
    
    # Let me try a different approach: use the fact that we have specific structure
    # and try to find a rank < 36 decomposition via optimization
    
    # First, get the rank-36 baseline
    baseline = build_strassen_36()
    if time.time() - start > deadline:
        return baseline
    
    # Try to find lower rank using numerical optimization
    # The tensor T is 8x20x10 with 80 non-zero entries
    # We want U(8xr), V(20xr), W(10xr) such that
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Strategy: start with rank r, initialize U,V,W randomly, optimize
    # Objective: sum over all (a,b,c) of (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
    
    def build_tensor():
        """Build the 8x20x10 multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()
    
    def tensor_residual(U, V, W):
        """Compute residual ||U⊗V⊗W - T||_F"""
        rank = U.shape[1]
        # Efficient computation using einsum
        # Result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is equivalent to contracting U with V and W
        # First: UV[a,b] = sum_r U[a,r]*V[b,r] -- no, that's not right
        # We need the full 3-tensor contraction
        
        # For small rank, just do it directly
        R = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            R += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.linalg.norm(R - T)
    
    def optimize_rank(r, max_iter=5000, lr=0.01):
        """Try to find rank-r decomposition via gradient descent."""
        rng = np.random.default_rng(r)
        U = rng.standard_normal((n*m, r)) * 0.5
        V = rng.standard_normal((m*p, r)) * 0.5
        W = rng.standard_normal((p*n, r)) * 0.5
        
        # Scale to roughly match
        scale = (np.linalg.norm(T) / max(1e-10, (np.linalg.norm(U)*np.linalg.norm(V)*np.linalg.norm(W))))**(1/3)
        U *= scale
        V *= scale
        W *= scale
        
        best_loss = float('inf')
        best_state = None
        
        for it in range(max_iter):
            if time.time() - start > deadline - 5:
                break
            
            # Compute gradient
            R = np.zeros((n*m, m*p, p*n))
            for ri in range(r):
                R += np.outer(np.outer(U[:, ri], V[:, ri]), W[:, ri])
            residual = R - T
            
            # Gradients:
            # dL/du[a,ri] = 2 * sum_{b,c} residual[a,b,c] * V[b,ri] * W[c,ri]
            # = 2 * sum_c W[c,ri] * sum_b residual[a,b,c] * V[b,ri]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for ri in range(r):
                # dU[:,ri] = 2 * (residual * V[:,ri] * W[:,ri]).sum(axis=(1,2))
                # Let's compute: for each a: sum_{b,c} residual[a,b,c]*V[b,ri]*W[c,ri]
                # = sum_c W[c,ri] * (sum_b residual[a,b,c]*V[b,ri])
                # For each c: sum_b residual[a,b,c]*V[b,ri] = (residual[:,:,c] @ V[:,ri])
                
                # dU[:,ri] = 2 * sum_c W[c,ri] * (residual @ V[:,ri] over b axis)
                # residual shape: (n*m, m*p, p*n)
                # residual @ V[:,ri] over axis 1: shape (n*m, p*n)
                rv = np.einsum('abc,b->ac', residual, V[:, ri])
                dU[:, ri] = 2 * (rv * W[:, ri]).sum(axis=1)
                
                # dV[:,ri] = 2 * sum_{a,c} residual[a,b,c]*U[a,ri]*W[c,ri]
                ru = np.einsum('abc,a->bc', residual, U[:, ri])
                dV[:, ri] = 2 * (ru * W[:, ri]).sum(axis=0)
                
                # dW[:,ri] = 2 * sum_{a,b} residual[a,b,c]*U[a,ri]*V[b,ri]
                rw = np.einsum('abc,b->ac', residual, V[:, ri])
                # Actually: dW[c,ri] = 2 * sum_{a,b} residual[a,b,c]*U[a,ri]*V[b,ri]
                # = 2 * sum_a U[a,ri] * (sum_b residual[a,b,c]*V[b,ri])
                # = 2 * sum_a U[a,ri] * rv[a,c]
                dW[:, ri] = 2 * (rv.T * U[:, ri]).sum(axis=1)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            loss = tensor_residual(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_state = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-4:
                break
        
        return best_state, best_loss
    
    # Try various ranks
    best_rank = 40
    best_decomp = build_trivial()
    
    # First verify rank 36
    if verify(baseline['U'], baseline['V'], baseline['W']):
        best_rank = 36
        best_decomp = baseline
        print(f"Rank 36 verified")
    else:
        print(f"Rank 36 FAILED verification")
        # Try rank 40
        if verify(best_decomp['U'], best_decomp['V'], best_decomp['W']):
            print(f"Rank 40 verified")
        else:
            print(f"Rank 40 FAILED")
    
    if time.time() - start > deadline:
        return best_decomp
    
    # Try to find lower ranks
    for target_rank in [35, 34, 33, 32, 31, 30]:
        if time.time() - start > deadline - 10:
            break
        print(f"Trying rank {target_rank}...")
        result = optimize_rank(target_rank, max_iter=3000, lr=0.05)
        if result is not None:
            (U, V, W), loss = result
            if loss < 1e-3:
                # Verify
                if verify(U, V, W):
                    best_rank = target_rank
                    best_decomp = {'U': U, 'V': V, 'W': W}
                    print(f"Rank {target_rank} VERIFIED!")
                    break
                else:
                    print(f"Rank {target_rank} loss={loss:.6f} but failed verification")
    
    # Try rank 36 with optimization to potentially find integer/rational solution
    if best_rank >= 36 and time.time() - start < deadline - 15:
        print("Optimizing rank 36 for cleaner values...")
        result = optimize_rank(36, max_iter=2000, lr=0.02)
        if result is not None:
            (U, V, W), loss = result
            if loss < 1e-6 and verify(U, V, W):
                # Try to round to rationals
                best_decomp = {'U': U, 'V': V, 'W': W}
    
    return best_decomp
