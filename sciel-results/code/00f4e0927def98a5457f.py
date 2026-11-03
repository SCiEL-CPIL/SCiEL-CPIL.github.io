"""Matrix multiplication 4x5x6: search for rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - elapsed()

    n, m, p = DIMENSIONS
    N = n * m  # 20
    M = m * p  # 30
    P = p * n  # 24

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def to_lists(mat):
        return mat.tolist()

    def verify(result):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
            return False
        if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
            return False
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target = np.zeros((N, M, P))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        return np.allclose(T, target, atol=1e-7)

    # Build target tensor once
    target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Hybrid block decomposition:
    # Partition 4x5x6 into blocks:
    # - 2x2x2 core (rows 0-1, cols 0-1, out 0-1): use Strassen rank-7
    # - Remaining strips: use schoolbook
    
    # Strassen's 2x2x2 multiplication:
    # C = A * B where A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = M5 - M4 + M6 + M7
    # C[0,1] = M1 + M7
    # C[1,0] = M3 + M4
    # C[1,1] = M5 - M1 + M2 + M3
    # Where:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        # For 2x2 matrices: A (2x2), B (2x2), C = A*B (2x2)
        # Indices: a = i*m+j, b = j*p+k, c = k*n+i
        # For 2x2x2: m=2, p=2, n=2
        # a in [0,3], b in [0,3], c in [0,3]
        # a = i*2+j, b = j*2+k, c = k*2+i
        
        # Strassen has 7 multiplications
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # A[0,0] -> a=0, B[0,0] -> b=0
        # A[1,1] -> a=3, B[1,1] -> b=3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C[0,0] += M1, C[0,1] += M1, C[1,0] -= M1, C[1,1] -= M1
        # C[0,0]: c=0, C[0,1]: c=1, C[1,0]: c=2, C[1,1]: c=3
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = -1; W[3, 0] = -1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # A[1,0] -> a=2, A[1,1] -> a=3, B[0,0] -> b=0
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C[1,1] += M2
        W[3, 1] = 1
        
        # M3 = A[0,0] * (B[1,0]-B[1,1])
        # A[0,0] -> a=0, B[1,0] -> b=2, B[1,1] -> b=3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        # C[1,0] += M3, C[1,1] += M3
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A[1,1] * (B[0,1]-B[0,0])
        # A[1,1] -> a=3, B[0,1] -> b=1, B[0,0] -> b=0
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        # C[0,0] -= M4, C[1,0] += M4
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # A[0,0] -> a=0, A[0,1] -> a=1, B[1,1] -> b=3
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C[0,0] += M5, C[1,1] += M5
        W[0, 4] = 1; W[3, 4] = 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # A[1,0] -> a=2, A[0,0] -> a=0, B[0,0] -> b=0, B[0,1] -> b=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C[0,0] += M6
        W[0, 5] = 1
        
        # M7 = (A[1,1]-A[0,1]) * (B[1,0]+B[1,1])
        # A[1,1] -> a=3, A[0,1] -> a=1, B[1,0] -> b=2, B[1,1] -> b=3
        U[3, 6] = 1; U[1, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C[0,1] += M7, C[1,0] += M7
        W[1, 6] = 1; W[2, 6] = 1
        
        return U, V, W

    # Build hybrid decomposition:
    # Core: rows 0-1 of A (i=0,1), cols 0-1 of B (j=0,1), out 0-1 (k=0,1)
    # This corresponds to:
    # A indices: a = i*5+j for i in [0,1], j in [0,1] -> a in [0,1,5,6]
    # B indices: b = j*6+k for j in [0,1], k in [0,1] -> b in [0,1,6,7]
    # C indices: c = k*4+i for k in [0,1], i in [0,1] -> c in [0,1,4,5]
    
    # Strassen operates on 2x2 blocks, so we need to map:
    # Local A indices (i,j) with i,j in [0,1] -> global a = i*5+j
    # Local B indices (j,k) with j,k in [0,1] -> global b = j*6+k
    # Local C indices (k,i) with k,i in [0,1] -> global c = k*4+i
    
    def build_hybrid():
        """Build hybrid decomposition with Strassen core + schoolbook strips."""
        # Strassen core: 2x2x2
        U_core, V_core, W_core = strassen_2x2x2()
        
        # Map local indices to global
        # Local a: 0->0, 1->1, 2->5, 3->6 (i*5+j)
        # Local b: 0->0, 1->1, 2->6, 3->7 (j*6+k)
        # Local c: 0->0, 1->1, 2->4, 3->5 (k*4+i)
        
        a_map = [0, 1, 5, 6]
        b_map = [0, 1, 6, 7]
        c_map = [0, 1, 4, 5]
        
        # Strassen uses 7 terms
        rank_strassen = 7
        
        # Schoolbook for remaining:
        # All (i,j,k) triples except those in the 2x2x2 core
        # Core: i in [0,1], j in [0,1], k in [0,1]
        # Total schoolbook: 4*5*6 = 120
        # Core schoolbook: 2*2*2 = 8
        # Remaining: 120 - 8 = 112
        
        # But we replace core schoolbook (8 terms) with Strassen (7 terms)
        # So total: 112 + 7 = 119
        
        total_rank = 119
        U = np.zeros((N, total_rank))
        V = np.zeros((M, total_rank))
        W = np.zeros((P, total_rank))
        
        # Place Strassen terms
        for r in range(rank_strassen):
            for local_a, global_a in enumerate(a_map):
                U[global_a, r] = U_core[local_a, r]
            for local_b, global_b in enumerate(b_map):
                V[global_b, r] = V_core[local_b, r]
            for local_c, global_c in enumerate(c_map):
                W[global_c, r] = W_core[local_c, r]
        
        # Place schoolbook terms for non-core
        col = rank_strassen
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if in core
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        assert col == total_rank, f"Expected {total_rank}, got {col}"
        
        return {"U": U, "V": V, "W": W}

    # Try hybrid decomposition
    if time_remaining() > 5:
        hybrid = build_hybrid()
        if verify(hybrid):
            best = hybrid
            best_rank = 119
            print(f"Found hybrid decomposition with rank {best_rank}")
        else:
            print("Hybrid decomposition failed verification")
    
    # Try to find even lower rank using ALS
    def als_step_fix_U(U, V, W, target, max_iter=20):
        """Fix U, update V and W alternately."""
        V = V.copy()
        W = W.copy()
        for _ in range(max_iter):
            if time_remaining() < 10:
                break
            # Fix V and W, update U
            H_U = np.zeros((V.shape[1], M * P))
            for r in range(V.shape[1]):
                H_U[r, :] = np.kron(V[:, r], W[:, r])
            target_reshaped = target.reshape(N, M * P)
            try:
                U_new, _, _, _ = np.linalg.lstsq(H_U.T, target_reshaped, rcond=None)
                U = U_new
            except:
                pass
            
            # Fix U and W, update V
            H_V = np.zeros((V.shape[1], N * P))
            for r in range(V.shape[1]):
                H_V[r, :] = np.kron(U[:, r], W[:, r])
            target_v_reshaped = target.transpose(1, 0, 2).reshape(M, N * P)
            try:
                V_new, _, _, _ = np.linalg.lstsq(H_V.T, target_v_reshaped, rcond=None)
                V = V_new
            except:
                pass
            
            # Fix U and V, update W
            H_W = np.zeros((W.shape[1], N * M))
            for r in range(W.shape[1]):
                H_W[r, :] = np.kron(U[:, r], V[:, r])
            target_w_reshaped = target.transpose(2, 0, 1).reshape(P, N * M)
            try:
                W_new, _, _, _ = np.linalg.lstsq(H_W.T, target_w_reshaped, rcond=None)
                W = W_new
            except:
                pass
        
        return U, V, W

    def try_rank(r, num_trials=3):
        """Try to find a decomposition of rank r."""
        if time_remaining() < 20:
            return None
        for trial in range(num_trials):
            if time_remaining() < 15:
                break
            # Random initialization
            U = np.random.randn(N, r) * 0.1
            V = np.random.randn(M, r) * 0.1
            W = np.random.randn(P, r) * 0.1
            
            # Run ALS
            U, V, W = als_step_fix_U(U, V, W, target, max_iter=30)
            
            # Check if we found a valid decomposition
            result = {"U": U, "V": V, "W": W}
            if verify(result):
                return result
        return None

    # Try ranks below current best
    for r in range(best_rank - 1, max(best_rank - 15, 100), -1):
        if time_remaining() < 30:
            break
        result = try_rank(r, num_trials=2)
        if result is not None:
            best = result
            best_rank = r
            print(f"Found decomposition with rank {r}")
            continue
    
    # Final verification
    if not verify(best):
        # Fall back to schoolbook
        best = schoolbook()
        best_rank = n * m * p

    return {"U": to_lists(best["U"]), "V": to_lists(best["V"]), "W": to_lists(best["W"])}
# EVOLVE-BLOCK-END
