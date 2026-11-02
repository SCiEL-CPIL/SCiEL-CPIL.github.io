"""
Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.

Strategy:
1. Build the schoolbook rank-96 decomposition as a baseline.
2. Try to find a lower-rank decomposition by exploiting structure.
   - The key observation: M_{3,4,8} can be viewed as M_{3,4,4} ⊗ M_{1,1,2}
   - If we can find a rank-r decomposition of M_{3,4,4}, and a rank-s decomposition
     of M_{1,1,2}, then we get a rank r*s decomposition of M_{3,4,8}.
   - M_{1,1,2} has rank 2 (schoolbook), so this gives rank 2*r.
   - But we can also try direct approaches.
3. Use a systematic search: for each candidate rank r from 95 down, try to solve
   the linear system with random initializations.

The tensor T of shape (n*m, m*p, p*n) = (12, 32, 24) has entries:
T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)

We seek U (12 x r), V (32 x r), W (24 x r) such that:
sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, T, atol=1e-6)

    def build_schoolbook():
        """Standard n*m*p rank decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # Baseline
    U_sb, V_sb, W_sb = build_schoolbook()
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p

    # Strategy 1: Tensor product decomposition
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # If rank(M_{3,4,4}) = r344 and rank(M_{1,1,2}) = r112,
    # then rank(M_{3,4,8}) <= r344 * r112
    # Schoolbook: r344 = 48, r112 = 2, total = 96
    # Can we do better?
    
    # For M_{1,1,2}: this is just scalar multiplication with a 2-element vector
    # A (1x1) * B (1x2) = C (1x2)
    # C[0,k] = A[0,0] * B[0,k] for k=0,1
    # Schoolbook rank = 2
    # This is already minimal (you need at least 2 multiplications)
    
    # For M_{3,4,4}: A (3x4) * B (4x4) = C (3x4)
    # Schoolbook rank = 48
    # Can we find a lower rank?
    
    # Let's try to find a rank-r decomposition of M_{3,4,4}
    # T344 of shape (12, 16, 12)
    T344 = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                T344[i*m + j, j*4 + k, k*n + i] = 1.0

    def check_decomposition_344(U, V, W):
        """Check if U, V, W exactly decompose T344."""
        result = np.zeros((n*m, m*4, 4*n), dtype=np.float64)
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, T344, atol=1e-6)

    def build_schoolbook_344():
        """Standard 3*4*4 rank decomposition."""
        rank = n * m * 4
        U = np.zeros((n*m, rank))
        V = np.zeros((m*4, rank))
        W = np.zeros((4*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(4):
                    U[i*m + j, col] = 1
                    V[j*4 + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a lower-rank decomposition of M_{3,4,4}
    U344_sb, V344_sb, W344_sb = build_schoolbook_344()
    best_344 = (U344_sb, V344_sb, W344_sb)
    best_rank_344 = n * m * 4

    # Random search for M_{3,4,4}
    np.random.seed(42)
    
    def try_rank_344(r, num_trials=200):
        """Try to find a rank-r decomposition of M_{3,4,4}."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Use values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, r))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*4, r))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(4*n, r))
            
            if check_decomposition_344(U, V, W):
                return (U, V, W)
        
        return None

    # Try ranks from 47 down to 20
    for r in range(47, 19, -1):
        if time.time() > deadline:
            break
        if r < best_rank_344:
            result = try_rank_344(r, num_trials=100)
            if result is not None:
                best_rank_344 = r
                best_344 = result
                break

    # If we found a better 344 decomposition, combine with 112
    if best_rank_344 < n * m * 4:
        U344, V344, W344 = best_344
        r344 = U344.shape[1]
        
        # M_{1,1,2} schoolbook: rank 2
        # U112 = [[1], [1]], V112 = [[1, 1]], W112 = [[1], [1]]
        # Wait, for M_{1,1,2}: A(1x1), B(1x2), C(1x2)
        # C[0,k] = A[0,0] * B[0,k]
        # U: (1*1, rank) = (1, 2): U[0,0]=1, U[0,1]=1
        # V: (1*2, rank) = (2, 2): V[0,0]=1, V[1,1]=1
        # W: (2*1, rank) = (2, 2): W[0,0]=1, W[1,1]=1
        U112 = np.array([[1.0, 1.0]])
        V112 = np.array([[1.0, 0.0], [0.0, 1.0]])
        W112 = np.array([[1.0, 0.0], [0.0, 1.0]])
        r112 = 2
        
        # Combine: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
        # The combined rank is r344 * r112
        # U_combined: (n*m, r344*r112)
        # V_combined: (m*p, r344*r112) where p=8, so V has shape (32, r344*2)
        # W_combined: (p*n, r344*r112) where p*n=24, so W has shape (24, r344*2)
        
        total_rank = r344 * r112
        U_comb = np.zeros((n*m, total_rank))
        V_comb = np.zeros((m*p, total_rank))
        W_comb = np.zeros((p*n, total_rank))
        
        for c1 in range(r344):
            for c2 in range(r112):
                col = c1 * r112 + c2
                U_comb[:, col] = U344[:, c1] * U112[0, c2]
                
                # V: index is (j, k) where j in 0..3, k in 0..7
                # For M_{3,4,4}: V344[j*4+k1, c1] where k1 in 0..3
                # For M_{1,1,2}: V112[k2, c2] where k2 in 0..1
                # Combined: k = k1 + 4*k2
                for j in range(m):
                    for k1 in range(4):
                        for k2 in range(r112):
                            k = k1 + 4 * k2
                            V_comb[j*p + k, col] = V344[j*4 + k1, c1] * V112[k2, c2]
                
                # W: index is (k, i) where k in 0..7, i in 0..2
                # For M_{3,4,4}: W344[k1*n+i, c1] where k1 in 0..3
                # For M_{1,1,2}: W112[0, c2] (since it's 1x1)
                # Combined: k = k1 + 4*k2
                for i in range(n):
                    for k1 in range(4):
                        for k2 in range(r112):
                            k = k1 + 4 * k2
                            W_comb[k*n + i, col] = W344[k1*n + i, c1] * W112[0, c2]
        
        if check_decomposition(U_comb, V_comb, W_comb):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = (U_comb, V_comb, W_comb)

    # Strategy 2: Direct random search for M_{3,4,8}
    # Try various ranks
    for r in [95, 90, 85, 80, 75, 70, 64, 60, 56, 52, 48, 44, 40]:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Use values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, r))
            V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, r))
            W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, r))
            
            if check_decomposition(U, V, W):
                best_rank = r
                best_result = (U, V, W)
                found = True
                break
        
        if found:
            break

    U, V, W = best_result

    # Convert to lists with integer/half-integer values
    def to_list(mat):
        return [[float(v) for v in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
