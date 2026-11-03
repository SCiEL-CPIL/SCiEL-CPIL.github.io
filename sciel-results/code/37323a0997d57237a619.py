"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import random
import time

def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    DEADLINE = 140  # Leave 10s margin before 150s kill
    
    # Fast verification using random test inputs (Schwartz-Zippel)
    def verify_fast(U, V, W, n, m, p, num_tests=20):
        """Verify decomposition using random integer test inputs."""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        
        for _ in range(num_tests):
            # Random integer matrices
            A = np.random.randint(-100, 100, (n, m))
            B = np.random.randint(-100, 100, (m, p))
            C_expected = A @ B
            
            # Compute C using decomposition
            # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # But wait, the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This doesn't look right. Let me reconsider.
            
            # The tensor decomposition: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
            # For matrix multiplication, we need: C_{i,k} = sum_j A_{i,j} B_{j,k}
            # The structure tensor has T_{i,j,k} = 1 for all valid (i,j,k)
            # So the decomposition must satisfy: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            # To verify: check that for random A, B, we get correct C
            # C_decomp[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should equal C_expected[i,k] = sum_j A[i,j] * B[j,k]
            # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            
            # Let's verify the tensor identity directly
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(U.shape[1]):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if val != 1:
                            return False
        
        # Also verify with actual matrix multiplication
        for _ in range(5):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C_expected = A @ B
            
            # Compute C using the decomposition
            C_decomp = np.zeros((n, p))
            for r in range(U.shape[1]):
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            C_decomp[i, k] += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                return False
        
        return True

    def verify_exact(U, V, W, n, m, p):
        """Exact verification of tensor identity."""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if val != 1:
                        return False
        return True

    # Fallback: Schoolbook Rank 45
    def get_schoolbook():
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

    # Try to find better decompositions using randomized search
    best_result = get_schoolbook()
    best_rank = 45
    
    # Strategy: Try to find rank-44, 43, etc. decompositions
    # Use random perturbations of schoolbook and verify
    
    def try_random_decomposition(target_rank, num_attempts=100):
        """Try to find a rank-r decomposition by random search."""
        for attempt in range(num_attempts):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Start from schoolbook and try to merge terms
            U = np.zeros((n*m, target_rank), dtype=np.int64)
            V = np.zeros((m*p, target_rank), dtype=np.int64)
            W = np.zeros((p*n, target_rank), dtype=np.int64)
            
            # Initialize with random values from small alphabet
            alphabet = [-2, -1, 0, 1, 2]
            for r in range(target_rank):
                # Randomly assign non-zero entries
                for idx in range(n*m):
                    if random.random() < 0.3:
                        U[idx, r] = random.choice(alphabet)
                for idx in range(m*p):
                    if random.random() < 0.3:
                        V[idx, r] = random.choice(alphabet)
                for idx in range(p*n):
                    if random.random() < 0.3:
                        W[idx, r] = random.choice(alphabet)
            
            # Verify
            if verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, num_tests=5):
                if verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try progressively lower ranks
    for target_rank in [44, 43, 42, 41, 40]:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = try_random_decomposition(target_rank, num_attempts=50)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
