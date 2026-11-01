"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=200, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute C via tensor decomposition:
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # C[i,k] = sum_{j,c} A[i,j] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] / ... 
        # Actually the decomposition is: C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        # Wait, that's not right. Let me think about the tensor structure.
        
        # The matrix multiplication tensor M has: M[i,j,k; a,b,c] where
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_{j,c} A[i,j] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This doesn't work directly. Let me reconsider.
        
        # Actually, the standard decomposition is:
        # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # No, that gives a product of two sums times W, which isn't bilinear in A and B.
        
        # The correct interpretation: The tensor T such that
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # can be written as C[i,k] = sum_c U[i*m+j, c]... 
        # 
        # Actually, the decomposition means:
        # For the bilinear map (A, B) -> C, we have:
        # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_l B[l,k] * V[j*p+l, c]) * W[k*n+i, c]
        # This still doesn't work because j appears in both U and V indices.
        
        # Let me reconsider. The matrix multiplication tensor has 3 input indices (i,j) for A, (j,k) for B,
        # and 2 output indices (i,k) for C. The shared index is j.
        # 
        # A rank-1 term in the tensor decomposition corresponds to:
        # C[i,k] += (sum_j A[i,j] * u_{i,j}) * (sum_j B[j,k] * v_{j,k}) * w_{i,k}
        # But this isn't quite right either because the j-sums are independent.
        
        # The correct form: C[i,k] = sum_c A[i,:] · U_c[i,:] * V_c[:,k] · B[:,k] * W_c[i,k]
        # where U_c is m-vector for each i, V_c is m-vector for each k, W_c is scalar for each (i,k).
        # But U has shape (n*m, rank), so U[i*m+j, c] is the (i,j) entry of the c-th factor's A-part.
        # V has shape (m*p, rank), so V[j*p+k, c] is the (j,k) entry of the c-th factor's B-part.
        # W has shape (p*n, rank), so W[k*n+i, c] is the (k,i) entry of the c-th factor's C-part.
        
        # So: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # This IS bilinear in A and B! The first sum is linear in A, second is linear in B, and W is a constant.
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # First factor: for each i, sum_j A[i,j] * U[i*m+j, c]
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            # Second factor: for each k, sum_j V[j*p+k, c] * B[j,k]
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            # Multiply and scale by W
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _try_rank_reduction(U, V, W, n=2, m=4, p=8, max_attempts=500):
    """Try to reduce rank by removing columns that are linearly dependent or zero."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = len(U[0])
    
    for _ in range(max_attempts):
        # Try removing each column
        removed = False
        for c in range(rank):
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, trials=50, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Initialize schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Try random perturbations and rank reduction
    random.seed(42)
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Create a perturbed version of schoolbook
        U_pert = [row[:] for row in U]
        V_pert = [row[:] for row in V]
        W_pert = [row[:] for row in W]
        
        # Add small random noise to some entries
        for _ in range(5):
            ri = random.randint(0, len(U_pert) - 1)
            ci = random.randint(0, schoolbook_rank - 1)
            U_pert[ri][ci] += random.gauss(0, 0.1)
            V_pert[ri % len(V_pert)][ci] += random.gauss(0, 0.1)
            W_pert[ri % len(W_pert)][ci] += random.gauss(0, 0.1)
        
        # Try to verify and reduce
        if _verify(U_pert, V_pert, W_pert, n, m, p, trials=100, tol=1e-8):
            U_r, V_r, W_r, rank_r = _try_rank_reduction(U_pert, V_pert, W_pert, n, m, p)
            if rank_r < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = rank_r
                print(f"Improved to rank {best_rank}")
    
    # Final verification
    if best_rank >= schoolbook_rank:
        # Return schoolbook
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Try structured lower-rank decompositions
    # For matrix multiplication, known results suggest rank < n*m*p may be possible
    # Let's try rank = n*m*p - 1, n*m*p - 2, etc.
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break
        
        # Try to find a valid decomposition at this rank
        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            random.seed(trial * 1000 + target_rank)
            np.random.seed(trial * 1000 + target_rank)
            
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Initialize with random values
            U_try = np.random.randn(n * m, target_rank) * 0.5
            V_try = np.random.randn(m * p, target_rank) * 0.5
            W_try = np.random.randn(p * n, target_rank) * 0.5
            
            # Verify
            if _verify(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p, trials=50, tol=1e-9):
                U_list = U_try.tolist()
                V_list = V_try.tolist()
                W_list = W_try.tolist()
                U_r, V_r, W_r, rank_r = _try_rank_reduction(U_list, V_list, W_list, n, m, p)
                if rank_r <= target_rank:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = rank_r
                    found = True
                    print(f"Found rank {best_rank} at target {target_rank}")
                    break
        
        if found:
            break
    
    if best_rank >= schoolbook_rank:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
