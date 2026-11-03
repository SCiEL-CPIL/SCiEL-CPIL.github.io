"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 75
    best_rank = rank_schoolbook
    best_decomp = None

    def make_schoolbook():
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

    # Start with schoolbook
    best_decomp = make_schoolbook()

    # Try to find a lower rank decomposition using structured search
    # For 3x5x5, known upper bound is around 55-63
    # We'll try a few candidate ranks and see if we can find valid decompositions

    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0]) if U and U[0] else 0
        # Build the tensor from U, V, W
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Compare with standard tensor T[i,j,k] = 1 if i,j,k are valid, else 0
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor entries are all 1 in the "standard" representation
        
        # Let's check: the tensor is defined as T_{ijk} = delta... no
        # Matrix multiplication: C_ik = sum_j A_ij B_jk
        # The tensor is T_{ijk} such that C_ik = sum_{j} A_ij B_jk T_{ijk}... 
        # Actually the standard tensor has T_{ijk} = 1 for all i,j,k
        # Wait, no. The tensor for matrix multiplication M_{n,m,p} has entries
        # T_{ijk} = 1 if we're multiplying A (n x m) by B (m x p)
        # The tensor is T_{ijk} where i indexes rows of A (and C), j indexes cols of A / rows of B, k indexes cols of B (and C)
        # T_{ijk} = 1 for all valid i,j,k
        
        # So we need: for all i,j,k: sum_c U_{im+j, c} V_{jp+k, c} W_{kn+i, c} = 1
        # (using the index convention from the problem: W[k*n+i, r])
        
        errors = 0
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    err = abs(s - 1.0)
                    max_err = max(max_err, err)
                    if err > 1e-6:
                        errors += 1
        return max_err < 1e-6, max_err

    # Try to use known decompositions
    # For small cases, we can try random search with structured initialization
    
    # Try rank from 50 up to 74
    for target_rank in range(50, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random structured matrix
        np.random.seed(target_rank * 7919 + 42)
        
        # Use a structured initialization based on schoolbook + noise
        U_base = [[0] * rank_schoolbook for _ in range(n * m)]
        V_base = [[0] * rank_schoolbook for _ in range(m * p)]
        W_base = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_base[i * m + j][col] = 1
                    V_base[j * p + k][col] = 1
                    W_base[k * n + i][col] = 1
                    col += 1
        
        # Try random projection to lower rank
        # This is a heuristic - not guaranteed to work
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(target_rank * 104729 + attempt * 31 + 7)
            
            # Create random mixing matrices to project from rank_schoolbook to target_rank
            P_U = np.random.randn(rank_schoolbook, target_rank)
            P_V = np.random.randn(rank_schoolbook, target_rank)
            P_W = np.random.randn(rank_schoolbook, target_rank)
            
            U_arr = np.array(U_base, dtype=float)
            V_arr = np.array(V_base, dtype=float)
            W_arr = np.array(W_base, dtype=float)
            
            U_new = U_arr @ P_U
            V_new = V_arr @ P_V
            W_new = W_arr @ P_W
            
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            
            valid, max_err = verify(U_list, V_list, W_list)
            if valid:
                # Found a valid decomposition with this rank
                # Try to clean up to integers/half-integers if possible
                best_rank = target_rank
                best_decomp = {"U": U_list, "V": V_list, "W": W_list}
                break
        if best_rank < rank_schoolbook:
            break

    # If we didn't find anything better, use schoolbook
    if best_rank == rank_schoolbook:
        best_decomp = make_schoolbook()

    return best_decomp
# EVOLVE-BLOCK-END
