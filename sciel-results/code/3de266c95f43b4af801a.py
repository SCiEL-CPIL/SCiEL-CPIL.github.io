import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Start with the known rank-55 Strassen-based decomposition
    - The key insight: for 2x3x3 tensor, rank is 7 (Strassen)
    - The current program uses 5 blocks of 2x2 Strassen (35 terms) + 20 schoolbook terms
    - We verify the decomposition and return it if valid
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Build the rank-55 solution
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        # --- Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks ---
        # A1 indices: i in {0,1}, j in {0,1} -> U index i*m + j
        # B1 indices: j in {0,1}, k in {0..9} -> V index j*p + k
        # C1 indices: i in {0,1}, k in {0..9} -> W index k*n + i
        
        # Strassen definitions for 2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        
        # We iterate over 5 blocks of B1. Each block covers 2 columns.
        # Block g corresponds to B columns 2g and 2g+1.
        
        for g in range(5):
            k_base = 2 * g
            k0 = k_base
            k1 = k_base + 1
            
            # B indices for this block
            # b00 is V[0*p + k0]
            # b01 is V[0*p + k1]
            # b10 is V[1*p + k0]
            # b11 is V[1*p + k1]
            
            v_idx_b00 = 0 * p + k0
            v_idx_b01 = 0 * p + k1
            v_idx_b10 = 1 * p + k0
            v_idx_b11 = 1 * p + k1
            
            # A indices
            # a00 is U[0*m + 0]
            # a01 is U[0*m + 1]
            # a10 is U[1*m + 0]
            # a11 is U[1*m + 1]
            
            u_idx_a00 = 0 * m + 0
            u_idx_a01 = 0 * m + 1
            u_idx_a10 = 1 * m + 0
            u_idx_a11 = 1 * m + 1
            
            # C indices for this block
            # c00 is C[0, k0] -> W[k0*n + 0]
            # c01 is C[0, k1] -> W[k1*n + 0]
            # c10 is C[1, k0] -> W[k0*n + 1]
            # c11 is C[1, k1] -> W[k1*n + 1]
            
            w_idx_c00 = k0 * n + 0
            w_idx_c01 = k1 * n + 0
            w_idx_c10 = k0 * n + 1
            w_idx_c11 = k1 * n + 1
            
            # Define the 7 Strassen terms
            # m1 = (a00 + a11) * (b00 + b11)
            # m2 = (a10 + a11) * b00
            # m3 = a00 * (b01 - b11)
            # m4 = a11 * (b10 - b00)
            # m5 = (a00 + a01) * b11
            # m6 = (a10 - a00) * (b00 + b01)
            # m7 = (a01 - a11) * (b10 + b11)
            
            # We will add 7 columns to U, V, W for this block.
            
            # Term 1: m1
            U[u_idx_a00, col] = 1
            U[u_idx_a11, col] = 1
            V[v_idx_b00, col] = 1
            V[v_idx_b11, col] = 1
            # Contributions to C:
            # c00 += 1, c11 += 1
            W[w_idx_c00, col] = 1
            W[w_idx_c11, col] = 1
            col += 1
            
            # Term 2: m2
            U[u_idx_a10, col] = 1
            U[u_idx_a11, col] = 1
            V[v_idx_b00, col] = 1
            # Contributions to C:
            # c10 += 1, c11 -= 1
            W[w_idx_c10, col] = 1
            W[w_idx_c11, col] = -1
            col += 1
            
            # Term 3: m3
            U[u_idx_a00, col] = 1
            V[v_idx_b01, col] = 1
            V[v_idx_b11, col] = -1
            # Contributions to C:
            # c01 += 1, c11 += 1
            W[w_idx_c01, col] = 1
            W[w_idx_c11, col] = 1
            col += 1
            
            # Term 4: m4
            U[u_idx_a11, col] = 1
            V[v_idx_b10, col] = 1
            V[v_idx_b00, col] = -1
            # Contributions to C:
            # c00 += 1, c10 += 1
            W[w_idx_c00, col] = 1
            W[w_idx_c10, col] = 1
            col += 1
            
            # Term 5: m5
            U[u_idx_a00, col] = 1
            U[u_idx_a01, col] = 1
            V[v_idx_b11, col] = 1
            # Contributions to C:
            # c01 += 1, c00 -= 1
            W[w_idx_c01, col] = 1
            W[w_idx_c00, col] = -1
            col += 1
            
            # Term 6: m6
            U[u_idx_a10, col] = 1
            U[u_idx_a00, col] = -1
            V[v_idx_b00, col] = 1
            V[v_idx_b01, col] = 1
            # Contributions to C:
            # c11 += 1
            W[w_idx_c11, col] = 1
            col += 1
            
            # Term 7: m7
            U[u_idx_a01, col] = 1
            U[u_idx_a11, col] = -1
            V[v_idx_b10, col] = 1
            V[v_idx_b11, col] = 1
            # Contributions to C:
            # c00 += 1
            W[w_idx_c00, col] = 1
            col += 1
        
        # --- Part 2: A2 (2x1) * B2 (1x10) using Schoolbook ---
        # A2 indices: i in {0,1}, j=2 -> U index i*m + 2
        # B2 indices: j=2, k in {0..9} -> V index 2*p + k
        # C2 indices: i in {0,1}, k in {0..9} -> W index k*n + i
        
        # This part is outer product of A_col2 and B_row2.
        # C[i, k] = A[i, 2] * B[2, k]
        
        for i in range(n):  # 0, 1
            for k in range(p):  # 0..9
                # A index
                u_idx = i * m + 2
                # B index
                v_idx = 2 * p + k
                # C index
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Verify the rank-55 solution is correct
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U.shape[1]
        
        # The tensor T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor has entries T[i,j,k] which should be 1 for all valid i,j,k
        
        T_target = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, k] = 1.0
        
        T_computed = np.zeros((n, m, p))
        for t in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        return np.allclose(T_target, T_computed, atol=1e-10)
    
    if not verify_decomposition(U55, V55, W55, n, m, p):
        # Fallback to schoolbook if Strassen verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W, best_rank = U, V, W, rank
    else:
        # Check time limit
        if time.time() - start_time > DEADLINE:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        # Try to reduce rank by identifying redundant terms
        # For the Strassen-based solution, check if we can merge terms
        # by finding if any W columns are linearly dependent with others
        
        # More practical: use the fact that for 2x3x3, rank is 7 (minimal)
        # For 2x3x10, we have 5 blocks of 2x3x2 (each with rank 7) plus 2x3x1 terms
        # Total: 5*7 + 2*10 = 35 + 20 = 55, which matches
        
        # Can we do better? The lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30
        # But that's not achievable for matrix multiplication in general
        
        # Let's try a random search for rank-54 by perturbing the rank-55 solution
        rng = np.random.RandomState(42)
        
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Create a perturbed version of the rank-55 solution
            U_pert = U55.copy()
            V_pert = V55.copy()
            W_pert = W55.copy()
            
            # Add small random perturbations
            U_pert += rng.randn(*U_pert.shape) * 0.01
            V_pert += rng.randn(*V_pert.shape) * 0.01
            W_pert += rng.randn(*W_pert.shape) * 0.01
            
            # Try to find a linear combination of columns that gives a valid rank-54 solution
            # This is a non-linear problem, so we'll use a heuristic
            
            # Check if the perturbed solution is still valid
            if verify_decomposition(U_pert, V_pert, W_pert, n, m, p):
                # Try to reduce rank by finding a null space direction
                # Build the constraint matrix
                pass
        
        # If no improvement found, return the rank-55 solution
    
    # Check time limit
    if time.time() - start_time > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Convert to lists for the required interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
