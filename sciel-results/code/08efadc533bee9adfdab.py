"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Attempts a rank-32 decomposition using two Laderman (rank 7) blocks 
    and one schoolbook (rank 18) block for the remaining 2 columns.
    Falls back to schoolbook (rank 72) if the Laderman factors are invalid.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS
    
    # --- 1. Schoolbook Decomposition (Rank 72) ---
    U_school = []
    V_school = []
    W_school = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = [0] * (n * m)
                u_row[i * m + j] = 1
                v_row = [0] * (m * p)
                v_row[j * p + k] = 1
                w_row = [0] * (p * n)
                w_row[k * n + i] = 1
                U_school.append(u_row)
                V_school.append(v_row)
                W_school.append(w_row)
    
    schoolbook_result = {
        "U": U_school,
        "V": V_school,
        "W": W_school
    }

    # --- 2. Laderman-based Decomposition (Rank 32) ---
    # Laderman factors for 3x3x3 multiplication
    # U_lad, V_lad, W_lad are 7x9 lists of lists
    # Source: Standard Laderman decomposition
    # Note: Indices are flattened (i*m + j)
    
    # These are the 7 factors (matrices) for U, V, W
    # Each factor is a 3x3 matrix flattened to 9 elements
    U_lad_factors = [
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1]
    ]
    V_lad_factors = [
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1]
    ]
    W_lad_factors = [
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1]
    ]
    
    # Note: The above factors are symmetric in A and B, but C reconstruction is different.
    # Actually, Laderman factors are usually presented as:
    # C = sum r (U_r . A) (V_r . B) W_r
    # The W_r matrices must be different from U_r and V_r.
    # The factors above are a GUESS. I will verify them.
    
    # Let's use a known correct set of Laderman factors if possible.
    # Since I am unsure, I will rely on the validation step.
    
    # Construct U, V, W for Rank 32
    # Block 1: cols 0,1,2 (Laderman)
    # Block 2: cols 3,4,5 (Laderman)
    # Block 3: cols 6,7 (Schoolbook)
    
    U_rank32 = []
    V_rank32 = []
    W_rank32 = []
    
    def add_block(k_offset, U_factors, V_factors, W_factors):
        for r in range(len(U_factors)):
            u_row = [0] * (n * m)
            v_row = [0] * (m * p)
            w_row = [0] * (p * n)
            
            # Fill U
            for idx, val in enumerate(U_factors[r]):
                u_row[idx] = val
            
            # Fill V (only for k in block)
            for j in range(m):
                for k_local in range(3):
                    # Map local k (0,1,2) to global k
                    k_global = k_offset + k_local
                    v_idx = j * p + k_global
                    # V_factors[r] is flattened 3x3 (j_local, k_local)
                    v_val = V_factors[r][j * 3 + k_local]
                    v_row[v_idx] = v_val
            
            # Fill W (only for k in block)
            for k_local in range(3):
                k_global = k_offset + k_local
                for i in range(n):
                    w_idx = k_global * n + i
                    # W_factors[r] is flattened 3x3 (k_local, i)?
                    # My convention: W_r is a matrix such that C_ik += scalar * W_ik.
                    # So W_factors[r] flattened is (i, k) or (k, i)?
                    # Task W index: k*n + i.
                    # So we need value at [k, i] of the W matrix.
                    w_val = W_factors[r][k_local * 3 + i] # Assuming W_factors is k,i flattened
                    w_row[w_idx] = w_val
            
            U_rank32.append(u_row)
            V_rank32.append(v_row)
            W_rank32.append(w_row)
            
    # Add Laderman Block 1 (cols 0-2)
    if time.time() < deadline:
        add_block(0, U_lad_factors, V_lad_factors, W_lad_factors)
        add_block(3, U_lad_factors, V_lad_factors, W_lad_factors)
        
    # Add Schoolbook Block 3 (cols 6-7)
    if time.time() < deadline:
        for i in range(n):
            for j in range(m):
                for k in [6, 7]:
                    u_row = [0] * (n * m)
                    u_row[i * m + j] = 1
                    v_row = [0] * (m * p)
                    v_row[j * p + k] = 1
                    w_row = [0] * (p * n)
                    w_row[k * n + i] = 1
                    U_rank32.append(u_row)
                    V_rank32.append(v_row)
                    W_rank32.append(w_row)
                    
    rank32_result = {
        "U": U_rank32,
        "V": V_rank32,
        "W": W_rank32
    }
    
    # --- 3. Validation ---
    def validate(result):
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        # Random test
        np.random.seed(42)
        A = np.random.randint(-5, 5, size=(n, m))
        B = np.random.randint(-5, 5, size=(m, p))
        
        C_expected = A @ B
        
        # Compute C_decomp = sum_r (U_r . A) (V_r . B) W_r
        # U_r is n*m x 1. A is n x m.
        # Dot product: sum_{i,j} U[i*m+j] * A[i,j]
        # Vectorize A
        A_flat = A.flatten() # Row-major: i*m + j
        
        # V_r is m*p x 1. B is m x p.
        # Dot product: sum_{j,k} V[j*p+k] * B[j,k]
        B_flat = B.flatten() # Row-major: j*p + k
        
        C_decomp = np.zeros((p, n)) # Note: W index is k*n + i -> Row k, Col i?
        # Wait, C is n x p.
        # W row index k*n + i. So row k, col i in W's matrix form (p x n).
        # C_ik = sum_r (..) W_r[k, i].
        # So C is n x p. C_decomp[i, k] = ... W_val[k, i].
        
        for r in range(U.shape[1]):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            scalar = np.dot(u_vec, A_flat) * np.dot(v_vec, B_flat)
            
            # Add scalar * W_r to C_decomp
            # W_r is p*n vector.
            # Reshape to p x n (rows k, cols i)
            W_mat = w_vec.reshape(p, n)
            
            # C[i, k] += scalar * W_mat[k, i]
            # C_decomp is n x p.
            # C_decomp[i, k] = scalar * W_mat[k, i]
            # So C_decomp += scalar * W_mat.T
            
            C_decomp += scalar * W_mat.T
            
        # Check if C_decomp equals C_expected
        if np.allclose(C_decomp, C_expected):
            return True
        return False
    
    if time.time() < deadline and len(U_rank32) < 72:
        if validate(rank32_result):
            return rank32_result
            
    # Fallback to schoolbook
    return schoolbook_result
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Constructed rank {len(result['U'][0])}")
