"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0

    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = sum(U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] for r in range(rank))
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(val - target) > 1e-6:
                            return False
        return True

    def to_half_int_list(arr):
        """Convert numpy array to list of lists with half-integer values."""
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                rounded = round(val * 2) / 2.0
                if rounded == int(rounded):
                    row.append(int(rounded))
                else:
                    row.append(rounded)
            result.append(row)
        return result

    # Build Strassen-based rank-36 construction
    def build_strassen_block(j_offset, k_offset, rank_offset, k_size):
        """Add a Strassen (2x2x2) or schoolbook (2x2x1) block.
        Returns new rank offset."""
        U = np.zeros((n*m, 0))  # placeholder
        V = np.zeros((m*p, 0))
        W = np.zeros((p*n, 0))
        
        if k_size == 2:
            # Strassen 2x2x2, rank 7
            # Local indices: a in {0,1} (row), b in {0,1} (col of A / row of B), c in {0,1} (col of B)
            # U local: [a*2+b]
            # V local: [b*2+c]
            # W local: [c*2+a]
            
            U_block = np.zeros((4, 7))
            V_block = np.zeros((4, 7))
            W_block = np.zeros((4, 7))
            
            # r=0: M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
            U_block[0, 0] = 1; U_block[3, 0] = 1  # A00, A11
            V_block[0, 0] = 1; V_block[3, 0] = 1  # B00, B11
            W_block[0, 0] = 1; W_block[3, 0] = 1  # C00, C11
            
            # r=1: M2 = (A10+A11)B00 -> C10+=1, C11-=1
            U_block[2, 1] = 1; U_block[3, 1] = 1  # A10, A11
            V_block[0, 1] = 1  # B00
            W_block[2, 1] = 1; W_block[3, 1] = -1  # C10, C11
            
            # r=2: M3 = A00(B01-B11) -> C01+=1, C11+=1
            U_block[0, 2] = 1  # A00
            V_block[1, 2] = 1; V_block[3, 2] = -1  # B01, B11
            W_block[1, 2] = 1; W_block[3, 2] = 1  # C01, C11
            
            # r=3: M4 = A11(B10-B00) -> C00+=1, C10+=1
            U_block[3, 3] = 1  # A11
            V_block[2, 3] = 1; V_block[0, 3] = -1  # B10, B00
            W_block[0, 3] = 1; W_block[2, 3] = 1  # C00, C10
            
            # r=4: M5 = (A00+A01)B11 -> C00-=1, C01+=1
            U_block[0, 4] = 1; U_block[1, 4] = 1  # A00, A01
            V_block[3, 4] = 1  # B11
            W_block[0, 4] = -1; W_block[1, 4] = 1  # C00, C01
            
            # r=5: M6 = (A10-A00)(B00+B01) -> C11+=1
            U_block[2, 5] = 1; U_block[0, 5] = -1  # A10, A00
            V_block[0, 5] = 1; V_block[1, 5] = 1  # B00, B01
            W_block[3, 5] = 1  # C11
            
            # r=6: M7 = (A01-A11)(B10+B11) -> C00+=1
            U_block[1, 6] = 1; U_block[3, 6] = -1  # A01, A11
            V_block[2, 6] = 1; V_block[3, 6] = 1  # B10, B11
            W_block[0, 6] = 1  # C00
            
            # Map local to global
            U_g = np.zeros((n*m, 7))
            V_g = np.zeros((m*p, 7))
            W_g = np.zeros((p*n, 7))
            
            for a in range(2):
                for b in range(2):
                    for r in range(7):
                        U_g[a * m + (j_offset + b), r] = U_block[a*2+b, r]
            
            for b in range(2):
                for c in range(2):
                    for r in range(7):
                        V_g[(j_offset + b) * p + (k_offset + c), r] = V_block[b*2+c, r]
            
            for c in range(2):
                for a in range(2):
                    for r in range(7):
                        W_g[(k_offset + c) * n + a, r] = W_block[c*2+a, r]
            
            return U_g, V_g, W_g, 7
        
        elif k_size == 1:
            # Schoolbook 2x2x1, rank 4
            U_g = np.zeros((n*m, 4))
            V_g = np.zeros((m*p, 4))
            W_g = np.zeros((p*n, 4))
            
            for a in range(2):
                for b in range(2):
                    r = a * 2 + b
                    U_g[a * m + (j_offset + b), r] = 1
                    V_g[(j_offset + b) * p + (k_offset), r] = 1
                    W_g[(k_offset) * n + a, r] = 1
            
            return U_g, V_g, W_g, 4
    
    # Build the full rank-36 construction
    U_parts = []
    V_parts = []
    W_parts = []
    
    # j_group0 (j=0,1), k_group0 (k=0,1): Strassen
    U_b, V_b, W_b, rk = build_strassen_block(0, 0, 0, 2)
    U_parts.append(U_b); V_parts.append(V_b); W_parts.append(W_b)
    
    # j_group0 (j=0,1), k_group1 (k=2,3): Strassen
    U_b, V_b, W_b, rk = build_strassen_block(0, 2, 0, 2)
    U_parts.append(U_b); V_parts.append(V_b); W_parts.append(W_b)
    
    # j_group0 (j=0,1), k_group2 (k=4): Schoolbook 2x2x1
    U_b, V_b, W_b, rk = build_strassen_block(0, 4, 0, 1)
    U_parts.append(U_b); V_parts.append(V_b); W_parts.append(W_b)
    
    # j_group1 (j=2,3), k_group0 (k=0,1): Strassen
    U_b, V_b, W_b, rk = build_strassen_block(2, 0, 0, 2)
    U_parts.append(U_b); V_parts.append(V_b); W_parts.append(W_b)
    
    # j_group1 (j=2,3), k_group1 (k=2,3): Strassen
    U_b, V_b, W_b, rk = build_strassen_block(2, 2, 0, 2)
    U_parts.append(U_b); V_parts.append(V_b); W_parts.append(W_b)
    
    # j_group1 (j=2,3), k_group2 (k=4): Schoolbook 2x2x1
    U_b, V_b, W_b, rk = build_strassen_block(2, 4, 0, 1)
    U_parts.append(U_b); V_parts.append(V_b); W_parts.append(W_b)
    
    U_36 = np.hstack(U_parts)
    V_36 = np.hstack(V_parts)
    W_36 = np.hstack(W_parts)
    
    rank_36 = U_36.shape[1]
    
    # Verify the rank-36 construction
    if check_decomposition(U_36, V_36, W_36, rank_36):
        best_rank = rank_36
        best_result = {
            "U": to_half_int_list(U_36),
            "V": to_half_int_list(V_36),
            "W": to_half_int_list(W_36)
        }
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        best_rank = rank_school
        best_result = {
            "U": to_half_int_list(U_school),
            "V": to_half_int_list(V_school),
            "W": to_half_int_list(W_school)
        }
    
    # Try to find lower rank using optimization
    def residuals(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = sum(U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] for r in range(rank))
                        target = 1.0 if j1 == j2 else 0.0
                        res.append(val - target)
        return np.array(res)
    
    # Try ranks below 36
    for rank in range(max(1, best_rank - 8), best_rank):
        if time.time() - start_time > deadline - 15:
            break
        
        found = False
        for seed in range(30):
            if time.time() - start_time > deadline - 15:
                break
            
            np.random.seed(seed * 104729 + rank * 7919 + 42)
            
            # Initialize with random values
            U0 = np.random.randn(n*m, rank) * 0.3
            V0 = np.random.randn(m*p, rank) * 0.3
            W0 = np.random.randn(p*n, rank) * 0.3
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, x0, args=(rank,), method='lm', max_nfev=8000)
                if result.cost < 1e-12:
                    U = result.x[:n*m*rank].reshape(n*m, rank)
                    V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Round to half-integers and check
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded, rank):
                        best_rank = rank
                        best_result = {
                            "U": to_half_int_list(U_rounded),
                            "V": to_half_int_list(V_rounded),
                            "W": to_half_int_list(W_rounded)
                        }
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
