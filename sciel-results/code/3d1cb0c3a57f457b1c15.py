import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the target tensor T[a,b,c]
    # a = i*m+j, b = j'*p+k', c = k''*n+i''
    # T[a,b,c] = 1 if i==i'', j==j', k'==k'' else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook decomposition (rank n*m*p)
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
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def is_valid_entries(U, V, W):
        """Check if all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-7:
                    return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0

    # Strategy: 
    # For a given rank r, pick random U and V.
    # For each c in 0..p*n-1, solve the linear system for W[c,:].
    # The system is: sum_t W[c,t] * (U[a,t] * V[b,t]) = T[a,b,c] for all a,b.
    # This is K_c * W_c = y_c, where K_c is (n*m*m*p) x r.
    
    min_possible_rank = max(n*m, m*p, n*p)
    
    for rank in range(min_possible_rank, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try a few random structured initializations
        # We want U and V to be such that the linear systems are solvable.
        # Using sparse matrices might help with numerical stability or exactness.
        
        attempts = 0
        max_attempts = 100
        found = False
        
        while attempts < max_attempts and time.time() < deadline - 2:
            attempts += 1
            
            # Generate random U and V with small integer/half-integer entries
            # Using {-1, 0, 1} for simplicity, or {-1, -0.5, 0, 0.5, 1}
            # Let's try integers first, then half-integers if needed.
            
            # Random sparse matrices
            density = 0.5
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Fill U and V with random values
            for i in range(n*m):
                for t in range(rank):
                    if random.random() < density:
                        U[i, t] = random.choice([-1, 1])
            for i in range(m*p):
                for t in range(rank):
                    if random.random() < density:
                        V[i, t] = random.choice([-1, 1])
                        
            # Check if U and V are "generic" enough (rank of submatrices?)
            # We need to solve for W.
            
            W = np.zeros((p*n, rank))
            solvable = True
            
            for c in range(p*n):
                # Build K_c: (n*m*m*p) x rank
                # Index (a, b) maps to row idx
                # a in 0..n*m-1, b in 0..m*p-1
                # K_c[row, t] = U[a, t] * V[b, t]
                
                K_c = np.zeros((n*m * m*p, rank))
                y_c = np.zeros(n*m * m*p)
                
                row_idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        K_c[row_idx, :] = U[a, :] * V[b, :]
                        y_c[row_idx] = T[a, b, c]
                        row_idx += 1
                        
                # Solve K_c * W_c = y_c
                # Use least squares
                try:
                    W_c, residuals, rank_K, s = np.linalg.lstsq(K_c, y_c, rcond=None)
                    W[c, :] = W_c
                except np.linalg.LinAlgError:
                    solvable = False
                    break
                
            if not solvable:
                continue
                
            # Check if W is close to half-integers
            # Snap W to half-integers
            W_snapped = snap_to_half_int(W)
            
            # Verify with snapped W
            if check_decomposition(U, V, W_snapped) and is_valid_entries(U, V, W_snapped):
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W_snapped
                found = True
                break
                
        if found:
            break
    
    # Convert to lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    def clean_list(lst):
        result = []
        for row in lst:
            clean_row = []
            for val in row:
                snapped = round(val * 2) / 2.0
                if abs(snapped - round(snapped)) < 1e-10:
                    clean_row.append(int(round(snapped)))
                else:
                    clean_row.append(snapped)
            result.append(clean_row)
        return result
    
    return {
        "U": clean_list(U_list),
        "V": clean_list(V_list),
        "W": clean_list(W_list)
    }
