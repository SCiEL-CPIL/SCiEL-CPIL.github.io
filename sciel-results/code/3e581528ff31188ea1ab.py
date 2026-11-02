import time
import numpy as np
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    deadline = time.time() + 145.0
    
    def build_rank25():
        """Guaranteed rank-25: Strassen for (0,1),(2,3),(4,5) + schoolbook for col 6."""
        rank = 25
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_strassen(col_start, k_start):
            # M1 = (A00+A11)*(B0,ks+B1,ks+1)
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0+k_start, col_start+0] += 1
            V[7*1+(k_start+1), col_start+0] += 1
            W[2*k_start+0, col_start+0] += 1
            W[2*(k_start+1)+1, col_start+0] += 1
            
            # M2 = (A10+A11)*B0,ks
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0+k_start, col_start+1] += 1
            W[2*k_start+1, col_start+1] += 1
            W[2*(k_start+1)+1, col_start+1] -= 1
            
            # M3 = A00*(B0,ks+1-B1,ks+1)
            U[0, col_start+2] += 1
            V[7*0+(k_start+1), col_start+2] += 1
            V[7*1+(k_start+1), col_start+2] -= 1
            W[2*(k_start+1)+0, col_start+2] += 1
            W[2*(k_start+1)+1, col_start+2] += 1
            
            # M4 = A11*(B1,ks-B0,ks)
            U[3, col_start+3] += 1
            V[7*1+k_start, col_start+3] += 1
            V[7*0+k_start, col_start+3] -= 1
            W[2*k_start+0, col_start+3] += 1
            W[2*k_start+1, col_start+3] += 1
            
            # M5 = (A00+A01)*B1,ks+1
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1+(k_start+1), col_start+4] += 1
            W[2*k_start+0, col_start+4] -= 1
            W[2*(k_start+1)+0, col_start+4] += 1
            
            # M6 = (A10-A00)*(B0,ks+B0,ks+1)
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0+k_start, col_start+5] += 1
            V[7*0+(k_start+1), col_start+5] += 1
            W[2*(k_start+1)+1, col_start+5] += 1
            
            # M7 = (A01-A11)*(B1,ks+B1,ks+1)
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1+k_start, col_start+6] += 1
            V[7*1+(k_start+1), col_start+6] += 1
            W[2*k_start+0, col_start+6] += 1
        
        add_strassen(0, 0)
        add_strassen(7, 2)
        add_strassen(14, 4)
        
        # Schoolbook for column 6
        cs = 21
        U[0, cs+0] += 1; V[7*0+6, cs+0] += 1; W[2*6+0, cs+0] += 1
        U[1, cs+1] += 1; V[7*1+6, cs+1] += 1; W[2*6+0, cs+1] += 1
        U[2, cs+2] += 1; V[7*0+6, cs+2] += 1; W[2*6+1, cs+2] += 1
        U[3, cs+3] += 1; V[7*1+6, cs+3] += 1; W[2*6+1, cs+3] += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U_mat, V_mat, W_mat, atol=1e-8):
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=atol)
    
    def is_half_integer(mat, atol=1e-8):
        return np.allclose(mat * 2, np.round(mat * 2), atol=atol)
    
    def to_half_int_list(mat):
        return np.round(mat * 2).astype(int).tolist()
    
    def try_rank24_perturbed():
        """
        Start from rank-25 solution, remove one schoolbook term, and try to 
        find a small perturbation that absorbs it into the Strassen structure.
        """
        r = 24
        # Get the rank-25 solution as arrays
        sol25 = build_rank25()
        U25 = np.array(sol25['U'], dtype=float)
        V25 = np.array(sol25['V'], dtype=float)
        W25 = np.array(sol25['W'], dtype=float)
        
        # The last 4 terms (indices 21-24) are schoolbook for column 6
        # Try removing each one and seeing if we can solve for it
        for remove_idx in [21, 22, 23, 24]:
            if time.time() > deadline:
                return None
            
            # Take first 24 terms, but we need to modify them
            # Strategy: keep 21 Strassen terms + 3 modified schoolbook terms
            # The 3 schoolbook terms should handle column 6 with rank 3 instead of 4
            
            U_mat = np.zeros((n*m, r))
            V_mat = np.zeros((m*p, r))
            
            # Copy Strassen terms (first 21)
            for c in range(21):
                U_mat[:, c] = U25[:, c]
                V_mat[:, c] = V25[:, c]
            
            # For the last 3 terms, try to handle column 6
            # We need: C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6]
            # With 3 rank-1 terms, we can do at most 3 independent products
            # Try: term21 = A00*B06, term22 = A01*B16, term23 = (A10*B06 + A11*B16)?? No, that's not rank-1
            # 
            # Alternative: use a Strassen-like trick for the last column
            # C[0,6] = A00*B06 + A01*B16
            # C[1,6] = A10*B06 + A11*B16
            #
            # With 3 terms: 
            # T1 = (A00+A11)*(B06+B16)
            # T2 = (A10+A11)*B06
            # T3 = A00*(B16-B06)  -- but this only gives 2 of the 4 needed products
            #
            # Actually, for a 2x2 matrix times a 2x1 vector, the tensor rank is 4.
            # So we can't do it in rank 3 for a single column.
            #
            # New approach: share terms between column 5 and column 6
            # The Strassen block for (4,5) already computes combinations involving B[:,4] and B[:,5]
            # If we extend one of those to also involve B[:,6], we might save a term.
            
            # Let's try a different strategy: random search with Strassen bias
            for trial in range(150):
                if time.time() > deadline:
                    return None
                
                U_mat = np.zeros((n*m, r))
                V_mat = np.zeros((m*p, r))
                
                # Initialize with Strassen-like structure for first 21 columns
                for c in range(21):
                    # Copy from rank-25
                    U_mat[:, c] = U25[:, c]
                    V_mat[:, c] = V25[:, c]
                
                # For last 3 columns, try random patterns for column 6
                for c in range(21, 24):
                    # Random sparse pattern
                    num_u = np.random.choice([1, 2], p=[0.3, 0.7])
                    u_pos = np.random.choice(n*m, size=num_u, replace=False)
                    U_mat[u_pos, c] = np.random.choice([-1, 1], size=num_u)
                    
                    # V should involve column 6 (indices 12, 13) and maybe others
                    v_candidates = list(range(12, 14)) + list(range(0, 12))
                    num_v = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                    v_pos = np.random.choice(v_candidates, size=min(num_v, len(v_candidates)), replace=False)
                    V_mat[v_pos, c] = np.random.choice([-1, 1], size=len(v_pos))
                
                # Solve for W
                W_mat = np.zeros((p*n, r))
                consistent = True
                
                for j in range(p):
                    for i in range(n):
                        if time.time() > deadline:
                            return None
                        
                        A_row = np.zeros((m, r))
                        for k in range(m):
                            A_row[k, :] = U_mat[i*m + k, :] * V_mat[k*p + j, :]
                        
                        b_row = np.ones(m)
                        
                        try:
                            sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=1e-10)
                            if not np.allclose(A_row @ sol, b_row, atol=1e-8):
                                consistent = False
                                break
                            W_mat[j*n + i, :] = sol
                        except Exception:
                            consistent = False
                            break
                    
                    if not consistent:
                        break
                
                if not consistent:
                    continue
                
                if verify(U_mat, V_mat, W_mat):
                    if (is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat)):
                        return {
                            "U": to_half_int_list(U_mat),
                            "V": to_half_int_list(V_mat),
                            "W": to_half_int_list(W_mat)
                        }
        
        return None
    
    def try_rank24_global():
        """
        Global random search for rank 24 with better initialization.
        """
        r = 24
        max_trials = 300
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            U_mat = np.zeros((n*m, r))
            V_mat = np.zeros((m*p, r))
            
            for c in range(r):
                # Bias toward Strassen-like patterns
                if c < 21:
                    # Use patterns similar to Strassen
                    pattern = c % 7
                    if pattern == 0:  # (A00+A11)
                        u_idx = np.array([0, 3])
                        U_mat[u_idx, c] = 1
                    elif pattern == 1:  # (A10+A11)
                        u_idx = np.array([2, 3])
                        U_mat[u_idx, c] = 1
                    elif pattern == 2:  # A00
                        u_idx = np.array([0])
                        U_mat[u_idx, c] = 1
                    elif pattern == 3:  # A11
                        u_idx = np.array([3])
                        U_mat[u_idx, c] = 1
                    elif pattern == 4:  # (A00+A01)
                        u_idx = np.array([0, 1])
                        U_mat[u_idx, c] = 1
                    elif pattern == 5:  # (A10-A00)
                        u_idx = np.array([2, 0])
                        U_mat[u_idx, c] = np.array([1, -1])
                    else:  # (A01-A11)
                        u_idx = np.array([1, 3])
                        U_mat[u_idx, c] = np.array([1, -1])
                    
                    # V: random but biased toward 2 non-zeros
                    num_v = np.random.choice([1, 2, 3], p=[0.1, 0.6, 0.3])
                    v_pos = np.random.choice(m*p, size=num_v, replace=False)
                    V_mat[v_pos, c] = np.random.choice([-1, 1], size=num_v)
                else:
                    # Last 3: for column 6
                    num_u = np.random.choice([1, 2], p=[0.3, 0.7])
                    u_pos = np.random.choice(n*m, size=num_u, replace=False)
                    U_mat[u_pos, c] = np.random.choice([-1, 1], size=num_u)
                    
                    # V biased toward column 6 indices
                    v_pos = np.random.choice([12, 13], size=np.random.choice([1, 2]), replace=False)
                    V_mat[v_pos, c] = np.random.choice([-1, 1], size=len(v_pos))
                    # Maybe add one more from other columns
                    if np.random.random() < 0.3:
                        extra = np.random.choice(list(range(0, 12)), size=1)
                        V_mat[extra[0], c] = np.random.choice([-1, 1])
            
            # Solve for W
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    
                    A_row = np.zeros((m, r))
                    for k in range(m):
                        A_row[k, :] = U_mat[i*m + k, :] * V_mat[k*p + j, :]
                    
                    b_row = np.ones(m)
                    
                    try:
                        sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=1e-10)
                        if not np.allclose(A_row @ sol, b_row, atol=1e-8):
                            consistent = False
                            break
                        W_mat[j*n + i, :] = sol
                    except Exception:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            if verify(U_mat, V_mat, W_mat):
                if (is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat)):
                    return {
                        "U": to_half_int_list(U_mat),
                        "V": to_half_int_list(V_mat),
                        "W": to_half_int_list(W_mat)
                    }
        
        return None
    
    def try_rank23():
        """Attempt rank-23."""
        r = 23
        max_trials = 80
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return None
            
            U_mat = np.zeros((n*m, r))
            V_mat = np.zeros((m*p, r))
            
            for c in range(r):
                num_u = np.random.choice([1, 2], p=[0.4, 0.6])
                u_pos = np.random.choice(n*m, size=num_u, replace=False)
                U_mat[u_pos, c] = np.random.choice([-1, 1], size=num_u)
                
                num_v = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                v_pos = np.random.choice(m*p, size=num_v, replace=False)
                V_mat[v_pos, c] = np.random.choice([-1, 1], size=num_v)
            
            W_mat = np.zeros((p*n, r))
            consistent = True
            
            for j in range(p):
                for i in range(n):
                    if time.time() > deadline:
                        return None
                    
                    A_row = np.zeros((m, r))
                    for k in range(m):
                        A_row[k, :] = U_mat[i*m + k, :] * V_mat[k*p + j, :]
                    
                    b_row = np.ones(m)
                    
                    try:
                        sol, residuals, rank_A, s = LA.lstsq(A_row, b_row, rcond=1e-10)
                        if not np.allclose(A_row @ sol, b_row, atol=1e-8):
                            consistent = False
                            break
                        W_mat[j*n + i, :] = sol
                    except Exception:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if not consistent:
                continue
            
            if verify(U_mat, V_mat, W_mat):
                if (is_half_integer(U_mat) and is_half_integer(V_mat) and is_half_integer(W_mat)):
                    return {
                        "U": to_half_int_list(U_mat),
                        "V": to_half_int_list(V_mat),
                        "W": to_half_int_list(W_mat)
                    }
        
        return None
    
    # Guaranteed fallback
    best = build_rank25()
    
    # Try rank 24 with perturbed approach
    sol = try_rank24_perturbed()
    if sol is not None:
        return sol
    
    # Try rank 24 with global search
    sol = try_rank24_global()
    if sol is not None:
        return sol
    
    # Try rank 23
    sol = try_rank23()
    if sol is not None:
        return sol
    
    return best
# EVOLVE-BLOCK-END
