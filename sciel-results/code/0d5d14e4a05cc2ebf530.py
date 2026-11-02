"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _try_border_rank_reduction(n, m, p, start_time, deadline):
    """
    Try to construct a decomposition using border rank reduction with rational functions.
    
    The idea: Find a low-rank decomposition that works over Q(x) (rational functions in x),
    then specialize x to an integer value to get exact integer/half-integer entries.
    
    For 4x4x7, we look for rank < 112 (schoolbook).
    """
    np.random.seed(42)
    best_rank = n * m * p
    best_result = _build_schoolbook(n, m, p)
    
    # Try ranks from 60 to 111
    for target_rank in range(60, 112):
        if time.time() > deadline:
            break
        
        max_attempts = 30 if target_rank < 80 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate sparse U and V with small integer entries
            U_rand = np.zeros((n*m, target_rank))
            V_rand = np.zeros((m*p, target_rank))
            
            for c in range(target_rank):
                # Sparse random with bias towards structured patterns
                density_u = 0.25 if target_rank < 80 else 0.2
                density_v = 0.25 if target_rank < 80 else 0.2
                
                u_nonzero = np.random.choice(n*m, size=max(1, int(n*m*density_u)), replace=False)
                v_nonzero = np.random.choice(m*p, size=max(1, int(m*p*density_v)), replace=False)
                
                for idx in u_nonzero:
                    U_rand[idx, c] = np.random.choice([-2, -1, 1, 2])
                for idx in v_nonzero:
                    V_rand[idx, c] = np.random.choice([-2, -1, 1, 2])
            
            # For each (i,k) pair, we need: sum_c U[i,j,c]*V[j,k,c] = 1/W[i,k,c]
            # The product P[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            # We need W[i,k,c] = 1/P[i,k,c] to be a half-integer
            
            W_mat = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                uc = U_rand[:, c]
                vc = V_rand[:, c]
                for k in range(p):
                    for i in range(n):
                        # Compute sum_j U[i*m+j, c] * V[j*p+k, c]
                        denom = 0.0
                        for j in range(m):
                            denom += uc[i*m+j] * vc[j*p+k]
                        if abs(denom) < 1e-12:
                            valid = False
                            break
                        W_mat[k*n+i, c] = 1.0 / denom
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W entries are nice (integer or half-integer)
            W_doubled = W_mat * 2.0
            W_rounded = np.round(W_doubled)
            if not np.allclose(W_doubled, W_rounded, atol=1e-9):
                continue
            
            # Verify the decomposition
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(int(round(W_mat[i, c] * 2))) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _is_nice_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    return abs(doubled - rounded) < tol

def _try_strassen_block(n, m, p, start_time, deadline):
    """
    Try to construct a decomposition using block Strassen-like methods.
    For 4x4x4, we can use 2x2 Strassen blocks. For 4x4x7, we handle the 7 columns separately.
    
    Strategy: Decompose the 4x4 A and B into 2x2 blocks, use Strassen for the 2x2 parts,
    then handle the extra columns of B.
    """
    # For 4x4x4 part: use 2x2 Strassen applied to 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B, each 2x2 block computed via Strassen
    
    # Strassen for 2x2: rank 7
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # This gives rank 7 for 2x2x2, so for 4x4x4 (four 2x2 blocks), we'd need 4*7=28
    # But we can share some computations... Actually, the standard approach gives 28 for 4x4x4
    
    # For 4x4x7: we have 4x4 A, 4x7 B
    # Split B into B_left (4x4) and B_right (4x3)
    # C_left = A * B_left (4x4) - can use rank 28 via Strassen blocks
    # C_right = A * B_right (4x3) - schoolbook would be 4*4*3=48, but we can do better
    
    # Actually, let me try a direct construction with small rank
    
    # Alternative: use the fact that we can express products of sums
    # Let me try a systematic search over structured patterns
    
    np.random.seed(42)
    best_rank = n * m * p
    best_result = _build_schoolbook(n, m, p)
    
    # Try ranks from 50 to 111
    for target_rank in range(50, 112):
        if time.time() > deadline:
            break
        
        # Use structured random search with constraints
        max_attempts = 200 if target_rank < 80 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate U and V with small integer entries
            # Bias towards sparse patterns
            density = 0.3 if target_rank < 80 else 0.2
            U_rand = np.zeros((n*m, target_rank))
            V_rand = np.zeros((m*p, target_rank))
            
            for c in range(target_rank):
                # Randomly select non-zero positions
                u_nonzero = np.random.choice(n*m, size=max(1, int(n*m*density)), replace=False)
                v_nonzero = np.random.choice(m*p, size=max(1, int(m*p*density)), replace=False)
                
                for idx in u_nonzero:
                    U_rand[idx, c] = np.random.choice([-1, 1, 2])
                for idx in v_nonzero:
                    V_rand[idx, c] = np.random.choice([-1, 1, 2])
            
            # Compute W
            W_mat = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                uc = U_rand[:, c]
                vc = V_rand[:, c]
                for k in range(p):
                    for i in range(n):
                        denom = 0.0
                        for j in range(m):
                            denom += uc[i*m+j] * vc[j*p+k]
                        if abs(denom) < 1e-12:
                            valid = False
                            break
                        W_mat[k*n+i, c] = 1.0 / denom
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W entries are nice (integer or half-integer)
            W_doubled = W_mat * 2.0
            W_rounded = np.round(W_doubled)
            if not np.allclose(W_doubled, W_rounded, atol=1e-9):
                continue
            
            # Verify
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(int(round(W_mat[i, c] * 2))) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def _try_systematic_construction(n, m, p, start_time, deadline):
    """
    Try to construct a decomposition using algebraic identities.
    Use the approach of expressing C[i,k] as sum of products of linear forms.
    """
    # For each (i,k) pair, C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to write this as sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    # Try using linear combinations
    # Key insight: if we can find U, V such that for all (i,k),
    # sum_j U[i,j,c] * V[j,k,c] = 1/W[i,k,c] for some nice W
    
    # Let me try a more structured approach: use Fourier-like bases
    
    np.random.seed(123)
    best_rank = n * m * p
    best_result = _build_schoolbook(n, m, p)
    
    # Try different target ranks
    for target_rank in range(55, 112):
        if time.time() > deadline:
            break
        
        max_attempts = 150
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate U and V with a mix of structured and random patterns
            U_rand = np.zeros((n*m, target_rank))
            V_rand = np.zeros((m*p, target_rank))
            
            for c in range(target_rank):
                # Mix of strategies
                strategy = np.random.random()
                if strategy < 0.3:
                    # Sparse random
                    u_mask = np.random.random(n*m) < 0.2
                    v_mask = np.random.random(m*p) < 0.2
                    U_rand[u_mask, c] = np.random.choice([-1, 1], size=u_mask.sum())
                    V_rand[v_mask, c] = np.random.choice([-1, 1], size=v_mask.sum())
                elif strategy < 0.6:
                    # Row-structured: non-zero in one row of U, one row of V
                    i_idx = np.random.randint(n)
                    j_idx = np.random.randint(m)
                    k_idx = np.random.randint(p)
                    U_rand[i_idx*m:(i_idx+1)*m, c] = np.random.choice([-1, 1], size=m)
                    V_rand[j_idx*p:(j_idx+1)*p, c] = np.random.choice([-1, 1], size=p)
                else:
                    # Dense with small values
                    U_rand[:, c] = np.random.randint(-2, 3, size=n*m)
                    V_rand[:, c] = np.random.randint(-2, 3, size=m*p)
            
            # Compute W
            W_mat = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                uc = U_rand[:, c]
                vc = V_rand[:, c]
                for k in range(p):
                    for i in range(n):
                        denom = 0.0
                        for j in range(m):
                            denom += uc[i*m+j] * vc[j*p+k]
                        if abs(denom) < 1e-12:
                            valid = False
                            break
                        W_mat[k*n+i, c] = 1.0 / denom
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W entries are nice
            W_doubled = W_mat * 2.0
            W_rounded = np.round(W_doubled)
            if not np.allclose(W_doubled, W_rounded, atol=1e-9):
                continue
            
            # Verify
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(int(round(W_mat[i, c] * 2))) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    
    # Try border rank reduction first (most promising)
    try:
        result = _try_border_rank_reduction(n, m, p, start_time, deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
    except Exception:
        pass
    
    # Try Strassen-block approach if time remains
    if time.time() < deadline - 30:
        try:
            result = _try_strassen_block(n, m, p, start_time, deadline)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    if len(result["U"][0]) < len(best["U"][0]):
                        best = result
        except Exception:
            pass
    
    # Try systematic construction if time remains
    if time.time() < deadline - 10:
        try:
            result = _try_systematic_construction(n, m, p, start_time, deadline)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    if len(result["U"][0]) < len(best["U"][0]):
                        best = result
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
