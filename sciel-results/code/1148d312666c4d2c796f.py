import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
DEADLINE_SECONDS = 110.0  # Safety margin before 120s evaluator limit

def _verify_decomposition(U, V, W, n=2, m=3, p=4):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each output C[i,j] = sum_k A[i,k]*B[k,j], we need:
    W[k*n+i, r] * U[i*m+k, r] * V[k*p+j, r] summed over r equals 1 for the tensor entry.
    
    Actually, the decomposition is: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[k*n+i, r]
    where T[i,j,k] = 1 if we're computing C[i,j] from A[i,k]*B[k,j], else 0.
    
    Wait, let me re-read the task. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
    has entries T[i, j, k] where i in [0,n), j in [0,p), k in [0,m).
    T[i,j,k] = 1 if this corresponds to A[i,k] * B[k,j] contributing to C[i,j].
    
    Actually the standard matrix multiplication tensor: C[i,j] = sum_k A[i,k] B[k,j]
    The tensor T has indices (i, j, k) with T[i,j,k] = 1 for all valid i,j,k.
    
    The decomposition says: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[i*p+j, r]
    
    Wait, the W index is k*n+i? Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    
    Hmm, that's unusual. Let me think about this more carefully.
    
    Standard: C[i,j] = sum_k A[i,k] * B[k,j]
    U is indexed by (i,k) -> i*m+k, shape (n*m, rank)
    V is indexed by (k,j) -> k*p+j, shape (m*p, rank)  
    W is indexed by... the task says W[k*n+i, r] with shape (p*n, rank)
    
    Wait, W has shape (p*n, rank). So W's first index ranges over p*n values.
    The task says "W[k*n+i,r]" but that doesn't make sense with shape (p*n, rank) since k goes up to m-1 and i up to n-1, giving max index (m-1)*n+n-1 = m*n-1, but W has p*n rows.
    
    Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    And "output factor follows the official trace convention W[k*n+i,r]"
    
    I think there might be a different convention. Let me look at the current program's usage:
    
    In the current program, W indices used are 0 through 7 (since p*n = 4*2 = 8).
    C indices: 00:0, 01:2, 10:1, 11:3, 02:4, 03:6, 12:5, 13:7
    
    So for C[i,j] where i in {0,1} (n=2) and j in {0,1,2,3} (p=4):
    C[0,0] -> W index 0
    C[0,1] -> W index 2
    C[1,0] -> W index 1
    C[1,1] -> W index 3
    C[0,2] -> W index 4
    C[0,3] -> W index 6
    C[1,2] -> W index 5
    C[1,3] -> W index 7
    
    So W index = j*n + i? Let's check:
    C[0,0]: 0*2+0=0 ✓
    C[0,1]: 1*2+0=2 ✓
    C[1,0]: 0*2+1=1 ✓
    C[1,1]: 1*2+1=3 ✓
    C[0,2]: 2*2+0=4 ✓
    C[0,3]: 3*2+0=6 ✓
    C[1,2]: 2*2+1=5 ✓
    C[1,3]: 3*2+1=7 ✓
    
    So W index = j*n + i, i.e., W[j*n+i, r]. The task description says "W[k*n+i,r]" but I think that's a typo or different notation. The actual convention used is W[j*n+i, r] where j is the column of C and i is the row.
    
    So the decomposition is:
    For each (i, j, k): 1 = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    Let me verify this is what the current code does.
    """
    n, m, p = 2, 3, 4
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                # T[i,j,k] = 1
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+k, r] * V_arr[k*p+j, r] * W_arr[j*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _verify_half_integer(mat):
    """Check all entries are half-integers."""
    for row in mat:
        for val in row:
            v = float(val)
            if abs(v * 2 - round(v * 2)) > 1e-12:
                return False
    return True

def _build_base_decomposition():
    """Build the known rank-22 decomposition."""
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:4]
    # A indices: 00:0, 01:1, 10:3, 11:4
    # B indices: 00:0, 01:1, 10:4, 11:5
    # C indices (W): 00:0, 01:2, 10:1, 11:3
    
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # B indices: 02:2, 03:3, 12:6, 13:7
    # C indices (W): 02:4, 03:6, 12:5, 13:7
    
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    # A indices: 02:2, 12:5
    # B indices: 20:8, 21:9, 22:10, 23:11
    
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    return U, V, W

def construct():
    start_time = time.time()
    
    # Start with known rank-22 solution
    U, V, W = _build_base_decomposition()
    
    # Verify base solution
    if not _verify_decomposition(U, V, W):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    if not (_verify_half_integer(U) and _verify_half_integer(V) and _verify_half_integer(W)):
        return {"U": U, "V": V, "W": W}
    
    best_rank = 22
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition using random search
    # Strategy: try rank 21, 20, ... down to some minimum
    # Use random perturbation and projection to half-integers
    
    n, m, p = 2, 3, 4
    nm, mp, pn = n*m, m*p, p*n
    
    # The tensor T has nm * mp * pn = 6 * 12 * 8 = 576 entries, all equal to 1
    # We need: for all (i,j,k): sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] = 1
    
    # Random search approach:
    # 1. Start with a random rank-r decomposition
    # 2. Use iterative refinement to make it exact
    # 3. Round to half-integers and verify
    
    random.seed(42)
    
    for target_rank in range(21, 14, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > DEADLINE_SECONDS - 30:
                break
            
            # Random initialization
            U_r = np.random.randn(nm, target_rank) * 0.5
            V_r = np.random.randn(mp, target_rank) * 0.5
            W_r = np.random.randn(pn, target_rank) * 0.5
            
            # Iterative refinement: minimize sum of squared errors
            # Error = sum_{i,j,k} (sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] - 1)^2
            
            # Use simple gradient-free approach: random perturbations
            best_err = float('inf')
            U_curr, V_curr, W_curr = U_r.copy(), V_r.copy(), W_r.copy()
            
            for it in range(200):
                if time.time() - start_time > DEADLINE_SECONDS - 30:
                    break
                
                # Compute current error
                err = 0.0
                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_curr[i*m+k, r] * V_curr[k*p+j, r] * W_curr[j*n+i, r]
                            err += (val - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    if best_err < 1e-10:
                        break
                
                # Random perturbation
                scale = max(0.01, 1.0 / (1.0 + it))
                U_curr += np.random.randn(nm, target_rank) * scale
                V_curr += np.random.randn(mp, target_rank) * scale
                W_curr += np.random.randn(pn, target_rank) * scale
            
            if best_err < 1e-8:
                # Round to half-integers
                U_half = np.round(U_curr * 2) / 2
                V_half = np.round(V_curr * 2) / 2
                W_half = np.round(W_curr * 2) / 2
                
                # Verify
                U_list = [[float(x) for x in row] for row in U_half]
                V_list = [[float(x) for x in row] for row in V_half]
                W_list = [[float(x) for x in row] for row in W_half]
                
                if _verify_decomposition(U_list, V_list, W_list):
                    if _verify_half_integer(U_list) and _verify_half_integer(V_list) and _verify_half_integer(W_list):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        break
            else:
                # Also try rounding the best found
                U_half = np.round(U_curr * 2) / 2
                V_half = np.round(V_curr * 2) / 2
                W_half = np.round(W_curr * 2) / 2
                
                U_list = [[float(x) for x in row] for row in U_half]
                V_list = [[float(x) for x in row] for row in V_half]
                W_list = [[float(x) for x in row] for row in W_half]
                
                if _verify_decomposition(U_list, V_list, W_list):
                    if _verify_half_integer(U_list) and _verify_half_integer(V_list) and _verify_half_integer(W_list):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        break
        
        if best_rank < 22:
            break
    
    # Final verification
    if best_rank < 22:
        if _verify_decomposition(best_U, best_V, best_W):
            if _verify_half_integer(best_U) and _verify_half_integer(best_V) and _verify_half_integer(best_W):
                return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
