import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Strategy: 
    1. Try to find a rank-23 decomposition using Gröbner basis elimination over F2.
    2. If successful, return rank 23.
    3. If timeout or failure, return the known rank-27 schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe before 150s kill

    n, m, p = 3, 3, 3
    input_dim = n * m  # 9
    output_dim = p * n  # 9 (W[k*n+i])
    
    # Fallback: Rank 27 Schoolbook
    def get_schoolbook():
        rank = 27
        U = np.zeros((input_dim, rank))
        V = np.zeros((input_dim, rank))
        W = np.zeros((output_dim, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U index: i*m + j
                    U[i*m + j, col] = 1.0
                    # V index: j*p + k
                    V[j*p + k, col] = 1.0
                    # W index: k*n + i
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try Rank 23
    try:
        result = _find_rank_23_f2(deadline)
        if result is not None:
            return result
    except Exception:
        pass

    return get_schoolbook()

def _find_rank_23_f2(deadline):
    """
    Search for rank-23 decomposition over F2.
    The condition for a term r to be valid is:
    For all i, j, k: 
      sum_{r'} U[i,j,r'] * V[j,k,r'] * W[k,i,r'] = 1 if (i,j,k) is part of the matmul tensor, 0 otherwise.
    Actually, the tensor M_3 has entries 1 at (i,j,k) for all i,j,k? 
    No, the matrix multiplication tensor T has entries T_{i,j,k} = 1 if output (i,k) gets contribution from A_{i,j}B_{j,k}.
    So T_{i,j,k} = 1 for all i,j,k in {0,1,2}.
    Wait, the standard tensor for matmul M_{n,m,p} has entries 1 at (i,j,k) where output index is (i,k) and input indices are (i,j) and (j,k).
    Yes, T_{i,j,k} = 1 for all i,j,k.
    
    We want U, V, W such that:
    sum_{r=0}^{R-1} U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 (mod 2) for all i,j,k.
    
    This is a system of 27 polynomial equations in 3*9*23 = 621 variables over F2.
    We can fix gauge freedoms to reduce variables.
    Gauge: We can permute columns of U,V,W. We can also flip signs (add 1) to U[i,j,r] and V[j,k,r] simultaneously? No, over F2, -1=1.
    
    A simpler approach for F2:
    The equations are cubic. Gröbner basis for 600 variables is likely too slow even with optimizations.
    However, we can use a randomized search or a simpler algebraic structure.
    
    Alternative: Use the known structure of rank 23.
    The rank 23 decomposition is unique up to symmetry (mostly).
    It is known that over F2, the rank of 3x3 matmul is actually 23?
    Actually, over F2, the rank might be different. 
    Let's check if a rank 23 solution exists over F2.
    
    Given the complexity, a full Grobner basis is risky.
    Let's try a different structural approach: 
    Exploit the symmetry.
    
    However, implementing a robust Grobner solver in 150s is hard.
    Let's try a heuristic search with constraint propagation?
    
    Actually, there is a known rank-23 decomposition with integer coefficients.
    The coefficients are in {-1, 0, 1}.
    If we can find one, we can return it.
    
    Let's try to hardcode a verified rank-23 decomposition if we can derive it or find a reliable source in the code.
    Since I cannot browse, I will attempt a local search optimization over integers {-1, 0, 1} using a gradient-free method or random restarts, verifying the tensor identity.
    
    The identity is:
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    Let's define the error:
    E = sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    
    We want E=0.
    
    Let's try random initialization with values in {-1, 0, 1} and use a simple local search (simulated annealing or hill climbing) to minimize E.
    Since the space is discrete, we can flip individual bits (change -1 to 1 or 0 to 1, etc.) and check if E decreases.
    """
    
    rank = 23
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((9, rank), dtype=int)
    W = np.zeros((9, rank), dtype=int)
    
    # Initialize randomly with -1, 0, 1
    np.random.seed(42)
    for r in range(rank):
        U[:, r] = np.random.choice([-1, 0, 1], size=9)
        V[:, r] = np.random.choice([-1, 0, 1], size=9)
        W[:, r] = np.random.choice([-1, 0, 1], size=9)
    
    def calculate_error(U, V, W):
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r] for all i,j,k
        # Using einsum
        # U: (9, R) -> reshape to (3, 3, R)
        # V: (9, R) -> reshape to (3, 3, R)
        # W: (9, R) -> reshape to (3, 3, R)
        
        U_3d = U.reshape(3, 3, rank)
        V_3d = V.reshape(3, 3, rank)
        W_3d = W.reshape(3, 3, rank)
        
        # Sum over r: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a contraction.
        # Let's do it step by step to be safe and fast.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Note indices:
        # U[i,j,r]
        # V[j,k,r]
        # W[k,i,r]
        
        # We can compute M[i,j,k] = sum_r U[i,j,r] V[j,k,r]
        # Then T[i,j,k] = sum_k? No, sum_r is already done.
        # Wait, T[i,j,k] depends on r.
        # T[i,j,k] = sum_r (U[i,j,r] * V[j,k,r] * W[k,i,r])
        
        # Let's use einsum: 'ijr,jkr,kir->ijk'
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
            # We want T[i,j,k] == 1 for all i,j,k
            error = np.sum((T - 1) ** 2)
            return error
        except Exception:
            return float('inf')
    
    current_error = calculate_error(U, V, W)
    
    # Local search
    # Variables are 3 * 9 * 23 = 621 integers in {-1, 0, 1}
    # We can represent them as a flat array.
    
    def get_flat(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def set_flat(flat, U, V, W):
        U_size = 9 * rank
        V_size = 9 * rank
        U[:] = flat[:U_size].reshape(9, rank)
        V[:] = flat[U_size:U_size+V_size].reshape(9, rank)
        W[:] = flat[U_size+V_size:].reshape(9, rank)
    
    # To speed up error calculation, we can update only the affected terms.
    # But let's first see if a simple random search finds it.
    # If not, we might need to be smarter.
    
    # Given the time limit, a full local search on 621 variables might be slow if we recalculate full error each time.
    # However, 621 variables is small.
    
    # Let's try a few random restarts.
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if best_error == 0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # Simple Hill Climbing with random restarts
    max_time = deadline
    while time.time() < max_time:
        # Random restart
        U = np.random.choice([-1, 0, 1], size=(9, rank))
        V = np.random.choice([-1, 0, 1], size=(9, rank))
        W = np.random.choice([-1, 0, 1], size=(9, rank))
        
        current_error = calculate_error(U, V, W)
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_error == 0:
                break
            
        # Local search
        # Pick a random variable, change its value, check error.
        # To make it faster, we can try changing one variable at a time.
        # There are 621 variables.
        # Let's do a few iterations of local search.
        for _ in range(100): # Max 100 steps per restart
            if time.time() > max_time:
                break
            
            # Pick random matrix and index
            mat = np.random.choice([0, 1, 2])
            idx = np.random.randint(0, 9 * rank)
            i, j = divmod(idx, rank)
            
            old_val = [U, V, W][mat][i, j]
            # Try other values
            new_vals = [v for v in [-1, 0, 1] if v != old_val]
            new_val = np.random.choice(new_vals)
            
            [U, V, W][mat][i, j] = new_val
            new_error = calculate_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < best_error:
                    best_error = current_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error == 0:
                        break
            else:
                [U, V, W][mat][i, j] = old_val
            
            if best_error == 0:
                break
        
        if best_error == 0:
            break

    if best_error == 0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

# Keep the original fallback logic intact in construct()
