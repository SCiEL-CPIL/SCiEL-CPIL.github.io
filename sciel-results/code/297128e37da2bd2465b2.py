"""Matrix multiplication 3x3 * 3x7 rank minimization via structured algebraic search."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook fallback
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
    
    best_result = None
    best_rank = n * m * p  # 63
    
    # Value set to search over (small rationals)
    value_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 0.25, -0.25],
    ]
    
    # Strategy 1: Try to find decompositions using structured patterns
    # For each rank, try different structured initializations
    
    # First, try known good ranks with structured search
    target_ranks = list(range(21, 45))
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Try different value sets and structured patterns
        for vs_idx, values in enumerate(value_sets):
            if time.time() > deadline - 5:
                break
            
            # Multiple random structured initializations
            max_attempts = 20 if rank <= 30 else 10
            if time.time() > deadline - 10:
                max_attempts = 5
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 3:
                    break
                
                try:
                    result = _structured_search(rank, n, m, p, values, attempt, deadline)
                    if result is not None:
                        found = True
                        best_rank = rank
                        best_result = result
                        break
                except Exception:
                    continue
            
            if found:
                break
        
        if found:
            # Check if we can go lower
            continue
    
    # Strategy 2: If structured search didn't find anything better,
    # try a more targeted approach using the 3x3x3 block structure
    if best_rank >= n * m * p:
        if time.time() < deadline - 15:
            result = _block_structure_search(n, m, p, deadline)
            if result is not None:
                r = len(result["U"][0])
                if r < best_rank:
                    best_rank = r
                    best_result = result
    
    if best_result is None:
        return make_schoolbook()
    
    return best_result

def _structured_search(rank, n, m, p, values, attempt_idx, deadline):
    """Search for rank-rank decomposition using structured patterns with small values."""
    rng = np.random.RandomState(42 + attempt_idx * 7919)
    
    # Create structured initialization
    # Pattern 1: Sparse with small values
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each rank term, assign structured patterns
    for c in range(rank):
        # Choose which elements to activate based on pattern
        pattern_type = (c + attempt_idx) % 4
        
        if pattern_type == 0:
            # Single element pattern (schoolbook-like but with shared terms)
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i * m + j, c] = rng.choice(values)
            V[j * p + k, c] = rng.choice(values)
            W[k * n + i, c] = rng.choice(values)
        elif pattern_type == 1:
            # Two-element pattern in U
            i1, i2 = rng.choice(n, 2, replace=False)
            j = c % m
            k = (c // m) % p
            U[i1 * m + j, c] = rng.choice(values)
            U[i2 * m + j, c] = rng.choice(values)
            V[j * p + k, c] = rng.choice(values)
            W[k * n + i1, c] = rng.choice(values)
            W[k * n + i2, c] = rng.choice(values)
        elif pattern_type == 2:
            # Cross pattern
            i = c % n
            j1, j2 = rng.choice(m, 2, replace=False)
            k = (c // n) % p
            U[i * m + j1, c] = rng.choice(values)
            U[i * m + j2, c] = rng.choice(values)
            V[j1 * p + k, c] = rng.choice(values)
            V[j2 * p + k, c] = rng.choice(values)
            W[k * n + i, c] = rng.choice(values)
        else:
            # Random sparse
            n_u = rng.randint(1, 3)
            n_v = rng.randint(1, 3)
            n_w = rng.randint(1, 3)
            
            u_indices = rng.choice(n * m, n_u, replace=False)
            v_indices = rng.choice(m * p, n_v, replace=False)
            w_indices = rng.choice(p * n, n_w, replace=False)
            
            for idx in u_indices:
                U[idx, c] = rng.choice(values)
            for idx in v_indices:
                V[idx, c] = rng.choice(values)
            for idx in w_indices:
                W[idx, c] = rng.choice(values)
    
    # Local refinement: try small perturbations and swaps
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Compute initial loss
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    loss = np.sum((T - 1.0) ** 2)
    
    if loss < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_loss = loss
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Iterative local search: swap values, try different assignments
    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        if loss < 1e-10:
            break
        
        # Random local move: change one value in one matrix
        mat_choice = it % 3
        if mat_choice == 0:
            idx = rng.randint(0, n * m - 1)
            c_idx = rng.randint(0, rank - 1)
            old_val = U[idx, c_idx]
            new_val = rng.choice(values)
            if new_val != old_val:
                U[idx, c_idx] = new_val
                Ur = U.reshape(n, m, rank)
                T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                new_loss = np.sum((T - 1.0) ** 2)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[idx, c_idx] = old_val
        elif mat_choice == 1:
            idx = rng.randint(0, m * p - 1)
            c_idx = rng.randint(0, rank - 1)
            old_val = V[idx, c_idx]
            new_val = rng.choice(values)
            if new_val != old_val:
                V[idx, c_idx] = new_val
                Vr = V.reshape(m, p, rank)
                T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                new_loss = np.sum((T - 1.0) ** 2)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[idx, c_idx] = old_val
        else:
            idx = rng.randint(0, p * n - 1)
            c_idx = rng.randint(0, rank - 1)
            old_val = W[idx, c_idx]
            new_val = rng.choice(values)
            if new_val != old_val:
                W[idx, c_idx] = new_val
                Wr = W.reshape(p, n, rank)
                T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                new_loss = np.sum((T - 1.0) ** 2)
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[idx, c_idx] = old_val
        
        # Update current state
        Ur = best_U.reshape(n, m, rank)
        Vr = best_V.reshape(m, p, rank)
        Wr = best_W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        loss = np.sum((T - 1.0) ** 2)
    
    if best_loss < 1e-8:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _block_structure_search(n, m, p, deadline):
    """Exploit block structure: 3x3x7 = three 3x3x1 blocks or one 3x3x3 + remainder."""
    # For 3x3x7, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..6
    # This is 7 separate 3x3 matrix-vector multiplications
    
    # A 3x3 matrix times a 3x1 vector can be done with rank 3 (schoolbook) 
    # or potentially less with clever decompositions.
    
    # Actually, for the full 3x3x3 tensor (3x3 matrix times 3x3 matrix),
    # Strassen gives rank 7. Let's try to use that structure.
    
    # Strassen's 7-term decomposition for 3x3x3:
    # M1 = (A11)(B11+B22)
    # M2 = (A11+A22)(B22)
    # M3 = (A11+A22)(B11)
    # M4 = (A22)(B22+B11)
    # M5 = (A11)(B11)
    # M6 = (A22)(B22)
    # M7 = (A11+A22)(B11+B22)
    # (This is not quite right - Strassen is for 2x2. For 3x3, rank is 23.)
    
    # For 3x3x3, the known minimal rank is 23 (from literature).
    # For 3x3x7, we need to extend this.
    
    # Let's try a direct approach: search for low-rank decompositions
    # using the fact that the tensor is all-ones.
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,7)
    # We want sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Key insight: if we can find U, V, W such that for each (i,j),
    # the vector (V[j,0,c]*W[0,i,c] + ... ) works out...
    
    # Let's try a parameterized approach:
    # Assume W has a specific structure based on k
    
    # For now, try random search with better initialization
    rng = np.random.RandomState(12345)
    
    for target_rank in range(25, 40):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            # Initialize with some structure
            U = rng.standard_normal((n * m, target_rank)) * 0.1
            V = rng.standard_normal((m * p, target_rank)) * 0.1
            W = rng.standard_normal((p * n, target_rank)) * 0.1
            
            # Add schoolbook-like components for first few terms
            for c in range(min(target_rank, 10)):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i * m + j, c] += 1.0
                V[j * p + k, c] += 1.0
                W[k * n + i, c] += 1.0
            
            # Quick local search
            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)
            
            best_loss = float('inf')
            for step in range(20):
                if time.time() > deadline - 2:
                    break
                
                T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
                loss = np.sum((T - 1.0) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    if loss < 1e-10:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Small perturbation
                U += rng.standard_normal((n * m, target_rank)) * 0.01
                V += rng.standard_normal((m * p, target_rank)) * 0.01
                W += rng.standard_normal((p * n, target_rank)) * 0.01
                Ur = U.reshape(n, m, target_rank)
                Vr = V.reshape(m, p, target_rank)
                Wr = W.reshape(p, n, target_rank)
    
    return None
# EVOLVE-BLOCK-END
