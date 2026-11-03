"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Build the multiplication tensor T[i*m+j, j*p+k] = 1 for valid (i,j,k)
    # Actually: T is a tensor T[i, j, k] where output C[i,k] = sum_j A[i,j]*B[j,k]
    # Flattened: U index = i*m+j, V index = j*p+k, W index = k*n+i (trace convention W[k*n+i, r])
    
    # The tensor T[a, b, c] where:
    # a = i*m + j (U index)
    # b = j*p + k (V index)  
    # c = k*n + i (W index)
    # T[a,b,c] = 1 if a,b,c are consistent (same j in a and b, same i in a and c, same k in b and c)
    
    # For a rank-r decomposition: T = sum_{r=1}^{rank} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # We need to find U (n*m × rank), V (m*p × rank), W (p*n × rank)
    
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build the tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a known valid solution (rank 36) and try to reduce
    
    def make_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        def add_term(u_dict, v_dict, w_dict):
            nonlocal col
            for idx, val in u_dict.items():
                U[idx, col] = val
            for idx, val in v_dict.items():
                V[idx, col] = val
            for idx, val in w_dict.items():
                W[idx, col] = val
            col += 1
        
        def add_strassen(a_i, a_j, b_i, b_j, c_i, c_j):
            def ma(l): return (a_i + l//2) * m + (a_j + l%2)
            def mb(l): return (b_i + l//2) * p + (b_j + l%2)
            def mc(l):
                i = c_i + l//2
                k = c_j + l%2
                return k * n + i
            add_term({ma(0): 1, ma(3): 1}, {mb(0): 1, mb(3): 1}, {mc(0): 1, mc(3): 1})
            add_term({ma(2): 1, ma(3): 1}, {mb(0): 1}, {mc(2): 1, mc(3): -1})
            add_term({ma(0): 1}, {mb(1): 1, mb(3): -1}, {mc(1): 1, mc(3): 1})
            add_term({ma(3): 1}, {mb(2): 1, mb(0): -1}, {mc(0): 1, mc(2): 1})
            add_term({ma(0): 1, ma(1): 1}, {mb(3): 1}, {mc(0): -1, mc(1): 1})
            add_term({ma(2): 1, ma(0): -1}, {mb(0): 1, mb(1): 1}, {mc(3): 1})
            add_term({ma(1): 1, ma(3): -1}, {mb(2): 1, mb(3): 1}, {mc(0): 1})
        
        def add_schoolbook_2x1(a_i, a_j, b_i, b_j, c_i, c_j):
            def ma(l): return (a_i + l//2) * m + (a_j + l%2)
            def mb(l): return (b_i + l) * p + b_j
            def mc(l):
                i = c_i + l
                k = c_j
                return k * n + i
            add_term({ma(0): 1}, {mb(0): 1}, {mc(0): 1})
            add_term({ma(1): 1}, {mb(1): 1}, {mc(0): 1})
            add_term({ma(2): 1}, {mb(0): 1}, {mc(1): 1})
            add_term({ma(3): 1}, {mb(1): 1}, {mc(1): 1})
        
        add_strassen(0, 0, 0, 0, 0, 0)
        add_strassen(0, 0, 0, 2, 0, 2)
        add_schoolbook_2x1(0, 0, 0, 4, 0, 4)
        add_strassen(0, 2, 2, 0, 0, 0)
        add_strassen(0, 2, 2, 2, 0, 2)
        add_schoolbook_2x1(0, 2, 2, 4, 0, 4)
        
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        # Compute T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is expensive for large rank, so do it smartly
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # But that's O(nm*mp*pn*rank) which is 8*20*10*rank = 1600*rank
        # For rank=36, that's 57600 ops, fine
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=1e-10)
    
    # Start with rank 36 solution
    U36, V36, W36 = make_rank36()
    
    # Try to reduce rank by removing terms or combining
    # Strategy: try rank from 20 to 35, attempt to find valid decomposition
    
    best_result = None
    best_rank = 36
    
    # First, try to see if we can just use fewer of the rank-36 terms
    # by checking if some terms are redundant
    
    # Try a local search: start with rank 36, try to zero out columns or merge
    # Better: try randomized initialization for lower ranks
    
    for target_rank in range(20, 36):
        if time.time() > deadline - 5:
            break
        
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and not found:
            if time.time() > deadline - 5:
                break
            attempts += 1
            
            # Random initialization with sparsity
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Initialize with random sparse patterns
            for r in range(target_rank):
                # Pick random non-zero entries
                nz_u = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(mp, size=np.random.randint(1, 4), replace=False)
                nz_w = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
                
                for idx in nz_u:
                    U[idx, r] = np.random.choice([-1, 1])
                for idx in nz_v:
                    V[idx, r] = np.random.choice([-1, 1])
                for idx in nz_w:
                    W[idx, r] = np.random.choice([-1, 1])
            
            if verify(U, V, W):
                found = True
                best_rank = target_rank
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    # If no better found, try a more structured approach
    # Try to modify the rank-36 solution to reduce rank
    if best_rank == 36 and time.time() < deadline - 10:
        # Try removing individual terms from rank 36 and checking
        for r in range(36):
            if time.time() > deadline - 10:
                break
            U_test = U36.copy()
            V_test = V36.copy()
            W_test = W36.copy()
            U_test[:, r] = 0
            V_test[:, r] = 0
            W_test[:, r] = 0
            if verify(U_test, V_test, W_test):
                best_rank = 35
                best_result = (U_test, V_test, W_test)
                break
    
    # Convert best result to lists
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U36, V36, W36
    
    rank = U.shape[1]
    U_list = [[float(U[i, r]) for r in range(rank)] for i in range(nm)]
    V_list = [[float(V[i, r]) for r in range(rank)] for i in range(mp)]
    W_list = [[float(W[i, r]) for r in range(rank)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
