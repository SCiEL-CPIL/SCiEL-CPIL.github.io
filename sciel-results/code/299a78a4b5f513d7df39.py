"""Combined program: structured Strassen decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the full matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank-minimal decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    
    # Start with the structured decomposition (rank 66)
    terms = []
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a = (i1, j1)
        b = (i1, j2)
        c = (i2, j1)
        d = (i2, j2)
        e = (j1, k1)
        f = (j1, k2)
        g = (j2, k1)
        h = (j2, k2)
        C11 = (i1, k1)
        C12 = (i1, k2)
        C21 = (i2, k1)
        C22 = (i2, k2)
        
        terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
        terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
        terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
        terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
        terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
        terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
        terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
    
    def add_schoolbook_term(i, j, k):
        terms.append((
            {(i, j): 1},
            {(j, k): 1},
            {(i, k): 1}
        ))
    
    # Base decomposition: rank 66
    for j in range(4):
        for k in range(6):
            add_schoolbook_term(0, j, k)
    
    for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
        add_strassen_block(1, 2, 0, 1, k1, k2)
        add_strassen_block(1, 2, 2, 3, k1, k2)
    
    # Convert to matrices
    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        
        return U, V, W
    
    U, V, W = terms_to_matrices(terms)
    current_rank = len(terms)
    
    # Try to reduce rank by finding linear dependencies
    # Check if we can combine terms
    if time.time() < deadline:
        # Try greedy rank reduction
        # Check if any term is a linear combination of others
        from itertools import combinations
        
        # Build the tensor T as a 3D array
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Verify current decomposition
        def verify(U, V, W):
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.allclose(recon, T, atol=1e-10)
        
        if not verify(U, V, W):
            # Fallback: return schoolbook
            terms_fallback = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms_fallback.append((
                            {(i, j): 1},
                            {(j, k): 1},
                            {(i, k): 1}
                        ))
            U, V, W = terms_to_matrices(terms_fallback)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try to reduce rank by checking for linear dependencies
        # If we have rank r, check if any column is in the span of others
        reduced_U = U.copy()
        reduced_V = V.copy()
        reduced_W = W.copy()
        active = list(range(current_rank))
        
        # Greedy: try removing each term
        for attempt in range(current_rank):
            if time.time() > deadline - 5:
                break
            removed = False
            for idx in range(len(active)):
                if time.time() > deadline - 5:
                    break
                test_active = [a for i, a in enumerate(active) if i != idx]
                if len(test_active) == 0:
                    continue
                
                # Check if removing this term still gives valid decomposition
                # This is expensive, so only try for a few
                if attempt > 5:
                    break
                
                U_test = reduced_U[:, test_active]
                V_test = reduced_V[:, test_active]
                W_test = reduced_W[:, test_active]
                
                recon = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
                if np.allclose(recon, T, atol=1e-10):
                    # We can remove this term
                    active = test_active
                    removed = True
                    break
            
            if not removed:
                break
        
        if len(active) < current_rank:
            U = reduced_U[:, active]
            V = reduced_V[:, active]
            W = reduced_W[:, active]
            current_rank = len(active)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
