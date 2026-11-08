"""Matrix multiplication 2x3 x 3x10 - rank-minimal decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces C = A*B using NumPy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Check required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(Ua[a, :rank] * Va[b, :rank], Wa[c, :rank])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j_prime = b // p
                k = b % p
                i_prime = c % n
                k_prime = c // n
                
                if i == i_prime and j == j_prime and k == k_prime:
                    continue
                
                val = np.dot(Ua[a, :rank] * Va[b, :rank], Wa[c, :rank])
                if abs(val) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _rank55(n, m, p):
    """Block-based rank-55 decomposition for 2x3 x 3x10."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # Direct terms for row 2 (A02, A12) with B2
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

def _try_rank54_smart(n, m, p, deadline):
    """
    Try to find rank-54 by modifying the rank-55 structure.
    Key idea: In the 2x3 x 3x2 block, try to save 1 term by combining
    the direct terms more cleverly with Strassen terms.
    
    For the 2x3 x 3x2 block:
    - Strassen handles A[0:2,0:2] * B[0:2,0:2] in 7 terms
    - Direct handles A[0:2,2] * B[2,0:2] in 4 terms
    - Total: 11 terms
    
    To save 1: try to merge one direct term with a Strassen term.
    For example, can we combine A02*B2k0 with some Strassen output?
    
    Alternative: use a different 10-term decomposition for the block.
    Let's try a systematic search over sign patterns.
    """
    import random
    random.seed(12345)
    
    rank = 54
    best_result = None
    
    # Strategy: Try different block size distributions
    # 5 blocks, total 54 terms. Options: [11,11,11,11,10] or [11,11,11,10,11] etc.
    
    # For the 10-term block, try various decompositions
    # The 2x3 x 3x2 tensor has 12 required entries.
    # Let's try to find a rank-10 decomposition.
    
    # Known: 2x2x2 has rank 7 (Strassen)
    # 2x1x2 has rank 4 (direct) or can it be done in 3?
    # 2x1x2: A[i,j]*B[j,k] where i in {0,1}, j={2}, k in {0,1}
    # That's A[i,2]*B[2,k] for i=0,1 and k=0,1
    # This is a 2x1 * 1x2 = 2x2 multiplication
    # Rank of 2x1x1x2 tensor: it's just 4 independent products
    # Actually no, it's rank 4 since each (i,k) pair is independent
    
    # So 7 + 4 = 11 is the natural decomposition.
    # To get 10, we need cross-block interaction or a smarter decomposition.
    
    # Let's try: use 10 terms for one block by having a different structure.
    # One idea: instead of separating the 2x2x2 and 2x1x2 parts,
    # use a unified decomposition.
    
    # For now, let's try a random search approach:
    # Start with rank-55, try to zero out one column by adjusting others.
    
    for attempt in range(500):
        if time.time() > deadline - 15:
            break
        
        # Pick which block to try to reduce
        block_to_reduce = attempt % 5
        
        # Start with rank-55 structure
        U = np.zeros((6, rank))
        V = np.zeros((30, rank))
        W = np.zeros((20, rank))
        
        col = 0
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            if t == block_to_reduce:
                # Try a 10-term decomposition for this block
                # Use a randomized approach: start with 11 terms, try to merge two
                terms_10 = _get_10_term_block(k0, k1, random)
                for term in terms_10:
                    for (ui, s) in term[0]:
                        U[ui, col] += s
                    for (vi, s) in term[1]:
                        V[vi, col] += s
                    for (wi, s) in term[2]:
                        W[wi, col] += s
                    col += 1
            else:
                # Standard 11-term block
                terms_11 = _get_11_term_block(k0, k1)
                for term in terms_11:
                    for (ui, s) in term[0]:
                        U[ui, col] += s
                    for (vi, s) in term[1]:
                        V[vi, col] += s
                    for (wi, s) in term[2]:
                        W[wi, col] += s
                    col += 1
        
        if col != rank:
            continue
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _get_11_term_block(k0, k1):
    """Return the standard 11-term decomposition for a 2x3 x 3x2 block."""
    terms = []
    
    # M1: (A00 + A11) * (B0,k0 + B1,k1)
    terms.append([
        [(0, 1), (4, 1)],
        [(k0, 1), (10+k1, 1)],
        [(2*k0, 1), (2*k1+1, 1)]
    ])
    
    # M2: (A10 + A11) * B0,k0
    terms.append([
        [(3, 1), (4, 1)],
        [(k0, 1)],
        [(2*k0+1, 1), (2*k1+1, -1)]
    ])
    
    # M3: A00 * (B0,k1 - B1,k1)
    terms.append([
        [(0, 1)],
        [(k1, 1), (10+k1, -1)],
        [(2*k1, 1), (2*k1+1, 1)]
    ])
    
    # M4: A11 * (B1,k0 - B0,k0)
    terms.append([
        [(4, 1)],
        [(10+k0, 1), (k0, -1)],
        [(2*k0, 1), (2*k0+1, 1)]
    ])
    
    # M5: (A00 + A01) * B1,k1
    terms.append([
        [(0, 1), (1, 1)],
        [(10+k1, 1)],
        [(2*k0, -1), (2*k1, 1)]
    ])
    
    # M6: (A10 - A00) * (B0,k0 + B0,k1)
    terms.append([
        [(3, 1), (0, -1)],
        [(k0, 1), (k1, 1)],
        [(2*k1+1, 1)]
    ])
    
    # M7: (A01 - A11) * (B1,k0 + B1,k1)
    terms.append([
        [(1, 1), (4, -1)],
        [(10+k0, 1), (10+k1, 1)],
        [(2*k0, 1)]
    ])
    
    # Direct terms
    terms.append([
        [(2, 1)],
        [(20+k0, 1)],
        [(2*k0, 1)]
    ])
    
    terms.append([
        [(2, 1)],
        [(20+k1, 1)],
        [(2*k1, 1)]
    ])
    
    terms.append([
        [(5, 1)],
        [(20+k0, 1)],
        [(2*k0+1, 1)]
    ])
    
    terms.append([
        [(5, 1)],
        [(20+k1, 1)],
        [(2*k1+1, 1)]
    ])
    
    return terms

def _get_10_term_block(k0, k1, rng):
    """
    Try to find a 10-term decomposition for a 2x3 x 3x2 block.
    This is the key challenge. Let's try several patterns.
    """
    # Pattern 1: Try to merge two direct terms
    # A02*B2k0 and A12*B2k1 can't be merged directly.
    # But maybe: (A02+A12)*B2k0 and (A02-A12)*B2k1?
    # That gives: C0k0 = (A02+A12)*B2k0, C1k1 = (A02-A12)*B2k1
    # But we also need C1k0 and C0k1.
    
    # Let's try a different approach: use 7 Strassen terms + 3 clever terms.
    # The 3 terms need to handle: A02*B2k0, A02*B2k1, A12*B2k0, A12*B2k1
    # That's 4 products with 3 terms - seems impossible unless we use linear combinations.
    
    # Idea: 
    # T8 = A02 * B2k0
    # T9 = A12 * B2k1  
    # T10 = (A02 + A12) * (B2k0 + B2k1)
    # Then:
    # C0k0 = T8 + T10 - T9 - (A02*B2k1 + A12*B2k0)  -- doesn't work cleanly
    
    # Let's try yet another approach:
    # Use the fact that we can have half-integer entries.
    # T8 = A02 * B2k0
    # T9 = A12 * B2k0
    # T10 = (A02 + A12) * B2k1
    # C0k1 = T10 - T9*B2k1/B2k0... no, this doesn't work.
    
    # For now, return a placeholder that will fail verification
    # This function needs a valid 10-term decomposition
    # Let's return the 11-term structure truncated (will fail)
    terms_11 = _get_11_term_block(k0, k1)
    return terms_11[:10]  # This will likely fail verification

def _try_global_rank54(n, m, p, deadline):
    """
    Try a completely different approach for rank 54.
    Instead of 5 independent blocks, use a global structure.
    
    The key insight: maybe we can share some terms across blocks
    if we use a different decomposition strategy.
    """
    import random
    random.seed(999)
    
    rank = 54
    best_result = None
    
    # Random restart approach: initialize with random sparse patterns
    for attempt in range(200):
        if time.time() > deadline - 20:
            break
        
        U = np.zeros((6, rank))
        V = np.zeros((30, rank))
        W = np.zeros((20, rank))
        
        # For each column, assign a random pattern
        for r in range(rank):
            # Randomly choose which (i,j) pairs to include in U
            num_u = rng.randint(1, 3)
            u_indices = random.sample(range(6), min(num_u, 6))
            for idx in u_indices:
                U[idx, r] = random.choice([-1, 1])
            
            num_v = random.randint(1, 4)
            v_indices = random.sample(range(30), min(num_v, 30))
            for idx in v_indices:
                V[idx, r] = random.choice([-1, 1])
            
            num_w = random.randint(1, 3)
            w_indices = random.sample(range(20), min(num_w, 20))
            for idx in w_indices:
                W[idx, r] = random.choice([-1, 1])
        
        # This random approach is unlikely to work, but let's try
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, try the known rank-55 decomposition
    try:
        res55 = _rank55(n, m, p)
        if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
            # Found valid rank-55, now try to improve
            if time.time() < deadline - 30:
                res54 = _try_rank54_smart(n, m, p, deadline)
                if res54 is not None:
                    return res54
            return res55
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
