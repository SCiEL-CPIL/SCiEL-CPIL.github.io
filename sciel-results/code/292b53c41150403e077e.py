"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
                
                if i == j_prime and j == j_prime and k == k_prime and i == i_prime:
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

def _try_rank54(n, m, p):
    """
    Attempt to find a rank-54 decomposition by modifying the rank-55 block structure.
    
    The idea: in each 2-column block, we use 11 terms. For rank 54, we need total 54 terms
    across 5 blocks, so we need some blocks with 10 terms.
    
    Strategy: Try to merge or eliminate one term per block by exploiting the structure.
    We'll use a parameterized search over the block structure.
    
    For the 2x3 x 3x2 sub-problem (one block), the tensor has 12 non-zero entries.
    We need to find a rank-10 decomposition for one block (saving 1 term).
    
    Let's try a different factorization pattern for the 2x3 x 3x2 case.
    """
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll use 10 terms for the first 4 blocks and 14 for the last? No, that's 54 total.
    # Actually: 5 blocks * 11 = 55. To get 54, we need one block with 10 terms.
    # Let's try: blocks 0-3 use 11 terms, block 4 uses 10 terms. Total = 44+10 = 54.
    # Or: all blocks use some mix. Let's try making the last block use 10 terms.
    
    # Alternative approach: use a global rank-54 structure.
    # Let's try to find a valid rank-54 by adjusting the rank-55 structure.
    
    # For now, let's try a different block pattern where we save 1 term in each of 5 blocks
    # by using a smarter decomposition. Total would be 5*10 = 50 if we could save 1 per block.
    # But we only need to save 1 total, so let's save 1 in one block.
    
    # Let's try: first 4 blocks use the standard 11-term pattern, last block uses 10 terms.
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        if t < 4:
            # Standard 11-term block
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
        else:
            # Last block: try 10-term decomposition
            # We need to handle: A[0:2, 0:3] * B[0:3, 8:10] -> C[0:2, 8:10]
            # That's a 2x3 * 3x2 -> 2x2 multiplication, 12 entries.
            # Try a different factorization with 10 terms.
            
            # Let's use the same structure but try to merge two terms.
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
            
            # Direct terms for row 2 - try to merge k0 and k1 for A02
            # A02 * B2,k0 -> C0,k0 and A02 * B2,k1 -> C0,k1
            # Can we do: A02 * (B2,k0 + B2,k1) -> C0,k0 + C0,k1? No, that gives sum.
            # Let's keep them separate but see if we can merge with something else.
            
            # For now, use 4 direct terms (no merging)
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
