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

def _try_rank54_search(n, m, p, deadline):
    """Search for a rank-54 decomposition using random restarts and local search."""
    import random
    random.seed(42)
    rank = 54
    best = None
    
    # The tensor has 2*3*10 = 60 required entries.
    # We need to find U (6x54), V (30x54), W (20x54) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1 for required (a,b,c) and 0 otherwise.
    
    # Strategy: start from rank-55 and try to zero out one column via perturbation
    # Or: random initialization with structured patterns
    
    # Let's try: use the rank-55 structure but try to merge two terms in one block
    # by finding a linear dependence.
    
    # Alternative: use a different block decomposition.
    # For 2x3 x 3x2, the tensor rank is known to be at most 11.
    # Let's try to find rank 10 for one block.
    
    # The 2x3 x 3x2 multiplication tensor:
    # A is 2x3 (indices i=0,1; j=0,1,2)
    # B is 3x2 (indices j=0,1,2; k=0,1)
    # C is 2x2 (indices i=0,1; k=0,1)
    # Required: T[i,j,j',k,i',k'] = 1 if i=i', j=j', k=k'
    
    # That's 2*3*2 = 12 required entries.
    
    # Let me try a structured rank-10 decomposition for the 2x3x3x2 block.
    # Use 10 terms with entries in {-1, 0, 1}.
    
    # Try all patterns where each term has at most 2 non-zero entries in U, V, W
    # This is a combinatorial search - let's do it smartly.
    
    # Actually, let's try a specific known pattern.
    # For 2x2x2 (Strassen), rank 7. For 2x3x3x2, we can use:
    # Split into: (2x2x2 for cols 0,1 of A and rows 0,1 of B) + (2x1x2 for col 2 of A and row 2 of B)
    # Strassen for 2x2x2: 7 terms
    # Direct for 2x1x2: 2 terms  
    # Total: 9? But we need cross terms...
    
    # Actually the block is 2x3 * 3x2, so:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # The (A[i,0]*B[0,k] + A[i,1]*B[1,k]) part is a 2x2 * 2x2 = 2x2 multiplication
    # which has rank 7 (Strassen).
    # The A[i,2]*B[2,k] part is 2*1*1*2 = 4 rank-1 terms (or 2 if we can combine).
    # A[i,2]*B[2,0] -> C[i,0] and A[i,2]*B[2,1] -> C[i,1]
    # These are 2 separate rank-1 terms (can't combine since different k).
    # So total: 7 + 2 + 2 = 11? No, 7 + 4 = 11.
    
    # Wait, for the 2x1 * 1x2 part: A[i,2]*B[2,k] for i=0,1 and k=0,1
    # That's 4 rank-1 terms: (A[0,2]*B[2,0], A[0,2]*B[2,1], A[1,2]*B[2,0], A[1,2]*B[2,1])
    # Can we do better? A[0,2]*B[2,0] and A[1,2]*B[2,1] can't be combined with others easily.
    # But A[0,2]*B[2,0] + A[1,2]*B[2,0] = (A[0,2]+A[1,2])*B[2,0] -> but this gives C[0,0] and C[1,0] both.
    # Hmm, this is getting complex.
    
    # Let me just try: 7 (Strassen) + 4 (direct) = 11, which is what we have.
    # To get 10, we need to save 1 term.
    
    # Idea: In Strassen, one of the 7 terms involves (A[0,0]+A[1,1]).
    # If we can combine one Strassen term with one direct term...
    
    # Let's try a computational search: random rank-10 decompositions and verify.
    
    for attempt in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with structure
        U = np.zeros((6, rank))
        V = np.zeros((30, rank))
        W = np.zeros((20, rank))
        
        # For each of 5 blocks (k-pairs), assign 10 or 11 terms
        # Try: 4 blocks with 11 terms, 1 block with 10 terms
        block_sizes = [11, 11, 11, 11, 10]
        random.shuffle(block_sizes)
        
        col = 0
        valid = True
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            size = block_sizes[t]
            
            if size == 11:
                # Use the standard 11-term pattern
                terms = [
                    # (U_indices, V_indices, W_indices, signs)
                    ([(0,0),(3,1)], [(0,k0),(1,k1)], [(k0,0),(k1,1)], [1,1]),
                    ([(3,0),(3,1)], [(0,k0)], [(k0,1),(k1,1)], [1,-1]),
                    ([(0,0)], [(0,k1),(1,k1)], [(k1,0),(k1,1)], [1,-1]),
                    ([(3,1)], [(1,k0),(0,k0)], [(k0,0),(k0,1)], [1,-1]),
                    ([(0,0),(0,1)], [(1,k1)], [(k0,0),(k1,0)], [1,-1]),
                    ([(3,0),(0,0)], [(0,k0),(0,k1)], [(k1,1)], [1,-1]),
                    ([(0,1),(3,1)], [(1,k0),(1,k1)], [(k0,0)], [1,-1]),
                    ([(2,0)], [(2,k0)], [(k0,0)], [1]),
                    ([(2,0)], [(2,k1)], [(k1,0)], [1]),
                    ([(2,1)], [(2,k0)], [(k0,1)], [1]),
                    ([(2,1)], [(2,k1)], [(k1,1)], [1]),
                ]
                for ui, vi, wi, signs in terms:
                    for (u_idx, s) in zip(ui, signs if len(signs)==len(ui) else [1]*len(ui)):
                        U[u_idx[0]*3+u_idx[1], col] += s
                    for (v_idx, s) in zip(vi, signs if len(signs)==len(vi) else [1]*len(vi)):
                        V[v_idx[0]*10+v_idx[1], col] += s
                    for (w_idx, s) in zip(wi, signs if len(signs)==len(wi) else [1]*len(wi)):
                        W[w_idx[0]*2+w_idx[1], col] += s
                    col += 1
            else:
                # Try 10-term pattern: use Strassen for the 2x2x2 part (7 terms)
                # + 3 terms for the remaining (instead of 4)
                # Strassen for A[0:2,0:2] * B[0:2,k0:k1]:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                
                # C00 = M1 - M2 - M4 + M6
                # C01 = M3 + M5 + M6
                # C10 = M2 + M4 + M7
                # C11 = M1 - M3 - M5 + M7
                
                # Plus: A02*B2k0 -> C0k0, A02*B2k1 -> C0k1, A12*B2k0 -> C1k0, A12*B2k1 -> C1k1
                # Try to combine: (A02+A12)*B2k0 -> C0k0 + C1k0? No, that gives both.
                # We need separate terms unless we can subtract.
                
                # Idea: M8 = A02*B2k0, M9 = A12*B2k0, M10 = (A02+A12)*B2k1
                # C0k1 = M10 - M9*B2k1/B2k0... no this doesn't work cleanly.
                
                # Let's try: M8 = A02*B2k0, M9 = A12*B2k1, M10 = (A02+A12)*B2k1
                # C0k0 = M8
                # C1k0 = ? (need A12*B2k0, but we don't have it directly)
                # This doesn't work.
                
                # Alternative: use 10 terms with a different structure.
                # Let's just use the 11-term pattern but with one term having zero weight
                # (effectively 10 terms). This won't work.
                
                # For now, use 11 terms for this block too (mark as invalid for rank 54)
                valid = False
                break
        
        if not valid:
            continue
        
        if col != rank:
            continue
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try rank-55 decomposition with verification
    try:
        res55 = _rank55(n, m, p)
        if time.time() < deadline - 5:
            if _verify(res55["U"], res55["V"], res55["W"], n, m, p, 55):
                return res55
    except Exception:
        pass
    
    # Try rank-54 search
    try:
        if time.time() < deadline - 20:
            res54 = _try_rank54_search(n, m, p, deadline)
            if res54 is not None:
                return res54
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
