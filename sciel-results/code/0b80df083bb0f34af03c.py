"""Matrix multiplication 3,4,8 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 120  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24

    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # U shape: (n*m, r) = (12, r)
    # V shape: (m*p, r) = (32, r)
    # W shape: (p*n, r) = (24, r)

    def verify(U, V, W, r):
        """Check if the decomposition is exact using Fraction arithmetic."""
        # For each (i, j, k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        Uf = [[Fraction(int(x*2))/2 for x in row] for row in U]
        Vf = [[Fraction(int(x*2))/2 for x in row] for row in V]
        Wf = [[Fraction(int(x*2))/2 for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W, r):
        """Fast verification using numpy with exact rational check via scaling by 2."""
        # Scale by 2 to make integers, then check
        U2 = np.array(U, dtype=np.int64) * 2
        V2 = np.array(V, dtype=np.int64) * 2
        W2 = np.array(W, dtype=np.int64) * 2
        # For each (i,j,k): sum_c U2[i*m+j,c]*V2[j*p+k,c]*W2[k*n+i,c] should equal 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U2[i*m+j, c] * V2[j*p+k, c] * W2[k*n+i, c]
                    if s != 8:
                        return False
        return True

    # Try known small-rank decompositions first
    # The theoretical lower bound for 3x4x4 is 22 (Landsberg), but 3x4x8 is different.
    # Let's try to find a good decomposition via random search.

    # Start with the schoolbook as fallback
    rank_schoolbook = n * m * p  # 96
    U_sb = [[0]*rank_schoolbook for _ in range(nm)]
    V_sb = [[0]*rank_schoolbook for _ in range(mp)]
    W_sb = [[0]*rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Strategy: Try ranks from lower to upper, use random search
    # For each rank r, try to find U, V, W with entries in {0, 1/2, 1} or {-1, -1/2, 0, 1/2, 1}
    
    # Known: for m x n x m (square), rank is around 2m^2. For 3x4x8, let's estimate.
    # The tensor has 3*4*8 = 96 entries. Rank lower bound is 96/max_product.
    # Let's try a structured approach: use the fact that this is a "thin" matrix mult.

    # Approach: Random search with local optimization
    # Represent entries as integers (multiply by 2)
    # U entries in {-2,-1,0,1,2} (i.e., -1,-1/2,0,1/2,1)
    # Actually, let's be more flexible: entries in {-3,-2,-1,0,1,2,3} / 2

    def random_decomposition(r, entries_scale=2, max_entry=3):
        """Generate random U, V, W with entries in {-max_entry,...,max_entry}/entries_scale."""
        U = np.zeros((nm, r), dtype=np.int64)
        V = np.zeros((mp, r), dtype=np.int64)
        W = np.zeros((pn, r), dtype=np.int64)
        for c in range(r):
            U[:, c] = np.random.randint(-max_entry, max_entry + 1, nm)
            V[:, c] = np.random.randint(-max_entry, max_entry + 1, mp)
            W[:, c] = np.random.randint(-max_entry, max_entry + 1, pn)
        return U, V, W

    def check_decomposition(U2, V2, W2, r):
        """Check if U2*V2*W2 (element-wise product summed) gives 8 for all (i,j,k).
        U2, V2, W2 are integer arrays (scaled by 2)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U2[i*m+j, :], V2[j*p+k, :] * W2[k*n+i, :])
                    if s != 8:
                        return False
        return True

    # Try increasing ranks from a reasonable lower bound
    # Lower bound: rank >= 96 / max(|U|*|V|*|W|) for each entry
    # With entries in {-1, 1/2, 0, 1/2, 1}, max product is 1, so rank >= 96.
    # With entries in {-1, -1/2, 0, 1/2, 1, 2}, max product is 2*1*1=2 or similar.
    # Let's try larger entries to get lower rank.
    
    # Actually, for matrix multiplication, there are known results:
    # rank(2x2x2) = 7 (Strassen)
    # For 3x3x3, rank is 23 (or 22 with 3x3x3).
    # For rectangular, it's less clear.
    
    # Let's try a different approach: use the structure of the problem.
    # The tensor T_{ijk} = 1 for all i,j,k.
    # We need sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1.
    
    # Try rank from 50 to 96 with random search
    start_rank = 50
    max_rank = 96
    
    # Use a smarter random: initialize with schoolbook and perturb
    for r in range(start_rank, max_rank + 1):
        if time.time() > DEADLINE - 30:
            break
        
        # Try multiple random initializations
        max_attempts = 200 if r < 70 else 500
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 30:
                break
            
            # Random initialization with small integers
            U2 = np.random.randint(-3, 4, (nm, r), dtype=np.int64)
            V2 = np.random.randint(-3, 4, (mp, r), dtype=np.int64)
            W2 = np.random.randint(-3, 4, (pn, r), dtype=np.int64)
            
            if check_decomposition(U2, V2, W2, r):
                # Found a valid decomposition!
                U_out = [[int(x)/2 for x in U2[i]] for i in range(nm)]
                V_out = [[int(x)/2 for x in V2[i]] for i in range(mp)]
                W_out = [[int(x)/2 for x in W2[i]] for i in range(pn)]
                return {"U": U_out, "V": V_out, "W": W_out}
        
        # Also try structured: use Strassen-like ideas
        # For 3x4x8, we can think of it as combining 2x2 blocks
        # 3x4 = 2x2 + 1x2, 4x8 = 2x2 + 2x2
        # This is getting complex. Let's try a local search approach.
        
        # Local search: start from schoolbook (96 rank), try to reduce rank
        # by finding linear dependencies or using Strassen tricks.
    
    # If random search fails, try a more structured approach
    # Use the fact that T is the all-ones tensor.
    # rank(T) for all-ones tensor of size n*m*p is... well, it's the matrix multiplication tensor.
    
    # Let's try one more thing: use the known 2x2x2 Strassen decomposition
    # and extend it to 3x4x8.
    # 3x4 matrix = [2x4; 1x4], 4x8 = [4x8]
    # Product = [2x8; 1x8]
    # Use Strassen for the 2x2 part: need 2x2x2 blocks.
    # 2x4 * 4x8: split 4 into 2+2, so we have (2x2)*(2x8) and (2x2)*(2x8)
    # Each (2x2)*(2x8) can be done with Strassen: 7 multiplications of (2x1)*(1x8)?
    # No, Strassen is for (2x2)*(2x2).
    
    # Alternative: use the rank-7 decomposition for 2x2x2 and tile it.
    # For 2x2 * 2x2, rank 7.
    # For 2x2 * 2x8: split 8 into four 2x2 blocks.
    # (2x2) * [2x2, 2x2, 2x2, 2x2] = [(2x2)*(2x2), (2x2)*(2x2), (2x2)*(2x2), (2x2)*(2x2)]
    # Each (2x2)*(2x2) needs rank 7, so 4*7 = 28 for the 2x2*2x8 part.
    # But we can share: the left matrix is the same 2x2, so we use the same 7 "multiplication" patterns.
    # Actually, for (2x2)*(2x8), we can do it in rank 14: 
    #   M1 = (A11+A22)(B11+B22) for the first 2x2 block
    #   etc. But the right side has 4 blocks.
    #   (2x2)*(2x8) = (2x2)*(2x2) concatenated 4 times.
    #   Using Strassen: 7 products, each product is (1x1)*(1x8) essentially? No.
    #   Strassen: C = A*B where A,B are 2x2. 7 scalar products.
    #   For (2x2)*(2x8): A is 2x2, B is 2x8. 
    #   Split B = [B1, B2] where B1, B2 are 2x4.
    #   C = A*B = [A*B1, A*B2].
    #   A*B1 is (2x2)*(2x4). Split B1 = [B11, B12] 2x2 each.
    #   A*B1 = [A*B11, A*B12]. Each A*B11 is (2x2)*(2x2), rank 7.
    #   So A*B1 needs 7+7 = 14, and A*B2 needs 14, total 28.
    #   But we can do better: A*B = A*[B1, B2] = [A*B1, A*B2].
    #   Using Strassen on the "width": 
    #   M1 = (A11+A22)(B11+B21)  -- this gives (1x1)*(2x2) product? No.
    
    # This is getting too complicated. Let's just return the best we have.
    
    # One more attempt: try rank 70-95 with more attempts
    for r in range(70, min(96, max_rank) + 1):
        if time.time() > DEADLINE - 15:
            break
        for attempt in range(300):
            if time.time() > DEADLINE - 15:
                break
            U2 = np.random.randint(-4, 5, (nm, r), dtype=np.int64)
            V2 = np.random.randint(-4, 5, (mp, r), dtype=np.int64)
            W2 = np.random.randint(-4, 5, (pn, r), dtype=np.int64)
            if check_decomposition(U2, V2, W2, r):
                U_out = [[int(x)/2 for x in U2[i]] for i in range(nm)]
                V_out = [[int(x)/2 for x in V2[i]] for i in range(mp)]
                W_out = [[int(x)/2 for x in W2[i]] for i in range(pn)]
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Return schoolbook
    return best_result
# EVOLVE-BLOCK-END
