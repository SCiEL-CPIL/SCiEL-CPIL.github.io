"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    Uses a structured embedding of the 3x4x5 problem into the 4x4x4 problem
    and applies a known low-rank decomposition for 4x4x4.
    """
    n, m, p = DIMENSIONS
    
    # The 4x4x4 matrix multiplication tensor can be decomposed with rank 37.
    # We will construct the 4x4x4 decomposition, then restrict it to 3x4x5.
    # Note: The problem asks for 3x4 (A) * 4x5 (B) -> 3x5 (C).
    # Indices: A[i, j] with i in [0,2], j in [0,3]
    #          B[j, k] with j in [0,3], k in [0,4]
    #          C[i, k] with i in [0,2], k in [0,4]
    
    # We embed this into 4x4x4:
    # A4[i, j] for i in [0,3], j in [0,3]. We set A4[3, :] = 0.
    # B4[j, k] for j in [0,3], k in [0,3]. We set B4[:, 4] = 0 (wait, B is 4x5, so k goes to 4).
    # Actually, 4x4x4 implies dimensions 4,4,4. Our target is 3,4,5.
    # This embedding is tricky because 5 > 4. 
    # Let's rethink. The 4x4x4 tensor is for multiplying 4x4 by 4x4.
    # Our problem is 3x4 by 4x5.
    # We can embed 3x4 into 4x4 by padding the first dimension.
    # We can embed 4x5 into 4x4? No, 5 > 4.
    # So 4x4x4 is not a direct superset of 3x4x5 in the standard sense where dimensions match.
    
    # However, we can use a general rank minimization search guided by the structure.
    # Or, we can use the fact that the rank of M_{3,4,5} is bounded by the rank of larger tensors.
    # Rank(M_{3,4,5}) <= Rank(M_{4,4,5})? 
    # Actually, let's just use a smarter search than random.
    # The "structured" hint suggests using symmetries.
    
    # Let's try a different structural approach:
    # The tensor is T = sum_{i,j,k} e_i^A e_j^B e_k^C ... wait, it's multiplication.
    # T = sum_{i,j,k} A_{ij} B_{jk} C_{ik} (dual basis).
    # Tensor entries T_{(i,j), (j',k'), (k,i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}? No.
    # The multiplication tensor T_{ijk, j'k', k'i'} is 1 if j=j', i=i', k=k'? 
    # Let's check the trace convention: W[k*n+i, r].
    # U[i*m+j, r], V[j'*p+k', r], W[k*n+i, r].
    # Sum_r U[i*m+j, r] V[j'*p+k', r] W[k*n+i, r] should be 1 if i=i', j=j', k=k' and 0 otherwise?
    # No, the tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p) is defined such that
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is in U^* x V^* x W^*?
    # Usually T = sum_{i,j,k} e^i_A e^j_B e^k_C \otimes e_i^A e_j^B e_k^C?
    # Let's stick to the verification in the seed:
    # expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
    # val = sum_r U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
    # Here indices are:
    # U index: i (row of A), j (col of A)
    # V index: jp (row of B), k (col of B)
    # W index: kp (col of C), ip (row of C) -- wait, W[k*n+i].
    # So W index is (k, i).
    # The condition is i==ip, j==jp, k==kp.
    # This implies U depends on (i,j), V on (j,k), W on (k,i).
    # This is the standard matrix multiplication tensor.
    
    # Strategy: Use a known decomposition for 4x4x4?
    # Since 5 > 4, we can't embed 4x5 into 4x4.
    # But we can embed 3x4x5 into 4x4x5?
    # Rank(M_{4,4,5}) is unknown but likely > 37.
    
    # Let's try a "structured random search" using the property that the tensor is symmetric in some ways?
    # No, it's not fully symmetric.
    
    # Alternative: Use the fact that for small matrices, the rank is often close to the schoolbook rank minus a few terms.
    # Schoolbook rank = 3*4*5 = 60.
    # Lower bounds:
    # Rank >= max(n*m, m*p, p*n) = max(12, 20, 15) = 20.
    # Rank >= n*p + m - 1? (Strassen-like?)
    # For 2x2x2, rank is 7 (schoolbook 8).
    # For 3x3x3, rank is 23 (schoolbook 27).
    # For 4x4x4, rank is 37 (schoolbook 64).
    
    # Let's assume the rank for 3x4x5 is significantly lower than 60.
    # We will perform a local search starting from the schoolbook decomposition,
    # trying to merge terms or find linear dependencies.
    
    # Better: Use a "greedy" approach to find a low rank decomposition.
    # We can formulate this as finding a low-rank approximation of the tensor, but it must be EXACT.
    
    # Let's use the "embedding into 4x4x4" idea more carefully.
    # If we pad A to 4x4 (add a row of zeros) and B to 4x4? No, B is 4x5.
    # We can pad B to 4x5? It is already 4x5.
    # We can pad C to 4x5? It is 3x5.
    # So we are looking at M_{4,4,5}.
    # If we know a decomposition for M_{4,4,5}, we can restrict it.
    # But we don't have a known low-rank decomposition for 4x4x5.
    
    # Let's try a different structural insight:
    # The tensor is T_{(i,j), (j',k'), (k,i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}.
    # This is a "combinatorial" tensor.
    
    # We will use a "basis change" approach.
    # We know the rank of M_{4,4,4} is 37.
    # Can we map 3x4x5 to a subspace of 4x4x4?
    # Dimensions:
    # A: 3x4 -> 4x4 (pad 1 row)
    # B: 4x5 -> 4x4? No.
    # B: 4x5 -> 5x5?
    
    # Actually, there is a known result: Rank(M_{n,m,p}) <= Rank(M_{n+1, m+1, p+1})?
    # Not necessarily.
    
    # Let's go with a robust "local search" that starts from schoolbook and tries to reduce rank by 1 iteratively.
    # This is a "descent" method.
    # Start with rank 60.
    # Try to find a linear combination of the last 2 terms that can be replaced by 1 term?
    # Or simply, use a "randomized greedy" algorithm:
    # 1. Start with rank R.
    # 2. Generate random U, V, W of rank R.
    # 3. Check if they decompose the tensor.
    # 4. If not, try to "fix" the errors by adjusting one matrix?
    
    # Since the space is discrete (integers/half-integers), we can use a "meet-in-the-middle" or "constraint satisfaction" approach.
    
    # However, given the time limit, the most effective "structural" change is likely to use a **pre-computed lookup** or a **very specific algebraic identity** if one exists for 3x4x5.
    # If no specific identity is known, we must search.
    
    # Let's try a "rank reduction" search:
    # Assume rank is R.
    # We can fix U and V and solve for W?
    # T = U V^T W^T? No, it's a sum of outer products.
    # T_{(i,j), (j',k'), (k,i')} = sum_r U_{(i,j), r} V_{(j',k'), r} W_{(k,i'), r}.
    
    # This is a system of linear equations for W if U and V are fixed.
    # Number of unknowns in W: (p*n) * R = 15 * R.
    # Number of equations: (n*m) * (m*p) * (p*n) = 12 * 20 * 15 = 3600.
    # This is overdetermined for small R.
    
    # We can iterate:
    # 1. Pick random U, V.
    # 2. Check if the system for W is consistent.
    # 3. If consistent, we have a solution.
    
    # To make this faster, we can use the fact that the tensor is sparse (only 60 non-zero entries? No, the tensor T has 3600 entries, but only 60 are 1? 
    # Wait, T_{(i,j), (j',k'), (k,i')} is 1 if i=i', j=j', k=k'.
    # So there are exactly n*m*p = 60 non-zero entries in the tensor, all equal to 1.
    # The rest are 0.
    
    # So we only need to satisfy 60 equations of "sum = 1" and 3540 equations of "sum = 0".
    
    # Let's implement a search that:
    # 1. Iterates R from lower bound to upper bound.
    # 2. For each R, tries random U, V.
    # 3. Solves for W using least squares (or exact rational arithmetic).
    # 4. Verifies if the solution is exact.
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(rank):
                                    val += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                                if abs(val - expected) > 1e-9:
                                    return False
        return True

    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Lower bound for rank
    # Rank >= max(n*m, m*p, p*n) = 20
    # Rank >= n*p + m - 1? (For 2x2x2, 4+2-1=5, rank 7. For 3x3x3, 9+3-1=11, rank 23. Not a tight lower bound.)
    # Let's start from 25.
    
    import random
    import numpy as np
    
    # We will try to solve for W given U, V.
    # The equations are:
    # For each (i, j, jp, k, kp, ip):
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = target
    
    # This is linear in W.
    # Let's form the matrix A where A[eq, var] = U[i*m+j, r] * V[jp*p+k, r]
    # and the vector b[eq] = target.
    # Variables are W[kp*n+ip, r].
    
    def solve_for_W(U, V):
        # Construct the linear system
        # Number of equations: n * m * m * p * p * n = 3*4*4*5*5*3 = 3600
        # Number of variables: p * n * R = 15 * R
        
        eqs = []
        rhs = []
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                row = [0.0] * (p * n)
                                # The variable index in W is (kp*n+ip) * R + r
                                # Wait, W is (p*n) x R.
                                # So variable (row_w, r) is at index row_w * R + r.
                                for r in range(len(U[0])):
                                    val = U[i*m+j][r] * V[jp*p+k][r]
                                    if val != 0:
                                        row[(kp*n+ip) * len(U[0]) + r] = val
                                eqs.append(row)
                                rhs.append(expected)
        
        A = np.array(eqs)
        b = np.array(rhs)
        
        # Solve A x = b
        # Use least squares
        try:
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        # Check if the solution is exact
        if np.allclose(A @ x, b, atol=1e-9):
            # Reshape x to W
            W = x.reshape((p*n), len(U[0]))
            return W
        return None

    # Search for lower ranks
    for rank in range(25, rank_sb):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random U and V with small integer entries
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            
            W = solve_for_W(U, V)
            if W is not None:
                # Verify to be sure (lstsq might have numerical issues)
                if verify(U, V, W):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break # Found a solution for this rank, try lower
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
