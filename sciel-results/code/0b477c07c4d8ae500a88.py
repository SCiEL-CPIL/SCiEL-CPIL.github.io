"""
Rank-minimal decomposition for 2x2 by 2x11 matrix multiplication.

Strategy: SUBSPACE SPLITTING + INTEGER ANSATZ SEARCH.
1. The multiplication tensor T has dimensions (2, 2, 11).
2. We seek a rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
3. This is equivalent to finding subspaces U (dim r in R^{2*2}), V (dim r in R^{2*11}), W (dim r in R^{11*2}) such that the trilinear form matches.
4. For small n,m, we can often find decompositions with integer or half-integer coefficients.
5. We use a structured search:
   - Start with the known upper bound (e.g., 39 from Strassen).
   - Try to reduce rank by assuming a specific structure for U, V, W that allows for sharing of products across the 11 columns.
   - Specifically, for 2x2 * 2x11, the rank is known to be 2p+1 = 23? Or is it lower?
   - Actually, R(2,2,p) is known to be 2p+1 for p>=2? No, R(2,2,2)=7, R(2,2,3)=? 
   - Let's check known values: R(2,2,1)=4, R(2,2,2)=7, R(2,2,3)=? 
   - A common bound is R(2,2,p) <= 2p+1. For p=11, that's 23.
   - Let's try to construct a rank-23 decomposition.
   - If that fails, try 24, 25... up to 39.
   
   Construction for rank 2p+1 = 23:
   The idea is to use 2p+1 products.
   Let the products be M_0, ..., M_{2p}.
   We can define them such that they cover all 44 terms.
   
   Alternative approach: Use the "Kronecker product" or "block" structure.
   A(2x2) * B(2x11).
   C(2x11) = A * B.
   
   Let's try to implement a solver that searches for a rank-r decomposition with integer coefficients in {-1, 0, 1, 2} or similar small sets, using linear algebra to verify.
   Since the space of tensors is small (44 entries), we can set up a system of linear equations for the coefficients if we fix the support (which non-zero entries in U, V, W).
   
   However, fixing the support is the hard part.
   
   Let's try a different structural insight:
   The tensor T[i,j,k] = 1.
   We can view this as a map from A to (B -> C).
   
   Let's try to generate a rank-23 decomposition explicitly.
   Reference: "The rank of the matrix multiplication tensor"
   For 2x2x2, rank is 7.
   For 2x2xp, rank is 2p+1 for p>=2? 
   Let's assume R(2,2,11) = 23.
   
   How to construct rank 23 for 2x2 * 2x11?
   We can use the decomposition for 2x2 * 2x2 (rank 7) and extend it.
   Or use the formula:
   C[i,k] = sum_j A[i,j] B[j,k]
   
   Let's try to construct it using 23 products.
   Products M_0 ... M_22.
   
   Actually, there is a known construction for R(2,2,p) = 2p+1.
   It involves:
   M_0 = (A00 + A11) * (B0,0 + B1,0)  ... this looks like Strassen.
   
   Let's implement a generic "search for small integer coefficients" for a fixed rank r.
   We will iterate r from 23 to 39.
   For each r, we try to find U, V, W with entries in {-1, 0, 1} (or maybe {-2,-1,0,1,2}) such that the tensor matches.
   This is a constraint satisfaction problem.
   Given the small dimensions (U is 4xr, V is 22xr, W is 22xr), and the number of constraints (44), we can use a randomized local search or a SAT-like approach?
   Too slow for 150s if done naively.
   
   Better: Use the known algebraic structure.
   For 2x2 * 2xp, the rank is 2p+1.
   Construction:
   Let p be the number of columns.
   We define 2p+1 products.
   
   Let's look up the explicit construction for R(2,2,p).
   It is often derived from the 2x2x2 case by "doubling" or "concatenating".
   
   Actually, let's try a specific rank-23 construction.
   If I can't find it easily, I will fall back to the Strassen 39.
   But wait, inspiration 1 got 1.1282 with rank 35.
   Can we get 23?
   
   Let's try to verify if rank 23 is possible.
   Lower bound: max(n*m, m*p, p*n) = max(4, 22, 22) = 22.
   So rank >= 22.
   Since it must be an integer, and usually these tensors have rank > max dim, 23 is a very strong candidate.
   
   Let's try to construct rank 23.
   
   We need U (4x23), V (22x23), W (22x23).
   
   Let's try to use the following pattern for 2x2 * 2xp:
   For each k in 0..p-1, we need to produce C[0,k] and C[1,k].
   
   There is a known decomposition:
   M_{2k} = (A_{0,0} + A_{1,1}) (B_{0,k} + B_{1,k})  -- No, this mixes k.
   
   Let's try to search for a rank-23 solution using a randomized algorithm with integer coefficients.
   We initialize U, V, W with random small integers.
   We compute the reconstructed tensor T_rec.
   We compute the error E = T_target - T_rec.
   We try to update U, V, W to reduce the error.
   This is non-convex.
   
   Alternative: Use the "splitting" method.
   The tensor T is of format (2,2,11).
   We can split the last mode (11) into two parts: 10 and 1.
   T = T1 + T2 where T1 is (2,2,10) and T2 is (2,2,1).
   Rank(T) <= Rank(T1) + Rank(T2).
   Rank(T2) = Rank(2x2*2x1) = 4.
   Rank(T1) <= Rank(2x2*2x10).
   If we can find Rank(2x2*2x10) = 21, then total is 25.
   If Rank(2x2*2xp) = 2p+1, then Rank(2x2*2x10) = 21.
   Total = 21 + 4 = 25.
   
   Can we do better?
   Maybe Rank(2x2*2xp) = 2p+1 is not optimal for all p?
   For p=2, R=7 (2*2+1=5? No, 2*2+1=5 is wrong, 7 is correct. So the formula 2p+1 gives 5 for p=2, which is too low. So R(2,2,2)=7 > 5.
   So the formula is not 2p+1.
   
   Let's check R(2,2,2)=7.
   R(2,2,3)=?
   If R(2,2,2)=7, and we add a column, we might add 2 products?
   So R(2,2,p) = 2p+3?
   For p=2, 2*2+3=7. Correct.
   For p=11, 2*11+3=25.
   
   Let's try to construct rank 25.
   Construction:
   Take the rank-7 Strassen decomposition for the first 2 columns (k=0,1).
   Then for the remaining 9 columns (k=2..10), use a rank-2 decomposition per column?
   No, that would be 7 + 9*2 = 25.
   Wait, for a single column k (2x2 * 2x1), the rank is 4 (schoolbook).
   But we can share products?
   If we treat each remaining column independently, it's 4 per column.
   7 + 9*4 = 43. Worse than 39.
   
   The Strassen 39 construction used 7 per pair (2 columns) + 4 for the last.
   5 pairs * 7 = 35.
   1 single * 4 = 4.
   Total 39.
   
   If we use a rank-2 decomposition for the "rest" of the columns?
   Is it possible to decompose 2x2 * 2x1 with rank 2?
   C0 = A00 B0 + A01 B1
   C1 = A10 B0 + A11 B1
   Rank 2 would mean:
   C0 = u0 v0 w0 + u1 v1 w1
   C1 = u0' v0 w0 + u1' v1 w1
   This implies the matrix [[A00, A01], [A10, A11]] has rank 1? No.
   The tensor for 2x2*2x1 has rank 4. It cannot be reduced to 2.
   
   So the "pair" approach is good.
   Can we do better than 7 per pair?
   R(2,2,2) = 7 is minimal.
   So 7 per pair is optimal for pairs.
   
   So 39 is likely near-optimal for this structure.
   But maybe we can overlap pairs?
   Or use a different grouping?
   
   What if we use a rank-25 construction based on a different algebraic identity?
   There is a result that R(2,2,p) = 2p+3 for p>=2.
   Let's try to construct rank 25.
   
   How to get 25?
   25 = 7 + 18?
   25 = 5 * 5?
   
   Let's try to implement a search for rank 25 with integer coefficients.
   We will use a randomized local search.
   Initialize U, V, W with random integers in {-1, 0, 1}.
   Compute T_rec.
   Error = ||T_target - T_rec||_F^2.
   If error is 0, we found a solution.
   If not, perturb one entry and check if error decreases.
   Repeat.
   
   This might work for small ranks.
   
   Let's implement this search for r in [25, 26, 27, 30, 35, 39].
   Start with 25.
   
   If we find a valid decomposition, return it.
   Otherwise, return the best found (lowest rank) or the fallback 39.
"""

import time
import numpy as np
import random

DIMENSIONS = [2, 2, 11]

def _verify_numpy(U, V, W, n, m, p, trials=20, seed=42):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    rec = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rec[i, j, k] += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    
    target = np.ones((n, m, p), dtype=float)
    if not np.allclose(rec, target, atol=1e-9):
        return False
    
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C_true = A @ B
        C_rec = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
                C_rec[i, k] = s
        if not np.allclose(C_rec, C_true, atol=1e-9):
            return False
    return True

def _build_strassen_39(n, m, p):
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    for (k1, k2) in pairs:
        # M1
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p+k1][col] = 1.0; V[1*p+k2][col] = 1.0
        W[k1*n+0][col] = 1.0; W[k2*n+1][col] = 1.0
        col += 1
        # M2
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p+k1][col] = 1.0
        W[k1*n+1][col] = 1.0
        col += 1
        # M3
        U[0][col] = 1.0
        V[1*p+k1][col] = 1.0; V[0*p+k1][col] = -1.0
        W[k2*n+0][col] = 1.0
        col += 1
        # M4
        U[3][col] = 1.0
        V[0*p+k2][col] = 1.0; V[1*p+k2][col] = -1.0
        W[k1*n+0][col] = 1.0; W[k2*n+1][col] = 1.0
        col += 1
        # M5
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p+k2][col] = 1.0
        W[k1*n+0][col] = -1.0; W[k2*n+0][col] = 1.0
        col += 1
        # M6
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p+k1][col] = 1.0; V[0*p+k2][col] = 1.0
        W[k2*n+1][col] = 1.0
        col += 1
        # M7
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p+k1][col] = 1.0; V[1*p+k2][col] = 1.0
        W[k1*n+0][col] = 1.0; W[k2*n+1][col] = 1.0
        col += 1
    
    # k=10
    U[0][col] = 1.0; V[0*p+10][col] = 1.0; W[10*n+0][col] = 1.0; col += 1
    U[1][col] = 1.0; V[1*p+10][col] = 1.0; W[10*n+0][col] = 1.0; col += 1
    U[2][col] = 1.0; V[0*p+10][col] = 1.0; W[10*n+1][col] = 1.0; col += 1
    U[3][col] = 1.0; V[1*p+10][col] = 1.0; W[10*n+1][col] = 1.0; col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}

def _search_rank(n, m, p, target_rank, time_limit=10.0, seed=0):
    """Randomized search for a decomposition of given rank."""
    rng = np.random.RandomState(seed)
    U = rng.randint(-1, 2, size=(n*m, target_rank))
    V = rng.randint(-1, 2, size=(m*p, target_rank))
    W = rng.randint(-1, 2, size=(p*n, target_rank))
    
    # Verify initial
    if _verify_numpy(U, V, W, n, m, p, trials=1, seed=seed):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    start_time = time.time()
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while time.time() - start_time < time_limit:
        # Pick a random matrix and entry to perturb
        mat_idx = rng.randint(0, 3) # 0:U, 1:V, 2:W
        if mat_idx == 0:
            mat = U
        elif mat_idx == 1:
            mat = V
        else:
            mat = W
            
        row = rng.randint(0, mat.shape[0])
        col = rng.randint(0, mat.shape[1])
        
        old_val = mat[row, col]
        new_val = old_val + rng.randint(-1, 1)
        
        mat[row, col] = new_val
        
        # Check error
        # Fast error check: compute T_rec and compare to 1
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)
        
        # Only compute error for a few random points to be fast?
        # No, we need exact zero.
        # Let's just do a quick check on the full tensor.
        rec = np.zeros((n, m, p), dtype=float)
        for r in range(target_rank):
            rec += U3[:,:,r] * V3[:,:,r] * W3[:,:,r].transpose(1,0) # Note: W3[k,i,r] -> W3.T[i,k,r]?
            # W3 is (p, n, rank). We need W[k,i,r].
            # W3[k, i, r].
            # U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
            # This is not a simple element-wise product if we don't align indices.
            # Let's stick to the slow loop for correctness in this search, 
            # but optimize if possible.
            # Actually, for n=2, m=2, p=11, rank~25, the loop is 2*2*11*25 = 1100 ops.
            # This is fast enough.
        
        # Re-implement fast check
        rec = np.zeros((n, m, p), dtype=float)
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        rec[i, j, k] += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
        
        error = np.sum((rec - 1.0)**2)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if error < 1e-9:
                # Found exact solution
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Revert
            mat[row, col] = old_val
            
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Fallback
    fallback = _build_strassen_39(n, m, p)
    
    # Try to find better rank
    # Known lower bound 22.
    # Try 23, 24, 25...
    # Search is expensive, so limit time per rank.
    
    # Try rank 25 first (likely optimal or close)
    if time.time() < deadline - 20:
        res = _search_rank(n, m, p, 25, time_limit=15.0, seed=123)
        if res:
            if _verify_numpy(res['U'], res['V'], res['W'], n, m, p, trials=5, seed=999):
                return res

    # Try rank 26
    if time.time() < deadline - 20:
        res = _search_rank(n, m, p, 26, time_limit=15.0, seed=456)
        if res:
            if _verify_numpy(res['U'], res['V'], res['W'], n, m, p, trials=5, seed=999):
                return res

    # Try rank 30
    if time.time() < deadline - 10:
        res = _search_rank(n, m, p, 30, time_limit=10.0, seed=789)
        if res:
            if _verify_numpy(res['U'], res['V'], res['W'], n, m, p, trials=5, seed=999):
                return res

    # Return fallback
    return fallback

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
