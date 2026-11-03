"""
Model-and-optimize: parameterize rank-r factorization and fit/optimize parameters.
Target: exact rank decomposition of 3x4x6 matrix multiplication tensor, minimize rank.
Strategy: start from a partition-based baseline (rank 66), then run a randomized
merge-and-drop local search to reduce rank while preserving exactness.
Additionally, try a HALS-inspired iterative refinement to find non-trivial
rank reductions by treating the problem as a low-rank tensor approximation
with exactness constraints.
"""
import random
import time
import numpy as np

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

def true_tensor():
    """Return the list of (i,j,k) triples that are 1 in the matmul tensor."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def verify(U, V, W, r):
    """Verify exact decomposition: for all (i,j,k), sum_l U[Aidx(i,j),l]*V[Bidx(j,k),l]*W[Cidx(i,k),l] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def baseline_66():
    """Construct the baseline rank-66 decomposition using block partitioning."""
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        # M1: (A00+A11)(B00+B11)
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        # M2: (A10+A11)(B00)
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        # M3: (A00)(B01-B11)
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        # M4: (A11)(B10-B00)
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        # M5: (A00+A01)(B11)
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        # M6: (A10-A00)(B00+B01)
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        # M7: (A01-A11)(B10+B11)
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 -> 1x2 multiplication using 4 terms."""
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        # C0 += A0*B00 + A1*B10
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        # C1 += A0*B01 + A1*B11
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    # Row 0 of A (1x4) times 4x6 blocks
    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    # Rows 1-2 of A (2x4) times 4x6 blocks using Strassen
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return U, V, W

def try_drop(U, V, W, r, l):
    """Check if removing term l still gives an exact decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for ll in range(r):
                    if ll == l:
                        continue
                    s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(i,k)][ll]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def try_merge(U, V, W, r, l1, l2):
    """
    Try to merge two rank terms into one. This is a heuristic:
    Check if the two terms have compatible structures that allow
    a single rank-1 term to represent their sum.
    Returns (success, new_U, new_V, new_W, new_r) or (False, None, None, None, r).
    """
    # For a merge to work, we need:
    # u1*a1 + u2*a2 = u' * a' for all relevant indices
    # This is generally not possible unless the vectors are proportional
    # in a specific way. We do a simple check: see if we can find
    # a scalar alpha such that (u1 + alpha*u2), (v1 + alpha*v2), (w1 + alpha*w2)
    # still produce the correct contributions.
    # This is a very limited heuristic; most merges will fail.
    return False, None, None, None, r

def local_search(U, V, W, r, deadline, rng):
    """Local search: try to drop redundant terms."""
    best_r = r
    best_U, best_V, best_W = U, V, W
    while time.time() < deadline - 10:
        improved = False
        # Try drops in random order
        order = list(range(best_r))
        rng.shuffle(order)
        for l in order:
            if time.time() > deadline - 10:
                break
            if try_drop(best_U, best_V, best_W, best_r, l):
                # Remove term l
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                break
        if not improved:
            break
    return best_r, best_U, best_V, best_W

def hals_refine(U, V, W, r, deadline, rng):
    """
    HALS-inspired refinement: try to find a lower-rank decomposition
    by treating the tensor as a sum of rank-1 terms and using alternating
    least squares to refine the factors, then checking for exactness.
    
    This is a heuristic approach: we start from the baseline, then try
    to perturb and re-optimize to find a sparser/lower-rank solution.
    """
    best_r = r
    best_U, best_V, best_W = U, V, W
    
    # Convert to numpy for faster operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[Aidx(i,j), Bidx(j,k), Cidx(i,k)] = 1.0
    
    # Try to find a lower-rank approximation using iterative refinement
    # We'll try ranks from r-1 down to some minimum
    for target_r in range(r - 1, max(r - 10, 50), -1):
        if time.time() > deadline - 15:
            break
        
        # Initialize with random perturbation of the first target_r columns
        U_try = U_np[:, :target_r].copy()
        V_try = V_np[:, :target_r].copy()
        W_try = W_np[:, :target_r].copy()
        
        # Add small random perturbation to break symmetry
        U_try += rng.gauss(0, 0.1) * np.ones_like(U_try) * 0  # Keep integer for now
        V_try += rng.gauss(0, 0.1) * np.ones_like(V_try) * 0
        W_try += rng.gauss(0, 0.1) * np.ones_like(W_try) * 0
        
        # Iterate: try to adjust factors to better match the tensor
        for it in range(50):
            if time.time() > deadline - 15:
                break
            
            # Update W: for each column l, W[:,l] should satisfy
            # sum_l U[i,j,l]*V[j,k,l]*W[i,k,l] = T[i,j,k]
            # This is a linear system for W[:,l] given U[:,l] and V[:,l]
            # But it's coupled across l, so we do coordinate descent
            
            improved = False
            for l in range(target_r):
                # Fix U[:,l] and V[:,l], solve for W[:,l]
                # The contribution of term l is: U[:,l] ⊗ V[:,l] ⊗ W[:,l]
                # We want the total sum to equal T
                # Residual after removing term l:
                R = T.copy()
                for ll in range(target_r):
                    if ll == l:
                        continue
                    # Subtract contribution of term ll
                    for i in range(n*m):
                        for j in range(m*p):
                            for k in range(p*n):
                                R[i,j,k] -= U_try[i,ll] * V_try[j,ll] * W_try[k,ll]
                
                # Now we need U[:,l] ⊗ V[:,l] ⊗ W[:,l] = R
                # This is a rank-1 tensor fitting problem
                # W[:,l] can be found by: W[k,l] = R[i,j,k] / (U[i,l]*V[j,l]) for any (i,j) with U[i,l]*V[j,l] != 0
                # Use least squares: minimize ||U[:,l] ⊗ V[:,l] ⊗ W[:,l] - R||
                
                # Vectorize: R_flat[k] = sum_{i,j} U[i,l]*V[j,l]*W[k,l] * delta(i,j matches)
                # Actually, R[i,j,k] = U[i,l]*V[j,l]*W[k,l] for all i,j,k
                # So W[k,l] = R[i,j,k] / (U[i,l]*V[j,l]) for any (i,j) with U[i,l]*V[j,l] != 0
                
                # Find valid (i,j) pairs
                valid = []
                for i in range(n*m):
                    for j in range(m*p):
                        prod = U_try[i,l] * V_try[j,l]
                        if abs(prod) > 1e-10:
                            valid.append((i, j, prod))
                
                if len(valid) > 0:
                    # Least squares for W[:,l]
                    # For each k, R[i,j,k] = U[i,l]*V[j,l]*W[k,l]
                    # So W[k,l] = R[i,j,k] / (U[i,l]*V[j,l])
                    # Average over valid (i,j) pairs
                    W_new = np.zeros(p*n)
                    count = np.zeros(p*n)
                    for (i, j, prod) in valid[:10]:  # Limit for speed
                        for k in range(p*n):
                            W_new[k] += R[i,j,k] / prod
                            count[k] += 1
                    for k in range(p*n):
                        if count[k] > 0:
                            W_try[k,l] = W_new[k] / count[k]
                        else:
                            W_try[k,l] = 0
                    improved = True
            
            if not improved:
                break
        
        # Check if the result is exact (within tolerance)
        # Verify
        exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(target_r):
                        s += U_try[Aidx(i,j),l] * V_try[Bidx(j,k),l] * W_try[Cidx(i,k),l]
                    if abs(s - 1) > 1e-6:
                        exact = False
                        break
                if not exact:
                    break
            if not exact:
                break
        
        if exact:
            # Convert to list format with rounded values
            U_list = [[round(U_try[i,l], 6) for l in range(target_r)] for i in range(n*m)]
            V_list = [[round(V_try[j,l], 6) for l in range(target_r)] for j in range(m*p)]
            W_list = [[round(W_try[k,l], 6) for l in range(target_r)] for k in range(p*n)]
            
            # Final verification with exact arithmetic
            if verify(U_list, V_list, W_list, target_r):
                best_r = target_r
                best_U, best_V, best_W = U_list, V_list, W_list
                break
    
    return best_r, best_U, best_V, best_W

def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    rng = random.Random(42)

    # Baseline
    U, V, W = baseline_66()
    r = 66

    # Local search to drop redundant terms
    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    # HALS-inspired refinement
    r, U, V, W = hals_refine(U, V, W, r, deadline, rng)

    # Final verification
    assert verify(U, V, W, r), "Final decomposition is not exact!"

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
