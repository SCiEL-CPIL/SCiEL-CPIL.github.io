"""Matrix multiplication 3x4x6: rank-minimal exact tensor decomposition.
Strategy: block-structured baseline (rank 66) + enhanced local search with
merge operations and randomized restarts to find lower rank.
"""
import numpy as np
import time
import random
from itertools import combinations

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

def baseline_66():
    """Construct rank-66 baseline using block partitioning."""
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
        # M1
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        # M2
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        # M3
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        # M4
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        # M5
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        # M6
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        # M7
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    # Partition: A rows 1+2, cols 2+2; B rows 2+2, cols 2+2+2
    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def verify_np(U, V, W):
    """Vectorized verification using NumPy."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    result = np.einsum('ijl,jkl,kil->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(result, 1.0, atol=1e-9)

def try_drop(U, V, W, l):
    """Check if removing term l still gives an exact decomposition."""
    U_new = np.delete(U, l, axis=1)
    V_new = np.delete(V, l, axis=1)
    W_new = np.delete(W, l, axis=1)
    return verify_np(U_new, V_new, W_new)

def try_merge_pair(U, V, W, l1, l2):
    """Try to merge terms l1 and l2 into a single term.
    
    Two rank-1 terms can be merged if there exist scalars a, b, c such that:
    a*U[:,l1] + b*U[:,l2] = c*U_merged
    a*V[:,l1] + b*V[:,l2] = c*V_merged  
    a*W[:,l1] + b*W[:,l2] = c*W_merged
    
    This is equivalent to checking if the 2x2 matrices formed by pairing
    corresponding entries have rank 1.
    """
    # Check if the two terms are proportional in all three factors
    U1, U2 = U[:, l1], U[:, l2]
    V1, V2 = V[:, l1], V[:, l2]
    W1, W2 = W[:, l1], W[:, l2]
    
    # For merge to work, we need: U1/V1/W1 proportional to U2/V2/W2
    # i.e., there exist alpha, beta such that U2 = alpha*U1, V2 = beta*V1, W2 = gamma*W1
    # and alpha = beta = gamma (same scaling)
    
    # Find scaling factors from non-zero entries
    def find_scale(x, y):
        """Find scale factor s such that y ≈ s*x, return s or None if not proportional."""
        mask = np.abs(x) > 1e-10
        if not np.any(mask):
            return None
        s_vals = y[mask] / x[mask]
        if not np.allclose(s_vals, s_vals[0], atol=1e-10):
            return None
        return s_vals[0]
    
    sU = find_scale(U1, U2)
    sV = find_scale(V1, V2)
    sW = find_scale(W1, W2)
    
    if sU is None or sV is None or sW is None:
        return None
    
    # Check if all scales are equal
    if not (np.isclose(sU, sV, atol=1e-10) and np.isclose(sV, sW, atol=1e-10)):
        return None
    
    # If proportional, we can merge: keep one term, remove the other
    # The merged term is just the original (they're the same up to scale)
    return (l1, l2, sU)

def local_search_enhanced(U, V, W, deadline, rng):
    """Enhanced local search with drop and merge operations."""
    r = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_r = r
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try drops first
        indices = list(range(best_r))
        rng.shuffle(indices)
        
        for l in indices:
            if time.time() > deadline - 10:
                break
            if try_drop(best_U, best_V, best_W, l):
                best_U = np.delete(best_U, l, axis=1)
                best_V = np.delete(best_V, l, axis=1)
                best_W = np.delete(best_W, l, axis=1)
                best_r -= 1
                improved = True
                break
        
        # If no drop found, try merges
        if not improved and best_r >= 2:
            pairs = list(combinations(range(best_r), 2))
            rng.shuffle(pairs)
            
            for l1, l2 in pairs:
                if time.time() > deadline - 10:
                    break
                merge_info = try_merge_pair(best_U, best_V, best_W, l1, l2)
                if merge_info is not None:
                    # Remove the second term (keep first)
                    best_U = np.delete(best_U, l2, axis=1)
                    best_V = np.delete(best_V, l2, axis=1)
                    best_W = np.delete(best_W, l2, axis=1)
                    best_r -= 1
                    improved = True
                    break
    
    return best_U, best_V, best_W, best_r

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Baseline
    U, V, W = baseline_66()
    
    # Local search to reduce rank
    U, V, W, r = local_search_enhanced(U, V, W, deadline, rng)
    
    # Final verification
    if not verify_np(U, V, W):
        # Fallback to baseline if something went wrong
        U, V, W = baseline_66()
        r = 66
    
    # Convert back to lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
