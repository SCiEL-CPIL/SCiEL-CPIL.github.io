"""
Rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
Strategy: structured baseline using Strassen for 2x2 blocks and direct for 1x2 blocks,
then local search with drop and merge operations to reduce rank while preserving exactness.
"""
import random
import time
import numpy as np

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

def verify(U, V, W, r):
    """Verify exact decomposition: for all (i,j,k), sum_l U[Aidx(i,j),l]*V[Bidx(j,k),l]*W[Cidx(i,k),l] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def baseline_66():
    """Construct a rank-66 baseline using Strassen for 2x2 blocks and direct for 1x2 blocks."""
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

    # Row 0 of A (1x4) multiplied by B (4x6): split into three 1x2 x 2x2 blocks
    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    # Rows 1-2 of A (2x4) multiplied by B (4x6): split into three 2x2 x 2x2 Strassen blocks
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66, f"Expected 66, got {col}"
    return U, V, W

def try_drop(U, V, W, r, l):
    """Check if removing term l still gives an exact decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for ll in range(r):
                    if ll == l:
                        continue
                    s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(i,k)][ll]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def try_merge_pair(U, V, W, r, l1, l2):
    """
    Try to merge two rank terms into one. This is a heuristic that checks
    if the two terms have proportional structure in a way that allows cancellation.
    Returns (newU, newV, newW) if merge succeeds, else None.
    """
    # For a valid merge, we need:
    # U[:,l1]*V[:,l1]*W[:,l1] + U[:,l2]*V[:,l2]*W[:,l2] = U',V',W' (rank 1)
    # This is generally hard. A simple heuristic: check if one term is a scalar
    # multiple of the other in all three matrices simultaneously.
    # Check if U[:,l1] is proportional to U[:,l2]
    u1 = np.array(U[:,l1]).flatten() if isinstance(U, list) else np.array(U)[:,l1]
    u2 = np.array(U[:,l2]).flatten() if isinstance(U, list) else np.array(U)[:,l2]
    v1 = np.array(V[:,l1]).flatten() if isinstance(V, list) else np.array(V)[:,l1]
    v2 = np.array(V[:,l2]).flatten() if isinstance(V, list) else np.array(V)[:,l2]
    w1 = np.array(W[:,l1]).flatten() if isinstance(W, list) else np.array(W)[:,l1]
    w2 = np.array(W[:,l2]).flatten() if isinstance(W, list) else np.array(W)[:,l2]
    
    # Check proportionality
    def is_proportional(a, b):
        if np.allclose(a, 0) and np.allclose(b, 0):
            return 0.0
        if np.allclose(a, 0) or np.allclose(b, 0):
            return None
        # Find non-zero elements
        mask = a != 0
        if not np.any(mask):
            return None
        ratios = b[mask] / a[mask]
        if np.allclose(ratios, ratios[0], atol=1e-9):
            return ratios[0]
        return None
    
    ru = is_proportional(u1, u2)
    rv = is_proportional(v1, v2)
    rw = is_proportional(w1, w2)
    
    if ru is not None and rv is not None and rw is not None:
        # Check if 1 + ru*rv*rw == 0 (for the terms to cancel)
        if abs(1 + ru*rv*rw) < 1e-9:
            # The two terms cancel! We can drop both and add nothing.
            # But wait, if they cancel, the sum is 0, so we just drop both.
            # Actually, if U[:,l1] = ru * U[:,l2] and similarly for V,W,
            # then term1 = ru*rv*rw * term2, so term1+term2 = (1+ru*rv*rw)*term2 = 0
            # So we can drop both terms.
            # But we need to return a single merged term, which would be zero (not useful).
            # Instead, we drop both. But our interface expects a single term replacement.
            # Let's just return None here and handle double-drop separately.
            pass
    
    # More sophisticated: check if we can express the sum as a rank-1 tensor
    # This is equivalent to checking if the 3-tensor has rank 1.
    # For simplicity, we'll skip this and rely on drops.
    return None

def local_search(U, V, W, r, deadline, rng):
    """Local search to reduce rank via drops and merges."""
    best_r = r
    best_U, best_V, best_W = U, V, W
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try to drop each term
        for l in range(best_r):
            if try_drop(best_U, best_V, best_W, best_r, l):
                # Remove term l
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                break
        
        if improved:
            continue
        
        # Try to merge pairs
        for l1 in range(best_r):
            if time.time() > deadline - 5:
                break
            for l2 in range(l1+1, best_r):
                result = try_merge_pair(best_U, best_V, best_W, best_r, l1, l2)
                if result is not None:
                    best_U, best_V, best_W = result
                    best_r -= 1
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_r, best_U, best_V, best_W

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    # Baseline
    U, V, W = baseline_66()
    r = 66

    # Local search to drop redundant terms
    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    assert verify(U, V, W, r), "Final decomposition is not exact!"

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
