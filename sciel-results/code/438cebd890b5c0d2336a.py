"""Model-and-optimize: parameterize rank-r factorization and fit/optimize parameters.
Target: exact rank decomposition of 3x4x6 matrix multiplication tensor, minimize rank.
Strategy: start from a partition-based baseline (rank 66), then run a randomized
merge-and-drop local search to reduce rank while preserving exactness.
"""
import random
import time

n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p

# Index helpers
def Aidx(i, j): return i * m + j          # U row index (n*m)
def Bidx(j, k): return j * p + k          # V row index (m*p)
def Cidx(i, k): return k * n + i          # W row index (p*n), trace convention W[k*n+i]

def true_tensor():
    """Return the list of (i,j,k) triples that are 1 in the matmul tensor."""
    return [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

def verify(U, V, W, r):
    """Verify exact decomposition using vectorized numpy."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[Aidx(i,j)], Va[Bidx(j,k)] * Wa[Cidx(i,k)])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def baseline_66():
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

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return U, V, W

def is_zero_term(U, V, W, r, l):
    """Check if term l contributes nothing to any tensor entry."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l] != 0:
                    return False
    return True

def try_merge(U, V, W, r, l1, l2):
    """Try to merge rank terms l1 and l2 into a single rank-1 term.
    
    For the sum of two rank-1 tensors u1⊗v1⊗w1 + u2⊗v2⊗w2 to be rank-1,
    we need the supports to be compatible. Specifically, for each (i,j,k):
      u1[i,j]*v1[j,k]*w1[i,k] + u2[i,j]*v2[j,k]*w2[i,k] = u3[i,j]*v3[j,k]*w3[i,k]
    
    A sufficient condition: if the "effective" 2D projections are proportional.
    We test: can we find scalars a,b (not both zero) such that
      a*u1⊗v1⊗w1 + b*u2⊗v2⊗w2 is rank-1?
    
    For rank-1, the tensor T_{ijk} = a*u1[i,j]*v1[j,k]*w1[i,k] + b*u2[i,j]*v2[j,k]*w2[i,k]
    must factor. We check by sampling: if for all (i,j,k), T_{ijk} = T_{i'j'k}*T_{ij'k'}/T_{i'j'k'} (when defined).
    
    Simpler approach: check if the two terms have proportional "marginals" in a way that allows merging.
    We'll use a numerical approach: form the combined tensor and check its rank via SVD.
    """
    import numpy as np
    # Build the combined 3-tensor for just terms l1, l2
    # T[i,j,k] = U[i,j,l1]*V[j,k,l1]*W[i,k,l1] + U[i,j,l2]*V[j,k,l2]*W[i,k,l2]
    # For this to be rank-1, it must equal u[i,j]*v[j,k]*w[i,k] for some u,v,w.
    
    # Extract the 2D slices and check consistency
    # A rank-1 3-tensor has the property that all 2x2 minors of any unfolding are zero.
    # Let's use the matricization: reshape to (n*m, p) matrix M[i,j;k] = T[i,j,k]
    # Actually reshape to (n*m*p) vector and check... no, we need the tensor structure.
    
    # Better: check if the tensor is rank-1 by verifying that for any two (i,j) pairs,
    # the vectors over k are proportional, and similarly for other unfoldings.
    
    # Build T as array of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = (U[Aidx(i,j)][l1] * V[Bidx(j,k)][l1] * W[Cidx(i,k)][l1] +
                           U[Aidx(i,j)][l2] * V[Bidx(j,k)][l2] * W[Cidx(i,k)][l2])
    
    # Check rank-1: all 2x2 minors of the (n*m) x p unfolding must be zero
    # Unfold along (i,j) vs k: matrix of shape (n*m, p)
    M1 = T.reshape(n*m, p)
    # Check all 2x2 minors
    for a in range(n*m):
        for b in range(a+1, n*m):
            for c in range(p):
                for d in range(c+1, p):
                    minor = M1[a,c]*M1[b,d] - M1[a,d]*M1[b,c]
                    if abs(minor) > 1e-9:
                        return None
    
    # Also check (i) vs (j,k) unfolding
    M2 = T.reshape(n, m*p)
    for a in range(n):
        for b in range(a+1, n):
            for c in range(m*p):
                for d in range(c+1, m*p):
                    minor = M2[a,c]*M2[b,d] - M2[a,d]*M2[b,c]
                    if abs(minor) > 1e-9:
                        return None
    
    # The combined tensor IS rank-1. Now extract the factors.
    # Find a nonzero entry to normalize
    found = False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i,j,k]) > 1e-12:
                    # Set u[i,j] = 1, then v[j,k]*w[i,k] = T[i,j,k]
                    # This is underdetermined; we need to find u,v,w such that u[i,j]*v[j,k]*w[i,k] = T[i,j,k]
                    # Use log-domain or iterative approach
                    found = True
                    break
            if found:
                break
        if found:
            break
    
    if not found:
        # Zero tensor, trivially rank-1 with all zeros
        return (np.zeros(n*m), np.zeros(m*p), np.zeros(p*n))
    
    # Extract factors from rank-1 tensor T
    # u[i,j] * v[j,k] * w[i,k] = T[i,j,k]
    # Fix u[0,0] = 1 (or some value), then solve for others
    # This is a system of equations; use iterative proportional fitting
    u = np.ones(n*m)
    v = np.ones(m*p)
    w = np.ones(p*n)
    
    for iteration in range(20):
        # Update u: u[i,j] = T[i,j,k] / (v[j,k]*w[i,k]) for some fixed k
        for i in range(n):
            for j in range(m):
                # Find a k where v[j,k]*w[i,k] != 0
                for k in range(p):
                    denom = v[Bidx(j,k)] * w[Cidx(i,k)]
                    if abs(denom) > 1e-12:
                        u[Aidx(i,j)] = T[i,j,k] / denom
                        break
        # Update v: v[j,k] = T[i,j,k] / (u[i,j]*w[i,k]) for some fixed i
        for j in range(m):
            for k in range(p):
                for i in range(n):
                    denom = u[Aidx(i,j)] * w[Cidx(i,k)]
                    if abs(denom) > 1e-12:
                        v[Bidx(j,k)] = T[i,j,k] / denom
                        break
        # Update w: w[i,k] = T[i,j,k] / (u[i,j]*v[j,k]) for some fixed j
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    denom = u[Aidx(i,j)] * v[Bidx(j,k)]
                    if abs(denom) > 1e-12:
                        w[Cidx(i,k)] = T[i,j,k] / denom
                        break
    
    return (u, v, w)

def try_drop(U, V, W, r, l):
    """Check if removing term l still gives an exact decomposition."""
    # Recompute without term l using numpy for speed
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[Aidx(i,j)], Va[Bidx(j,k)] * Wa[Cidx(i,k)])
                s -= U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def local_search(U, V, W, r, deadline, rng):
    best_r = r
    best_U, best_V, best_W = U, V, W
    
    while time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Try to drop zero-contribution terms
        for l in range(best_r):
            if is_zero_term(best_U, best_V, best_W, best_r, l):
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try to drop terms that are redundant (removing them still gives exact decomp)
        for l in range(best_r):
            if try_drop(best_U, best_V, best_W, best_r, l):
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 3: Try to merge pairs of terms into one
        merged = False
        for l1 in range(best_r):
            if time.time() > deadline - 5:
                break
            for l2 in range(l1+1, best_r):
                if time.time() > deadline - 5:
                    break
                result = try_merge(best_U, best_V, best_W, best_r, l1, l2)
                if result is not None:
                    u_new, v_new, w_new = result
                    # Remove terms l1 and l2, add the merged term
                    newU = []
                    newV = []
                    newW = []
                    for row_u, row_v, row_w in zip(best_U, best_V, best_W):
                        pass
                    # Build new matrices
                    keep_indices = [l for l in range(best_r) if l != l1 and l != l2]
                    newU = [row[:l1] + row[l1+1:l2] + row[l2+1:] for row in best_U]
                    newV = [row[:l1] + row[l1+1:l2] + row[l2+1:] for row in best_V]
                    newW = [row[:l1] + row[l1+1:l2] + row[l2+1:] for row in best_W]
                    # Add the new merged column
                    for idx in range(len(newU)):
                        newU[idx].append(float(u_new[idx]))
                    for idx in range(len(newV)):
                        newV[idx].append(float(v_new[idx]))
                    for idx in range(len(newW)):
                        newW[idx].append(float(w_new[idx]))
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
                    improved = True
                    merged = True
                    break
            if merged:
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

    # Local search to drop/merge redundant terms
    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    # Final verification
    if not verify(U, V, W, r):
        # Fallback to baseline if something went wrong
        U, V, W = baseline_66()
        r = 66

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
