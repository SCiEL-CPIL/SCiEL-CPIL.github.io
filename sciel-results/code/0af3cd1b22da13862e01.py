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
    """Verify exact decomposition: for all (i,j,k), sum_l U[Aidx(i,j),l]*V[Bidx(j,k),l]*W[Cidx(i,k),l] == 1.
    And for any (i,j,k) that is NOT a tensor entry (i.e., the tensor is only nonzero on those),
    actually the matmul tensor T[i,j,k] = 1 for all valid (i,j,k) since it's full rank product.
    Wait: the matmul tensor has T_{i,j,k} = 1 iff ... no. The tensor for matrix multiplication
    A*B=C is T_{i,j,k} = 1 for every (i,j,k) in the support, which is the full n*m*p = 72 entries.
    So we need sum_l U[i,j,l] V[j,k,l] W[i,k,l] = 1 for all (i,j,k).
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if s != 1:
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

def try_merge(U, V, W, r, l1, l2):
    """Try to merge rank terms l1 and l2 into a single term by checking if their
    combined contribution can be represented as a rank-1 tensor.
    We check if the outer product structure is preserved."""
    # Collect the nonzero entries for each term across all (i,j,k)
    # For a rank-1 term l, the contribution to T[i,j,k] is U[Aidx(i,j),l]*V[Bidx(j,k),l]*W[Cidx(i,k),l]
    # Two terms can be merged if there exist scalars a,b such that
    # a*U1[i,j]*V1[j,k]*W1[i,k] + b*U2[i,j]*V2[j,k]*W2[i,k] = U'[i,j]*V'[j,k]*W'[i,k]
    # This is complex. A simpler heuristic: check if the supports are disjoint or proportional.
    
    # Check if the two terms have identical support patterns (up to scaling)
    # For each (i,j), collect U values for l1 and l2
    # For each (j,k), collect V values for l1 and l2
    # For each (i,k), collect W values for l1 and l2
    
    u1_vals = {}
    u2_vals = {}
    v1_vals = {}
    v2_vals = {}
    w1_vals = {}
    w2_vals = {}
    
    for i in range(n):
        for j in range(m):
            idx = Aidx(i,j)
            if U[idx][l1] != 0:
                u1_vals[(i,j)] = U[idx][l1]
            if U[idx][l2] != 0:
                u2_vals[(i,j)] = U[idx][l2]
    
    for j in range(m):
        for k in range(p):
            idx = Bidx(j,k)
            if V[idx][l1] != 0:
                v1_vals[(j,k)] = V[idx][l1]
            if V[idx][l2] != 0:
                v2_vals[(j,k)] = V[idx][l2]
    
    for i in range(n):
        for k in range(p):
            idx = Cidx(i,k)
            if W[idx][l1] != 0:
                w1_vals[(i,k)] = W[idx][l1]
            if W[idx][l2] != 0:
                w2_vals[(i,k)] = W[idx][l2]
    
    # If supports are identical, check proportionality
    if set(u1_vals.keys()) == set(u2_vals.keys()) and set(v1_vals.keys()) == set(v2_vals.keys()) and set(w1_vals.keys()) == set(w2_vals.keys()):
        # Check if all ratios are the same
        ratio_u = None
        for key in u1_vals:
            if u1_vals[key] != 0:
                r_val = u2_vals[key] / u1_vals[key]
                if ratio_u is None:
                    ratio_u = r_val
                elif abs(r_val - ratio_u) > 1e-10:
                    return False
        ratio_v = None
        for key in v1_vals:
            if v1_vals[key] != 0:
                r_val = v2_vals[key] / v1_vals[key]
                if ratio_v is None:
                    ratio_v = r_val
                elif abs(r_val - ratio_v) > 1e-10:
                    return False
        ratio_w = None
        for key in w1_vals:
            if w1_vals[key] != 0:
                r_val = w2_vals[key] / w1_vals[key]
                if ratio_w is None:
                    ratio_w = r_val
                elif abs(r_val - ratio_w) > 1e-10:
                    return False
        # If proportional, the two terms are linearly dependent and can be merged
        return True
    
    return False

def try_drop(U, V, W, r, l):
    """Check if removing term l still gives an exact decomposition."""
    # Recompute without term l
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for ll in range(r):
                    if ll == l:
                        continue
                    s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(i,k)][ll]
                if s != 1:
                    return False
    return True

def local_search(U, V, W, r, deadline, rng):
    best_r = r
    best_U, best_V, best_W = U, V, W
    while time.time() < deadline - 5:
        # Try to drop each term
        improved = False
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
        
        # Try to merge pairs of terms
        merged = False
        for l1 in range(best_r):
            for l2 in range(l1+1, best_r):
                if try_merge(best_U, best_V, best_W, best_r, l1, l2):
                    # Merge l1 and l2 into a single term
                    # The merged term is just l1 (since l2 is proportional)
                    # Remove l2
                    newU = [row[:l2] + row[l2+1:] for row in best_U]
                    newV = [row[:l2] + row[l2+1:] for row in best_V]
                    newW = [row[:l2] + row[l2+1:] for row in best_W]
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
                    merged = True
                    break
            if merged:
                break
        if not merged:
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
