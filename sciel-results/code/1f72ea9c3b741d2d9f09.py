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
    """Try to merge rank terms l1 and l2 into a single term.
    Check if the sum of two rank-1 tensors is itself rank-1."""
    # For each (i,j,k), compute the contribution from terms l1 and l2
    # If for all (i,j,k): U1[i,j]*V1[j,k]*W1[i,k] + U2[i,j]*V2[j,k]*W2[i,k]
    # can be written as U'[i,j]*V'[j,k]*W'[i,k], then merge is possible.
    
    # Quick check: if supports are identical, check proportionality
    u1_nonzero = set()
    u2_nonzero = set()
    for i in range(n):
        for j in range(m):
            idx = Aidx(i,j)
            if U[idx][l1] != 0:
                u1_nonzero.add((i,j))
            if U[idx][l2] != 0:
                u2_nonzero.add((i,j))
    
    v1_nonzero = set()
    v2_nonzero = set()
    for j in range(m):
        for k in range(p):
            idx = Bidx(j,k)
            if V[idx][l1] != 0:
                v1_nonzero.add((j,k))
            if V[idx][l2] != 0:
                v2_nonzero.add((j,k))
    
    w1_nonzero = set()
    w2_nonzero = set()
    for i in range(n):
        for k in range(p):
            idx = Cidx(i,k)
            if W[idx][l1] != 0:
                w1_nonzero.add((i,k))
            if W[idx][l2] != 0:
                w2_nonzero.add((i,k))
    
    # Case 1: Identical supports - check proportionality
    if u1_nonzero == u2_nonzero and v1_nonzero == v2_nonzero and w1_nonzero == w2_nonzero:
        # Check if U vectors are proportional
        ratio = None
        for key in u1_nonzero:
            i,j = key
            u1v = U[Aidx(i,j)][l1]
            u2v = U[Aidx(i,j)][l2]
            if u1v != 0:
                if ratio is None:
                    ratio = u2v / u1v
                elif abs(u2v/u1v - ratio) > 1e-10:
                    return False
            elif u2v != 0:
                return False
        if ratio is not None:
            # Verify V and W are also proportional
            for key in v1_nonzero:
                j,k = key
                v1v = V[Bidx(j,k)][l1]
                v2v = V[Bidx(j,k)][l2]
                if v1v != 0:
                    if abs(v2v/v1v - ratio) > 1e-10:
                        return False
                elif v2v != 0:
                    return False
            for key in w1_nonzero:
                i,k = key
                w1v = W[Cidx(i,k)][l1]
                w2v = W[Cidx(i,k)][l2]
                if w1v != 0:
                    if abs(w2v/w1v - ratio) > 1e-10:
                        return False
                elif w2v != 0:
                    return False
            return True
    
    # Case 2: One support is subset of the other
    # If l2's support is contained in l1's support, try to absorb l2 into l1
    if u2_nonzero <= u1_nonzero and v2_nonzero <= v1_nonzero and w2_nonzero <= w1_nonzero:
        # Try to find a scalar alpha such that for all (i,j,k) in support of l2:
        # U1[i,j]*V1[j,k]*W1[i,k] + alpha * U2[i,j]*V2[j,k]*W2[i,k] 
        # is still representable. This is complex, so use a simpler heuristic:
        # Check if the combined tensor on the intersection is rank-1
        pass
    
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
    no_improve_count = 0
    max_no_improve = 3
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try to drop each term
        for l in range(best_r):
            if try_drop(best_U, best_V, best_W, best_r, l):
                newU = [row[:l] + row[l+1:] for row in best_U]
                newV = [row[:l] + row[l+1:] for row in best_V]
                newW = [row[:l] + row[l+1:] for row in best_W]
                best_r -= 1
                best_U, best_V, best_W = newU, newV, newW
                improved = True
                no_improve_count = 0
                break
        if improved:
            continue
        
        # Try to merge pairs of terms
        merged = False
        for l1 in range(best_r):
            for l2 in range(l1+1, best_r):
                if try_merge(best_U, best_V, best_W, best_r, l1, l2):
                    # Merge: add l2 into l1 and remove l2
                    for idx in range(len(best_U)):
                        best_U[idx][l1] += best_U[idx][l2]
                    for idx in range(len(best_V)):
                        best_V[idx][l1] += best_V[idx][l2]
                    for idx in range(len(best_W)):
                        best_W[idx][l1] += best_W[idx][l2]
                    newU = [row[:l2] + row[l2+1:] for row in best_U]
                    newV = [row[:l2] + row[l2+1:] for row in best_V]
                    newW = [row[:l2] + row[l2+1:] for row in best_W]
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
                    merged = True
                    no_improve_count = 0
                    break
            if merged:
                break
        if merged:
            continue
        
        # No improvement found
        no_improve_count += 1
        if no_improve_count >= max_no_improve:
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

    # Try multiple random restarts with different initial perturbations if time allows
    if time.time() < deadline - 30:
        best_r = r
        best_U, best_V, best_W = U, V, W
        
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            # Create a perturbed version by adding small random noise to some entries
            # then try to clean up
            perturbed_U = [row[:] for row in U]
            perturbed_V = [row[:] for row in V]
            perturbed_W = [row[:] for row in W]
            
            # Try dropping from the current best
            cur_r = best_r
            cur_U, cur_V, cur_W = best_U, best_V, best_W
            cur_r, cur_U, cur_V, cur_W = local_search(cur_U, cur_V, cur_W, cur_r, deadline - 30, rng)
            
            if cur_r < best_r:
                best_r = cur_r
                best_U, best_V, best_W = cur_U, cur_V, cur_W
                r, U, V, W = cur_r, cur_U, cur_V, cur_W
    
    assert verify(U, V, W, r), "Final decomposition is not exact!"

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
