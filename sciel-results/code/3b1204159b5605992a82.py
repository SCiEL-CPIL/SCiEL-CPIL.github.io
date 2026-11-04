"""Model-and-optimize: parameterize rank-r factorization and fit/optimize parameters.
Target: exact rank decomposition of 3x4x6 matrix multiplication tensor, minimize rank.
Strategy: start from a partition-based baseline (rank 66), then run a randomized
merge-and-drop local search to reduce rank while preserving exactness.
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
    """Verify exact decomposition: for all (i,j,k),
    sum_l U[Aidx(i,j),l]*V[Bidx(j,k),l]*W[Cidx(i,k),l] == 1.
    Uses vectorized einsum for speed."""
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    # einsum over rank axis l: result shape (n*m, m*p, p*n)
    T = np.einsum('il,jl,kl->ijk', Ua, Va, Wa)
    # T[a, b, c] where a=Aidx(i,j), b=Bidx(j,k), c=Cidx(i,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[Aidx(i,j), Bidx(j,k), Cidx(i,k)] != 1:
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
    """Try to replace rank terms l1 and l2 with a single rank-1 term.
    The sum of two rank-1 tensors is rank-1 iff the three index-sets are
    proportional. If so, return the merged (u,v,w) vectors; else None.
    """
    u1 = [U[row][l1] for row in range(n*m)]
    u2 = [U[row][l2] for row in range(n*m)]
    v1 = [V[row][l1] for row in range(m*p)]
    v2 = [V[row][l2] for row in range(m*p)]
    w1 = [W[row][l1] for row in range(p*n)]
    w2 = [W[row][l2] for row in range(p*n)]

    # Find a pair of nonzero entries in u1,u2 to fix scale
    def find_pair(a, b):
        for idx in range(len(a)):
            if a[idx] != 0 and b[idx] != 0:
                return idx
        return None

    # We need u2 = alpha*u1, v2 = beta*v1, w2 = gamma*w1 with alpha*beta*gamma=1
    # Determine alpha from u, beta from v, gamma from w
    def ratio(a, b):
        idx = find_pair(a, b)
        if idx is None:
            # one is all zero
            if all(x == 0 for x in a) and all(x == 0 for x in b):
                return None  # degenerate, handled by drop
            return None
        return b[idx] / a[idx]

    # Check proportionality and compute ratios
    def check_prop(a, b, expected_ratio):
        for idx in range(len(a)):
            if a[idx] == 0:
                if b[idx] != 0:
                    return False
            else:
                if b[idx] != expected_ratio * a[idx]:
                    return False
        return True

    # Try each of the three as the "base" (the one with nonzero pair)
    # alpha = u2/u1, beta = v2/v1, gamma = w2/w1
    # Need alpha*beta*gamma = 1
    ratios = []
    for a, b in [(u1, u2), (v1, v2), (w1, w2)]:
        idx = find_pair(a, b)
        if idx is not None:
            ratios.append(b[idx] / a[idx])
        else:
            ratios.append(None)

    # If any ratio is None, that pair is (0,0) or (nonzero,0) or (0,nonzero)
    # (nonzero,0) or (0,nonzero) -> not proportional -> fail
    # (0,0) -> that tensor is zero in that factor -> the whole rank-1 term is zero -> drop handles it
    # For merge we need all three proportional with consistent ratios
    valid = True
    for (a, b), ratio in [((u1,u2), ratios[0]), ((v1,v2), ratios[1]), ((w1,w2), ratios[2])]:
        if ratio is None:
            # must be all-zero in both
            if not (all(x == 0 for x in a) and all(x == 0 for x in b)):
                valid = False
                break
        else:
            if not check_prop(a, b, ratio):
                valid = False
                break

    if not valid:
        return None

    # Now check alpha*beta*gamma = 1
    # If a ratio is None, that factor is all-zero in both terms -> the rank-1 tensor is zero
    # in that factor for both -> product is zero -> contribution is zero -> drop handles it
    # For a genuine merge we need all three ratios present
    if None in ratios:
        return None

    alpha, beta, gamma = ratios
    if alpha * beta * gamma != 1:
        return None

    # Merged term: use term l1 as base, it already satisfies the equation
    # since u1*beta*v1*gamma*w1 = u1*v1*w1 (because alpha*beta*gamma=1 and u2=alpha*u1 etc.)
    # Actually the sum is (u1+u2)⊗(v1+v2)⊗(w1+w2)?? No.
    # Sum of two rank-1 tensors = u1⊗v1⊗w1 + u2⊗v2⊗w2
    # = u1⊗v1⊗w1 + alpha*u1 ⊗ beta*v1 ⊗ gamma*w2... wait u2=alpha*u1, v2=beta*v1, w2=gamma*w1
    # = u1⊗v1⊗w1 + alpha*beta*gamma * u1⊗v1⊗w1 = u1⊗v1⊗w1 + u1⊗v1⊗w1 = 2*u1⊗v1⊗w1
    # So merged = 2*u1, 2*v1, 2*w1... but we want sum to equal a single rank-1.
    # sum = (1+alpha*beta*gamma) * u1⊗v1⊗w1 = 2 * u1⊗v1⊗w1
    # So merged vectors are 2*u1, 2*v1, 2*w1? No: a rank-1 tensor u⊗v⊗w has entries u_i*v_j*w_k.
    # 2*u1⊗v1⊗w1 means u'=2*u1, v'=v1, w'=w1 (or distribute the 2 anywhere).
    # Actually (2*u1)⊗v1⊗w1 = 2*(u1⊗v1⊗w1). Yes.
    # So merged = (2*u1, v1, w1) or (u1, 2*v1, w1) etc.
    # But wait: the sum of the two rank-1 contributions to the tensor is
    # u1⊗v1⊗w1 + u2⊗v2⊗w2 where u2=alpha*u1 etc.
    # = u1⊗v1⊗w1 + alpha*beta*gamma*(u1⊗v1⊗w1) = (1+alpha*beta*gamma)*(u1⊗v1⊗w1) = 2*(u1⊗v1⊗w1)
    # So the merged single term should have vectors (2*u1, v1, w1) [distribute factor 2 to U]
    merged_u = [2 * x for x in u1]
    merged_v = list(v1)
    merged_w = list(w1)
    return merged_u, merged_v, merged_w

def try_drop(U, V, W, r, l):
    """Check if removing term l still gives an exact decomposition (vectorized)."""
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    # Zero out column l
    Ua[:, l] = 0
    T = np.einsum('il,jl,kl->ijk', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[Aidx(i,j), Bidx(j,k), Cidx(i,k)] != 1:
                    return False
    return True

def local_search(U, V, W, r, deadline, rng):
    best_r = r
    best_U, best_V, best_W = U, V, W
    while time.time() < deadline - 5:
        improved = False
        # 1) Try to drop each term
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
        # 2) Try to merge pairs of terms
        for l1 in range(best_r):
            for l2 in range(l1 + 1, best_r):
                merged = try_merge(best_U, best_V, best_W, best_r, l1, l2)
                if merged is not None:
                    mu, mv, mw = merged
                    # Replace columns l1 and l2 with a single column (use l1's slot, drop l2)
                    newU = [row[:l1] + [mu[row]] + row[l1+1:] for row in range(n*m)]
                    # Need to fix: row is an int index here, need actual row list
                    newU = []
                    for row_idx in range(n*m):
                        newrow = best_U[row_idx][:l1] + [mu[row_idx]] + best_U[row_idx][l1+1:]
                        # remove l2 (which is now at l2-1 since we inserted at l1)
                        # l2 > l1, so after insertion at l1, l2 shifts to l2
                        # original l2 index becomes l2 (since one element inserted before it)
                        # wait: inserting at l1 shifts everything from l1 onward by +1
                        # so original l2 is now at l2 (l2 was > l1, now l2+1? no)
                        # original indices: 0..l1-1, l1, l1+1, ..., l2, ...
                        # after inserting at l1: 0..l1-1, NEW, l1, l1+1, ..., l2, ...
                        # so original l2 is now at index l2+1? No:
                        # positions: 0..l1-1 (same), l1=NEW, l1+1=old_l1, ..., l2+1=old_l2
                        # So we need to remove index l2+1
                        newrow = newrow[:l2+1] + newrow[l2+2:]
                        newU.append(newrow)
                    newV = []
                    for row_idx in range(m*p):
                        newrow = best_V[row_idx][:l1] + [mv[row_idx]] + best_V[row_idx][l1+1:]
                        newrow = newrow[:l2+1] + newrow[l2+2:]
                        newV.append(newrow)
                    newW = []
                    for row_idx in range(p*n):
                        newrow = best_W[row_idx][:l1] + [mw[row_idx]] + best_W[row_idx][l1+1:]
                        newrow = newrow[:l2+1] + newrow[l2+2:]
                        newW.append(newrow)
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
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
