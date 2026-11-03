DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from collections import defaultdict

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (u_dict, v_dict, w_dict) with keys (i,j), (j,k), (k,i) in local 2x2 coords.
    """
    terms = []
    # P1 = (A00+A11)(B00+B11)
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # P2 = (A10+A11)B00
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    # P3 = A00(B10-B00)
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    # P4 = A11(B01-B10)
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    # P5 = (A00+A01)B11
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    # P6 = (A10-A00)(B00+B01)
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # P7 = (A01-A11)(B10+B11)
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _construct_2x3x9():
    """
    Construct base decomposition for 2x3 * 3x9 using Strassen block strategy.
    Split 9 columns into 4 pairs + 1 single for the 2x2 A-block, plus direct 2x1x9.
    Returns (U, V, W) with rank 50.
    """
    n, m, p = 2, 3, 9
    U_terms = []
    V_terms = []
    W_terms = []

    strassen_terms = _strassen_2x2x2_terms()

    # Part 1: 4 Strassen blocks on A[0:2, 0:2] x B[0:2, (k1,k2)] for 4 column pairs
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                actual_k = k1 if lk == 0 else k2
                V_term[(lj, actual_k)] = coeff
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                actual_k = k1 if lk == 0 else k2
                W_term[(li, actual_k)] = coeff
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)

    # Part 2: single column k=8 for A[0:2, 0:2] x B[0:2, 8] -> direct (4 terms)
    for i in range(2):
        for j in range(2):
            U_terms.append({(i, j): 1})
            V_terms.append({(j, 8): 1})
            W_terms.append({(i, 8): 1})

    # Part 3: A[0:2, 2] x B[2, 0:9] -> direct (18 terms)
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})

    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff

    return U, V, W

def _decompose_to_terms(U, V, W):
    """Convert matrices to list of rank-1 terms (sparse nonzero dicts)."""
    nU = len(U)
    nV = len(V)
    nW = len(W)
    rank = len(U[0]) if U else 0
    terms = []
    for c in range(rank):
        u = {i: U[i][c] for i in range(nU) if U[i][c] != 0}
        v = {j: V[j][c] for j in range(nV) if V[j][c] != 0}
        w = {k: W[k][c] for k in range(nW) if W[k][c] != 0}
        if u and v and w:
            terms.append((u, v, w))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of rank-1 terms back to (U, V, W) matrices."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(terms):
        for (i, j), coeff in u.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w.items():
            W[k * n + i][c] = coeff
    return U, V, W

def _term_tensor(term):
    """Return the dense (n*m, m*p, p*n) tensor for one rank-1 term (as dict of nonzero)."""
    u, v, w = term
    out = {}
    for i, ui in u.items():
        for j, vj in v.items():
            for k, wk in w.items():
                key = (i, j, k)
                out[key] = ui * vj * wk
    return out

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition exactly reconstructs the MM tensor."""
    rank = len(U[0]) if U else 0
    T = {}
    for c in range(rank):
        u = {i: U[i][c] for i in range(n * m) if U[i][c] != 0}
        v = {j: V[j][c] for j in range(m * p) if V[j][c] != 0}
        w = {k: W[k][c] for k in range(p * n) if W[k][c] != 0}
        for i, ui in u.items():
            ii, jj = divmod(i, m)
            for j, vj in v.items():
                jj2, kk = divmod(j, p)
                if jj != jj2:
                    continue
                for k, wk in w.items():
                    kk2, ii2 = divmod(k, n)
                    if kk != kk2 or ii2 != ii:
                        continue
                    key = (i, j, k)
                    T[key] = T.get(key, 0.0) + ui * vj * wk
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                got = T.get((i * m + j, j * p + k, k * n + i), 0.0)
                if abs(got - expected) > tol:
                    return False
    for key, val in T.items():
        i, j, k = key
        ii, jj = divmod(i, m)
        jj2, kk = divmod(j, p)
        kk2, ii2 = divmod(k, n)
        if jj != jj2 or kk != kk2 or ii2 != ii:
            if abs(val) > tol:
                return False
    return True

def _try_reduce(terms, n, m, p, deadline, max_iter=200):
    """
    Greedily try to remove terms that are linear combinations of the rest.
    Returns a possibly smaller list of terms.
    """
    current = list(terms)
    for _ in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        removed = False
        for idx in range(len(current)):
            if time.time() > deadline - 10.0:
                break
            target = _term_tensor(current[idx])
            others = [current[j] for j in range(len(current)) if j != idx]
            keys = list(target.keys())
            if not keys:
                continue
            A = np.zeros((len(keys), len(others)))
            b = np.array([target[k] for k in keys], dtype=float)
            for col, ot in enumerate(others):
                otensor = _term_tensor(ot)
                for r, key in enumerate(keys):
                    A[r, col] = otensor.get(key, 0.0)
            try:
                coeff, res, rnk, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                if len(res) > 0:
                    resid = res[0]
                else:
                    resid = np.linalg.norm(A @ coeff - b)
                bnorm = np.linalg.norm(b)
                if bnorm == 0 or resid < 1e-8 * max(bnorm, 1e-12):
                    current.pop(idx)
                    removed = True
                    break
            except Exception:
                continue
        if not removed:
            break
    return current

def _try_group_reduce(terms, n, m, p, deadline):
    """
    Detect groups of terms with identical (U,V) patterns where W vectors
    are linearly dependent, and eliminate redundant terms.
    """
    current = list(terms)
    
    # Build tensor representations once
    tensors = [_term_tensor(t) for t in current]
    
    # Group by (U, V) fingerprint
    groups = defaultdict(list)
    for idx in range(len(current)):
        u = tuple(sorted(current[idx][0].items()))
        v = tuple(sorted(current[idx][1].items()))
        groups[(u, v)].append(idx)
    
    # Within each group, check if W vectors are linearly dependent
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        if time.time() > deadline - 10.0:
            break
        w_vecs = []
        all_w_indices = set()
        for idx in idxs:
            w = current[idx][2]
            for k in w:
                all_w_indices.add(k)
        all_w_indices = sorted(all_w_indices)
        for idx in idxs:
            w = current[idx][2]
            w_vecs.append(np.array([w.get(k, 0.0) for k in all_w_indices]))
        
        w_matrix = np.array(w_vecs).T
        rank_w = np.linalg.matrix_rank(w_matrix, tol=1e-10)
        if rank_w < len(idxs):
            null_dim = len(idxs) - rank_w
            to_remove = set(idxs[-null_dim:])
            for idx in sorted(to_remove, reverse=True):
                current.pop(idx)
            tensors = [_term_tensor(t) for t in current]
    
    return current

def _try_global_linear_reduce(terms, n, m, p, deadline):
    """
    Global linear algebra approach to find minimal basis.
    """
    current = list(terms)
    tensors = [_term_tensor(t) for t in current]
    
    all_keys = set()
    for t in tensors:
        all_keys.update(t.keys())
    all_keys = sorted(all_keys)
    
    if not all_keys:
        return current
    
    M = np.zeros((len(all_keys), len(current)))
    for col in range(len(current)):
        for r, key in enumerate(all_keys):
            M[r, col] = tensors[col].get(key, 0.0)
    
    rank = np.linalg.matrix_rank(M, tol=1e-10)
    if rank < len(current):
        remaining = list(range(len(current)))
        for _ in range(len(current) - rank):
            if time.time() > deadline - 10.0:
                break
            removed = False
            for i in range(len(remaining)):
                if time.time() > deadline - 10.0:
                    break
                idx = remaining[i]
                if i == 0:
                    continue
                others = [remaining[j] for j in range(len(remaining)) if j != i]
                if not others:
                    continue
                A = M[:, others]
                b = M[:, idx]
                try:
                    coeff, res, rnk, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    if len(res) > 0:
                        resid = res[0]
                    else:
                        resid = np.linalg.norm(A @ coeff - b)
                    bnorm = np.linalg.norm(b)
                    if bnorm == 0 or resid < 1e-8 * max(bnorm, 1e-12):
                        remaining.pop(i)
                        removed = True
                        break
                except Exception:
                    continue
            if not removed:
                break
        current = [current[i] for i in remaining]
    
    return current

def _rank1_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Base Strassen block decomposition (rank 50)
    try:
        U, V, W = _construct_2x3x9()
        if not _verify(U, V, W, n, m, p):
            raise ValueError("base decomposition failed verification")
    except Exception:
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])

    # Refinement: multiple reduction strategies
    try:
        terms = _decompose_to_terms(U, V, W)
        
        # Strategy 1: Group-based reduction (fast, exploits structure)
        reduced1 = _try_group_reduce(terms, n, m, p, deadline)
        if len(reduced1) < best_rank:
            rU, rV, rW = _terms_to_matrices(reduced1, n, m, p)
            if _verify(rU, rV, rW, n, m, p):
                best_U, best_V, best_W = rU, rV, rW
                best_rank = len(rU[0])
        
        # Strategy 2: Greedy single-term elimination on the best so far
        if time.time() < deadline - 20.0:
            terms2 = _decompose_to_terms(best_U, best_V, best_W)
            reduced2 = _try_reduce(terms2, n, m, p, deadline, max_iter=200)
            if len(reduced2) < best_rank:
                rU, rV, rW = _terms_to_matrices(reduced2, n, m, p)
                if _verify(rU, rV, rW, n, m, p):
                    best_U, best_V, best_W = rU, rV, rW
                    best_rank = len(rU[0])
        
        # Strategy 3: Global linear algebra on the best so far
        if time.time() < deadline - 15.0:
            terms3 = _decompose_to_terms(best_U, best_V, best_W)
            reduced3 = _try_global_linear_reduce(terms3, n, m, p, deadline)
            if len(reduced3) < best_rank:
                rU, rV, rW = _terms_to_matrices(reduced3, n, m, p)
                if _verify(rU, rV, rW, n, m, p):
                    best_U, best_V, best_W = rU, rV, rW
                    best_rank = len(rU[0])
    except Exception:
        pass

    # Final safety check
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _rank1_decomposition(n, m, p)

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
