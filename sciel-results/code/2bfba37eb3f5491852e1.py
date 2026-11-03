import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    n, m, p = DIMENSIONS
    
    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, l): return j * p + l
    def w_idx(l, i): return l * n + i
    
    # Strassen 2x2x2 block: replaces 8 schoolbook terms with 7
    def strassen_block(i1, i2, j1, j2, l1, l2):
        """
        Returns list of 7 rank-1 terms as (u_dict, v_dict, w_dict)
        where each dict maps index -> coefficient
        """
        terms = []
        
        # M1 = (a11+a22)(b11+b22) -> contributes +1 to C11, +1 to C22
        terms.append((
            {u_idx(i1,j1): 1, u_idx(i2,j2): 1},
            {v_idx(j1,l1): 1, v_idx(j2,l2): 1},
            {w_idx(l1,i1): 1, w_idx(l2,i2): 1}
        ))
        # M2 = (a21+a22)b11 -> contributes -1 to C11, +1 to C22
        terms.append((
            {u_idx(i2,j1): 1, u_idx(i2,j2): 1},
            {v_idx(j1,l1): 1},
            {w_idx(l1,i1): -1, w_idx(l2,i2): 1}
        ))
        # M3 = a11(b21-b22) -> contributes +1 to C11, +1 to C21
        terms.append((
            {u_idx(i1,j1): 1},
            {v_idx(j2,l1): 1, v_idx(j2,l2): -1},
            {w_idx(l1,i1): 1, w_idx(l1,i2): 1}
        ))
        # M4 = a22(b11-b12) -> contributes -1 to C11, +1 to C12
        terms.append((
            {u_idx(i2,j2): 1},
            {v_idx(j1,l1): 1, v_idx(j1,l2): -1},
            {w_idx(l1,i1): -1, w_idx(l2,i1): 1}
        ))
        # M5 = (a11+a12)b22 -> contributes +1 to C12, -1 to C22
        terms.append((
            {u_idx(i1,j1): 1, u_idx(i1,j2): 1},
            {v_idx(j2,l2): 1},
            {w_idx(l2,i1): 1, w_idx(l2,i2): -1}
        ))
        # M6 = (a12-a21)(b11+b21) -> contributes +1 to C21, +1 to C22
        terms.append((
            {u_idx(i1,j2): 1, u_idx(i2,j1): -1},
            {v_idx(j1,l1): 1, v_idx(j2,l1): 1},
            {w_idx(l1,i2): 1, w_idx(l2,i2): 1}
        ))
        # M7 = (a21-a11)(b12+b22) -> contributes +1 to C11
        terms.append((
            {u_idx(i2,j1): 1, u_idx(i1,j1): -1},
            {v_idx(j1,l2): 1, v_idx(j2,l2): 1},
            {w_idx(l1,i1): 1}
        ))
        
        return terms
    
    # Define Strassen blocks
    # n=3: can have one pair (0,1), leaving index 2 as single
    # m=5: can have two pairs (0,1), (2,3), leaving index 4 as single
    # p=7: can have three pairs (0,1), (2,3), (4,5), leaving index 6 as single
    
    i_pairs = [(0, 1)]
    j_pairs = [(0, 1), (2, 3)]
    l_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # Collect all Strassen terms
    all_terms = []
    covered = set()
    
    for (i1, i2) in i_pairs:
        for (j1, j2) in j_pairs:
            for (l1, l2) in l_pairs:
                block_terms = strassen_block(i1, i2, j1, j2, l1, l2)
                all_terms.extend(block_terms)
                # Mark covered schoolbook indices
                for i in (i1, i2):
                    for j in (j1, j2):
                        for l in (l1, l2):
                            covered.add((i, j, l))
    
    # Number of Strassen terms
    num_strassen = len(all_terms)
    
    # Schoolbook terms for uncovered (i,j,k)
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_terms.append((i, j, k))
    
    num_schoolbook = len(schoolbook_terms)
    total_rank = num_strassen + num_schoolbook
    
    # Build factor matrices
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    # Add Strassen terms
    for (u_dict, v_dict, w_dict) in all_terms:
        for idx, coeff in u_dict.items():
            U[idx, col] += coeff
        for idx, coeff in v_dict.items():
            V[idx, col] += coeff
        for idx, coeff in w_dict.items():
            W[idx, col] += coeff
        col += 1
    
    # Add schoolbook terms
    for (i, j, k) in schoolbook_terms:
        U[u_idx(i, j), col] = 1
        V[v_idx(j, k), col] = 1
        W[w_idx(k, i), col] = 1
        col += 1
    
    assert col == total_rank, f"col={col}, rank={total_rank}"
    
    # Verify correctness
    # Check all valid entries T[i,j,k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = u_idx(i, j)
                b = v_idx(j, k)
                c = w_idx(k, i)
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    raise ValueError(f"Missing: T[{a},{b},{c}] = {val}")
    
    # Spot check some zero entries
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = u_idx(i, j)
                    b = v_idx(j2, k)
                    c = w_idx(k, i)
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-9:
                        raise ValueError(f"Spurious: T[{a},{b},{c}] = {val}")
    
    # Convert to lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
