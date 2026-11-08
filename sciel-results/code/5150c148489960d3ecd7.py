"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, rank):
        """Verify that the decomposition is exact using NumPy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_strassen_terms():
        """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
        terms = []
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}))
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}))
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    def generate_partitions(dim):
        """Generate all partitions of range(dim) into groups of size 1 or 2."""
        if dim == 0:
            return [[]]
        if dim == 1:
            return [[[0]]]
        if dim == 2:
            return [[[0,1]], [[0],[1]]]
        if dim == 3:
            return [[[0,1],[2]], [[0],[1,2]], [[0],[1],[2]]]
        if dim == 4:
            return [[[0,1],[2,3]], [[0,1],[2],[3]], [[0],[1,2],[3]],
                    [[0],[1],[2,3]], [[0],[1],[2],[3]]]
        if dim == 5:
            results = []
            for first in range(2):  # first group size 1 or 2
                if first == 0:
                    rest = generate_partitions(dim - 1)
                    for r in rest:
                        results.append([[0]] + r)
                else:
                    rest = generate_partitions(dim - 2)
                    for r in rest:
                        results.append([[0,1]] + r)
            return results
        if dim == 6:
            results = []
            for first in range(2):
                if first == 0:
                    rest = generate_partitions(dim - 1)
                    for r in rest:
                        results.append([[0]] + r)
                else:
                    rest = generate_partitions(dim - 2)
                    for r in rest:
                        results.append([[0,1]] + r)
            return results
        return []

    def build_decomposition(i_groups, j_groups, k_groups, strassen_terms):
        """Build a decomposition from given groupings."""
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        if col != total_rank:
            return None, None
            
        result = {"U": U, "V": V, "W": W}
        return result, total_rank

    def try_exhaustive_partition_search():
        """
        Try all possible partitions of each dimension into groups of size 1 or 2,
        and find the one that minimizes total rank.
        """
        if time.time() > deadline:
            return None, None
        
        strassen_terms = get_strassen_terms()
        
        i_parts = generate_partitions(n)
        j_parts = generate_partitions(m)
        k_parts = generate_partitions(p)
        
        best_result = None
        best_rank = float('inf')
        
        for ig in i_parts:
            if time.time() > deadline:
                break
            for jg in j_parts:
                if time.time() > deadline:
                    break
                for kg in k_parts:
                    if time.time() > deadline:
                        break
                    
                    # Quick rank estimate
                    est_rank = 0
                    for a in ig:
                        for b in jg:
                            for c in kg:
                                if len(a)==2 and len(b)==2 and len(c)==2:
                                    est_rank += 7
                                else:
                                    est_rank += len(a)*len(b)*len(c)
                    
                    if est_rank >= best_rank:
                        continue
                    
                    result, rank = build_decomposition(ig, jg, kg, strassen_terms)
                    
                    if result is None:
                        continue
                    
                    if verify_numpy(result['U'], result['V'], result['W'], rank):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = result
        
        if best_result is not None:
            return best_result, best_rank
        
        return None, None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Try exhaustive partition search
    improved, improved_rank = try_exhaustive_partition_search()
    if improved is not None and improved_rank < best_rank:
        best_result = improved
        best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
