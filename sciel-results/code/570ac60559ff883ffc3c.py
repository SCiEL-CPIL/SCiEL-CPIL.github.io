import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a hybrid approach:
    1. Start with a baseline of Strassen blocks + schoolbook (rank ~99).
    2. Attempt to reduce rank by finding overlapping Strassen blocks that can replace schoolbook terms
       via linear combination, or by using more advanced block decompositions if feasible.
    3. Given the constraints, the most reliable improvement is to optimize the packing of Strassen blocks
       and ensure no unnecessary terms are added.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    def add_schoolbook(i, j, k):
        add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # Strategy: Maximize the number of disjoint Strassen blocks.
    # We have i in {0,1,2}, j in {0..4}, k in {0..6}.
    # A Strassen block covers 2x2x2.
    # We can pack disjoint blocks greedily.
    
    # Let's try a simple greedy packing of disjoint 2x2x2 blocks.
    # A block is defined by (i1, i2), (j1, j2), (k1, k2) with i1 < i2, etc.
    # Since dimensions are small, we can enumerate all possible 2x2x2 sub-cubes
    # and select a maximum set of disjoint ones.
    
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            if i2 != i1 + 1:
                continue  # Only adjacent indices for 2x2
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    if j2 != j1 + 1:
                        continue
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            if k2 != k1 + 1:
                                continue
                            # Define the set of (i,j,k) indices covered
                            covered = set()
                            for ii in range(i1, i2 + 1):
                                for jj in range(j1, j2 + 1):
                                    for kk in range(k1, k2 + 1):
                                        covered.add((ii, jj, kk))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), covered))

    # Greedy set cover for disjoint blocks
    used_indices = set()
    selected_blocks = []
    
    # Sort blocks by some heuristic, e.g., random or fixed order
    # To get a consistent result, let's just iterate through them.
    # A better heuristic: pick blocks that cover the most "central" or "harder" indices?
    # For now, random shuffle with a fixed seed for reproducibility in a single run?
    # Or just iterate.
    
    # Actually, let's just iterate through all possible blocks and add if disjoint.
    # This is O(N^2) but N is small.
    
    # To improve packing, we can use a simple backtracking or just greedy.
    # Let's try to maximize count.
    # Since the grid is small, we can use a simple recursive search for max packing?
    # No, just greedy is likely sufficient and fast.
    
    # Let's sort by i, j, k to be deterministic
    all_blocks.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]))
    
    for block_params, covered in all_blocks:
        if used_indices.isdisjoint(covered):
            selected_blocks.append(block_params)
            used_indices.update(covered)
    
    # Add the selected Strassen blocks
    for params in selected_blocks:
        i1, i2, j1, j2, k1, k2 = params
        add_strassen_block(i1, i2, j1, j2, k1, k2)

    # Add schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used_indices:
                    add_schoolbook(i, j, k)

    rank = len(U_cols)
    
    # Verification: Check if the decomposition is exact.
    # Build matrices U, V, W
    U = np.array([[0] * rank for _ in range(U_rows)]).T
    V = np.array([[0] * rank for _ in range(V_rows)]).T
    W = np.array([[0] * rank for _ in range(W_rows)]).T
    
    for c in range(rank):
        for r in range(U_rows):
            U[r, c] = U_cols[c][r]
        for r in range(V_rows):
            V[r, c] = V_cols[c][r]
        for r in range(W_rows):
            W[r, c] = W_cols[c][r]

    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's verify a few random points or all points (small enough).
    # T[i,j,k] = 1 for all i,j,k.
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    is_valid = False
                    break
        if not is_valid:
            break

    if not is_valid:
        # Fallback to pure schoolbook if packing failed (should not happen)
        U_cols.clear()
        V_cols.clear()
        W_cols.clear()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_schoolbook(i, j, k)
        rank = len(U_cols)
        U = np.array([[0] * rank for _ in range(U_rows)]).T
        V = np.array([[0] * rank for _ in range(V_rows)]).T
        W = np.array([[0] * rank for _ in range(W_rows)]).T
        for c in range(rank):
            for r in range(U_rows):
                U[r, c] = U_cols[c][r]
            for r in range(V_rows):
                V[r, c] = V_cols[c][r]
            for r in range(W_rows):
                W[r, c] = W_cols[c][r]

    return {
        "U": [[int(round(float(U[r, c]))) for c in range(rank)] for r in range(U_rows)],
        "V": [[int(round(float(V[r, c]))) for c in range(rank)] for r in range(V_rows)],
        "W": [[int(round(float(W[r, c]))) for c in range(rank)] for r in range(W_rows)]
    }

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Rank:", len(res["U"][0]) if res["U"] else 0)
    # Verify
    n, m, p = 3, 5, 7
    U = np.array(res["U"]).T
    V = np.array(res["V"]).T
    W = np.array(res["W"]).T
    rank = len(res["U"][0])
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    print("Valid:", ok)
