"""Fresh design: search for rank-minimal exact tensor decomposition of 3x3x4 matmul."""
import time
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS
N = n * m          # 9
M = m * p          # 12
P = p * n          # 12

def _verify(U, V, W, rank):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True

def _solve_linear(A, b, deadline):
    num_eqs = len(b)
    num_unknowns = len(A[0]) if A else 0
    
    aug = []
    for i in range(num_eqs):
        row = list(A[i]) + [b[i]]
        aug.append(row)
    
    n_rows = len(aug)
    n_cols = num_unknowns
    pivot_row = 0
    pivot_cols = []
    
    for col in range(n_cols):
        if time.time() > deadline - 5:
            return None
        pivot = -1
        for row in range(pivot_row, n_rows):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        aug[pivot_row], aug[pivot] = aug[pivot], aug[pivot_row]
        pivot_val = aug[pivot_row][col]
        if pivot_val != 1:
            for j in range(col, n_cols + 1):
                aug[pivot_row][j] = aug[pivot_row][j] / pivot_val
        for row in range(n_rows):
            if row == pivot_row:
                continue
            factor = aug[row][col]
            if factor != 0:
                for j in range(col, n_cols + 1):
                    aug[row][j] -= factor * aug[pivot_row][j]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row >= n_rows:
            break
    
    for row in range(pivot_row, n_rows):
        for j in range(n_cols):
            if aug[row][j] != 0:
                return None
        if aug[row][n_cols] != 0:
            return None
    
    x = [Fraction(0) for _ in range(n_cols)]
    for i, col in enumerate(pivot_cols):
        x[col] = aug[i][n_cols]
    
    return x

def _search_rank(deadline):
    """
    Structured search: fix W to unit vectors, distribute columns among (i,k) pairs.
    For each distribution, solve the resulting system.
    """
    import random
    random.seed(12345)
    
    # (i,k) pairs: i in [0,3), k in [0,4) -> 12 pairs
    ik_pairs = [(i, k) for i in range(n) for k in range(p)]
    
    for target_rank in range(23, 28):
        if time.time() > deadline - 10:
            break
        
        # Try different distributions of columns among (i,k) pairs
        num_attempts = 500
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Randomly assign each column to an (i,k) pair
            col_to_ik = [random.choice(ik_pairs) for _ in range(target_rank)]
            
            # Build W: unit vectors
            W = [[0] * target_rank for _ in range(P)]
            for c in range(target_rank):
                i_c, k_c = col_to_ik[c]
                W[k_c * n + i_c][c] = 1
            
            # For each (i,k), get the set of columns assigned to it
            ik_to_cols = {}
            for c in range(target_rank):
                ik = col_to_ik[c]
                if ik not in ik_to_cols:
                    ik_to_cols[ik] = []
                ik_to_cols[ik].append(c)
            
            # Now solve for U and V.
            # Unknowns: U[i*m+j, c] for i,j,c and V[j*p+k, c] for j,k,c
            # But many are zero by construction.
            
            # For each (i,k) pair with columns C_ik:
            #   For each j: sum_{c in C_ik} U[i*m+j, c] * V[j*p+k, c] = 1
            # This is m equations with 2*|C_ik| unknowns (U values and V values).
            
            # Collect all unknowns
            unknown_indices = {}
            idx = 0
            for c in range(target_rank):
                i_c, k_c = col_to_ik[c]
                for j in range(m):
                    # U[i*m+j, c]
                    key_u = ('U', i_c, j, c)
                    if key_u not in unknown_indices:
                        unknown_indices[key_u] = idx
                        idx += 1
                    # V[j*p+k_c, c]
                    key_v = ('V', j, k_c, c)
                    if key_v not in unknown_indices:
                        unknown_indices[key_v] = idx
                        idx += 1
            
            num_unknowns = idx
            
            # Build equations
            A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(n * m * p)]
            b = [Fraction(1) for _ in range(n * m * p)]
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ik = (i, k)
                        if ik in ik_to_cols:
                            for c in ik_to_cols[ik]:
                                # U[i*m+j, c] * V[j*p+k, c]
                                # This is bilinear! Not linear.
                                # We need a different approach.
                                pass
                        eq_idx += 1
            
            # The system is bilinear, not linear. We need to fix one variable set.
            # Fix U to be sparse (1 nonzero per row per column group), then solve for V.
            
            # Fix U: for each (i, j, c), set U[i*m+j, c] = 1 if c is the first column for (i,k_c)
            U = [[0] * target_rank for _ in range(N)]
            for c in range(target_rank):
                i_c, k_c = col_to_ik[c]
                # Set U[i_c*m+j, c] = 1 for all j
                for j in range(m):
                    U[i_c * m + j][c] = 1
            
            # Now solve for V linearly:
            # For each (i,j,k): sum_{c in C_{(i,k)}} U[i*m+j, c] * V[j*p+k, c] = 1
            # Since U[i*m+j, c] = 1 for c in C_{(i,k_c)} where i_c=i:
            # sum_{c in C_{(i,k)}} V[j*p+k, c] = 1
            
            num_v_unknowns = M * target_rank
            
            def v_idx(j, k, c):
                return j * p * target_rank + k * target_rank + c
            
            A = [[Fraction(0) for _ in range(num_v_unknowns)] for _ in range(n * m * p)]
            b = [Fraction(1) for _ in range(n * m * p)]
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ik = (i, k)
                        if ik in ik_to_cols:
                            for c in ik_to_cols[ik]:
                                A[eq_idx][v_idx(j, k, c)] = Fraction(1)
                        eq_idx += 1
            
            v = _solve_linear(A, b, deadline)
            if v is None:
                continue
            
            V = [[0] * target_rank for _ in range(M)]
            for j in range(m):
                for k in range(p):
                    for c in range(target_rank):
                        V[j * p + k][c] = v[v_idx(j, k, c)]
            
            if _verify(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start = time.time()
    deadline = start + 120
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    schoolbook_result = {"U": U, "V": V, "W": W}
    
    if time.time() < deadline - 15:
        result = _search_rank(deadline)
        if result is not None:
            return result
    
    return schoolbook_result
