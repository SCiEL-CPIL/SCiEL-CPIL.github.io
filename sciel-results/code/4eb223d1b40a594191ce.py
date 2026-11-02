"""Exact low-rank tensor decomposition search via projected integer lattice search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # safety margin under 150s wall-clock limit


def _target_tensor(n, m, p):
    """T[i,j,k] = 1 for all (i,j,k)."""
    return np.ones((n, m, p), dtype=object)


def _search_m322(r, deadline, seed):
    """
    Search for a rank-r decomposition of M_{3,2,2}.
    n=3, m=2, p=2.
    U (6 x r), V (4 x r), W (6 x r).
    """
    n, m, p = 3, 2, 2
    rng = random.Random(seed)
    target = np.ones((n, m, p), dtype=int)
    
    # Precompute indices for speed
    # u_idx[i, j] = i * m + j
    # v_idx[j, k] = j * p + k
    # w_idx[k, i] = k * n + i
    
    def check_terms(terms):
        total = np.zeros((n, m, p), dtype=int)
        for u, v, w in terms:
            for i in range(n):
                for j in range(m):
                    uv = u[i * m + j]
                    if uv == 0:
                        continue
                    for k in range(p):
                        total[i, j, k] += uv * v[j * p + k] * w[k * n + i]
        return np.all(total == target)

    max_iter = 500000
    for it in range(max_iter):
        if time.time() > deadline:
            return None
        terms = []
        for t in range(r):
            u = np.zeros(n * m, dtype=int)
            v = np.zeros(m * p, dtype=int)
            w = np.zeros(p * n, dtype=int)
            # Sparse random vectors
            # Try to keep them sparse
            for _ in range(rng.randint(1, 3)):
                pos = rng.randint(0, n * m - 1)
                u[pos] = rng.choice([-1, 1])
            for _ in range(rng.randint(1, 3)):
                pos = rng.randint(0, m * p - 1)
                v[pos] = rng.choice([-1, 1])
            for _ in range(rng.randint(1, 3)):
                pos = rng.randint(0, p * n - 1)
                w[pos] = rng.choice([-1, 1])
            terms.append((u, v, w))
        
        if check_terms(terms):
            return terms
    return None


def _combine_decompositions(decomp1, decomp2, n, m1, p1, m2, p2):
    """
    Combine two decompositions:
    M_{n, m1, p1} and M_{n, m2, p2} ?
    Actually, we need to compose based on the split logic.
    
    For M_{3,4,7} split into two M_{3,2,7}:
    Decomp A for M_{3,2,7}.
    Decomp B for M_{3,2,7}.
    Total U = [U_A, U_B] (concatenate columns)
    Total V = [V_A, V_B]
    Total W = [W_A, W_B]
    
    But we need to map the indices correctly.
    M_{3,4,7}:
    A (3x4), B (4x7).
    Split A into A1 (3x2), A2 (3x2).
    Split B into B1 (2x7), B2 (2x7).
    C = A1 B1 + A2 B2.
    
    Decomp for M_{3,2,7} gives U1 (6 x r1), V1 (14 x r1), W1 (6 x r1).
    U1 corresponds to A1 (3x2) indices.
    V1 corresponds to B1 (2x7) indices.
    W1 corresponds to C (3x7) indices? No, W is p*n = 7*3 = 21.
    Wait, W[k*n+i].
    
    For the combined M_{3,4,7}:
    U (12 x R).
    V (28 x R).
    W (21 x R).
    
    Term t from Decomp A:
    U_A (6 x 1). Indices 0..5 correspond to A1 (3x2).
    In full U (12 x 1), A1 is rows 0..5 (cols 0,1 of A).
    A2 is rows 6..11 (cols 2,3 of A).
    So U_A goes into rows 0..5.
    
    V_A (14 x 1). Indices 0..13 correspond to B1 (2x7).
    In full V (28 x 1), B1 is rows 0..13 (rows 0,1 of B).
    B2 is rows 14..27 (rows 2,3 of B).
    So V_A goes into rows 0..13.
    
    W_A (21 x 1). Indices 0..20 correspond to C (3x7) -> k*n+i.
    In full W (21 x 1), it's the same.
    So W_A goes into rows 0..20.
    
    Term t from Decomp B:
    U_B (6 x 1). Indices 0..5 correspond to A2 (3x2).
    In full U, A2 is rows 6..11.
    So U_B goes into rows 6..11.
    
    V_B (14 x 1). Indices 0..13 correspond to B2 (2x7).
    In full V, B2 is rows 14..27.
    So V_B goes into rows 14..27.
    
    W_B (21 x 1). Indices 0..20 correspond to C (3x7).
    In full W, it's the same.
    So W_B goes into rows 0..20.
    
    So we can just concatenate the columns of U, V, W from the two decompositions,
    but we need to place them in the correct rows.
    """
    pass


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 84
    
    # Build schoolbook as fallback
    rank = schoolbook_rank
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
    best = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find better decomposition using block composition
    # Target: M_{3,2,2} rank 11
    # If found, construct M_{3,4,7}
    
    # Search for M_{3,2,2} rank 11
    # Use a tight deadline for the search
    search_deadline = start_time + 20.0
    result_322 = _search_m322(11, search_deadline, seed=42)
    
    if result_322 is not None:
        # Construct M_{3,2,2} decomposition
        r_322 = 11
        U_322 = np.zeros((6, r_322), dtype=int)
        V_322 = np.zeros((4, r_322), dtype=int)
        W_322 = np.zeros((6, r_322), dtype=int)
        for t in range(r_322):
            u, v, w = result_322[t]
            U_322[:, t] = u
            V_322[:, t] = v
            W_322[:, t] = w
        
        # Construct M_{3,2,1} (Schoolbook, rank 6)
        r_321 = 6
        U_321 = np.zeros((6, r_321), dtype=int)
        V_321 = np.zeros((2, r_321), dtype=int) # m*p = 2*1 = 2
        W_321 = np.zeros((3, r_321), dtype=int) # p*n = 1*3 = 3
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(1):
                    U_321[i * 2 + j, col] = 1
                    V_321[j * 1 + k, col] = 1
                    W_321[k * 3 + i, col] = 1
                    col += 1
        
        # Construct M_{3,2,7} from M_{3,2,2} and M_{3,2,1}
        # M_{3,2,7} = M_{3,2,4} + M_{3,2,3}
        # M_{3,2,4} = M_{3,2,2} + M_{3,2,2}
        # M_{3,2,3} = M_{3,2,1} + M_{3,2,2}
        # So M_{3,2,7} uses 3 copies of M_{3,2,2} and 1 copy of M_{3,2,1}
        
        # We need to map indices for M_{3,2,7}
        # n=3, m=2, p=7
        # U (6 x R), V (14 x R), W (21 x R)
        
        # Copy 1: M_{3,2,2} for p=0,1
        # Copy 2: M_{3,2,2} for p=2,3
        # Copy 3: M_{3,2,2} for p=4,5
        # Copy 4: M_{3,2,1} for p=6
        
        # Wait, M_{3,2,2} covers p=2 elements.
        # M_{3,2,1} covers p=1 element.
        # Total p = 2+2+2+1 = 7.
        
        # Let's build M_{3,2,7} decomposition
        U_327 = np.zeros((6, 3 * r_322 + r_321), dtype=int)
        V_327 = np.zeros((14, 3 * r_322 + r_321), dtype=int)
        W_327 = np.zeros((21, 3 * r_322 + r_321), dtype=int)
        
        offset = 0
        # Block 1: p=0,1
        for t in range(r_322):
            U_327[:, offset] = U_322[:, t]
            # V maps j (0,1) and k (0,1) to j*2+k (0..3)
            for j in range(2):
                for k in range(2):
                    V_327[j * 2 + k, offset] = V_322[j * 2 + k, t]
            # W maps k (0,1) and i (0..2) to k*3+i (0..5)
            for k in range(2):
                for i in range(3):
                    W_327[k * 3 + i, offset] = W_322[k * 3 + i, t]
            offset += 1
            
        # Block 2: p=2,3
        for t in range(r_322):
            U_327[:, offset] = U_322[:, t]
            for j in range(2):
                for k in range(2):
                    V_327[j * 2 + (k + 2), offset] = V_322[j * 2 + k, t]
            for k in range(2):
                for i in range(3):
                    W_327[(k + 2) * 3 + i, offset] = W_322[k * 3 + i, t]
            offset += 1
            
        # Block 3: p=4,5
        for t in range(r_322):
            U_327[:, offset] = U_322[:, t]
            for j in range(2):
                for k in range(2):
                    V_327[j * 2 + (k + 4), offset] = V_322[j * 2 + k, t]
            for k in range(2):
                for i in range(3):
                    W_327[(k + 4) * 3 + i, offset] = W_322[k * 3 + i, t]
            offset += 1
            
        # Block 4: p=6 (M_{3,2,1})
        for t in range(r_321):
            U_327[:, offset] = U_321[:, t]
            for j in range(2):
                for k in range(1):
                    V_327[j * 1 + (k + 6), offset] = V_321[j * 1 + k, t]
            for k in range(1):
                for i in range(3):
                    W_327[(k + 6) * 3 + i, offset] = W_321[k * 3 + i, t]
            offset += 1

        # Now construct M_{3,4,7} from two M_{3,2,7}
        # M_{3,4,7} = M_{3,2,7} (for A1, B1) + M_{3,2,7} (for A2, B2)
        # U (12 x R_total)
        # V (28 x R_total)
        # W (21 x R_total)
        
        R_total = 2 * (3 * r_322 + r_321)
        U_final = np.zeros((12, R_total), dtype=int)
        V_final = np.zeros((28, R_total), dtype=int)
        W_final = np.zeros((21, R_total), dtype=int)
        
        # Part 1: A1 (rows 0-5 of U), B1 (rows 0-13 of V)
        for t in range(3 * r_322 + r_321):
            U_final[0:6, t] = U_327[:, t]
            V_final[0:14, t] = V_327[:, t]
            W_final[:, t] = W_327[:, t]
            
        # Part 2: A2 (rows 6-11 of U), B2 (rows 14-27 of V)
        offset2 = 3 * r_322 + r_321
        for t in range(3 * r_322 + r_321):
            U_final[6:12, offset2 + t] = U_327[:, t]
            V_final[14:28, offset2 + t] = V_327[:, t]
            W_final[:, offset2 + t] = W_327[:, t]
            
        best_rank = R_total
        best = {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }
        
        if best_rank < schoolbook_rank:
            return best
            
    return best
