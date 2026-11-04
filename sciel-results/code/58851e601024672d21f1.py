"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    total_triples = n * m * p  # 120
    
    # Generate all candidate rank-1 tensors (0-1 vectors)
    # A rank-1 tensor is defined by (S1, S2, S3) where:
    # S1 subset of [n]x[m], S2 subset of [m]x[p], S3 subset of [p]x[n]
    # It covers triple (i,j,k) iff (i,j) in S1, (j,k) in S2, (k,i) in S3
    
    # To keep it tractable, we'll use a greedy + local search approach
    # First, let's try to find a good solution via randomized greedy set cover
    
    # Represent each triple as (i, j, k)
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # A "box" is defined by (I, J, K) where I subset of [n], J subset of [m], K subset of [p]
    # It covers (i,j,k) iff i in I, j in J, k in K
    # This corresponds to u having 1s at (i,j) for i in I, j in J
    # v having 1s at (j,k) for j in J, k in K
    # w having 1s at (k,i) for k in K, i in I
    
    # So a box is a product set I x J x K
    # The number of possible boxes is 2^n * 2^m * 2^p = 2^4 * 2^5 * 2^6 = 16*32*64 = 32768
    # That's manageable!
    
    import random
    random.seed(42)
    
    # Generate all boxes
    all_boxes = []
    for I in range(1, 2**n):  # non-empty subsets of [n]
        for J in range(1, 2**m):  # non-empty subsets of [m]
            for K in range(1, 2**p):  # non-empty subsets of [p]
                # Check if this box covers at least one triple
                # It covers (i,j,k) for i in I, j in J, k in K
                # All such triples exist, so it covers |I|*|J|*|K| triples
                all_boxes.append((I, J, K))
    
    # For each box, compute which triples it covers
    # triple_index[i,j,k] = i*m*p + j*p + k
    def triple_idx(i, j, k):
        return i * m * p + j * p + k
    
    # Precompute box coverage as bitmasks or sets
    # With 120 triples, we can use a 128-bit integer or a set
    box_covers = []
    for (I, J, K) in all_boxes:
        covered = set()
        for i in range(n):
            if I & (1 << i):
                for j in range(m):
                    if J & (1 << j):
                        for k in range(p):
                            if K & (1 << k):
                                covered.add(triple_idx(i, j, k))
        box_covers.append(covered)
    
    # Greedy set cover
    uncovered = set(range(total_triples))
    selected_boxes = []
    
    while uncovered and (time.time() - start_time) < deadline - 30:
        best_box = -1
        best_count = 0
        for idx, covered in enumerate(box_covers):
            count = len(covered & uncovered)
            if count > best_count:
                best_count = count
                best_box = idx
        if best_box == -1 or best_count == 0:
            break
        selected_boxes.append(best_box)
        uncovered -= box_covers[best_box]
    
    greedy_rank = len(selected_boxes)
    
    # Try to improve with local search / simulated annealing
    # Actually, let's try a more systematic approach: ILP
    # But first, let's see what greedy gives us
    
    # Try multiple random restarts of greedy
    best_solution = None
    best_rank = greedy_rank
    
    for restart in range(20):
        if (time.time() - start_time) > deadline - 30:
            break
        random.shuffle(all_boxes)
        uncovered = set(range(total_triples))
        selected = []
        for idx, covered in enumerate(box_covers):
            if (time.time() - start_time) > deadline - 30:
                break
            if covered & uncovered:
                selected.append(idx)
                uncovered -= covered
                if not uncovered:
                    break
        if len(selected) < best_rank:
            best_rank = len(selected)
            best_solution = selected.copy()
    
    # Now try to reduce the solution further with local search
    # Remove redundant boxes
    if best_solution:
        for attempt in range(100):
            if (time.time() - start_time) > deadline - 20:
                break
            improved = False
            for idx in list(best_solution):
                # Try removing box idx
                test_solution = [b for b in best_solution if b != idx]
                # Check if still covers all triples
                covered = set()
                for b in test_solution:
                    covered |= box_covers[b]
                if len(covered) == total_triples:
                    best_solution = test_solution
                    best_rank = len(best_solution)
                    improved = True
                    break
            if not improved:
                break
    
    # Build the factor matrices from the solution
    # Each selected box (I, J, K) gives:
    # u_{i,j} = 1 if i in I and j in J, else 0
    # v_{j,k} = 1 if j in J and k in K, else 0
    # w_{k,i} = 1 if k in K and i in I, else 0
    
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, box_idx in enumerate(best_solution):
        I, J, K = all_boxes[box_idx]
        for i in range(n):
            if I & (1 << i):
                for j in range(m):
                    if J & (1 << j):
                        U[i * m + j][col] = 1
        for j in range(m):
            if J & (1 << j):
                for k in range(p):
                    if K & (1 << k):
                        V[j * p + k][col] = 1
        for k in range(p):
            if K & (1 << k):
                for i in range(n):
                    if I & (1 << i):
                        W[k * n + i][col] = 1
    
    # Verify the decomposition
    # For each (i,j,k), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
