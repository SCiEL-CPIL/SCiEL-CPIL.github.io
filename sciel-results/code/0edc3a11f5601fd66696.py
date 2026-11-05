import numpy as np
from fractions import Fraction
import itertools
import time
import random

DIMENSIONS = [2, 3, 5]
TIME_LIMIT = 140  # Leave 10s safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] * V[:,r] * W[:,r] == tensor T where T[i,j,k] = 1 for all i,j,k
    """
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return U, V, W, rank

def _try_rank_reduction(U, V, W, target_rank, n, m, p, time_budget):
    """
    Try to reduce the rank by attempting to merge terms.
    We look for pairs of terms that can be combined.
    """
    start_time = time.time()
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    # Try to find a lower rank by solving a system of equations
    # For each pair of terms, check if they can be merged
    # This is a heuristic approach
    
    # Convert to numpy for faster computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try removing one term at a time and see if the remaining terms can still work
    # by adjusting coefficients
    for r_remove in range(current_rank):
        if time.time() - start_time > time_budget:
            break
        
        # Create masks for all terms except r_remove
        mask = [i for i in range(current_rank) if i != r_remove]
        
        # Check if the remaining terms can span the same space
        # This is a simplified check - we need to verify the tensor decomposition
        U_new = U_arr[:, mask]
        V_new = V_arr[:, mask]
        W_new = W_arr[:, mask]
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r in range(len(mask)):
                        s += U_new[idx_u, r] * V_new[idx_v, r] * W_new[idx_w, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            # Found a valid lower rank decomposition
            U_list = [[float(x) for x in row] for row in U_new]
            V_list = [[float(x) for x in row] for row in V_new]
            W_list = [[float(x) for x in row] for row in W_new]
            return U_list, V_list, W_list, len(mask)
    
    return U, V, W, current_rank

def _search_lower_rank(U, V, W, n, m, p, time_budget):
    """
    Search for a lower rank decomposition by trying various transformations.
    """
    start_time = time.time()
    current_rank = len(U[0])
    best_U, best_V, best_W, best_rank = U, V, W, current_rank
    
    # Try Winograd-style transformations
    # The idea is to use the structure of the problem to find a more efficient decomposition
    
    # For 2x3 times 3x5, we can try to exploit the rectangular structure
    # One approach: group the middle dimension differently
    
    # Try various random perturbations and see if we can find a lower rank
    # This is a heuristic search
    
    for attempt in range(100):
        if time.time() - start_time > time_budget:
            break
        
        # Try a random linear combination of the existing terms
        # and see if it leads to a lower effective rank
        pass
    
    return best_U, best_V, best_W, best_rank

def _strassen_2x2(U, V, W, n, m, p, i_start, j_start, k_start, r_offset):
    """
    Apply Strassen's algorithm to a 2x2 subblock.
    Returns the new rank offset.
    """
    r = r_offset
    
    # M1 = (A11 + A22)(B11 + B22)
    U[i_start * m + j_start][r] = 1
    U[(i_start+1) * m + (j_start+1)][r] = 1
    V[j_start * p + k_start][r] = 1
    V[(j_start+1) * p + (k_start+1)][r] = 1
    W[k_start * n + i_start][r] = 1
    W[(k_start+1) * n + (i_start+1)][r] = 1
    r += 1
    
    # M2 = (A21 + A22)B11
    U[(i_start+1) * m + j_start][r] = 1
    U[(i_start+1) * m + (j_start+1)][r] = 1
    V[j_start * p + k_start][r] = 1
    W[k_start * n + (i_start+1)][r] = 1
    W[(k_start+1) * n + (i_start+1)][r] = -1
    r += 1
    
    # M3 = A11(B21 - B11)
    U[i_start * m + j_start][r] = 1
    V[(j_start+1) * p + (k_start+1)][r] = 1
    V[j_start * p + k_start][r] = -1
    W[(k_start+1) * n + i_start][r] = 1
    W[(k_start+1) * n + (i_start+1)][r] = 1
    r += 1
    
    # M4 = A22(B12 - B22)
    U[(i_start+1) * m + (j_start+1)][r] = 1
    V[(j_start+1) * p + k_start][r] = 1
    V[j_start * p + k_start][r] = -1
    W[k_start * n + i_start][r] = 1
    W[k_start * n + (i_start+1)][r] = 1
    r += 1
    
    # M5 = (A11 + A12)B22
    U[i_start * m + j_start][r] = 1
    U[i_start * m + (j_start+1)][r] = 1
    V[(j_start+1) * p + (k_start+1)][r] = 1
    W[k_start * n + i_start][r] = -1
    W[k_start * n + (i_start+1)][r] = 1
    r += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[(i_start+1) * m + j_start][r] = 1
    U[i_start * m + j_start][r] = -1
    V[j_start * p + k_start][r] = 1
    V[j_start * p + (k_start+1)][r] = 1
    W[(k_start+1) * n + (i_start+1)][r] = 1
    r += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[i_start * m + (j_start+1)][r] = 1
    U[(i_start+1) * m + (j_start+1)][r] = -1
    V[(j_start+1) * p + k_start][r] = 1
    V[(j_start+1) * p + (k_start+1)][r] = 1
    W[k_start * n + i_start][r] = 1
    r += 1
    
    return r

def _build_decomposition_with_partitions(n, m, p, partitions, time_budget):
    """
    Build a decomposition based on given partitions of the middle dimension.
    partitions: list of (i_slice, j_slice, k_slice) tuples
    """
    start_time = time.time()
    
    # Determine the total rank needed
    total_rank = 0
    for i_slice, j_slice, k_slice in partitions:
        i_size = len(i_slice)
        j_size = len(j_slice)
        k_size = len(k_slice)
        
        if i_size == 2 and j_size == 2 and k_size == 2:
            total_rank += 7  # Strassen
        else:
            total_rank += i_size * j_size * k_size  # Schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    r = 0
    for i_slice, j_slice, k_slice in partitions:
        i_list = list(i_slice)
        j_list = list(j_slice)
        k_list = list(k_slice)
        
        if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
            # Use Strassen
            r = _strassen_2x2(U, V, W, n, m, p, i_list[0], j_list[0], k_list[0], r)
        else:
            # Use schoolbook
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
                        r += 1
    
    return U, V, W, r

def _generate_partitions(n, m, p):
    """
    Generate all possible partitions of the dimensions.
    """
    partitions = []
    
    # Try all ways to partition the middle dimension j into groups
    # and for each group, partition i and k
    
    # First, partition j into contiguous or arbitrary groups
    j_indices = list(range(m))
    
    # Generate all set partitions of j_indices
    def set_partitions(lst):
        if not lst:
            yield []
            return
        first = lst[0]
        for partition in set_partitions(lst[1:]):
            # Add first to each existing subset
            for i in range(len(partition)):
                new_partition = [list(s) for s in partition]
                new_partition[i].append(first)
                yield [tuple(s) for s in new_partition]
            # Add first as a new subset
            yield partition + [(first,)]
    
    for j_partition in set_partitions(j_indices):
        for i_partition in set_partitions(list(range(n))):
            for k_partition in set_partitions(list(range(p))):
                # For each combination of i, j, k subsets
                # We need to pair them up
                # This is complex, so let's try a simpler approach
                pass
    
    # Simpler approach: try specific known good partitions
    # Partition 1: Current approach
    partitions.append([
        (range(n), [2], range(p)),  # j=2, all i, all k
        (range(n), [0,1], [4]),     # j=0,1, k=4
        (range(n), [0,1], [0,1]),   # j=0,1, k=0,1 (Strassen)
        (range(n), [0,1], [2,3]),   # j=0,1, k=2,3 (Strassen)
    ])
    
    # Partition 2: Different grouping
    partitions.append([
        (range(n), [0], range(p)),  # j=0, all i, all k
        (range(n), [1], range(p)),  # j=1, all i, all k
        (range(n), [2], range(p)),  # j=2, all i, all k
    ])
    
    # Partition 3: Group k differently
    partitions.append([
        (range(n), [0,1], [0,1,2]),  # j=0,1, k=0,1,2
        (range(n), [0,1], [3,4]),    # j=0,1, k=3,4
        (range(n), [2], [0,1,2,3,4]), # j=2, all k
    ])
    
    # Partition 4: Try to use Strassen on different 2x2 blocks
    partitions.append([
        ([0,1], [0,1], [0,1]),  # Strassen on i=0,1, j=0,1, k=0,1
        ([0,1], [0,1], [2,3]),  # Strassen on i=0,1, j=0,1, k=2,3
        ([0,1], [0,1], [4]),    # schoolbook on i=0,1, j=0,1, k=4
        ([0,1], [2], [0,1,2,3,4]),  # schoolbook on i=0,1, j=2, all k
    ])
    
    return partitions

def _search_lower_rank(U, V, W, n, m, p, time_budget):
    """
    Search for a lower rank decomposition by trying various partitions.
    """
    start_time = time.time()
    current_rank = len(U[0])
    best_U, best_V, best_W, best_rank = U, V, W, current_rank
    
    # Generate and try different partitions
    partitions = _generate_partitions(n, m, p)
    
    for partition in partitions:
        if time.time() - start_time > time_budget:
            break
        
        U_new, V_new, W_new, rank = _build_decomposition_with_partitions(n, m, p, partition, time_budget)
        
        if rank < best_rank and _verify_decomposition(U_new, V_new, W_new, n, m, p):
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, build the known rank-28 decomposition
    U, V, W, rank = _build_rank28()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Initial decomposition verification failed")
    
    # Try to reduce the rank
    time_remaining = TIME_LIMIT - (time.time() - start_time)
    if time_remaining > 5:
        U, V, W, rank = _search_lower_rank(U, V, W, n, m, p, time_remaining - 5)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Final decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
