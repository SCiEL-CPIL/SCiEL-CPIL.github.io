"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # A: n x m, B: m x p, C: n x p
    dim_A = n * m  # 9
    dim_B = m * p  # 15
    dim_C = n * p  # 15
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Build target tensor T[a, b, c] = 1 if C[c] depends on A[a] and B[b]
    # a = i * m + j, b = j * p + k, c = k * n + i
    T = np.zeros((dim_A, dim_B, dim_C), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook solution: rank = n*m*p = 45
    rank_sb = n * m * p
    U_sb = np.zeros((dim_A, rank_sb))
    V_sb = np.zeros((dim_B, rank_sb))
    W_sb = np.zeros((dim_C, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        # Rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        rec = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(rec, 1.0, atol=1e-9)
    
    # Generate a pool of candidate rank-1 components using Strassen-like patterns
    # Each component is (u_vec, v_vec, w_vec) where these are sparse vectors
    candidate_pool = []
    
    # Add schoolbook components as base
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                u = np.zeros(dim_A)
                v = np.zeros(dim_B)
                w = np.zeros(dim_C)
                u[a] = 1
                v[b] = 1
                w[c] = 1
                candidate_pool.append((u, v, w))
    
    # Generate Strassen-like combinations for 3x3 blocks
    # For each pair of output columns, try to combine them
    for k1 in range(p):
        for k2 in range(k1 + 1, p):
            # Try to combine columns k1 and k2 using Strassen-like formulas
            for sign1 in [-1, 1]:
                for sign2 in [-1, 1]:
                    u = np.zeros(dim_A)
                    v = np.zeros(dim_B)
                    w = np.zeros(dim_C)
                    # A-like combination: use all i,j for both k1 and k2
                    for i in range(n):
                        for j in range(m):
                            a = i * m + j
                            u[a] = 1
                            b1 = j * p + k1
                            b2 = j * p + k2
                            v[b1] = sign1
                            v[b2] = sign2
                            c1 = k1 * n + i
                            c2 = k2 * n + i
                            w[c1] = 1
                            w[c2] = 1
                    candidate_pool.append((u, v, w))
    
    # Try to find a minimal set of components that covers all tensor entries
    # Each tensor entry (a,b,c) with T[a,b,c]=1 must be covered by at least one component
    # where u[a]*v[b]*w[c] contributes appropriately
    
    # Get all required tensor entries
    required_entries = []
    for a in range(dim_A):
        for b in range(dim_B):
            for c in range(dim_C):
                if T[a, b, c] == 1:
                    required_entries.append((a, b, c))
    
    # For each candidate, determine which required entries it "covers"
    # A candidate covers entry (a,b,c) if u[a]*v[b]*w[c] != 0
    candidate_coverage = []
    for u, v, w in candidate_pool:
        covered = set()
        for (a, b, c) in required_entries:
            if u[a] * v[b] * w[c] != 0:
                covered.add((a, b, c))
        if covered:
            candidate_coverage.append(covered)
    
    # Greedy set cover: repeatedly pick the candidate that covers the most uncovered entries
    uncovered = set(required_entries)
    selected_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_count = 0
        for idx, covered in enumerate(candidate_coverage):
            if idx in selected_indices:
                continue
            count = len(covered & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered -= candidate_coverage[best_idx]
    
    # Build the solution from selected candidates
    if selected_indices:
        R = len(selected_indices)
        U = np.zeros((dim_A, R))
        V = np.zeros((dim_B, R))
        W = np.zeros((dim_C, R))
        
        for r, idx in enumerate(selected_indices):
            u, v, w = candidate_pool[idx]
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        
        # Verify and if valid, update best
        if verify(U, V, W):
            best_U = U
            best_V = V
            best_W = W
            best_rank = R
    
    # If greedy didn't improve, try local search on the schoolbook solution
    # Try removing one component at a time and see if the decomposition still works
    if best_rank == rank_sb and time.time() < deadline - 10:
        # Try to remove components one by one
        current_U = best_U.copy()
        current_V = best_V.copy()
        current_W = best_W.copy()
        current_rank = best_rank
        
        for remove_col in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing column remove_col
            if remove_col == 0:
                test_U = current_U[:, 1:]
                test_V = current_V[:, 1:]
                test_W = current_W[:, 1:]
            elif remove_col == current_rank - 1:
                test_U = current_U[:, :-1]
                test_V = current_V[:, :-1]
                test_W = current_W[:, :-1]
            else:
                test_U = np.column_stack([current_U[:, :remove_col], current_U[:, remove_col+1:]])
                test_V = np.column_stack([current_V[:, :remove_col], current_V[:, remove_col+1:]])
                test_W = np.column_stack([current_W[:, :remove_col], current_W[:, remove_col+1:]])
            
            if verify(test_U, test_V, test_W):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                # After removing, restart the removal process
                remove_col = 0
                if current_rank < best_rank:
                    best_U = current_U
                    best_V = current_V
                    best_W = current_W
                    best_rank = current_rank
    
    # Convert to lists of lists with integer/half-integer values
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
