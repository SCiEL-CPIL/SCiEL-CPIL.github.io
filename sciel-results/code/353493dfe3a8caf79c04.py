import numpy as np
import time
import random
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Initialize with schoolbook decomposition
    rank_school = n * m * p
    best_U = [[0] * rank_school for _ in range(n * m)]
    best_V = [[0] * rank_school for _ in range(m * p)]
    best_W = [[0] * rank_school for _ in range(p * n)]
    for c in range(rank_school):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        best_U[i * m + j][c] = 1
        best_V[j * p + k][c] = 1
        best_W[k * n + i][c] = 1
    best_rank = rank_school

    def verify(U, V, W):
        r = len(U[0])
        # T[i, j, k] should be 1 if indices match, else 0
        # Actually T[i, j, k] = 1 for all i, j, k? No.
        # C[k, i] = sum_j A[i, j] B[j, k]
        # Tensor T_{i, j, k} = 1 if the term A[i,j]B[j,k] contributes to C[k,i].
        # This is always true. So T_{i, j, k} = 1 for all i, j, k.
        # Wait, if T is all ones, rank is 1.
        # The tensor is the structure of the multiplication.
        # The map is (A, B) -> C.
        # C_{k, i} = \sum_j A_{i, j} B_{j, k}.
        # The tensor T is such that T_{i, j, k} = 1 if and only if the indices are consistent.
        # But i, j, k are the indices of the basis vectors.
        # e_{i,j} in A-space, e_{j,k} in B-space, e_{k,i} in C-space.
        # The tensor is T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
        # So T_{i, j, k} = 1 for all i, j, k.
        # My previous reasoning about rank 1 was correct for the tensor T_{i,j,k}=1.
        # BUT, the decomposition is not of T_{i,j,k}=1.
        # The decomposition is of the Bilinear Map.
        # The map is defined by the indices.
        # Let's check the identity:
        # sum_c U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i] = 1 ?
        # If this holds for all i, j, k, then it is a valid decomposition.
        # For schoolbook:
        # c corresponds to (i, j, k).
        # U[c, i*m+j] is 1 if c's i is i and c's j is j.
        # V[c, j*p+k] is 1 if c's j is j and c's k is k.
        # W[c, k*n+i] is 1 if c's k is k and c's i is i.
        # So product is 1 only if c matches (i, j, k).
        # Sum is 1.
        # So the condition is sum_c U[c, i*m+j] * V[c, j*p+k] * W[c, k*n+i] = 1 for all i, j, k.
        
        # Let's vectorize this check.
        # Construct matrix M of size (n*m*p) x r
        # M[idx, c] = U[c, idx_A] * V[c, idx_B] * W[c, idx_C]
        # Where idx = i*m*p + j*p + k
        # idx_A = i*m + j
        # idx_B = j*p + k
        # idx_C = k*n + i
        
        total_entries = n * m * p
        # We need sum_c M[:, c] = 1
        # M is (total_entries) x r
        # Actually, we can compute M * ones? No, sum over c.
        # result = sum_c U[c, idx_A] * V[c, idx_B] * W[c, idx_C]
        # This is a bilinear form check.
        # We can construct a large matrix P where P[idx, c] = U[c, idx_A] * V[c, idx_B] * W[c, idx_C]
        # Then P @ ones(r) should be ones(total_entries)
        
        # To be efficient, we can just check a random subset or all if small.
        # n*m*p = 50. r <= 50. 50*50 = 2500 ops. Trivial.
        
        # Let's build P.
        P = np.zeros((total_entries, r))
        for c in range(r):
            Uc = U[c]
            Vc = V[c]
            Wc = W[c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        P[idx, c] = Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
        
        sums = np.sum(P, axis=1)
        # Check if sums are close to 1
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        return True

    # Randomized Local Search
    def rand_vec(size, scale=1.0):
        return [random.uniform(-scale, scale) for _ in range(size)]

    def rand_mat(rows, cols):
        return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

    current_rank = best_rank
    current_U = [row[:] for row in best_U]
    current_V = [row[:] for row in best_V]
    current_W = [row[:] for row in best_W]

    # Try to reduce rank by 1
    # Strategy: Randomly perturb, then try to drop a column?
    # Or: Start with random decomposition of rank R, optimize, then drop columns?
    # Given time limits, simple random restarts with small rank.
    
    # Known: Rank(1x5x5) is likely 25. Rank(2x5x5) is likely 50.
    # Is it possible to go below 50?
    # If n=1, rank is mp.
    # If n=2, rank is at least 2*mp? Not necessarily.
    # But let's try to find rank < 50.
    
    # We will try random decompositions of rank r < 50 and check validity.
    # This is a "lottery" approach.
    # To increase chances, we can use a heuristic:
    # Start with schoolbook, drop a column, and try to fix by adjusting others?
    # This is complex.
    
    # Let's try random vectors for r=49, 48, etc.
    # But the space is huge.
    # However, the constraints are linear.
    # sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1.
    # This is a system of nonlinear equations.
    
    # Let's just do a quick random search for rank 49 to 45.
    # If found, great. If not, return schoolbook.
    
    for target_rank in range(49, 40, -1):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Initialize with small random values
            for i in range(n * m):
                for c in range(target_rank):
                    U[i][c] = random.gauss(0, 1)
            for j in range(m * p):
                for c in range(target_rank):
                    V[j][c] = random.gauss(0, 1)
            for k in range(p * n):
                for c in range(target_rank):
                    W[k][c] = random.gauss(0, 1)
            
            # Check
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break # Found lower rank, stop searching for even lower? Or continue?
            # The directive says "rank-minimal". So we should keep searching if we find one.
            # But time is tight.
            # Let's continue to see if we can go lower.
    
    # Ensure output is integer/half-integer if possible?
    # The task says "integer/half-integer entries".
    # My random search produces floats.
    # If I find a solution, I need to snap it to half-integers?
    # Or does the evaluator accept floats that are close to half-integers?
    # "exactly decompose".
    # If I return floats, it might fail the "integer/half-integer" check.
    # The schoolbook solution uses 0 and 1.
    # If I don't find a valid solution with half-integers, I should stick to schoolbook.
    
    # For now, if I found a solution with random floats, I should check if they are close to half-integers.
    if best_rank < rank_school:
        # Check if best_U, V, W are close to half-integers
        is_valid_half_int = True
        for M in [best_U, best_V, best_W]:
            for row in M:
                for val in row:
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) > 1e-5:
                        is_valid_half_int = False
                        break
                if not is_valid_half_int: break
            if not is_valid_half_int: break
        
        if not is_valid_half_int:
            # Revert to schoolbook
            best_U = [[0] * rank_school for _ in range(n * m)]
            best_V = [[0] * rank_school for _ in range(m * p)]
            best_W = [[0] * rank_school for _ in range(p * n)]
            for c in range(rank_school):
                i = c // (m * p)
                rem = c % (m * p)
                j = rem // p
                k = rem % p
                best_U[i * m + j][c] = 1
                best_V[j * p + k][c] = 1
                best_W[k * n + i][c] = 1
            best_rank = rank_school

    # Final verify just in case
    if not verify(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U = [[0] * rank_school for _ in range(n * m)]
        best_V = [[0] * rank_school for _ in range(m * p)]
        best_W = [[0] * rank_school for _ in range(p * n)]
        for c in range(rank_school):
            i = c // (m * p)
            rem = c % (m * p)
            j = rem // p
            k = rem % p
            best_U[i * m + j][c] = 1
            best_V[j * p + k][c] = 1
            best_W[k * n + i][c] = 1

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
