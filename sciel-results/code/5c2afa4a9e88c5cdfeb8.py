"""Structured search for matrix multiplication tensor decomposition 4x4 * 4x5."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the all-ones multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    r = U_np.shape[1]
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        u_c = U_np[:, c]  # shape (NM,)
        v_c = V_np[:, c]  # shape (MP,)
        w_c = W_np[:, c]  # shape (PN,)
        
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_c[j * p + k]
                    w_val = w_c[k * n + i]
                    tensor[i, j, k] += u_val * v_val * w_val
    
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)

def _strassen_4x4_4x5():
    """
    Build a Strassen-inspired decomposition for 4x4 * 4x5.
    
    Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    Split B (4x5) into 2x2 blocks with right half being 2x3:
      B11 (2x2), B12 (2x3)
      B21 (2x2), B22 (2x3)
    
    Use Strassen's 7 multiplications for the 2x2 blocks,
    but handle the rectangular right side carefully.
    
    Actually, let's use a different approach:
    - For the left 2x2 part (4x4 * 4x2), use Strassen (rank 7)
    - For the right 2x2 part (4x4 * 4x3), we need a different scheme
    
    Simpler approach: Use Kronecker product structure.
    4x4 * 4x5 can be seen as combining two 2x2 * 2x2 blocks and one 2x2 * 2x3 block.
    
    Let's try a direct structured approach:
    Partition rows of A into top 2 and bottom 2.
    Partition cols of A into left 2 and right 2.
    Partition rows of B into top 2 and bottom 2.
    Partition cols of B into left 2 and right 3.
    
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    C[0:2, 2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:4]*B[2:4,2:5]
    C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    C[2:4, 2:5] = A[2:4,0:2]*B[0:2,2:5] + A[2:4,2:4]*B[2:4,2:5]
    
    This is getting complex. Let's use a known result:
    The rank of 4x4 * 4x4 is at most 49 (Strassen-based).
    For 4x4 * 4x5, we can pad B to 4x6 and use rank <= 56,
    but that's not minimal.
    
    Let's try a simpler structured approach:
    Use the fact that 4x4 * 4x5 = (2x2 * 2x2) ⊗ (2x2 * 2x2) partially.
    
    Actually, let's just do a systematic search with small integer entries.
    """
    pass

def _build_schoolbook():
    """Build the schoolbook (rank 80) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _random_candidate(rank, entries=(-1, 0, 1), seed=None):
    """Generate a random candidate of given rank with specified entry set."""
    if seed is not None:
        random.seed(seed)
    U = [[random.choice(entries) for _ in range(rank)] for _ in range(NM)]
    V = [[random.choice(entries) for _ in range(rank)] for _ in range(MP)]
    W = [[random.choice(entries) for _ in range(rank)] for _ in range(PN)]
    return U, V, W

def _structured_block_search():
    """
    Try structured decompositions based on block matrix multiplication.
    
    For 4x4 * 4x5, we can use the following approach:
    - Split into 2x2 blocks
    - Use Strassen for the square parts
    - Handle the rectangular part separately
    
    This should give rank around 35-40.
    """
    # Let's try a specific structured decomposition.
    # Key insight: if we can find U, V, W with small integer entries
    # that satisfy the tensor equation, we're done.
    
    # Try rank 32 first with a structured pattern
    # Then try lower ranks
    
    results = []
    
    for rank in [32, 28, 24, 20, 18, 16]:
        # Try a few structured patterns
        for pattern_seed in range(5):
            U, V, W = _random_candidate(rank, entries=(-1, 0, 1), seed=pattern_seed + rank * 100)
            if _verify(U, V, W):
                results.append((rank, U, V, W))
                return rank, U, V, W
        # Also try with {-1, -0.5, 0, 0.5, 1}
        for pattern_seed in range(3):
            U, V, W = _random_candidate(rank, entries=(-1, -0.5, 0, 0.5, 1), seed=pattern_seed + rank * 200)
            if _verify(U, V, W, tol=1e-8):
                results.append((rank, U, V, W))
                return rank, U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as guaranteed valid solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = n * m * p  # 80
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try structured search first (fast, may find good solutions)
    result = _structured_block_search()
    if result is not None:
        rank, U, V, W = result
        if rank < best_rank:
            best_rank = rank
            best_solution = {"U": U, "V": V, "W": W}
    
    # Now do random search for even lower ranks
    min_rank = max(10, best_rank - 15)
    max_rank = best_rank - 1
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - time.time() - 5
        if remaining <= 0:
            break
        
        # Number of trials proportional to remaining time
        num_trials = min(2000, int(remaining * 300))
        num_trials = max(num_trials, 50)
        
        found = False
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Alternate between integer and half-integer entries
            if trial % 2 == 0:
                entries = (-1, 0, 1)
            else:
                entries = (-1, -0.5, 0, 0.5, 1)
            
            U, V, W = _random_candidate(r, entries=entries)
            tol = 1e-9 if trial % 2 == 0 else 1e-8
            if _verify(U, V, W, tol=tol):
                best_rank = r
                best_solution = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            # Found a solution, continue to try lower ranks
            continue
    
    return best_solution
