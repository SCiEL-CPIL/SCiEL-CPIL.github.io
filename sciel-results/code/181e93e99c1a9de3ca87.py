"""Tensor rank reduction for 2x3 by 3x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = 2, 3, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor: T[i,j,k] = delta_{i,j,k} for the multiplication tensor
    # The multiplication tensor T has entries where T[i,j,k] = 1 if this is a valid (i,j,k) triple
    # For C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)
    # W has shape (p*n, rank), indexed by (k,i) following W[k*n+i, r]
    
    # Check: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] == 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                total = float(np.dot(np.dot(u_val, v_val), w_val))
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 140.0
    n, m, p = 2, 3, 5
    
    # Start with schoolbook decomposition (rank 30)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {
        "U": [[U_sb[i][j] for j in range(rank_sb)] for i in range(n*m)],
        "V": [[V_sb[i][j] for j in range(rank_sb)] for i in range(m*p)],
        "W": [[W_sb[i][j] for j in range(rank_sb)] for i in range(p*n)]
    }
    best_rank = rank_sb
    
    # Try to find lower rank decompositions using structured search
    # Key insight: for 2x3*3x5, we can try ranks below 30
    
    # Try rank 29, 28, 27... down to some minimum
    # Use a greedy/structured approach: try to merge pairs of rank-1 terms
    # that are "compatible"
    
    def try_rank_search(U_init, V_init, W_init, current_rank):
        """Try to reduce rank by finding linear dependencies."""
        nonlocal best_result, best_rank
        deadline = time.time() + 140.0
        
        for target_rank in range(current_rank - 1, 15, -1):
            if time.time() > deadline:
                break
            
            # Try random structured reductions
            attempts = 0
            while time.time() < deadline and attempts < 500:
                attempts += 1
                
                # Randomly select columns to perturb and try to re-solve
                # This is a heuristic: pick target_rank random columns from the current rank
                perm = np.random.permutation(current_rank)
                selected = perm[:target_rank]
                
                U_sub = U_init[:, selected]
                V_sub = V_init[:, selected]
                W_sub = W_init[:, selected]
                
                if _validate(U_sub.tolist(), V_sub.tolist(), W_sub.tolist()):
                    best_rank = target_rank
                    best_result = {
                        "U": [[U_sub[i][j] for j in range(target_rank)] for i in range(n*m)],
                        "V": [[V_sub[i][j] for j in range(target_rank)] for i in range(m*p)],
                        "W": [[W_sub[i][j] for j in range(target_rank)] for i in range(p*n)]
                    }
                    break
            else:
                continue
            break
    
    # More systematic approach: use the fact that the tensor has structure
    # For matrix multiplication, the tensor T[i,j,k] = delta_{ij} delta_{jk} (in tensor product sense)
    # Let's try a direct parameterized search
    
    # Actually, let's use a smarter approach: the 2x3x5 multiplication tensor
    # can be decomposed more efficiently using the structure
    
    # Known result: the rank of 2x3 * 3x5 matrix multiplication is known to be 
    # at most n*m*p = 30, and potentially lower.
    
    # Try a structured family: group terms by the middle index j
    # For each j, we have n*p terms involving j
    # Total = sum_j n*p = 3*10 = 30 (same as schoolbook)
    
    # Try to find if any rank-1 terms can be combined
    # A rank-1 term is (u, v, w) where u = e_{i,j}, v = e_{j,k}, w = e_{k,i}
    # Two terms can be combined if their sum is also rank-1 in some transformed space
    
    # Let's try a numerical optimization approach for lower ranks
    # We'll try ranks 29 down to 25 with random initialization
    
    for target_rank in range(29, 24, -1):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with structure
            # Use the fact that most entries should be small
            rng = np.random.RandomState(attempt * 7 + target_rank)
            
            # Initialize with zeros, then fill strategically
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # Each column r should contribute to some (i,j,k) pattern
            # Assign each column a "role" based on which (i,j) pair it primarily serves
            for r in range(target_rank):
                # Randomly assign which (i,j) this column is "for"
                ij = rng.randint(0, n*m)
                i, j = ij // m, ij % m
                
                # U part: primarily at position (i,j)
                U_r[i*m+j, r] = 1.0
                # Add small perturbations to other positions
                for _ in range(2):
                    pos = rng.randint(0, n*m)
                    if pos != i*m+j:
                        U_r[pos, r] = rng.uniform(-0.5, 0.5)
                
                # V part: must connect from j to some k
                k = rng.randint(0, p)
                V_r[j*p+k, r] = 1.0
                for _ in range(2):
                    pos = rng.randint(0, m*p)
                    if pos != j*p+k:
                        V_r[pos, r] = rng.uniform(-0.5, 0.5)
                
                # W part: must connect from k back to i (W[k*n+i, r])
                W_r[k*n+i, r] = 1.0
                for _ in range(2):
                    pos = rng.randint(0, p*n)
                    if pos != k*n+i:
                        W_r[pos, r] = rng.uniform(-0.5, 0.5)
            
            if _validate(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                best_rank = target_rank
                best_result = {
                    "U": [[U_r[i][j] for j in range(target_rank)] for i in range(n*m)],
                    "V": [[V_r[i][j] for j in range(target_rank)] for i in range(m*p)],
                    "W": [[W_r[i][j] for j in range(target_rank)] for i in range(p*n)]
                }
                found = True
                break
            elif time.time() > deadline:
                break
        
        if found:
            break
    
    if time.time() > deadline:
        pass
    
    # Convert numpy to lists if best_result was updated
    if best_rank < rank_sb:
        # Re-validate and convert
        U_arr = np.array(best_result["U"], dtype=float)
        V_arr = np.array(best_result["V"], dtype=float)
        W_arr = np.array(best_result["W"], dtype=float)
        if not _validate(best_result["U"], best_result["V"], best_result["W"]):
            best_result = {
                "U": [[U_sb[i][j] for j in range(rank_sb)] for i in range(n*m)],
                "V": [[V_sb[i][j] for j in range(rank_sb)] for i in range(m*p)],
                "W": [[W_sb[i][j] for j in range(rank_sb)] for i in range(p*n)]
            }
    
    return best_result
# EVOLVE-BLOCK-END
