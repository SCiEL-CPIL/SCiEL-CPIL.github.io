"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    V[2, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1  # k=0,i=0
    W[1, 0] = 1  # k=1,i=0
    W[2, 0] = 1  # k=0,i=1
    W[3, 0] = 1  # k=1,i=1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # k=0,i=1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1  # k=1,i=0
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1
    V[0, 3] = -1
    W[2, 3] = 1  # k=0,i=1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # k=1,i=0
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[2, 5] = 1
    W[0, 5] = 1  # k=0,i=0
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1
    U[3, 6] = -1
    V[1, 6] = 1
    V[3, 6] = 1
    W[0, 6] = -1  # k=0,i=0
    
    return U, V, W

def _verify_222():
    """Verify the Strassen 2x2x2 decomposition."""
    U, V, W = _strassen_222()
    n = m = p = 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(7):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _build_2x5x6():
    """
    Build 2x5x6 decomposition using block Strassen.
    Split 5 = 2+3, 6 = 2+2+2.
    
    A = [A1 (2x2) | A2 (2x3)]
    B = [B1 (2x6); B2 (3x6)]
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2x6, split B1 into three 2x2 blocks. Each uses Strassen (rank 7). Total: 21.
    A2*B2: 2x3x6, split A2 = [A2a (2x1) | A2b (2x2)], B2 = [B2a (1x6); B2b (2x6)].
      A2a*B2a: 2x1x6 (rank 12 schoolbook)
      A2b*B2b: 2x2x6 (rank 21 via Strassen)
      Total for A2*B2: 33
    Grand total: 21 + 33 = 54
    """
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n*m, rank))  # 10 x 54
    V = np.zeros((m*p, rank))  # 30 x 54
    W = np.zeros((p*n, rank))  # 12 x 54
    
    sU, sV, sW = _strassen_222()  # 4x7 each
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A1 indices, i in {0,1}, j in {0,1} -> U[i*5+j, col]
            U[0*5 + 0, col] = sU[0, r]  # i=0,j=0
            U[0*5 + 1, col] = sU[1, r]  # i=0,j=1
            U[1*5 + 0, col] = sU[2, r]  # i=1,j=0
            U[1*5 + 1, col] = sU[3, r]  # i=1,j=1
            
            # V: B1_block indices, j in {0,1} globally, k in {k_offset, k_offset+1}
            V[0*6 + k_offset + 0, col] = sV[0, r]  # j=0,k=k_offset
            V[0*6 + k_offset + 1, col] = sV[1, r]  # j=0,k=k_offset+1
            V[1*6 + k_offset + 0, col] = sV[2, r]  # j=1,k=k_offset
            V[1*6 + k_offset + 1, col] = sV[3, r]  # j=1,k=k_offset+1
            
            # W: C indices, k in {k_offset, k_offset+1}, i in {0,1}
            W[(k_offset+0)*2 + 0, col] = sW[0, r]  # k=k_offset, i=0
            W[(k_offset+0)*2 + 1, col] = sW[1, r]  # k=k_offset, i=1
            W[(k_offset+1)*2 + 0, col] = sW[2, r]  # k=k_offset+1, i=0
            W[(k_offset+1)*2 + 1, col] = sW[3, r]  # k=k_offset+1, i=1
            
            col += 1
    # col should be 21 here
    
    # Part 2a: A2a * B2a (2x1 x 6), schoolbook rank 12
    for i in range(2):
        for k in range(6):
            U[i*5 + 2, col] = 1   # A2a: i in {0,1}, j=2
            V[2*6 + k, col] = 1   # B2a: j=2, k in {0..5}
            W[k*2 + i, col] = 1   # C: k, i
            col += 1
    # col should be 33 here
    
    # Part 2b: A2b * B2b (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            # U: A2b indices, i in {0,1}, j in {3,4}
            U[0*5 + 3, col] = sU[0, r]  # i=0,j=0 (local) -> global j=3
            U[0*5 + 4, col] = sU[1, r]  # i=0,j=1 (local) -> global j=4
            U[1*5 + 3, col] = sU[2, r]  # i=1,j=0 (local) -> global j=3
            U[1*5 + 4, col] = sU[3, r]  # i=1,j=1 (local) -> global j=4
            
            # V: B2b_block, j in {3,4} globally, k in {k_offset, k_offset+1}
            V[3*6 + k_offset + 0, col] = sV[0, r]  # j=3, k=k_offset
            V[3*6 + k_offset + 1, col] = sV[1, r]  # j=3, k=k_offset+1
            V[4*6 + k_offset + 0, col] = sV[2, r]  # j=4, k=k_offset
            V[4*6 + k_offset + 1, col] = sV[3, r]  # j=4, k=k_offset+1
            
            # W: same as before
            W[(k_offset+0)*2 + 0, col] = sW[0, r]
            W[(k_offset+0)*2 + 1, col] = sW[1, r]
            W[(k_offset+1)*2 + 0, col] = sW[2, r]
            W[(k_offset+1)*2 + 1, col] = sW[3, r]
            
            col += 1
    # col should be 54 here
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _milp_search(rank_target, time_limit=30.0):
    """
    Use MILP to search for a rank-rank_target decomposition.
    Returns (U, V, W) as numpy arrays if found, None otherwise.
    
    The approach:
    1. Fix the sparsity pattern (which entries are non-zero) via binary variables
    2. Solve for the values (restricted to {-1, 0, 1} for simplicity)
    3. Verify the tensor identity holds
    """
    n, m, p = 2, 5, 6
    rank = rank_target
    
    # Total number of variables: rank * (n*m + m*p + p*n)
    # U: n*m * rank = 10 * rank
    # V: m*p * rank = 30 * rank  
    # W: p*n * rank = 12 * rank
    # Total: 52 * rank binary variables
    
    # For a given rank, we need to find values in {-1, 0, 1} such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a constraint satisfaction problem. Let's use a greedy approach:
    # Try to build the decomposition incrementally.
    
    start_time = time.time()
    
    # Precompute all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For each rank position r, we need to assign:
    # U_r: vector of length n*m
    # V_r: vector of length m*p
    # W_r: vector of length p*n
    # such that the product contributions sum correctly
    
    # Greedy: for each r, try all combinations of sparse patterns
    # This is too expensive for large rank, so we use a heuristic
    
    # Alternative: Use the known Strassen structure as a starting point
    # and try to merge/simplify
    
    # For now, let's try a direct search for small ranks
    # by checking if we can express the tensor as sum of rank-1 terms
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # This is equivalent to finding a low-rank decomposition of the 3D tensor
    
    # For rank 54 (current), we already have a valid decomposition
    # Let's try to find lower ranks by random search + verification
    
    rng = np.random.RandomState(42)
    
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random sparse patterns
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Each entry is -1, 0, or 1 with some probability
        for r in range(rank):
            # U column
            for idx in range(n*m):
                if rng.random() < 0.3:
                    U[idx, r] = rng.choice([-1, 1])
            
            # V column
            for idx in range(m*p):
                if rng.random() < 0.3:
                    V[idx, r] = rng.choice([-1, 1])
            
            # W column
            for idx in range(p*n):
                if rng.random() < 0.3:
                    W[idx, r] = rng.choice([-1, 1])
        
        # Verify
        valid = True
        for i, j, k in triples:
            total = 0
            for r in range(rank):
                total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if abs(total - 1.0) > 1e-9:
                valid = False
                break
        
        if valid:
            return U, V, W
    
    return None

def _try_improve_from_strassen(base_rank=54, max_reduction=6, time_limit=60.0):
    """
    Try to improve upon the Strassen-based decomposition by merging terms.
    Returns improved decomposition or None.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    
    base_result = _build_2x5x6()
    U_base = np.array(base_result['U'])
    V_base = np.array(base_result['V'])
    W_base = np.array(base_result['W'])
    base_rank = len(U_base[0])
    
    # Try to find linear dependencies among the rank terms
    # If some columns are linearly dependent, we might be able to reduce rank
    
    # For each target rank from base_rank-1 down to base_rank - max_reduction
    for target_rank in range(base_rank - 1, max(base_rank - max_reduction - 1, 20), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Check if we can express the base decomposition using fewer terms
        # by finding a linear transformation
        
        # The tensor is: sum_r u_r ⊗ v_r ⊗ w_r
        # If there's a linear dependency, we can reduce rank
        
        # Try random projections to see if we can find a lower-rank representation
        rng = np.random.RandomState(42 + target_rank)
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a rank-target_rank decomposition
            # by solving a system of equations
            
            # Number of constraints: n*m*p = 60
            # Number of free variables: target_rank * (n*m + m*p + p*n) - 3*target_rank
            # (we can fix one entry per term to normalize)
            
            # This is underdetermined for small target_rank, so use random init + optimization
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random values in {-1, 0, 1}
            for r in range(target_rank):
                U[:, r] = rng.choice([-1, 0, 1], size=n*m)
                V[:, r] = rng.choice([-1, 0, 1], size=m*p)
                W[:, r] = rng.choice([-1, 0, 1], size=p*n)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(target_rank):
                            total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _verify_full(U, V, W, n=2, m=5, p=6):
    """Verify the full decomposition."""
    rank = len(U[0])
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _check_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is integer or half-integer
                if abs(val * 2 - round(val * 2)) > 1e-12:
                    return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 140  # safety margin
    best_result = None
    best_rank = float('inf')
    
    # First, get the Strassen-based rank-54 decomposition as baseline
    try:
        if _verify_222():
            result = _build_2x5x6()
            if _verify_full(result['U'], result['V'], result['W']):
                if _check_valid_entries(result['U'], result['V'], result['W']):
                    best_result = result
                    best_rank = len(result['U'][0])
    except Exception:
        pass
    
    # If we didn't get a valid result, use schoolbook
    if best_result is None:
        best_result = _schoolbook_decomposition()
        best_rank = len(best_result['U'][0])
    
    # Try to improve using MILP-based search
    if time.time() < deadline - 40:  # Leave 40s for final verification and safety
        try:
            improved = _try_improve_from_strassen(base_rank=best_rank, max_reduction=4, time_limit=35.0)
            if improved is not None:
                if _verify_full(improved['U'], improved['V'], improved['W']):
                    if _check_valid_entries(improved['U'], improved['V'], improved['W']):
                        improved_rank = len(improved['U'][0])
                        if improved_rank < best_rank:
                            best_result = improved
                            best_rank = improved_rank
        except Exception:
            pass
    
    # Try direct MILP search for even lower ranks if time permits
    if time.time() < deadline - 20:
        for target_rank in range(best_rank - 1, max(best_rank - 3, 30), -1):
            if time.time() > deadline - 20:
                break
            try:
                found = _milp_search(target_rank, time_limit=10.0)
                if found is not None:
                    U, V, W = found
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify_full(U_list, V_list, W_list):
                        if _check_valid_entries(U_list, V_list, W_list):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            break  # Found the lowest rank we tried
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
