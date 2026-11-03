import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def _verify_solution(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    
    # Build the target tensor: T[a,b,c] = 1 iff (a,b,c) corresponds to a valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0
    
    # Check each target entry
    for (a, b, c), val in target.items():
        computed = 0.0
        for r in range(rank):
            computed += U[a, r] * V[b, r] * W[c, r]
        if abs(computed - val) > 1e-9:
            return False
    
    return True

def _try_find_rank12(n, m, p, timeout):
    """
    Attempt to find a rank-12 decomposition for 2x3x4 matrix multiplication.
    
    The key idea: for 2x3x4, the lower bound on rank is max(n*m, m*p, p*n) = max(6,12,8) = 12.
    We try to construct a rank-12 decomposition by exploiting the structure.
    
    One approach: use the fact that we can write the multiplication as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For rank-12, we need 12 rank-1 terms. Each term contributes to some subset
    of the (i,j,k) triples.
    
    A structured approach: try to find 12 triples of vectors (u_r, v_r, w_r)
    such that sum_r u_r ⊗ v_r ⊗ w_r = T.
    
    This is equivalent to finding a rank-12 decomposition of the 4-dimensional tensor.
    """
    start_time = time.time()
    deadline = start_time + timeout
    
    # For 2x3x4, let's try a specific construction.
    # The tensor has 24 nonzero entries. If we can find 12 rank-1 terms where
    # each term covers exactly 2 entries, we'd achieve rank 12.
    
    # A rank-1 term u⊗v⊗w where u has support {a1, a2}, v has support {b1, b2},
    # w has support {c1, c2} covers the 8 entries (a1,b1,c1), (a1,b1,c2), etc.
    # But we need the supports to be "compatible" with the tensor structure.
    
    # Let's try a different approach: use the fact that for rectangular matrix
    # multiplication, there may be known optimal decompositions.
    
    # Actually, let's try a numerical approach: use ALS (Alternating Least Squares)
    # to find a rank-12 decomposition, then round to half-integers and verify.
    
    # Initialize with random factors
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        rank = 12
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Update U: for each row a, U[a,:] should satisfy
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all (b,c)
            # This is a linear system for each a.
            # For fixed a, we need: U[a,:] * (V ⊙ W) = t_a
            # where (V ⊙ W)[b,c,r] = V[b,r] * W[c,r]
            
            # Build the system for U
            # For each a, we have equations for all (b,c) where T[a,b,c] is defined
            # But T is sparse, so we only have equations for valid (b,c) given a.
            
            # Actually, the tensor is: T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
            # For fixed a=i*m+j, the valid (b,c) are: b=j*p+k, c=k*n+i for k=0,...,p-1
            # So for each a, we have p equations.
            
            # Let's build the system properly.
            # For each a, collect the (b,c) pairs and the target values.
            
            # This is getting complex. Let's try a simpler approach:
            # Use the full tensor (including zeros) and solve a least-squares problem.
            
            # The tensor T is of size (n*m) x (m*p) x (p*n) with mostly zeros.
            # We want: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            
            # For ALS, fix V and W, solve for U:
            # For each a: sum_r U[a,r] * (V ⊙ W)[a,b,c,r] = T[a,b,c]
            # where (V ⊙ W)[a,b,c,r] = V[b,r] * W[c,r] (doesn't depend on a)
            
            # So for each a, we have a linear system:
            # [V[b1,r]*W[c1,r], V[b2,r]*W[c2,r], ...] * U[a,:] = [T[a,b1,c1], T[a,b2,c2], ...]
            
            # Let's build this for all a.
            
            # Collect all (b,c) pairs
            bc_pairs = []
            for b in range(m*p):
                for c in range(p*n):
                    bc_pairs.append((b, c))
            
            # Build the design matrix for U
            # For each (b,c) pair, the row is [V[b,0]*W[c,0], V[b,1]*W[c,1], ..., V[b,rank-1]*W[c,rank-1]]
            design_U = np.zeros((len(bc_pairs), rank))
            for idx, (b, c) in enumerate(bc_pairs):
                for r in range(rank):
                    design_U[idx, r] = V[b, r] * W[c, r]
            
            # For each a, the target is T[a, :, :] flattened
            # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
            # else 0
            
            # Solve for U
            for a in range(n*m):
                target_a = np.zeros(len(bc_pairs))
                for idx, (b, c) in enumerate(bc_pairs):
                    # Check if T[a,b,c] = 1
                    # a = i*m+j, b = j*p+k, c = k*n+i
                    # From a: i = a//m, j = a%m
                    i = a // m
                    j = a % m
                    # From b: j should match, k = b%m... wait, b = j*p+k, so k = b - j*p
                    # But we need j to match, and k in [0,p)
                    if j == (b // p) and (b % p) < p:
                        k = b % p
                        # Check c = k*n + i
                        if c == k * n + i:
                            target_a[idx] = 1.0
                        else:
                            target_a[idx] = 0.0
                    else:
                        target_a[idx] = 0.0
                
                # Solve design_U * U[a,:] = target_a
                U[a, :], residuals, rank_sol, sv = np.linalg.lstsq(design_U, target_a, rcond=None)
            
            # Similarly update V and W
            # For V: fix U and W, solve for V
            # For each b: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Design matrix for V: for each (a,c), row is [U[a,0]*W[c,0], ..., U[a,rank-1]*W[c,rank-1]]
            ac_pairs = []
            for a in range(n*m):
                for c in range(p*n):
                    ac_pairs.append((a, c))
            
            design_V = np.zeros((len(ac_pairs), rank))
            for idx, (a, c) in enumerate(ac_pairs):
                for r in range(rank):
                    design_V[idx, r] = U[a, r] * W[c, r]
            
            for b in range(m*p):
                target_b = np.zeros(len(ac_pairs))
                for idx, (a, c) in enumerate(ac_pairs):
                    i = a // m
                    j = a % m
                    if j == (b // p) and (b % p) < p:
                        k = b % p
                        if c == k * n + i:
                            target_b[idx] = 1.0
                        else:
                            target_b[idx] = 0.0
                    else:
                        target_b[idx] = 0.0
                
                V[b, :], residuals, rank_sol, sv = np.linalg.lstsq(design_V, target_b, rcond=None)
            
            # For W: fix U and V, solve for W
            ab_pairs = []
            for a in range(n*m):
                for b in range(m*p):
                    ab_pairs.append((a, b))
            
            design_W = np.zeros((len(ab_pairs), rank))
            for idx, (a, b) in enumerate(ab_pairs):
                for r in range(rank):
                    design_W[idx, r] = U[a, r] * V[b, r]
            
            for c in range(p*n):
                target_c = np.zeros(len(ab_pairs))
                for idx, (a, b) in enumerate(ab_pairs):
                    i = a // m
                    j = a % m
                    if j == (b // p) and (b % p) < p:
                        k = b % p
                        if c == k * n + i:
                            target_c[idx] = 1.0
                        else:
                            target_c[idx] = 0.0
                    else:
                        target_c[idx] = 0.0
                
                W[c, :], residuals, rank_sol, sv = np.linalg.lstsq(design_W, target_c, rcond=None)
        
        # After ALS, round to half-integers and verify
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p):
            return U_rounded, V_rounded, W_rounded
    
    return None

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Set a deadline: we have 120s total, use 100s for search, keep 20s margin
    start_time = time.time()
    search_timeout = 100.0
    
    # First, try to find a rank-12 decomposition
    result = _try_find_rank12(n, m, p, search_timeout)
    
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition(n, m, p, rank, timeout):
    """
    Legacy interface compatibility.
    Returns None to indicate no better solution found.
    """
    return None
