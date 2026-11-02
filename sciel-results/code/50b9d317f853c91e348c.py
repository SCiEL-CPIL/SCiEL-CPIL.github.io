"""Minimal-rank tensor decomposition for 2x2 * 2x7 matrix multiplication.

Approach family: Combinatorial Greedy Deflation with Exact Arithmetic.
We search for a decomposition using a greedy algorithm that picks rank-1
components with small integer entries (-1, 0, 1) to exactly match the tensor.
If a lower rank decomposition is found, it is returned; otherwise, the
schoolbook decomposition (rank 28) is returned.
"""
import itertools
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 2, 7]


def _build_tensor(n, m, p):
    """Return the (n*m)*(m*p)*(p*n) matrix-multiplication tensor T with the
    official trace convention  W[k*n + i, r]  (i.e. C index i*k, B index j*k,
    A index i*j)."""
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _schoolbook(n, m, p):
    """Return the canonical schoolbook rank-(n*m*p) decomposition as integer lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _greedy_exact_decomposition(T, target_rank, max_iters=1000):
    """
    Attempts to find a rank-`target_rank` decomposition of T using entries in {-1, 0, 1}.
    Uses a greedy deflation strategy: repeatedly pick a rank-1 tensor that best
    matches the remaining tensor, subtract it, and continue.
    Returns (U, V, W) as lists of lists if found, else None.
    """
    I, J, K = T.shape
    T_rem = T.copy()
    U = []
    V = []
    W = []
    
    # Non-zero indices of T_rem
    def get_nonzero_indices(T_mat):
        return list(zip(*np.nonzero(T_mat)))
    
    for c in range(target_rank):
        # Find best rank-1 component
        best_u, best_v, best_w = None, None, None
        best_score = -1
        
        # Strategy: Try to match a random non-zero entry in T_rem
        nz_indices = get_nonzero_indices(T_rem)
        if not nz_indices:
            break
            
        # Try a few random non-zero entries as anchors
        random.seed(c) # Deterministic for reproducibility
        anchors = random.sample(nz_indices, min(10, len(nz_indices)))
        
        for (i0, j0, k0) in anchors:
            val0 = T_rem[i0, j0, k0]
            # Initialize u, v, w with 0
            u = [0] * I
            v = [0] * J
            w = [0] * K
            
            # Set anchor
            u[i0] = 1
            v[j0] = 1
            w[k0] = val0 # val0 is 1 or -1
            
            # Optimize other entries to maximize alignment with T_rem
            # We want sum u_i v_j w_k to be close to T_rem
            # Iterate a few times to refine
            for _ in range(5):
                # Update u
                for i in range(I):
                    if i == i0: continue
                    # Calculate contribution of u_i
                    # We want u_i * (sum_j v_j w_k T_rem[i,j,k]) to be maximized
                    # This is a bit complex. Simpler: try all 3 values for each u_i
                    # and pick the one that minimizes error on a subset
                    pass # Too slow. Use a simpler heuristic.
                    
            # Simpler Heuristic: 
            # For each index i, set u_i to sign of (v * w) dot T_rem[i,:,:]
            # This is approximate.
            
            # Let's just use the anchor and set all other entries to 0 initially,
            # then see if we can add more 1s.
            
            # Actually, for matrix multiplication tensors, the rank-1 components
            # in the schoolbook algorithm are disjoint.
            # If we just pick disjoint anchors, we might reconstruct the schoolbook.
            
            # Let's try a different tactic: 
            # Pick u, v, w such that u_i v_j w_k = 1 if T_rem[i,j,k]=1 and 0 otherwise?
            # This is only possible if T_rem is a rank-1 tensor.
            
            # Greedy Deflation Standard:
            # Pick u, v, w to maximize <T_rem, u v w>
            # We restrict u,v,w to {-1, 0, 1}.
            
            # Let's try all combinations of u, v, w? No, too many.
            # Let's try random sparse u, v, w.
            
            pass
            
        # If the above is too complex, fall back to:
        # Pick a non-zero entry (i,j,k). Set u_i=1, v_j=1, w_k=T_rem[i,j,k].
        # Set all other entries to 0.
        # This creates a rank-1 tensor that matches that entry.
        # Subtract it.
        # This will take exactly rank(T) steps if we are lucky, but might not be optimal.
        # For matrix multiplication, this effectively reconstructs the schoolbook decomposition
        # if we pick entries that don't overlap.
        
        i0, j0, k0 = anchors[0]
        val0 = T_rem[i0, j0, k0]
        
        u = [0] * I
        v = [0] * J
        w = [0] * K
        u[i0] = 1
        v[j0] = 1
        w[k0] = val0
        
        # Check if this rank-1 tensor is "valid" (doesn't introduce errors in zero spots)
        # T_rem is mostly zeros.
        # The rank-1 tensor R = u v w has non-zeros at (i0, j, k) for all j, k? No.
        # R[i,j,k] = u[i] v[j] w[k].
        # If u has only one non-zero, v has one, w has one, then R has only ONE non-zero.
        # So R[i0, j0, k0] = val0. All other entries are 0.
        # So T_rem - R will just zero out that entry.
        # This is just zeroing out entries one by one.
        # This will take exactly number_of_non_zero_entries steps.
        # For 2x2x7, there are 28 non-zero entries.
        # So this will find a rank 28 decomposition.
        
        # To find a LOWER rank, we need rank-1 components with MORE non-zeros.
        # i.e., u, v, w must be dense.
        
        # Let's try to find a dense rank-1 component.
        # Randomly initialize u, v, w with -1, 0, 1.
        # Calculate error.
        # Optimize.
        
        found = False
        for _ in range(100): # 100 random tries
            u_rand = np.random.choice([-1, 0, 1], size=I)
            v_rand = np.random.choice([-1, 0, 1], size=J)
            w_rand = np.random.choice([-1, 0, 1], size=K)
            
            # Calculate R = u v w
            R = np.einsum('i,j,k->ijk', u_rand, v_rand, w_rand)
            
            # Check if R is a subset of T_rem (i.e., T_rem - R has no negative values? No, T_rem can have negatives? No, T_rem starts positive. But if we subtract, it can go negative.)
            # We want R to match T_rem as closely as possible.
            # Score = sum(T_rem * R) - sum(R*R) ?
            
            # If R has an entry 1 where T_rem has 0, that's bad.
            # If R has an entry 1 where T_rem has 1, that's good.
            
            # Let's check if R is "compatible": 
            # For all i,j,k, if R[i,j,k] != 0, then T_rem[i,j,k] must be non-zero?
            # If so, we can subtract R.
            
            compatible = True
            for i in range(I):
                if u_rand[i] == 0: continue
                for j in range(J):
                    if v_rand[j] == 0: continue
                    for k in range(K):
                        if w_rand[k] == 0: continue
                        if T_rem[i, j, k] == 0:
                            compatible = False
                            break
                    if not compatible: break
                if not compatible: break
            
            if compatible:
                # Also check if values match
                # T_rem[i,j,k] must equal u_rand[i] v_rand[j] w_rand[k]
                ok = True
                for i in range(I):
                    if u_rand[i] == 0: continue
                    for j in range(J):
                        if v_rand[j] == 0: continue
                        for k in range(K):
                            if w_rand[k] == 0: continue
                            if T_rem[i, j, k] != u_rand[i] * v_rand[j] * w_rand[k]:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                
                if ok:
                    # Found a valid rank-1 component
                    U.append(u_rand.tolist())
                    V.append(v_rand.tolist())
                    W.append(w_rand.tolist())
                    T_rem -= R
                    found = True
                    break
        
        if not found:
            # Fallback: pick a single entry
            i0, j0, k0 = anchors[0]
            val0 = T_rem[i0, j0, k0]
            u = [0] * I
            v = [0] * J
            w = [0] * K
            u[i0] = 1
            v[j0] = 1
            w[k0] = val0
            R = np.einsum('i,j,k->ijk', u, v, w)
            U.append(u)
            V.append(v)
            W.append(w)
            T_rem -= R
            
        if not get_nonzero_indices(T_rem):
            # Check if we used exactly target_rank columns
            if len(U) == target_rank:
                return U, V, W
            elif len(U) < target_rank:
                # Pad with zeros? No, rank is minimal.
                # If we finished early, we found a lower rank.
                return U, V, W
            else:
                # Used too many?
                return None

    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    schoolbook_rank = n * m * p
    
    # Search for lower ranks
    # Start from a reasonable lower bound. 
    # For 2x2x7, try 20 to 27.
    for r in range(20, schoolbook_rank):
        U, V, W = _greedy_exact_decomposition(T, r)
        if U is not None:
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
