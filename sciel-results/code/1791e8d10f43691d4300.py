"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from itertools import combinations, combinations_with_replacement

def _is_sidon(sets):
    """Check if all pair sums (including a+a) are unique within each set."""
    for s in sets:
        if not s:
            continue
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
    return True

def _diffs_disjoint(sets):
    """Check if nonzero difference sets are disjoint."""
    d0 = set()
    d1 = set()
    for a, b in combinations(sets[0], 2):
        d0.add(abs(b - a))
    for a, b in combinations(sets[1], 2):
        d1.add(abs(b - a))
    return not (d0 & d1)

def _valid(sets):
    return _is_sidon(sets) and _diffs_disjoint(sets)

def _score(sets):
    return len(sets[0]) + len(sets[1])

def _greedy_sidon_with_diffs(elements, allows, max_size=64):
    """
    Greedily build a Sidon set from a list of candidate elements,
    using only differences from 'allows'.
    """
    s = []
    sums = set()
    
    # Sort elements to try smaller differences first? Or random?
    # Random is better for exploration, but we call this multiple times.
    for x in elements:
        if len(s) >= max_size:
            break
        
        # Check difference constraint
        ok_diff = True
        for a in s:
            if abs(x - a) not in allows:
                ok_diff = False
                break
        
        if not ok_diff:
            continue
            
        # Check Sidon constraint
        ok_sidon = True
        for a in s:
            if x + a in sums:
                ok_sidon = False
                break
        
        if not ok_sidon:
            continue
        
        # Add x
        s.append(x)
        for a in s:
            sums.add(x + a)
            
    return s

def _spectral_partition(n=256, d_a=None, d_b=None):
    """
    Use spectral clustering to partition vertices 1..n into two groups
    based on the compatibility graphs defined by d_a and d_b.
    Returns two lists of vertices (0-indexed for internal use, +1 for value).
    """
    # Build adjacency matrix for combined compatibility
    # Edge (i, j) exists if |i-j| is in d_a OR d_b?
    # Actually, we want to separate them.
    # Let's create a graph where edges represent "conflict" if we try to put them in the same set?
    # No, let's use the idea:
    # If we put i in A and j in A, we need |i-j| in d_a.
    # If we put i in B and j in B, we need |i-j| in d_b.
    
    # Construct a graph G where W[i,j] = 1 if |i-j| in d_a, 0 otherwise.
    # And H where W[i,j] = 1 if |i-j| in d_b.
    # We want to find a cut that maximizes edges in G within cluster 1 and H within cluster 2?
    # This is a correlation clustering / max-cut variant.
    
    # Simple Spectral Approach:
    # Create a single graph where edge weight is 1 if compatible with A, -1 if compatible with B?
    # Or just use the Laplacian of the union graph to find a natural cut.
    
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff = abs((i+1) - (j+1))
            if diff in d_a:
                adj[i, j] = 1
                adj[j, i] = 1
            elif diff in d_b:
                adj[i, j] = 1
                adj[j, i] = 1
            else:
                adj[i, j] = 0
                adj[j, i] = 0
                
    # Compute Laplacian
    # L = D - A
    # Use scipy for efficiency
    A_sparse = sp.csr_matrix(adj)
    L = csgraph.laplacian(A_sparse, normed=True)
    
    try:
        # Get eigenvalues and eigenvectors
        # We need the second smallest eigenvector (Fiedler vector)
        # scipy.linalg.eigh for dense, or scipy.sparse.linalg.eigsh for sparse
        from scipy.sparse.linalg import eigsh
        
        # k=2 to get the first two
        # sigma=0 to find smallest eigenvalues
        vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
        
        # Fiedler vector is the second one (index 1)
        # Sort eigenvalues to be sure
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        
        # Partition based on sign
        group_0 = []
        group_1 = []
        for i in range(n):
            if fiedler[i] > 0:
                group_1.append(i+1)
            else:
                group_0.append(i+1)
                
        return group_0, group_1
    except Exception:
        # Fallback to random partition if spectral fails
        mid = n // 2
        return list(range(1, mid+1)), list(range(mid+1, n+1))

def _refine_sidon_set(s, allows, n=256, max_size=64):
    """
    Try to improve the Sidon set by adding more elements or swapping.
    """
    s = list(s)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.add(s[i] + s[j])
            
    # Calculate current diffs
    current_diffs = set()
    for i in range(len(s)):
        for j in range(len(s)):
            if i != j:
                current_diffs.add(abs(s[i] - s[j]))
                
    # Try to add elements from 1..n not in s
    candidates = [x for x in range(1, n+1) if x not in s]
    # Sort candidates by how "compatible" they are (e.g., smaller differences first?)
    # Or just iterate.
    
    improved = False
    for x in candidates:
        if len(s) >= max_size:
            break
            
        # Check diff constraint
        ok_diff = True
        for a in s:
            if abs(x - a) not in allows:
                ok_diff = False
                break
        if not ok_diff:
            continue
            
        # Check Sidon constraint
        ok_sidon = True
        for a in s:
            if x + a in sums:
                ok_sidon = False
                break
        if not ok_sidon:
            continue
            
        # Add it
        s.append(x)
        for a in s:
            sums.add(x + a)
        improved = True
        
    return s

def construct():
    n = 256
    max_size = 64
    best_sets = [[], []]
    best_score = 0
    rng = random.Random(42)
    
    all_diffs = list(range(1, n))
    
    # Generate a few diverse difference partitions
    partitions = []
    
    # 1. Even/Odd
    p_even = set(d for d in all_diffs if d % 2 == 0)
    p_odd = set(d for d in all_diffs if d % 2 == 1)
    partitions.append((p_even, p_odd))
    partitions.append((p_odd, p_even))
    
    # 2. Mod 3
    p_m3_0 = set(d for d in all_diffs if d % 3 == 0)
    p_m3_1 = set(d for d in all_diffs if d % 3 == 1)
    p_m3_2 = set(d for d in all_diffs if d % 3 == 2)
    partitions.append((p_m3_0 | p_m3_1, p_m3_2))
    partitions.append((p_m3_2, p_m3_0 | p_m3_1))
    partitions.append((p_m3_0, p_m3_1 | p_m3_2))
    partitions.append((p_m3_1, p_m3_0 | p_m3_2))
    
    # 3. Random partitions
    for _ in range(50):
        shuffled_diffs = list(all_diffs)
        rng.shuffle(shuffled_diffs)
        split = len(shuffled_diffs) // 2
        pa = set(shuffled_diffs[:split])
        pb = set(shuffled_diffs[split:])
        partitions.append((pa, pb))
        partitions.append((pb, pa))

    for pa, pb in partitions:
        # Use spectral clustering to suggest initial element groups
        g0, g1 = _spectral_partition(n, pa, pb)
        
        # Attempt 1: g0 uses pa, g1 uses pb
        A = _greedy_sidon_with_diffs(g0, pa, max_size)
        B = _greedy_sidon_with_diffs(g1, pb, max_size)
        
        # Refine
        A = _refine_sidon_set(A, pa, n, max_size)
        B = _refine_sidon_set(B, pb, n, max_size)
        
        if _valid([A, B]):
            score = _score([A, B])
            if score > best_score:
                best_score = score
                best_sets = [A, B]
                
        # Attempt 2: Swap pools
        A2 = _greedy_sidon_with_diffs(g1, pa, max_size)
        B2 = _greedy_sidon_with_diffs(g0, pb, max_size)
        
        A2 = _refine_sidon_set(A2, pa, n, max_size)
        B2 = _refine_sidon_set(B2, pb, n, max_size)
        
        if _valid([A2, B2]):
            score = _score([A2, B2])
            if score > best_score:
                best_score = score
                best_sets = [A2, B2]

    # Ensure non-empty
    if not best_sets[0]:
        best_sets[0] = [1]
    if not best_sets[1]:
        best_sets[1] = [2]
        
    # Final validation
    if not _valid(best_sets):
        best_sets = [[1], [2]]
        
    # Cap sizes
    best_sets[0] = best_sets[0][:64]
    best_sets[1] = best_sets[1][:64]
    
    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
