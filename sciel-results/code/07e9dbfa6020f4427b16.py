"""Program for large pairwise noncoprime sets containing the interval endpoint."""
# EVOLVE-BLOCK-START
import math

def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy:
    1. Identify candidates: Numbers in [1, 255] divisible by 3, 5, or 17.
    2. Build a graph where an edge exists between u and v if gcd(u, v) > 1.
    3. Find the Maximum Clique in this graph using Bron-Kerbosch with pivot selection.
    4. Return the clique (which is a valid pairwise non-coprime set).
    """
    # Step 1: Generate candidates
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Step 2: Build adjacency bitsets
    # adj[i] is a bitmask where bit j is 1 if gcd(candidates[i], candidates[j]) > 1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Step 3: Bron-Kerbosch with Pivot
    best_clique = []
    
    def bronk(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Choose pivot u from P union X to minimize branching
        # We want a pivot with the most neighbors in P
        # If P is empty, we can't pivot effectively, but base case handles P=0.
        # Actually, standard BK: if P and X are empty, expand.
        # Pivot selection: pick u in P ∪ X that maximizes |N(u) ∩ P|
        
        # Optimization: If P is empty, return (handled above).
        # We need to iterate over vertices in P.
        # To choose pivot, we check all vertices in P | X.
        
        # For efficiency with bitsets, we can just pick an arbitrary pivot or the first one if we don't implement full optimization, 
        # but "with pivot selection" is requested.
        
        # Let's find the pivot u.
        # We need to calculate |N(u) & P| for u in (P | X).
        # Since n is small (127), we can just loop.
        
        best_u = -1
        max_neighbors = -1
        
        # Get list of indices in P and X
        # P and X are passed as bitmasks (integers) for efficiency? 
        # Let's use bitmasks for P and X.
        
        # Re-define BK to use bitmasks for P and X
        pass

    # Let's re-implement BK using bitmasks for P and X to make pivot selection and intersection fast.
    
    def bronk_mask(R, P, X):
        nonlocal best_clique
        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Choose pivot u from P | X
        # We want to maximize |N(u) & P|
        # Iterate through all possible u in P | X
        # Since n=127, we can iterate 0 to n-1 and check if bit is set in P|X
        
        U = P | X
        best_u = -1
        max_count = -1
        
        # To speed up, we can just pick the first available vertex if we are not worried about worst-case,
        # but pivot selection is key for BK performance.
        # We'll iterate through set bits of U.
        temp_U = U
        while temp_U:
            # Get lowest set bit
            lsb = temp_U & -temp_U
            u = lsb.bit_length() - 1
            temp_U ^= lsb
            
            # Count neighbors of u in P
            # N(u) is adj[u]
            # Intersection with P
            count = (adj[u] & P).bit_count()
            
            if count > max_count:
                max_count = count
                best_u = u
                
        # Now iterate over vertices in P \ N(best_u)
        # This is the standard BK optimization
        candidates_to_branch = P & ~(adj[best_u] | (1 << best_u)) # Exclude best_u itself? 
        # Actually, the set to iterate is P \ N(u). 
        # Note: u is in P or X. If u is in P, it is not in N(u) (no self loops). 
        # So P & ~adj[best_u] is correct.
        
        temp_P = candidates_to_branch
        while temp_P:
            lsb = temp_P & -temp_P
            v = lsb.bit_length() - 1
            temp_P ^= lsb
            
            # R' = R ∪ {v}
            # P' = P ∩ N(v)
            # X' = X ∩ N(v)
            
            R.append(candidates[v])
            bronk_mask(R, P & adj[v], X & adj[v])
            R.pop()
            
            # Move v from P to X
            P &= ~(1 << v)
            X |= (1 << v)

    # Initial call
    # P is all vertices, X is empty
    init_P = (1 << n) - 1
    init_X = 0
    bronk_mask([], init_P, init_X)
    
    return best_clique

# EVOLVE-BLOCK-END
