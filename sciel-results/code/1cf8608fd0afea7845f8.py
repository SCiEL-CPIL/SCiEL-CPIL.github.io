"""
Optimized search for Ramsey graphs R(5,5) > n.
Starts with Paley(37) and attempts to add vertices using fast bit-parallel
clique/independent-set checks and local search (edge flipping).
"""

import time
import random
import sys

# Deadline guard: evaluator kills at 330s, use 280s internal for safety
_DEADLINE = 280.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _paley_37():
    """Return adjacency matrix (list of lists of int) and edge list for Paley(37)."""
    q = 37
    # Quadratic residues mod 37
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    adj = [[0] * q for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
    return q, adj, edges

def _to_bitsets(adj, n):
    """Convert adjacency matrix to list of Python ints (bitsets)."""
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in range(n):
            if adj[i][j]:
                b |= (1 << j)
        bits[i] = b
    return bits

def _has_k_clique(bits, n, k):
    """
    Check if graph (bitsets) has a clique of size k.
    Uses backtracking with bitsets and degree pruning.
    """
    if n < k:
        return False
    
    # Precompute degrees for pruning
    # We use a recursive helper that tracks the current clique size and candidate set
    
    # To speed up, we can order vertices by degree in the induced subgraph?
    # Simpler: Standard Bron-Kerbosch style with bitsets.
    
    def backtrack(candidates_mask, size_needed):
        if size_needed == 0:
            return True
        if size_needed == 1:
            return candidates_mask != 0
        
        # Pruning: if number of candidates is less than size_needed, fail
        if candidates_mask.bit_count() < size_needed:
            return False
            
        cm = candidates_mask
        while cm:
            # Get lowest set bit
            lb = cm & (-cm)
            v = lb.bit_length() - 1
            cm ^= lb
            
            # New candidates must be neighbors of v AND in current candidates
            new_cands = candidates_mask & bits[v]
            
            if new_cands.bit_count() < size_needed - 1:
                continue
                
            if backtrack(new_cands, size_needed - 1):
                return True
        return False
    
    all_mask = (1 << n) - 1
    
    # Optimization: Start from vertices with higher degree?
    # For small k=5, it's often fast enough without complex ordering.
    # But we can skip vertices with degree < k-1.
    
    for v in range(n):
        cands = all_mask & bits[v]
        if cands.bit_count() < k - 1:
            continue
        if backtrack(cands, k - 1):
            return True
    return False

def _has_k_independent(bits, n, k):
    """Check if graph (bitsets) has an independent set of size k."""
    if n < k:
        return False
    # Independent set in G = clique in complement
    comp_bits = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = all_mask & ~bits[i]
        comp_bits[i] &= ~(1 << i)  # remove self
    return _has_k_clique(comp_bits, n, k)

def _is_valid(bits, n):
    """Check (5,5)-free: no K5, no independent 5."""
    if _has_k_clique(bits, n, 5):
        return False
    if _has_k_independent(bits, n, 5):
        return False
    return True

def _try_add_vertex(bits, n, new_vertex_neighbors):
    """
    Try to add a vertex 'n' connected to vertices in new_vertex_neighbors (a bitset).
    Returns True if the resulting graph is (5,5)-free.
    """
    new_n = n + 1
    new_bits = bits[:] + [0]
    
    # Set up new vertex's connections
    # new_bits[new_n-1] is the neighbors of the new vertex
    new_bits[new_n-1] = new_vertex_neighbors
    
    # Update existing vertices' bitsets to include the new vertex
    # If vertex i is connected to new vertex, set bit (n) in bits[i]
    for i in range(n):
        if (new_vertex_neighbors >> i) & 1:
            new_bits[i] |= (1 << new_n - 1)
            
    return _is_valid(new_bits, new_n)

def _find_valid_addition(bits, n, rng):
    """
    Search for a bitset 'neighbors' such that adding a vertex connected to 'neighbors'
    results in a valid (5,5)-free graph.
    Uses random sampling of neighborhoods, biased by density.
    """
    # Strategy: 
    # 1. Randomly sample a subset of existing vertices.
    # 2. Check if adding this vertex is valid.
    # 3. If not, try to modify the subset (flip one bit) to make it valid? 
    #    This is complex. Instead, we just sample many random subsets.
    
    # To be smarter, we can look at the "constraints".
    # If adding vertex v with neighbors S creates a K5, it means there is a K4 in G[S] 
    # such that v connects to all 4, OR there is a K3 in G[S] and v connects to them 
    # and there is another vertex w in S connected to the K3? 
    # Actually, simpler: 
    # A K5 involving the new vertex must consist of the new vertex + 4 vertices from S that form a K4 in G.
    # So, if G[S] contains a K4, adding v connected to S creates a K5.
    # Similarly, an Independent Set of size 5 involving v requires 4 vertices in S that form 
    # an Independent Set of size 4 in G (i.e., a K4 in complement).
    
    # So, a necessary condition for S to be valid is:
    # 1. G[S] has no K4.
    # 2. G[S] has no Independent Set of size 4.
    
    # This is a much faster check than checking the whole graph!
    # Let's implement this pre-check.
    
    def _subset_is_safe(S_mask, n):
        # Check if G[S] has K4 or Ind 4
        # We can reuse _has_k_clique but restricted to S?
        # It's easier to just run the full check if we pass the pre-check.
        # But checking K4 in S is cheaper than K5 in G+S.
        
        # Let's just do the full check but with a quick rejection.
        # If S contains a K4, reject.
        # If S contains an Ind 4, reject.
        
        # We can check K4 in S by looking at all 4-subsets? No, too slow.
        # We can use the same backtracking logic on the induced subgraph.
        
        # For now, let's just try random S and run _is_valid.
        # To speed up, we'll try to be smart about the initial S.
        pass

    # Heuristic: Start with a random S of density ~0.5
    # If it fails, flip bits in S until it works or we give up.
    
    # Let's try a local search on the neighbor set S.
    # Start with a random S.
    current_S = 0
    for i in range(n):
        if rng.random() < 0.5:
            current_S |= (1 << i)
            
    # Check validity
    if _try_add_vertex(bits, n, current_S):
        return current_S
        
    # If not, try to flip bits to fix it.
    # This is a heuristic. We'll just sample more random S.
    # To improve, we can bias towards S that are "structurally similar" to valid neighborhoods.
    
    # For this iteration, we'll just do random sampling with a limit.
    for _ in range(100):
        # Random S
        S = 0
        for i in range(n):
            if rng.random() < 0.5:
                S |= (1 << i)
        if _try_add_vertex(bits, n, S):
            return S
            
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with Paley(37)
    n, adj, edges = _paley_37()
    bits = _to_bitsets(adj, n)
    
    # Verify seed
    if not _is_valid(bits, n):
        # Shouldn't happen, but return as-is
        return n, edges
    
    best_n = n
    best_edges = edges[:]
    
    rng = random.Random(42)
    
    # Try to add vertices
    # We will try to add one vertex at a time.
    # If we succeed, we update n, adj, bits, edges and continue.
    # If we fail, we stop.
    
    added = 0
    while _time_left() > 30:
        # Try to find a valid neighborhood for a new vertex
        new_S = _find_valid_addition(bits, n, rng)
        
        if new_S is not None:
            # Success! Update graph
            new_n = n + 1
            new_edges = edges[:]
            for i in range(n):
                if (new_S >> i) & 1:
                    new_edges.append((i, new_n - 1))
                    
            # Update bits
            bits = bits[:] + [0]
            bits[new_n - 1] = new_S
            for i in range(n):
                if (new_S >> i) & 1:
                    bits[i] |= (1 << new_n - 1)
                    
            n = new_n
            edges = new_edges
            best_n = n
            best_edges = edges[:]
            added += 1
            
            # Update adj if we need it for other checks? 
            # We only use bits for checks. We need edges for output.
            # We don't strictly need adj for the search, only bits and edges.
            # But _try_add_vertex uses bits.
            
        else:
            # Failed to add a vertex with random sampling.
            # Try a more structured approach? 
            # Or try to modify the existing graph?
            # For now, we stop.
            break
            
    return best_n, best_edges

# Keep the required entry point name
# The function is already named construct()
