"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_regular_subset(adj, subset):
    """Check if the induced subgraph on 'subset' is regular."""
    k = len(subset)
    if k < 6:
        return False
    # Build a mapping from vertex to index in subset for quick lookup
    # For small k (6-22), we can just compute degrees
    degs = []
    for v in subset:
        d = 0
        for u in subset:
            if u != v and adj[v][u]:
                d += 1
        degs.append(d)
    # Check if all degrees are equal
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def _check_validity_fast(adj, n):
    """
    Check if the graph has no regular induced subgraph on 6 or more vertices.
    Uses a smarter approach: precompute adjacency as bitmasks for speed.
    """
    # Convert adjacency matrix to bitmasks for faster subset degree calculation
    # adj_mask[v] is an integer where bit u is set if (v,u) is an edge
    adj_mask = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if adj[v][u]:
                mask |= (1 << u)
        adj_mask[v] = mask
    
    # We need to check all subsets of size >= 6
    # For n up to 22, this is sum(C(n,k) for k=6..n)
    # C(22,11) is about 705432, total subsets is manageable with early exit
    
    # Optimization: For a subset to be regular, all vertices must have the same degree within the subset.
    # We can iterate over subset sizes from 6 to n.
    
    # Precompute list of vertices for each size? No, just use combinations.
    
    # To speed up, we can check if a subset is a candidate first.
    # A necessary condition for a regular induced subgraph of size k:
    # The sum of degrees in the subset must be k * d for some integer d.
    # But computing degrees is the main cost.
    
    # Let's use a different strategy: iterate over subsets, but use bit operations.
    # For a subset S, the degree of v in S is popcount(adj_mask[v] & mask_S).
    
    # Iterate over all masks from 0 to (1<<n) - 1? No, that's 2^22 = 4M, which is fine.
    # But we only care about masks with popcount >= 6.
    
    # Actually, iterating 2^22 masks is fast in Python if we do it carefully?
    # 4 million iterations might take a few seconds in Python.
    # Let's stick to combinations for clarity and potentially better pruning.
    
    # Better: Use combinations of size 6, 7, ..., n.
    # For each combination, check if it's regular.
    
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            # Compute degrees
            degs = []
            for v in subset:
                # Count neighbors in subset
                # Using bitmask would be faster if we had the subset as a mask
                # But we have a tuple of vertices.
                d = 0
                for u in subset:
                    if u != v and adj[v][u]:
                        d += 1
                degs.append(d)
            
            if len(set(degs)) == 1:
                return False  # Found a regular subgraph
    
    return True

def _check_validity_bitmask(adj, n):
    """
    Faster check using bitmasks.
    """
    adj_mask = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if adj[v][u]:
                mask |= (1 << u)
        adj_mask[v] = mask
    
    # Iterate over all possible subsets represented as bitmasks
    # We can skip masks with popcount < 6
    # 2^22 = 4,194,304 iterations. This is feasible within 150s if the inner loop is fast.
    
    for mask in range(1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        # Check if induced subgraph is regular
        # We need to find the degree of each vertex in the subset.
        # If all degrees are equal, it's regular.
        
        # Optimization: The sum of degrees must be k * d.
        # Let's just compute degrees for each vertex in the subset.
        
        # To avoid iterating over all vertices 0..n-1, we can iterate over bits in mask.
        # But we need to check each vertex's degree.
        
        # Let's pick one vertex first to get the target degree.
        # Find the first set bit in mask.
        v0 = (mask & -mask).bit_length() - 1
        d0 = (adj_mask[v0] & mask).bit_count()
        
        # Now check all other vertices in mask have degree d0.
        m = mask
        while m:
            # Get next set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            if v != v0:
                dv = (adj_mask[v] & mask).bit_count()
                if dv != d0:
                    break  # Not regular, move to next mask
            m ^= lsb
        else:
            # All vertices had degree d0, so it is regular.
            return False
    
    return True

def _generate_random_graph(n, p):
    """Generate a random graph G(n, p)."""
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _graph_to_edges(adj, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    best_n = 5
    best_edges = []
    
    # We know 20-vertex graphs exist. Let's try to find one.
    # If we find a 21-vertex graph, that's even better.
    
    # Strategy:
    # 1. Try random graphs on 21 vertices with various p.
    # 2. If none found, try random graphs on 20 vertices.
    # 3. If none found, return the known 5-vertex empty graph (or a better small one).
    
    # The probability that a random graph G(n,p) has no regular induced subgraph on >= 6 vertices
    # is highest for p around 0.5, but we need to search.
    
    # For n=21, C(21, 10) ~ 352716, total subsets ~ 2^21 = 2M.
    # The bitmask check iterates 2^n masks. For n=21, that's 2M masks.
    # Each mask check takes O(k) time where k is popcount.
    # Average k is n/2 = 10.5. So ~2M * 10 = 20M operations. This should be fast in Python.
    
    # Let's try n=21 first.
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        
        # Try a few random graphs
        # For n=21, we might need many tries.
        # Let's try p values that are likely to work.
        # Regular subgraphs are less likely if degrees are varied.
        # p=0.5 gives binomial degrees, which is good.
        
        attempts = 0
        max_attempts = 1000 if n <= 18 else 500
        
        for p in [0.5, 0.4, 0.6, 0.3, 0.7, 0.45, 0.55]:
            if time.time() > deadline:
                break
            
            for _ in range(max_attempts):
                if time.time() > deadline:
                    break
                
                adj = _generate_random_graph(n, p)
                
                # Quick check: if the whole graph is regular, it's invalid.
                # But we need to check all subsets.
                
                if _check_validity_bitmask(adj, n):
                    edges = _graph_to_edges(adj, n)
                    return {'n': n, 'edges': edges}
    
    # If we didn't find a large graph, return the best small one we can guarantee.
    # The empty graph on 5 vertices is valid.
    # Can we do better? A 6-vertex graph with no regular induced subgraph on 6 vertices.
    # The only regular subgraph on 6 vertices is the whole graph.
    # So any non-regular graph on 6 vertices is valid!
    # Wait, the condition is "No set of 6 or more vertices may induce a regular subgraph".
    # So for n=6, we just need the whole graph to be non-regular.
    # For n=7, we need no 6-subset to be regular and the 7-graph to be non-regular.
    
    # Let's try to find a small valid graph if the large search fails.
    # But the large search should succeed for n=20 or 21.
    
    # Fallback: return a known valid small graph.
    # Empty graph on 5 vertices.
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
