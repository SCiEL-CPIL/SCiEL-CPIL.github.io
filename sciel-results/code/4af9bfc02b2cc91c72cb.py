"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    
    Strategy: Local Search / Hill Climbing on Edge Swaps.
    1. Initialize with a random graph of 91 edges.
    2. Iteratively swap a random edge with a random non-edge if it reduces the total number of 4-cycles.
    3. Use a time limit to ensure we finish within 120s.
    """
    n = 31
    target_edges = 91
    
    # Helper to count C4s containing a specific edge (u, v)
    # A C4 containing (u, v) corresponds to a path of length 3 between u and v: u-a-b-v
    # Edges: (u, a), (a, b), (b, v).
    # We require a != v, b != u, a != b.
    def count_c4s_for_edge(u, v, adj):
        count = 0
        # a must be a neighbor of u (excluding v)
        # b must be a neighbor of v (excluding u)
        # a and b must be adjacent
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[v]:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    count += 1
        return count

    # Helper to count total C4s
    def count_total_c4s(adj, n):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    total += count_c4s_for_edge(u, v, adj)
        return total // 4 # Each C4 is counted 4 times (once per edge)

    # Initialize random graph
    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    edges = all_pairs[:target_edges]
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
    
    current_c4s = count_total_c4s(adj, n)
    
    # Local Search
    start_time = time.time()
    time_limit = 10.0 # Use 10 seconds to be safe
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        u, v = random.choice(list(edge_set))
        
        # Calculate C4s removed by removing (u, v)
        c4s_removed = count_c4s_for_edge(u, v, adj)
        
        # Temporarily remove (u, v)
        adj[u].remove(v)
        adj[v].remove(u)
        edge_set.remove((u, v))
        
        # Pick a random non-edge to add
        # To avoid bias, pick from a list of non-edges or just try random pairs
        # Efficient way: pick random x, y
        found_better = False
        best_gain = 0
        best_swap = None
        
        # Try a few random non-edges to find one that reduces C4s
        # We can just pick one random non-edge and check
        # If it doesn't improve, we revert and try another? 
        # Or we can search for the best non-edge?
        # Given n=31, number of non-edges is ~465.
        # Checking all non-edges for each edge removal is 465 * 91 * (cost of check).
        # Cost of check is small.
        # Let's try to find the best non-edge to add after removing (u, v).
        
        best_new_c4s = float('inf')
        best_new_edge = None
        
        for x in range(n):
            for y in range(x + 1, n):
                if (x, y) in edge_set or (y, x) in edge_set:
                    continue
                # Check if adding (x, y) creates C4s
                # Note: (u, v) is already removed from edge_set and adj
                new_c4s = count_c4s_for_edge(x, y, adj)
                if new_c4s < best_new_c4s:
                    best_new_c4s = new_c4s
                    best_new_edge = (x, y)
        
        if best_new_edge is not None and best_new_c4s < c4s_removed:
            # Accept the swap
            x, y = best_new_edge
            adj[x].add(y)
            adj[y].add(x)
            edge_set.add((x, y))
            current_c4s -= c4s_removed
            current_c4s += best_new_c4s
        else:
            # Revert removal of (u, v)
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
    
    # Convert to list of lists
    result = [[u, v] for u, v in edge_set]
    
    # Ensure exactly 91 distinct edges
    if len(result) != target_edges:
        # This should not happen, but just in case
        pass
        
    return result

# EVOLVE-BLOCK-END
