"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a Q3-free graph on 16 vertices (0..15) maximizing edge count.
    
    Strategy:
    1. Start with a known Q3-free graph (e.g., complete bipartite K8,8 minus some edges,
       or a random construction).
    2. Use a local search / simulated annealing approach:
       - Try adding random edges. If Q3-free, keep it.
       - Try removing edges to allow adding more? No, we want to maximize edges.
       - Actually, since we want to maximize edges, we should try to add edges.
       - If stuck, try to swap: remove one edge, add another?
       
       Wait, a better approach for maximization under a hereditary property (like being
       Q3-free, which is hereditary):
       - Start with a sparse graph.
       - Randomly add edges. If the property is violated, reject.
       - To escape local optima, we might need to remove edges and re-add.
       
       However, Q3-free is a property that is *not* monotone in the sense that adding edges
       can violate it, but removing edges always preserves it.
       
       A standard approach for Turan-type problems with forbidden subgraphs is to use
       a "greedy" algorithm with backtracking or randomized restarts.
       
       Given the small size (16 vertices), we can use a bit-adjacency matrix and a
       fast check for Q3.
       
       Let's use a randomized local search:
       1. Initialize with a random Q3-free graph (e.g., by adding random edges and checking).
       2. Repeat:
          a. Pick a random non-edge (u, v).
          b. Check if adding (u, v) creates a Q3.
          c. If not, add it.
          d. If yes, with some probability, try to "swap": remove a random edge (x, y)
             and add (u, v), checking if the new graph is Q3-free.
       
       To make this faster, we need an efficient `has_q3(adj)` function.
       
       Given 16 vertices, the number of 8-subsets is C(16, 8) = 12870.
       For each 8-subset, we can check if it induces a Q3.
       Checking if a specific 8-vertex induced subgraph is a Q3 can be done by
       checking if it is 3-regular and bipartite.
       
       But checking all 12870 subsets every time we add an edge is too slow if we do
       thousands of iterations.
       
       Optimization:
       When adding edge (u, v), any new Q3 must contain (u, v).
       So we only need to check 8-subsets S such that u, v in S.
       The number of such subsets is C(14, 6) = 3003.
       For each such S, check if G[S] (with the new edge) is a Q3.
       
       How to check if an 8-vertex induced subgraph is a Q3?
       1. All vertices in S must have degree 3 within S.
       2. The graph must be bipartite.
       
       We can precompute the degrees within S? No, S changes.
       
       Let's implement a helper `is_q3(adj, S)` where S is a tuple of 8 vertices.
       adj is a list of 16 integers (bitmasks).
       
       Pseudocode for is_q3(adj, S):
       - Map S to indices 0..7.
       - Create a 8x8 adjacency matrix or bitmasks for these 8 vertices.
       - Check degrees: for each v in S, count neighbors in S. Must be 3.
       - Check bipartiteness: BFS/DFS on the 8-vertex graph.
       
       This is fast enough. 3003 subsets * (degree check + BFS) per edge addition.
       
       We will implement this.
    """
    n = 16
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    # Precompute all 8-subsets
    # C(16, 8) = 12870
    all_8_subsets = list(combinations(range(n), 8))
    
    # For faster checking, let's precompute for each pair (u, v),
    # the list of 8-subsets containing both u and v.
    # subsets_by_pair[u][v] = list of subsets S (as frozensets or tuples) such that u,v in S
    subsets_by_pair = [[None for _ in range(n)] for _ in range(n)]
    for S in all_8_subsets:
        for u, v in combinations(S, 2):
            if subsets_by_pair[u][v] is None:
                subsets_by_pair[u][v] = []
                subsets_by_pair[v][u] = subsets_by_pair[u][v]
            subsets_by_pair[u][v].append(S)
            
    def is_q3_subset(S):
        """Check if the induced subgraph on S is a Q3."""
        # S is a tuple of 8 vertices
        # Check degrees
        # Create a local mapping from vertex to 0..7
        # But we can just use the global adj
        for v in S:
            # Count neighbors of v in S
            mask = 0
            for w in S:
                if w == v: continue
                if (adj[v] >> w) & 1:
                    mask |= (1 << w)
            # Count bits in mask
            if bin(mask).count('1') != 3:
                return False
        # Check bipartiteness
        # BFS on the induced subgraph
        # Start from S[0]
        visited = {S[0]: 0}
        queue = deque([S[0]])
        while queue:
            u = queue.popleft()
            for w in S:
                if w == u: continue
                if (adj[u] >> w) & 1:
                    if w not in visited:
                        visited[w] = 1 - visited[u]
                        queue.append(w)
                    elif visited[w] == visited[u]:
                        return False # Odd cycle found
        return True
        
    def has_q3_after_adding(u, v):
        """Check if adding edge (u, v) creates a Q3."""
        # Set the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        for S in subsets_by_pair[u][v]:
            if is_q3_subset(S):
                # Revert and return True
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                return True
        # Revert
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        return False
        
    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
        
    # Initialization: Random greedy
    random.seed(42) # Fixed seed for reproducibility, or random?
    # Let's try to start with a known good graph or a random one.
    # A random graph with p=0.5 might already have Q3.
    # Let's start with an empty graph and add edges randomly.
    
    best_adj = [0] * n
    best_edges = 0
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10 seconds margin
    
    # Greedy initialization
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5: # Try to add
                if not has_q3_after_adding(u, v):
                    pass # Edge is added inside the function? 
                    # Wait, my has_q3_after_adding reverts.
                    # I need a function that ADDS if valid.
    
    # Let's restructure.
    # adj is modified in place.
    
    # Greedy Phase
    for _ in range(50): # 50 random restarts?
        adj = [0] * n
        # Try to add edges in random order
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() > deadline: break
            if not has_q3_after_adding(u, v):
                # The function reverts. I need to modify it to NOT revert if valid.
                # Let's write a helper `try_add_edge(u, v)`
                pass
                
    # Okay, let's implement `try_add_edge(u, v)` properly.
    
    def try_add_edge(u, v):
        """Adds edge (u, v) if it doesn't create a Q3. Returns True if added."""
        if (adj[u] >> v) & 1:
            return False
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        for S in subsets_by_pair[u][v]:
            if is_q3_subset(S):
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                return False
        
        return True
        
    # Restart loop
    global_best_adj = [0] * n
    global_best_count = 0
    
    for restart in range(100): # Limit restarts
        if time.time() > deadline: break
        adj = [0] * n
        # Random edge order
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if try_add_edge(u, v):
                pass
        
        # Local Search / Swaps
        # We are stuck. Try to improve.
        # Since we can't add more edges, we must swap.
        # Remove a random edge (x, y), then try to add the missing edges.
        
        for _ in range(200): # Iterations per restart
            if time.time() > deadline: break
            
            # Pick a random edge to remove
            edge_to_remove = None
            # Find all existing edges
            existing = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if v > u:
                        existing.append((u, v))
                    mask ^= lsb
            
            if not existing:
                break
                
            x, y = random.choice(existing)
            # Remove (x, y)
            adj[x] ^= (1 << y)
            adj[y] ^= (1 << x)
            
            # Try to add random non-edges
            improved = False
            # Try all non-edges? Or random ones?
            # Let's try random non-edges first, then maybe sweep.
            for _ in range(50):
                u, v = random.choice(edges)
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(u, v):
                        improved = True
                        break
            
            # If we didn't improve, maybe we should try to re-add (x, y) if it was good?
            # Or keep it removed if we added something else?
            # If we added something, the edge count is the same (removed 1, added 1).
            # But the structure changed.
            
            if not improved:
                # Try to re-add (x, y)
                if try_add_edge(x, y):
                    pass # Back to original state
                else:
                    pass # Stay in new state? But we didn't add anything else.
                    # If we removed (x,y) and couldn't add anything, we are worse off.
                    # So we MUST re-add (x,y) if we didn't add anything else.
                    # But if try_add_edge(x,y) fails, it means the current state (without x,y)
                    # allows (x,y)? No, it means adding (x,y) creates Q3.
                    # But we removed (x,y) from a Q3-free graph. So adding it back should be safe?
                    # Wait. G was Q3-free. G - (x,y) is Q3-free.
                    # Does G - (x,y) + (x,y) = G have Q3? No, G was Q3-free.
                    # So try_add_edge(x, y) SHOULD succeed.
                    # Why would it fail?
                    # Ah, `try_add_edge` checks for NEW Q3s.
                    # If G is Q3-free, then adding (x,y) to G-(x,y) restores G, which is Q3-free.
                    # So it should always succeed.
                    pass
            
        # Update global best
        current_count = count_edges()
        if current_count > global_best_count:
            global_best_count = current_count
            global_best_adj = adj[:]
            
    # Return the best graph found
    result = []
    for u in range(n):
        mask = global_best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v > u:
                result.append([u, v])
            mask ^= lsb
            
    return result

# EVOLVE-BLOCK-END
