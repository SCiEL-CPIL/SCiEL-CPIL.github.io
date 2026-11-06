import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    N = 256
    # 1. Identify nodes with self-conflict (2*a is a perfect square)
    excluded = set()
    for a in range(1, N + 1):
        s = 2 * a
        r = int(s ** 0.5)
        if r * r == s or (r + 1) * (r + 1) == s:
            excluded.add(a)
    
    # Valid nodes are those not excluded
    valid_nodes = [a for a in range(1, N + 1) if a not in excluded]
    
    # 2. Build adjacency list for valid nodes
    # a ~ b if a+b is a perfect square and a != b
    # We only care about edges between valid nodes.
    adj = {a: set() for a in valid_nodes}
    squares = [i * i for i in range(1, 23)] # up to 22^2=484, max sum 511
    
    for a in valid_nodes:
        for b in valid_nodes:
            if a >= b:
                continue
            s = a + b
            if s in squares:
                adj[a].add(b)
                adj[b].add(a)
                
    # 3. Greedy Max Independent Set with Random Restarts
    best_set = set()
    best_size = 0
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Randomize the order of nodes to break ties
        # We use a greedy removal strategy:
        # Start with all valid nodes.
        # While there is a conflict, remove the node with the highest degree in the current set.
        
        current_set = set(valid_nodes)
        
        # To make it faster, we can pre-compute degrees or just calculate on the fly.
        # N=245, so on-the-fly is fine.
        
        # Optimization: 
        # We can simulate the removal.
        # Actually, a simpler greedy:
        # Sort nodes by degree (static).
        # Iterate through sorted nodes. If node has no neighbors in *selected* set, add it.
        # This is "Greedy MIS".
        
        # Let's try the "Remove Highest Degree" approach which is often better for sparse graphs.
        # But "Add if no conflict" (Greedy MIS) is faster.
        # Let's compare.
        # Greedy MIS:
        # 1. Sort nodes by degree descending.
        # 2. For each node, if it has no neighbors in the result set, add it.
        
        # Let's try multiple strategies in the loop.
        
        # Strategy 1: Greedy MIS (Add if possible)
        # Sort by degree
        nodes_by_degree = sorted(valid_nodes, key=lambda x: len(adj[x]), reverse=True)
        
        # Randomize tie-breaking by shuffling equal degrees?
        # Or just shuffle the whole list slightly?
        # Let's shuffle the nodes with the same degree.
        
        # To ensure diversity, we can shuffle the input order before sorting if degrees are equal.
        # Or just run the standard greedy on a shuffled list of nodes?
        # Standard greedy on shuffled list:
        shuffled_nodes = valid_nodes[:]
        random.shuffle(shuffled_nodes)
        
        # Greedy Add
        s1 = set()
        for a in shuffled_nodes:
            # Check if a has any neighbor in s1
            conflict = False
            for n in adj[a]:
                if n in s1:
                    conflict = True
                    break
            if not conflict:
                s1.add(a)
        
        if len(s1) > best_size:
            best_size = len(s1)
            best_set = s1
            
        # Strategy 2: Greedy Remove
        # Start with all nodes.
        # While not independent, remove node with max degree in induced subgraph.
        s2 = set(valid_nodes)
        # We need to efficiently find the node with max degree in the current set.
        # Since N is small, we can just recalculate.
        
        # To speed up removal, we can just pick a random node with a conflict?
        # Or the one with max degree.
        
        # Let's try a hybrid:
        # Start with s2.
        # While time allows:
        #   Find all conflicts.
        #   If none, break.
        #   Pick node with max degree in s2.
        #   Remove it.
        
        # This might be slow if done naively in Python loops.
        # But 245 nodes, degree ~10-20.
        
        # Let's just stick to Strategy 1 (Greedy Add) with many restarts.
        # It is very fast.
        
        # We can also try "Greedy Add" but sorting by degree dynamically?
        # No, static degree is a good proxy.
        
        # Let's try to improve s1 with local search.
        # Local Search:
        # Try to swap: remove u from s1, add v not in s1.
        # If len increases, keep.
        # If len same, keep if it reduces "future conflicts"? (Hard to define).
        
        # Simple Hill Climbing:
        # For each u in s1:
        #   Remove u.
        #   For each v not in s1:
        #     If v can be added to (s1 - {u}):
        #        New set = (s1 - {u}) + {v}
        #        If len(New set) > len(s1): s1 = New set
        # This is O(N^2) per step.
        
        # Given the time limit, let's just run many Greedy Add restarts.
        # 245 nodes, check ~245 * avg_degree (~15) = ~3600 ops per restart.
        # 100,000 restarts = 360M ops. Might be too slow in Python.
        # Let's estimate speed.
        # 1 restart ~ 0.001s?
        # 50s / 0.001 = 50,000 restarts. Should be enough.
        
        pass # Loop continues

    return list(best_set)
