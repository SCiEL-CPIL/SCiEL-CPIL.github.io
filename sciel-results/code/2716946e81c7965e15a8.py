import random
import time

def construct():
    N = 256
    nodes = list(range(1, N + 1))
    
    # Precompute adjacency list for conflicts
    # adj[i] contains j if i and j conflict
    adj = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    def greedy_independent_set(order):
        selected = []
        selected_set = set()
        for u in order:
            # Check if u conflicts with any currently selected node
            # Optimization: if u has no neighbors in selected_set, it's safe
            # We can just iterate through selected_set
            is_conflict = False
            for v in selected_set:
                if v in adj[u]:
                    is_conflict = True
                    break
            if not is_conflict:
                selected.append(u)
                selected_set.add(u)
        return selected

    best_set = []
    start_time = time.time()
    timeout = 85.0 # Leave 5s margin for 90s limit
    
    # Try a few deterministic orders first
    # 1. Ascending
    s = greedy_independent_set(nodes)
    if len(s) > len(best_set):
        best_set = s
        
    # 2. Descending
    s = greedy_independent_set(sorted(nodes, reverse=True))
    if len(s) > len(best_set):
        best_set = s
        
    # 3. Order by degree (ascending) - nodes with fewer conflicts first
    degrees = [len(adj[i]) for i in range(N + 1)]
    order_by_degree = sorted(nodes, key=lambda x: degrees[x])
    s = greedy_independent_set(order_by_degree)
    if len(s) > len(best_set):
        best_set = s
        
    # 4. Random restarts
    # The greedy step is O(N^2) roughly. N=256.
    # 256^2 = 65536 operations.
    # Python can do millions of ops per second.
    # So we can do many restarts.
    
    while time.time() - start_time < timeout:
        random_order = nodes[:]
        random.shuffle(random_order)
        s = greedy_independent_set(random_order)
        if len(s) > len(best_set):
            best_set = s
            
    return best_set
