import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 180
    A = list(range(1, 91))
    B = list(range(91, 181))
    
    # Graph on A
    # Node i represents number i+1
    # Edge if conflict
    adj = {i: set() for i in range(90)}
    
    # Map number to index
    idx = {v: i for i, v in enumerate(A)}
    
    # Generate edges
    for a in A:
        a2 = a * a
        # Find divisors
        divs = []
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                divs.append(d)
                if d != a2 // d:
                    divs.append(a2 // d)
        
        for x in divs:
            y = a2 // x
            if x == y:
                continue
            # WLOG x < y
            if x > y:
                x, y = y, x
            
            b = a + x
            c = a + y
            
            # Check if b in A and c in B
            if b in idx and c >= 91:
                i_a = idx[a]
                i_b = idx[b]
                adj[i_a].add(i_b)
                adj[i_b].add(i_a)
                
            # Check if b in B and c in B
            # This is a conflict with B
            # We will ignore for now, or mark a
            # If b >= 91 and c >= 91, we can't add a without removing b or c
            # We'll handle this in a second pass if time permits
            
    # Find Max Independent Set on adj
    # Greedy heuristic or exact if small
    # Graph is small (90 nodes)
    # Use a simple backtracking or greedy
    
    # Greedy: pick node with lowest degree, remove neighbors
    # Or use networkx if available? No, stdlib only.
    # Let's use a simple recursive backtracking with pruning, or greedy.
    # Given the structure, it's likely very sparse.
    
    # Let's try a simple greedy:
    # Sort nodes by degree
    # Pick node, remove it and neighbors
    # Repeat
    
    nodes = list(range(90))
    # Calculate degrees
    deg = [len(adj[i]) for i in nodes]
    
    # Simple greedy MIS
    selected = []
    remaining = set(nodes)
    
    # Sort by degree dynamically?
    # Let's just do a basic greedy
    # Actually, for small graphs, we can try to be smarter.
    # But 90 nodes is small enough for a decent heuristic.
    
    # Let's use a random restart greedy?
    # Or just deterministic greedy by degree.
    
    # To ensure robustness, let's try a few heuristics.
    best_mis = []
    
    # Heuristic 1: Greedy by min degree
    import random
    random.seed(42)
    
    def greedy_mis(adj, nodes):
        rem = set(nodes)
        sel = []
        while rem:
            # Pick node with min degree in rem
            min_deg = float('inf')
            min_node = None
            for n in rem:
                deg_n = len(adj[n] & rem)
                if deg_n < min_deg:
                    min_deg = deg_n
                    min_node = n
            sel.append(min_node)
            rem.discard(min_node)
            for n in adj[min_node]:
                if n in rem:
                    rem.discard(n)
        return sel
    
    # Run greedy multiple times with random tie-breaking?
    # Or just once.
    # The graph is likely a forest or near-forest.
    
    mis_nodes = greedy_mis(adj, nodes)
    
    # Construct S
    S = set(B)
    for i in mis_nodes:
        S.add(A[i])
        
    # Check validity
    def is_valid(S):
        S = set(S)
        for a in S:
            if a > 90: continue # No triples with a in B
            a2 = a*a
            for d in range(1, int(a2**0.5)+1):
                if a2 % d == 0:
                    pairs = [(d, a2//d)]
                    if d != a2//d:
                        pairs.append((a2//d, d))
                    for x, y in pairs:
                        if x == y: continue
                        b = a + x
                        c = a + y
                        if a in S and b in S and c in S:
                            return False
        return True
        
    if is_valid(S):
        return sorted(S)
    else:
        # Fallback
        return list(range(91, 181))
