import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 40
    # adj[i] is a set of neighbors
    adj = [set() for _ in range(n)]
    
    def get_edges():
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges
    
    # Initial setup: 8 K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a].add(base+b)
                adj[base+b].add(base+a)
                
    def check_and_get_witness(u, v):
        """
        Returns (is_valid, witness_path)
        """
        if v in adj[u]:
            return True, None
        
        A = adj[u] - {v}
        B = adj[v] - {u}
        
        if not A or not B:
            return True, None

        # We need a path a-w2-w3-b where a in A, b in B
        # a, w2, w3, b must be distinct and not u, v
        
        for a in A:
            # w2 must be neighbor of a, not u, v
            N_a = adj[a] - {u, v}
            for w2 in N_a:
                if w2 == a: continue
                # w3 must be neighbor of w2, not u, v
                N_w2 = adj[w2] - {u, v}
                for w3 in N_w2:
                    if w3 == a or w3 == w2: continue
                    # b must be neighbor of w3, not u, v, and in B
                    N_w3 = adj[w3] - {u, v}
                    # Check intersection with B
                    # Also ensure b is distinct from a, w2, w3
                    for b in (N_w3 & B):
                        if b == a or b == w2 or b == w3:
                            continue
                        # Check all distinct
                        if len({u, a, w2, w3, b, v}) == 6:
                            return False, [a, w2, w3, b]
        
        return True, None

    def is_valid_swap(u, v, x, y):
        """
        Check if G - {(x, y)} + {(u, v)} is C6-free.
        Assumes (u, v) is not in G, (x, y) is in G.
        """
        # Temporarily remove x, y
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Check if (u, v) can be added
        valid, _ = check_and_get_witness(u, v)
        
        # Restore x, y
        adj[x].add(y)
        adj[y].add(x)
        
        return valid

    best_edges = get_edges()
    
    while time.time() < deadline:
        # 1. Greedy Add
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
        random.shuffle(missing)
        
        added_any = False
        for u, v in missing:
            if time.time() > deadline:
                break
            valid, witness = check_and_get_witness(u, v)
            if valid:
                adj[u].add(v)
                adj[v].add(u)
                added_any = True
                # Update missing list? No, just continue.
                # The added edge might block others, but we already checked them.
                # Actually, adding an edge doesn't unblock others.
                
        if not added_any:
            # 2. Local Search
            if not missing:
                break
                
            # Pick a random missing edge
            u, v = random.choice(missing)
            valid, witness = check_and_get_witness(u, v)
            
            if not valid and witness:
                a, w2, w3, b = witness
                # C6: u-a-w2-w3-b-v
                # Edges: (u,a), (a,w2), (w2,w3), (w3,b), (b,v)
                candidates = [(u, a), (a, w2), (w2, w3), (w3, b), (b, v)]
                
                valid_candidates = []
                for x, y in candidates:
                    if x > y: x, y = y, x
                    if y in adj[x]:
                        valid_candidates.append((x, y))
                
                if valid_candidates:
                    x, y = random.choice(valid_candidates)
                    if is_valid_swap(u, v, x, y):
                        # Perform swap
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
    return get_edges()

# EVOLVE-BLOCK-END
