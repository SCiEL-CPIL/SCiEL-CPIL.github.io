import random
import time

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                
    def is_valid_adj(adj):
        """Check if adj is K3,3-free. adj is list of ints (bitmasks)."""
        for a, b, c in triples:
            if (adj[a] & adj[b] & adj[c]).bit_count() > 2:
                return False
        return True
        
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) to adj. Return True if successful."""
        if (adj[u] >> v) & 1:
            return True # Already exists
        
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check validity for triples involving u or v
        # We only need to check triples where the intersection size might have increased.
        # The intersection N(a)&N(b)&N(c) can only increase if a,b,c includes u or v.
        
        valid = True
        # Triples containing u
        for x in range(n):
            if x == u: continue
            for y in range(n):
                if y == u or y == x: continue
                # Triple (u, x, y)
                if (adj[u] & adj[x] & adj[y]).bit_count() > 2:
                    valid = False
                    break
            if not valid: break
        if not valid:
            # Rollback
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False
            
        # Triples containing v (but not u, as those were checked above if x,y != u)
        # Wait, the loop above checked all (u, x, y). 
        # We also need to check triples (v, x, y) where x, y != u.
        # Actually, if a triple (a,b,c) does not contain u or v, its intersection didn't change.
        # If it contains u, we checked it.
        # If it contains v but not u, we need to check it.
        # The loop above iterated x,y over all other vertices. 
        # If x or y is v, it's covered?
        # If triple is (v, x, y), and x,y != u.
        # In the first loop, we fixed u. We iterated x, y.
        # If x=v, we checked (u, v, y).
        # If y=v, we checked (u, x, v).
        # But we haven't checked (v, x, y) where x,y != u.
        
        for x in range(n):
            if x == v or x == u: continue
            for y in range(n):
                if y == v or y == u or y == x: continue
                if (adj[v] & adj[x] & adj[y]).bit_count() > 2:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False
            
        return True

    best_edges = 0
    best_adj = None
    
    while time.time() < deadline:
        # Start with a random graph or empty
        # Let's start with a random graph with p=0.3 to get a good head start
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Clean up: remove edges until valid
        # Find a violation and remove an edge from it
        changed = True
        while changed:
            changed = False
            for a, b, c in triples:
                common = adj[a] & adj[b] & adj[c]
                if common.bit_count() > 2:
                    # Find an edge to remove
                    # The K3,3 is formed by {a,b,c} and 3 vertices in common.
                    # Removing any edge between {a,b,c} and common breaks this specific K3,3.
                    # But it might break others.
                    # Pick a random vertex from common and a random vertex from {a,b,c}
                    common_list = [i for i in range(n) if (common >> i) & 1]
                    # Actually, just pick any edge (a, v) where v in common
                    v = random.choice(common_list)
                    adj[a] &= ~(1 << v)
                    adj[v] &= ~(1 << a)
                    changed = True
                    break # Restart search for violation after change
            
        # Now adj is valid. Try to add edges.
        # Randomized Greedy Addition
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if try_add_edge(adj, u, v):
                pass # Added
        
        # Local Search: Try to add any missing edge
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
            random.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if try_add_edge(adj, u, v):
                    improved = True
        
        # Count edges
        edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result
