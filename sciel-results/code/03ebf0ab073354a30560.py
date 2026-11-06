import time
import random

def _elapsed(start_time):
    return time.time() - start_time

def _deadline_exceeded(start_time, limit=110.0):
    return _elapsed(start_time) > limit

def _has_triangle(adj):
    n = len(adj)
    for i in range(n):
        # Check if neighbors of i contain an edge
        neighbors = adj[i]
        for j in neighbors:
            for k in neighbors:
                if j < k and adj[j][k]:
                    return True
    return False

def _max_independent_set_size(adj, limit=10):
    """
    Returns the size of the maximum independent set.
    Stops if size >= limit.
    """
    n = len(adj)
    # Use a simple branch and bound with bitsets for speed
    # adj[i] is a set of neighbors
    
    # Convert to bitmask for speed? n <= 40, so bitmask is easy.
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    best = 0
    # Order vertices by degree descending for better pruning?
    # Simple recursive search
    def search(available_mask, current_size, current_set):
        nonlocal best
        if current_size > best:
            best = current_size
            if best >= limit:
                return True
        if available_mask == 0:
            return False
            
        # Pick a vertex
        # Heuristic: pick vertex with max degree in available subgraph?
        # For now, pick lowest index in available_mask
        v = (available_mask & -available_mask).bit_length() - 1
        
        # Option 1: Include v
        # New available mask: remove v and its neighbors
        new_mask = available_mask & ~(masks[v] | (1 << v))
        if search(new_mask, current_size + 1, current_set | (1 << v)):
            return True
            
        # Option 2: Exclude v
        new_mask = available_mask & ~(1 << v)
        if search(new_mask, current_size, current_set):
            return True
            
        return False
        
    all_mask = (1 << n) - 1
    search(all_mask, 0, 0)
    return best

def construct():
    start_time = time.time()
    
    # Base graph: And(9)
    def get_and_9():
        n = 26
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % 26 % 3 == 1:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj, n

    adj, n = get_and_9()
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to extend to 39
    target_n = 39
    # We will try to add (target_n - n) vertices
    
    # To improve success rate, we can try multiple random restarts of the extension process
    # But first, let's just try one deterministic-ish pass with random tie-breaking
    
    current_adj = [set(s) for s in adj]
    current_n = n
    
    for step in range(target_n - current_n):
        if _deadline_exceeded(start_time):
            break
            
        added = False
        # Try to find an independent set S in current_adj such that alpha(current_adj \ S) <= 8
        # We will try random independent sets
        
        # Generate random independent sets
        # A simple way: pick a random vertex, add it to S, remove its neighbors, repeat.
        # Or pick a random subset and check if it's independent.
        
        # Better: Enumerate small independent sets? No, size can be large.
        # Let's just try random sets and filter.
        
        # Actually, we can try to pick S from a large independent set in G.
        # Let I be a max independent set in G.
        # If we pick S subset of I, then G \ S is still triangle free.
        # We need alpha(G \ S) <= 8.
        
        # Let's try to pick S as a "random" independent set.
        # We will try 1000 random attempts to find a valid S.
        
        for _ in range(1000):
            if _deadline_exceeded(start_time):
                break
                
            # Generate a random independent set S
            S = set()
            available = set(range(current_n))
            while available:
                v = random.choice(list(available))
                S.add(v)
                # Remove v and its neighbors from available
                available -= (current_adj[v] | {v})
            
            # Check if S is independent (it is by construction)
            # Check if alpha(G \ S) <= 8
            # Build adj for G \ S
            # We can just check if there is an independent set of size 9 in G \ S
            
            # To speed up, we can just run the alpha checker on the subgraph induced by V\S
            # But building the subgraph is O(n^2).
            # Let's optimize: 
            # If |S| is small, G\S is large.
            # If |S| is large, G\S is small.
            
            # Let's just check if alpha(G\S) >= 9.
            # If yes, this S is invalid.
            
            # We can approximate by checking if there is a clique of size 9 in the complement of G\S?
            # No, just run the alpha checker.
            
            # To make it faster, we can pre-compute something?
            # Let's just run the alpha checker. It's fast for n<=40.
            
            # Create a list of vertices not in S
            vertices = [v for v in range(current_n) if v not in S]
            if len(vertices) < 9:
                # alpha must be <= 8
                valid = True
            else:
                # Map vertices to 0..len(vertices)-1
                mapping = {v: i for i, v in enumerate(vertices)}
                sub_adj = [set() for _ in range(len(vertices))]
                for i, u in enumerate(vertices):
                    for v in vertices:
                        if v in current_adj[u] and v > u:
                            sub_adj[i].add(mapping[v])
                            sub_adj[mapping[v]].add(i)
                
                alpha_sub = _max_independent_set_size(sub_adj, limit=9)
                if alpha_sub < 9:
                    valid = True
                else:
                    valid = False
            
            if valid:
                # Add new vertex connected to S
                new_v = current_n
                current_adj.append(set(S))
                for u in S:
                    current_adj[u].add(new_v)
                current_n += 1
                added = True
                break
        
        if not added:
            # Failed to add a vertex. Try to restart or give up.
            # For now, break and return current best.
            break
            
    # Update best if we did better
    if current_n > best_n:
        best_n = current_n
        best_adj = current_adj
        
    # If we didn't reach 39, maybe we can try a different strategy?
    # For now, return the best found.
    
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
