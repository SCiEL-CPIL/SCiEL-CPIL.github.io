import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to find a larger graph (e.g., 42 vertices)
    # We will attempt to construct a valid (5,5)-free graph on n=42
    
    def find_clique(adj, n, size):
        # adj is a list of bitmasks
        # Returns a list of vertices forming a clique of 'size' or None
        if n < size:
            return None
        
        def dfs(candidates, current):
            if len(current) == size:
                return current[:]
            
            # Pruning: if not enough candidates left
            if candidates.bit_count() < size - len(current):
                return None
            
            while candidates:
                # Pick a vertex
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_candidates = candidates & adj[v]
                res = dfs(new_candidates, current + [v])
                if res:
                    return res
            return None
        
        return dfs((1 << n) - 1, [])

    def is_valid(adj, n):
        # Check for K5
        if find_clique(adj, n, 5):
            return False
        # Check for Independent Set of 5 (Clique of 5 in complement)
        # Complement adj: adj_comp[i] = ((1<<n)-1) ^ (1<<i) ^ adj[i]
        adj_comp = [((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)]
        if find_clique(adj_comp, n, 5):
            return False
        return True

    def repair_graph(adj, n, max_steps=1000):
        # Iteratively remove edges to break K5 and add edges to break I5
        # adj is list of bitmasks
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return False, adj
            
            # Check for K5
            clique = find_clique(adj, n, 5)
            if clique:
                # Pick a random edge in the clique to remove
                u, v = random.sample(clique, 2)
                # Remove edge (u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                steps += 1
                continue
            
            # Check for I5 (Clique in complement)
            adj_comp = [((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)]
            indep = find_clique(adj_comp, n, 5)
            if indep:
                # Pick a random non-edge in the independent set to add
                u, v = random.sample(indep, 2)
                # Add edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                steps += 1
                continue
            
            # No violations found
            return True, adj
        
        return False, adj

    # Try to find a 42-vertex graph
    n_target = 42
    found_valid = False
    valid_adj = None
    
    # Try a few random initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize random graph
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        success, adj = repair_graph(adj, n_target, max_steps=2000)
        if success:
            found_valid = True
            valid_adj = adj
            break

    if found_valid:
        edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return n_target, edges

    # Fallback to seed
    return q, seed_edges
