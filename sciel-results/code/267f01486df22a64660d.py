import time
import random

def construct():
    n = 15
    start = time.time()
    timeout = 55.0
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_index = {p: i for i, p in enumerate(all_pairs)}
    
    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_degrees(adj):
        return [bin(a).count('1') for a in adj]

    def is_valid(adj):
        deg = get_degrees(adj)
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    # Check edge (u, v)
                    # Neighbors of u excluding v
                    nu = adj[u] & ~(1 << v)
                    nv = adj[v] & ~(1 << u)
                    
                    # Get set of degrees of neighbors
                    degs_u = set()
                    mask = nu
                    while mask:
                        lsb = mask & -mask
                        s = lsb.bit_length() - 1
                        degs_u.add(deg[s])
                        mask ^= lsb
                        
                    degs_v = set()
                    mask = nv
                    while mask:
                        lsb = mask & -mask
                        t = lsb.bit_length() - 1
                        degs_v.add(deg[t])
                        mask ^= lsb
                        
                    if degs_u & degs_v:
                        return False
        return True

    def random_valid_seed():
        adj = [0] * n
        # Randomly add edges if valid
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_valid(adj):
                continue
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        return adj

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def simulate_annealing(initial_adj, time_limit):
        adj = initial_adj[:]
        deg = get_degrees(adj)
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        T = 10.0
        cooling = 0.995
        
        while time.time() - start < time_limit:
            # Generate move
            move_type = random.random()
            if move_type < 0.5:
                # Add edge
                # Pick random non-edge
                u, v = random.choice(all_pairs)
                if adj[u] & (1 << v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    new_edges = current_edges + 1
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = adj[:]
                    current_edges = new_edges
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            elif move_type < 0.8:
                # Remove edge
                u, v = random.choice(all_pairs)
                if not (adj[u] & (1 << v)):
                    continue
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                # Always valid after removal
                new_edges = current_edges - 1
                # Accept with probability
                delta = current_edges - new_edges # 1
                if random.random() < pow(2.718, -delta / T):
                    current_edges = new_edges
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Swap: Remove one, Add one
                u, v = random.choice(all_pairs)
                if not (adj[u] & (1 << v)):
                    continue
                x, y = random.choice(all_pairs)
                if adj[x] & (1 << y):
                    continue
                
                # Try swap
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                if is_valid(adj):
                    # Edge count remains same
                    # We might accept to explore, but usually we want to increase.
                    # However, swap can enable future additions.
                    # Accept with high prob to move around
                    if random.random() < 0.8:
                        pass # Accept
                    else:
                        # Reject
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        adj[x] ^= (1 << y)
                        adj[y] ^= (1 << x)
                else:
                    # Reject
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] ^= (1 << y)
                    adj[y] ^= (1 << x)
            
            T *= cooling
            if T < 0.01:
                break
                
        return best_adj, best_edges

    # Run multiple SA runs with different seeds
    best_edges_global = 0
    best_adj_global = [0]*n
    
    for i in range(20):
        if time.time() - start > timeout:
            break
        seed_adj = random_valid_seed()
        sa_adj, sa_edges = simulate_annealing(seed_adj, timeout - 1.0)
        if sa_edges > best_edges_global:
            best_edges_global = sa_edges
            best_adj_global = sa_adj
        
        # Also try greedy on the SA result?
        # Convert adj to edge list
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if best_adj_global[u] & (1 << v):
                    edges.append((u, v))
                    
        # Greedy add on the best found so far
        # This is a hybrid: SA to get good basin, Greedy to polish
        if time.time() - start < timeout - 5.0:
            # Convert to set
            edge_set = set(edges)
            changed = True
            while changed and time.time() - start < timeout - 2.0:
                changed = False
                # Shuffle pairs for randomness
                pairs = all_pairs[:]
                random.shuffle(pairs)
                for u, v in pairs:
                    if (u, v) in edge_set:
                        continue
                    edge_set.add((u, v))
                    adj_test = make_adj(list(edge_set))
                    if is_valid(adj_test):
                        changed = True
                    else:
                        edge_set.remove((u, v))
                        
            new_edges = list(edge_set)
            new_count = len(new_edges)
            if new_count > best_edges_global:
                best_edges_global = new_count
                # Update global best adj? Not strictly needed if we just return edges
                best_adj_global = make_adj(new_edges)

    # Return best edges
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj_global[u] & (1 << v):
                final_edges.append([u, v])
                
    return final_edges
