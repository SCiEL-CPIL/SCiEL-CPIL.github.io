import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 20
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_pairs = len(all_pairs)  # 190
    
    def is_valid(adj):
        """Check if the graph represented by adj bitmasks is K3,3-free."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True
    
    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges
    
    def find_violation(adj):
        """Find a violating triple and its common neighbors, or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k, cijk)
        return None
    
    def make_valid(adj):
        """Remove edges until the graph is valid."""
        attempts = 0
        while time.time() < deadline - 20:
            viol = find_violation(adj)
            if viol is None:
                return True
            i, j, k, cijk = viol
            # Remove an edge from one of the triple vertices to one of the common neighbors
            bits = cijk
            x = (bits & (-bits)).bit_length() - 1
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            attempts += 1
            if attempts > 2000:
                break
        return is_valid(adj)
    
    def greedy_add(adj):
        """Try to add edges greedily. Returns number of edges added."""
        added = 0
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    added += 1
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
        return added
    
    def local_search(adj, time_limit):
        """Local search with 1-swap moves."""
        edges_in = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        best_count = count_edges(adj)
        
        while time.time() < time_limit and non_edges and edges_in:
            # Try a few random swaps
            for _ in range(min(50, len(edges_in) * len(non_edges))):
                e_rm = random.choice(edges_in)
                e_add = random.choice(non_edges)
                u1, v1 = e_rm
                u2, v2 = e_add
                
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                if adj[u2] & (1 << v2):
                    continue
                
                # Try the swap
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                if is_valid(adj):
                    greedy_add(adj)
                    c = count_edges(adj)
                    if c > best_count:
                        best_count = c
                        edges_in = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
                        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                    break
                else:
                    # Undo
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
        
        return best_count
    
    def generate_paley_19():
        """Paley-type graph on 19 vertices + one extra vertex."""
        adj = [0] * 19
        residues = set((x * x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u + 1, 19):
                if (v - u) % 19 in residues:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Add vertex 19 connected to some vertices
        # Connect to a few to maintain structure
        for v in range(0, 19, 3):
            adj[18] |= (1 << v)
            adj[v] |= (1 << 18)
        adj.append(adj[18])  # vertex 19's adjacency
        for v in range(19):
            if adj[19] & (1 << v):
                adj[v] |= (1 << 19)
        return adj
    
    def generate_random(p):
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj
    
    best_adj = [0] * n
    best_count = 0
    best_edges = []
    
    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)
    
    random.seed(12345)
    
    # Multiple restarts with different strategies
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        # Choose initial graph
        if s == 0:
            adj = generate_paley_19()
        elif s == 1:
            adj = generate_random(0.6)
        elif s == 2:
            adj = generate_random(0.5)
        elif s == 3:
            adj = generate_random(0.7)
        elif s == 4:
            adj = generate_random(0.55)
        elif s == 5:
            adj = generate_random(0.65)
        else:
            adj = generate_random(random.uniform(0.3, 0.7))
        
        # Make valid
        if not make_valid(adj):
            continue
        
        # Greedy add
        greedy_add(adj)
        
        # Local search with time limit
        remaining = deadline - 15 - time.time()
        if remaining > 5:
            local_search(adj, time.time() + min(remaining * 0.7, 10))
        
        update_best(adj)
    
    # Fallback
    if not best_edges:
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if (j - i) % 2 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if not is_valid(adj):
            return [[i, i+1] for i in range(n-1)]
        return adj_to_edges(adj)
        
    return best_edges

# EVOLVE-BLOCK-END
