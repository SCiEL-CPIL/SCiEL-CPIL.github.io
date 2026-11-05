import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Fast C6 check using bitsets
    def path5(u, v, adj):
        # Checks if there is a simple path of length 5 from u to v
        # u-a-b-c-d-v
        a_mask = adj[u] & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d must be in N(c) and N(v), and not u, a, b, v
                    d_mask = adj[c] & adj[v] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    if d_mask:
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj[i]:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def is_c6_free(adj):
        """Check if graph is C6-free by verifying no edge (u,v) has a path of length 5 between u and v."""
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj[u]:
                    if path5(u, v, adj):
                        return False
        return True

    def greedy_saturate(adj, deadline):
        """Add all possible edges until no more can be added."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return
                    if not ((1 << j) & adj[i]):
                        if not path5(i, j, adj):
                            add_edge(adj, i, j)
                            improved = True

    def remove_and_fill(adj, deadline):
        """Local search: remove one edge, then greedily add all possible new edges.
        Keep the change if net edge count increases."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Get current edges
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))
            
            if not current_edges:
                break
                
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() > deadline:
                    break
                
                # Remove edge (u, v)
                remove_edge(adj, u, v)
                old_count = count_edges(adj)
                
                # Greedily add all possible edges
                added_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if time.time() > deadline:
                            break
                        if not ((1 << j) & adj[i]):
                            if not path5(i, j, adj):
                                add_edge(adj, i, j)
                                added_edges.append((i, j))
                
                new_count = count_edges(adj)
                
                if new_count > old_count + 1:  # Net gain > 0
                    improved = True
                else:
                    # Revert
                    for i, j in added_edges:
                        remove_edge(adj, i, j)
                    add_edge(adj, u, v)

    for restart in range(80):
        if time.time() > deadline:
            break
        
        # Initialize: random graph with density ~0.4 (about 320 edges out of 780 possible)
        # This is dense enough to be near optimal but sparse enough to be C6-free
        adj = [0] * n
        p = 0.35 + 0.1 * (restart % 3)  # Vary density slightly
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    add_edge(adj, i, j)
        
        # Phase 1: Remove edges to make C6-free (if needed)
        # Check and fix C6s
        for _ in range(50):
            if time.time() > deadline:
                break
            # Find a C6 and remove an edge from it
            found = False
            for u in range(n):
                if found:
                    break
                for v in range(u + 1, n):
                    if found:
                        break
                    if (1 << v) & adj[u]:
                        if path5(u, v, adj):
                            # Remove edge (u, v) to break this C6
                            remove_edge(adj, u, v)
                            found = True
            if not found:
                break
        
        # Phase 2: Greedy saturation
        greedy_saturate(adj, deadline)
        
        # Phase 3: Remove-and-fill local search
        remove_and_fill(adj, deadline)
        
        # Update best
        current_edges_list = get_edges(adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges
