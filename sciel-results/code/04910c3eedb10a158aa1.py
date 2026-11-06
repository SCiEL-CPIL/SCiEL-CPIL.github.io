import time
import random

def construct():
    deadline = time.time() + 115
    n = 40
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {}
    for idx, (i, j) in enumerate(all_edges):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx

    def has_c6_with_edge(u, v, adj):
        """Check if adding edge (u,v) to graph with adjacency adj creates a C6."""
        a_mask = adj[u] & ~masks[v]
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = adj[a] & ~masks[u] & ~masks[v]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                c_mask = adj[b] & ~masks[u] & ~masks[a] & ~masks[v]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    d_mask = adj[c] & adj[v]
                    forbidden = masks[u] | masks[a] | masks[b] | masks[c]
                    if d_mask & ~forbidden:
                        return True
        return False

    def greedy_fill(adj, order=None):
        """Try to add all possible edges in given order (or random)."""
        if order is None:
            order = all_edges[:]
            random.shuffle(order)
        for u, v in order:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v, adj):
                    adj[u] |= masks[v]
                    adj[v] |= masks[u]
        return adj

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def random_init():
        """Create a random sparse starting graph."""
        adj = [0] * n
        # Start with a few random edges
        num_initial = random.randint(10, 25)
        for _ in range(num_initial):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (adj[u] & masks[v]) == 0:
                adj[u] |= masks[v]
                adj[v] |= masks[u]
        # Greedy fill
        greedy_fill(adj)
        return adj

    def local_search(adj, max_iterations=50):
        """Perform local search: try edge swaps to improve."""
        current = adj[:]
        current_count = count_edges(current)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try several random swaps
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge to remove
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v or (current[u] & masks[v]) == 0:
                    continue
                
                # Pick random non-edge to add
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y or (current[x] & masks[y]) != 0:
                    continue
                
                # Try the swap
                test = current[:]
                test[u] &= ~masks[v]
                test[v] &= ~masks[u]
                
                if not has_c6_with_edge(x, y, test):
                    test[x] |= masks[y]
                    test[y] |= masks[x]
                    # Greedy fill on the test graph
                    greedy_fill(test)
                    new_count = count_edges(test)
                    if new_count > current_count:
                        current = test
                        current_count = new_count
                        improved = True
                        break
            
            if not improved:
                # Perturb: remove 1-2 random edges
                for _ in range(random.randint(1, 2)):
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v and (current[u] & masks[v]):
                        current[u] &= ~masks[v]
                        current[v] &= ~masks[u]
                greedy_fill(current)
                current_count = count_edges(current)
        
        return current, current_count

    def systematic_swap_search(adj, max_iterations=100):
        """Systematically try edge exchanges."""
        current = adj[:]
        current_count = count_edges(current)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try all pairs of (remove edge, add edge) in random order
            edges_to_remove = []
            for i in range(n):
                for j in range(i + 1, n):
                    if current[i] & masks[j]:
                        edges_to_remove.append((i, j))
            
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (current[i] & masks[j]):
                        non_edges.append((i, j))
            
            random.shuffle(edges_to_remove)
            random.shuffle(non_edges)
            
            # Try a subset of swaps
            for rem in edges_to_remove[:20]:
                if time.time() > deadline - 5:
                    break
                for add in non_edges[:20]:
                    if time.time() > deadline - 5:
                        break
                    
                    u, v = rem
                    x, y = add
                    
                    test = current[:]
                    test[u] &= ~masks[v]
                    test[v] &= ~masks[u]
                    
                    if not has_c6_with_edge(x, y, test):
                        test[x] |= masks[y]
                        test[y] |= masks[x]
                        greedy_fill(test)
                        new_count = count_edges(test)
                        if new_count > current_count:
                            current = test
                            current_count = new_count
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return current, current_count

    best_adj = [0] * n
    best_count = 0

    # Phase 1: Multi-start greedy (30 seconds)
    phase1_deadline = time.time() + 30
    while time.time() < phase1_deadline:
        adj = random_init()
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    # Phase 2: Local search on best solution (40 seconds)
    phase2_deadline = time.time() + 40
    if best_count > 0:
        improved_adj, improved_count = local_search(best_adj[:], max_iterations=200)
        if improved_count > best_count:
            best_count = improved_count
            best_adj = improved_adj

    # Phase 3: Systematic swap search (remaining time)
    if best_count > 0 and time.time() < deadline - 10:
        improved_adj, improved_count = systematic_swap_search(best_adj[:], max_iterations=200)
        if improved_count > best_count:
            best_count = improved_count
            best_adj = improved_adj

    # Final greedy fill
    greedy_fill(best_adj)
    final_count = count_edges(best_adj)
    if final_count > best_count:
        best_count = final_count

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & masks[j]:
                edges.append([i, j])
    return edges
