import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (killed at 150s, but keep buffer)
    
    n = 20
    all_vertices = range(n)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) to the graph defined by adj 
        violates the K3,3-free condition.
        Assumes (u, v) is NOT currently an edge.
        """
        if adj[u] & (1 << v):
            return False
            
        # Case 1: u and v are in the same part of K3,3
        # Common neighbors of u and v (old)
        common_uv = adj[u] & adj[v]
        
        # We need to check if there exists 'a' such that 
        # |N(u) & N(v) & N(a)| >= 3
        # Note: N(u) and N(v) are old.
        # If we pick a from common_uv, a is neighbor of u and v.
        # We need 3 neighbors in common_uv that are also neighbors of a.
        
        # Iterate through bits in common_uv
        temp = common_uv
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check if a has at least 3 neighbors in common_uv
            if (common_uv & adj[a]).bit_count() >= 3:
                return False
                
        # Case 2: u and v are in different parts of K3,3
        # Parts: {u, a, b} and {v, c, d}
        # u is connected to v (new), a, b, c, d
        # v is connected to u (new), a, b, c, d
        # So a, b must be neighbors of v (old)
        # c, d must be neighbors of u (old), a, and b
        
        # Iterate over pairs (a, b) in N(v)
        neighbors_v = adj[v]
        temp_v = neighbors_v
        
        while temp_v:
            lsb_v = temp_v & -temp_v
            a = lsb_v.bit_length() - 1
            temp_v ^= lsb_v
            
            # Now iterate b in N(v) with b != a
            temp_v2 = temp_v
            while temp_v2:
                lsb_v2 = temp_v2 & -temp_v2
                b = lsb_v2.bit_length() - 1
                temp_v2 ^= lsb_v2
                
                # Check if |N(u) & N(a) & N(b)| >= 2
                if (adj[u] & adj[a] & adj[b]).bit_count() >= 2:
                    return False
                    
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_all_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def get_all_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def improved_local_search(adj, max_steps=50000):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        edges_list = get_all_edges(current_adj)
        non_edges_list = get_all_non_edges(current_adj)
        
        T = 1.0
        cooling_rate = 0.9995
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if T < 0.01:
                break
                
            move_type = random.randint(0, 1) # 0: Add, 1: Swap
            
            if move_type == 0:
                # Try to add an edge
                if not non_edges_list:
                    continue
                u, v = random.choice(non_edges_list)
                if is_valid_addition(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    
                    non_edges_list.remove((u, v))
                    edges_list.append((u, v))
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
            else:
                # Try swap
                if not edges_list or not non_edges_list:
                    continue
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                
                if (x == u and y == v) or (x == v and y == u):
                    continue
                
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                if is_valid_addition(current_adj, x, y):
                    current_adj[x] |= (1 << y)
                    current_adj[y] |= (1 << x)
                    
                    edges_list.remove((u, v))
                    non_edges_list.append((u, v))
                    non_edges_list.remove((x, y))
                    edges_list.append((x, y))
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
            T *= cooling_rate
            
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i)
        adj, edges = improved_local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish
    if time.time() < deadline - 5:
        adj, edges = improved_local_search(best_adj, max_steps=100000)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
