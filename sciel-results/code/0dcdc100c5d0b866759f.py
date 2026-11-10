import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            mask_v ^= lsb
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            mask_u ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update cc
        old_mask_v = adj[v] ^ (1 << u)
        m = old_mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            m ^= lsb
            
        old_mask_u = adj[u] ^ (1 << v)
        m = old_mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update cc
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            m ^= lsb
            
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            m ^= lsb

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges():
        c = 0
        for u in range(n):
            c += popcount(adj[u])
        return c // 2

    best_edges = []
    best_count = 0
    random.seed(42)
    
    # Try multiple restarts
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Reset graph
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Greedy initialization
        # Randomize order of vertices or edges
        possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible)
        for u, v in possible:
            if can_add(u, v):
                add_edge(u, v)
                
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges()
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.99995
        steps = 0
        max_steps = 100000
        
        while time.time() < deadline - 5 and steps < max_steps:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                steps += 1
                continue
                
            if (adj[u] >> v) & 1:
                # Remove edge
                remove_edge(u, v)
                # Try to add a random edge
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 != v2 and not ((adj[u2] >> v2) & 1):
                    if can_add(u2, v2):
                        add_edge(u2, v2)
                        # Swap accepted (count unchanged)
                    else:
                        # Revert removal
                        add_edge(u, v)
                else:
                    # Revert removal
                    add_edge(u, v)
            else:
                # Try to add edge
                if can_add(u, v):
                    add_edge(u, v)
                    # Accepted
                else:
                    pass
                    
            T *= cooling
            if T < 0.01:
                T = 0.01
            steps += 1
            
        # Final greedy pass
        possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible)
        for u, v in possible:
            if can_add(u, v):
                add_edge(u, v)
                
        final_count = count_edges()
        if final_count > best_count:
            best_count = final_count
            best_edges = get_edges()
            
    return best_edges

if __name__ == "__main__":
    # For local testing
    edges = construct()
    print(f"Edges: {len(edges)}")
