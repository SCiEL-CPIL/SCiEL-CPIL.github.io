import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 50.0

    # Partitions for Bipartite Graph K_7,8
    # A: 0..6 (7 vertices)
    # B: 7..14 (8 vertices)
    
    # Initialize with K_7,8
    adj = [0] * n
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    def get_degree(adj, i):
        return bin(adj[i]).count('1')

    def is_valid_adj(adj):
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        for u in range(n):
            au = adj[u]
            for v in range(n):
                if not (au & (1 << v)):
                    continue
                if u > v:
                    continue
                # Edge (u, v) exists
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj[v] & ~(1 << u)
                
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    
                    t_mask = av_minus_u & ~(1 << s)
                    while t_mask:
                        t_bit = t_mask & -t_mask
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        
                        if degrees[s] == degrees[t]:
                            return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    best_adj = adj[:]
    best_count = count_edges(best_adj)
    
    if not is_valid_adj(best_adj):
        # Fallback to disjoint triangles if K_7,8 somehow invalid (it shouldn't be)
        adj = [0] * n
        for i in range(0, n, 3):
            adj[i] |= (1 << (i+1)) | (1 << (i+2))
            adj[i+1] |= (1 << i) | (1 << (i+2))
            adj[i+2] |= (1 << i) | (1 << (i+1))
        best_adj = adj[:]
        best_count = count_edges(best_adj)

    # Local Search to try to improve
    random.seed(42)
    current_adj = best_adj[:]
    current_count = best_count
    
    max_iters = 50000
    iter_count = 0
    stall_count = 0
    max_stall = 2000
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        improved = False
        
        # Try to add a random non-edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and not (current_adj[u] & (1 << v)):
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if is_valid_adj(new_adj):
                current_adj = new_adj
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                improved = True
                stall_count = 0
                continue
        
        # Try swap
        # Remove random edge, add random non-edge
        u1 = random.randint(0, n - 1)
        v1 = random.randint(0, n - 1)
        if u1 != v1 and (current_adj[u1] & (1 << v1)):
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 != v2 and not (current_adj[u2] & (1 << v2)) and (u2, v2) != (u1, v1) and (u2, v2) != (v1, u1):
                new_adj = current_adj[:]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                if is_valid_adj(new_adj):
                    current_adj = new_adj
                    # Count unchanged
                    improved = True
                    stall_count = 0
                    continue

        if not improved:
            stall_count += 1
        else:
            stall_count = 0
            
        if stall_count > max_stall:
            current_adj = best_adj[:]
            current_count = best_count
            stall_count = 0

    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([i, j])
                
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    return final_edges
