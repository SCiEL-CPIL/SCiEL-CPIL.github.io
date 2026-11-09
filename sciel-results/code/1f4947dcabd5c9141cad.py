import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    p = 29
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    def make_coloring(n, residues, prime):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % prime
                colors.append(0 if diff in residues else 1)
        return {'n': n, 'colors': colors}
    
    base = make_coloring(29, squares, 29)
    
    def count_violations_np(adj, n):
        """Count K2,2,2 subgraphs using numpy for speed. adj is n x n bool matrix."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                S = adj[u] & adj[v]
                verts = np.where(S)[0]
                if len(verts) < 4:
                    continue
                # For each pair (x,y) in S, check if they share >= 2 neighbors in S
                # Use matrix multiplication approach: for subgraph induced by S,
                # count edges between pairs
                sub = adj[np.ix_(verts, verts)]
                # Number of K2,2 in sub: for each pair (x,y), count common neighbors
                # This is sum over pairs of C(common_neighbors, 2)
                # Faster: for each x in S, for each y > x in S, count common
                for xi in range(len(verts)):
                    x = verts[xi]
                    for yi in range(xi + 1, len(verts)):
                        y = verts[yi]
                        common = adj[x] & adj[y] & S
                        cnt = np.count_nonzero(common)
                        if cnt >= 2:
                            count += 1
        return count
    
    def count_violations_fast(adj, n):
        """Faster violation count using bit operations with early exit."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                S = adj[u] & adj[v]
                # Quick check: need at least 4 vertices in S
                if bin(S).count('1') < 4:
                    continue
                verts = [i for i in range(n) if (S >> i) & 1]
                for xi in range(len(verts)):
                    x = verts[xi]
                    for yi in range(xi + 1, len(verts)):
                        y = verts[yi]
                        common = adj[x] & adj[y] & S
                        cnt = bin(common).count('1')
                        if cnt >= 2:
                            count += 1
        return count
    
    def build_adj(colors, n):
        """Build adjacency bitsets from colors list."""
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        return adj0, adj1
    
    def edge_index(i, j, n):
        """Get the index of edge (i,j) in lexicographic order, i < j."""
        # Sum of edges in rows 0..i-1
        # Row k has (n-1-k) edges
        # sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        row_sum = i * (n - 1) - i * (i - 1) // 2
        # Within row i, edge (i, j) is at position j - i - 1
        return row_sum + (j - i - 1)
    
    def local_search_30(colors, adj0, adj1, n, max_time):
        """Local search trying to eliminate all violations."""
        total_viol = count_violations_fast(adj0, n) + count_violations_fast(adj1, n)
        
        if total_viol == 0:
            return colors, 0
        
        # Precompute all edge indices
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                all_edges.append((i, j, edge_index(i, j, n)))
        
        improved = True
        stall_count = 0
        max_stall = 50
        
        while improved and time.time() < max_time:
            improved = False
            stall_count += 1
            if stall_count > max_stall:
                break
            
            # Shuffle edges for diversification
            random.shuffle(all_edges)
            
            for i, j, idx in all_edges:
                if time.time() > max_time:
                    return colors, total_viol
                
                old_c = colors[idx]
                new_c = 1 - old_c
                
                # Flip
                colors[idx] = new_c
                if old_c == 0:
                    adj0[i] &= ~(1 << j)
                    adj0[j] &= ~(1 << i)
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                else:
                    adj1[i] &= ~(1 << j)
                    adj1[j] &= ~(1 << i)
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                
                new_v0 = count_violations_fast(adj0, n)
                new_v1 = count_violations_fast(adj1, n)
                new_total = new_v0 + new_v1
                
                if new_total < total_viol:
                    total_viol = new_total
                    improved = True
                    stall_count = 0
                    if total_viol == 0:
                        return colors, 0
                else:
                    # Revert
                    colors[idx] = old_c
                    if old_c == 0:
                        adj0[i] |= (1 << j)
                        adj0[j] |= (1 << i)
                        adj1[i] &= ~(1 << j)
                        adj1[j] &= ~(1 << i)
                    else:
                        adj1[i] |= (1 << j)
                        adj1[j] |= (1 << i)
                        adj0[i] &= ~(1 << j)
                        adj0[j] &= ~(1 << i)
        
        return colors, total_viol
    
    # Try to extend to n=30
    n30 = 30
    best_total_viol = float('inf')
    best_colors_30 = None
    
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Initialize: start with base 29 coloring + random edges for vertex 29
        colors = list(base['colors'])
        for i in range(29):
            colors.append(random.choice([0, 1]))
        
        adj0, adj1 = build_adj(colors, n30)
        
        # Quick check
        tv0 = count_violations_fast(adj0, n30)
        tv1 = count_violations_fast(adj1, n30)
        total_v = tv0 + tv1
        
        if total_v == 0:
            return {'n': 30, 'colors': colors}
        
        # Local search
        search_deadline = min(deadline - 5, time.time() + 2.0)
        colors, total_v = local_search_30(colors, adj0, adj1, n30, search_deadline)
        
        if total_v == 0:
            return {'n': 30, 'colors': colors}
        
        if total_v < best_total_viol:
            best_total_viol = total_v
            best_colors_30 = colors[:]
    
    # If we found a very good coloring (few violations), try more aggressive search
    if best_total_viol is not None and best_total_viol <= 5 and time.time() < deadline - 20:
        # Try simulated annealing style: allow some worsening moves
        colors = best_colors_30[:]
        adj0, adj1 = build_adj(colors, n30)
        total_v = count_violations_fast(adj0, n30) + count_violations_fast(adj1, n30)
        
        all_edges = []
        for i in range(n30):
            for j in range(i + 1, n30):
                all_edges.append((i, j, edge_index(i, j, n30)))
        
        temp = 1.0
        cooling = 0.995
        
        while time.time() < deadline - 10 and total_v > 0:
            random.shuffle(all_edges)
            for i, j, idx in all_edges:
                if time.time() > deadline - 10:
                    break
                
                old_c = colors[idx]
                new_c = 1 - old_c
                
                colors[idx] = new_c
                if old_c == 0:
                    adj0[i] &= ~(1 << j)
                    adj0[j] &= ~(1 << i)
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                else:
                    adj1[i] &= ~(1 << j)
                    adj1[j] &= ~(1 << i)
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                
                new_total = count_violations_fast(adj0, n30) + count_violations_fast(adj1, n30)
                delta = new_total - total_v
                
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    total_v = new_total
                    if total_v == 0:
                        return {'n': 30, 'colors': colors}
                else:
                    colors[idx] = old_c
                    if old_c == 0:
                        adj0[i] |= (1 << j)
                        adj0[j] |= (1 << i)
                        adj1[i] &= ~(1 << j)
                        adj1[j] &= ~(1 << i)
                    else:
                        adj1[i] |= (1 << j)
                        adj1[j] |= (1 << i)
                        adj0[i] &= ~(1 << j)
                        adj0[j] &= ~(1 << i)
            
            temp *= cooling
        
        if total_v == 0:
            return {'n': 30, 'colors': colors}
    
    return base
