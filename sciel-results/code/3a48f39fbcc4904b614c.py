import random
import time
import math

def construct():
    # Target n values
    target_ns = [26, 25, 24]
    
    start_time = time.time()
    time_limit = 115.0
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Partition vertices into 5 groups
        # Try balanced partition
        base_size = n // 5
        remainder = n % 5
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(5)]
        
        # Create groups
        groups = []
        current = 0
        for size in group_sizes:
            groups.append(list(range(current, current + size)))
            current += size
            
        # Precompute a valid C4-free coloring for each group size
        # We need a function to generate a C4-free coloring for K_k
        # We will use a greedy random approach with backtracking or just random retry
        def get_base_coloring(k):
            # Returns a dict {(i,j): color} for i<j in 0..k-1
            # Such that no monochromatic C4 exists
            for _ in range(1000):
                colors = {}
                valid = True
                # Generate random colors
                for i in range(k):
                    for j in range(i+1, k):
                        colors[(i,j)] = random.randint(0, 4)
                        
                # Check for C4s
                # For each color c, check if any C4 exists
                adj = [set() for _ in range(5)]
                for (u,v), c in colors.items():
                    adj[c].add((u,v))
                    adj[c].add((v,u))
                
                # Check C4s in each color class
                # A C4 is u-v-w-x-u.
                # For each pair (u,w), check common neighbors in color c.
                # If two common neighbors v,x exist, and (v,x) is an edge in color c? No.
                # C4 in color c means the subgraph induced by color c edges contains a C4.
                # We can check by iterating all 4-tuples? k is small (5 or 6).
                # 4-tuples: 6 choose 4 = 15. Very fast.
                found_c4 = False
                for c in range(5):
                    edges_c = set()
                    for (u,v) in adj[c]:
                        if u < v:
                            edges_c.add((u,v))
                    
                    # Check all subsets of 4 vertices
                    for a in range(k):
                        for b in range(a+1, k):
                            for d in range(b+1, k):
                                for e in range(d+1, k):
                                    # Check if (a,b), (b,d), (d,e), (e,a) are in edges_c
                                    # Or any permutation forming a cycle
                                    # Cycles on 4 nodes: (a,b,d,e), (a,b,e,d)
                                    # Edges: (a,b), (b,d), (d,e), (e,a)
                                    if ((a,b) in edges_c or (b,a) in edges_c) and \
                                       ((b,d) in edges_c or (d,b) in edges_c) and \
                                       ((d,e) in edges_c or (e,d) in edges_c) and \
                                       ((e,a) in edges_c or (a,e) in edges_c):
                                        found_c4 = True
                                        break
                                    # Other cycle: (a,b,e,d)
                                    if ((a,b) in edges_c or (b,a) in edges_c) and \
                                       ((b,e) in edges_c or (e,b) in edges_c) and \
                                       ((e,d) in edges_c or (d,e) in edges_c) and \
                                       ((d,a) in edges_c or (a,d) in edges_c):
                                        found_c4 = True
                                        break
                                if found_c4: break
                            if found_c4: break
                        if found_c4: break
                    if found_c4: break
                    if found_c4: break
                
                if not found_c4:
                    return colors
            # Fallback: return a random one (should not happen for k=5,6)
            return { (i,j): random.randint(0,4) for i in range(k) for j in range(i+1, k) }

        # Assign base colorings to groups
        # Map global (i,j) to local (u,v) in group
        # We will store the full colors array
        colors = [0] * (n * (n - 1) // 2)
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)

        # Initialize intra-group edges
        for g_idx, group in enumerate(groups):
            k = len(group)
            base_col = get_base_coloring(k)
            # Map local indices to global
            for u in range(k):
                for v in range(u + 1, k):
                    gi = group[u]
                    gj = group[v]
                    idx = get_idx(gi, gj)
                    colors[idx] = base_col[(u, v)]
                    
        # Initialize inter-group edges randomly
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i and j are in the same group
                # Simple check: find group of i and j
                # Optimize: precompute group_id for each vertex
                pass
        
        group_id = [0] * n
        for g_idx, group in enumerate(groups):
            for v in group:
                group_id[v] = g_idx
                
        for i in range(n):
            for j in range(i + 1, n):
                if group_id[i] != group_id[j]:
                    idx = get_idx(i, j)
                    colors[idx] = random.randint(0, 4)
                    
        # Build adjacency for delta calculation
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
                
        # SA on inter-group edges only
        # List of inter-group edge indices
        inter_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if group_id[i] != group_id[j]:
                    inter_edges.append(get_idx(i, j))
                    
        # Initial energy
        def count_c4s():
            total = 0
            for c in range(5):
                # Count C4s in color c
                # Using the 4-tuple check is too slow for n=26 in SA loop.
                # Use edge-based delta.
                pass
            # We need a fast initial count.
            # For n=26, 15*4=60 4-tuples per color? No, 26 choose 4 is ~14k.
            # 14k * 5 colors = 70k checks. Fast enough for init.
            total = 0
            for c in range(5):
                edges_c = set()
                for u in range(n):
                    for v in adj[c][u]:
                        if u < v:
                            edges_c.add((u,v))
                for a in range(n):
                    for b in range(a+1, n):
                        for d in range(b+1, n):
                            for e in range(d+1, n):
                                if ((a,b) in edges_c) and ((b,d) in edges_c) and ((d,e) in edges_c) and ((e,a) in edges_c):
                                    total += 1
                                if ((a,b) in edges_c) and ((b,e) in edges_c) and ((e,d) in edges_c) and ((d,a) in edges_c):
                                    total += 1
            return total

        current_energy = count_c4s()
        best_energy = current_energy
        best_colors = colors.copy()
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        steps = 0
        max_steps = 200000 # More steps allowed since we only flip inter edges
        
        def edge_c4_contribution(c, i, j):
            cnt = 0
            for x in adj[c][j]:
                if x == i: continue
                for y in adj[c][i]:
                    if y == j: continue
                    if x == y: continue
                    if y in adj[c][x]:
                        cnt += 1
            return cnt

        while T > min_T and steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            idx_ij = random.choice(inter_edges)
            # Find i, j from idx_ij
            # Reverse of get_idx
            # i*n - i(i+1)/2 + (j-i-1) = idx
            # This is hard to reverse. Store (i,j) in parallel list.
            pass
            
        # Re-implement with stored (i,j)
        inter_edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if group_id[i] != group_id[j]:
                    inter_edge_list.append((i, j, get_idx(i, j)))
                    
        current_energy = count_c4s()
        best_energy = current_energy
        best_colors = colors.copy()
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        steps = 0
        max_steps = 50000
        
        while T > min_T and steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            i, j, idx_ij = random.choice(inter_edge_list)
            c_old = colors[idx_ij]
            c_new = random.randint(0, 4)
            if c_old == c_new: continue
            
            delta_old = -edge_c4_contribution(c_old, i, j)
            delta_new = edge_c4_contribution(c_new, i, j)
            delta_E = delta_old + delta_new
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                colors[idx_ij] = c_new
                adj[c_old][i].remove(j)
                adj[c_old][j].remove(i)
                adj[c_new][i].add(j)
                adj[c_new][j].add(i)
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return {'n': n, 'colors': best_colors}
                        
            T *= cooling
            steps += 1
            
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
            
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
