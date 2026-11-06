import random
import time

def construct():
    start_time = time.time()
    
    for n in range(27, 10, -1):
        if time.time() - start_time > 110:
            break
        # Try multiple random initializations
        for attempt in range(1000):
            if time.time() - start_time > 110:
                break
            # Random initial coloring
            colors = [random.randint(0, 4) for _ in range(n*(n-1)//2)]
            
            # Convert to adjacency
            adj = [set() for _ in range(5)]
            for c in range(5):
                adj[c] = [set() for _ in range(n)]
            
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx]
                    adj[c][i].add(j)
                    adj[c][j].add(i)
                    idx += 1
            
            # Local search: find C4s and flip edges
            improved = True
            while improved and time.time() - start_time < 110:
                improved = False
                # Find a C4
                c4_found = None
                for c in range(5):
                    for u in range(n):
                        for v in range(u+1, n):
                            common = adj[c][u] & adj[c][v]
                            if len(common) >= 2:
                                # Find a C4: u, v share at least 2 common neighbors
                                # Pick two common neighbors w, x
                                common_list = list(common)
                                w, x = common_list[0], common_list[1]
                                c4_found = (c, u, v, w, x)
                                break
                            if c4_found:
                                break
                        if c4_found:
                            break
                    if c4_found:
                        break
                
                if c4_found is None:
                    # No C4s, we have a valid coloring!
                    return {'n': n, 'colors': colors}
                
                c, u, v, w, x = c4_found
                # Flip one of the edges in the C4
                # Edges: (u,w), (w,v), (v,x), (x,u) in color c
                # Pick a random edge to flip
                edges_in_c4 = [(u,w), (w,v), (v,x), (x,u)]
                edge = random.choice(edges_in_c4)
                a, b = min(edge), max(edge)
                idx2 = a*n - a*(a+1)//2 + (b - a - 1)
                # Actually, let me compute index properly
                # Edge (i,j) with i<j has index: sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
                # = i*(n-1) - i*(i-1)//2 + (j-i-1)
                # Let me just recompute
                idx2 = 0
                for k in range(a):
                    idx2 += n - 1 - k
                idx2 += b - a - 1
                
                old_c = colors[idx2]
                new_c = random.choice([c2 for c2 in range(5) if c2 != old_c])
                # Remove old edge
                adj[old_c][a].discard(b)
                adj[old_c][b].discard(a)
                # Add new edge
                colors[idx2] = new_c
                adj[new_c][a].add(b)
                adj[new_c][b].add(a)
                improved = True
        
        # If we couldn't find a valid coloring for this n, try smaller n
        continue
    
    # Fallback: return a small valid coloring
    return {'n': 1, 'colors': []}
