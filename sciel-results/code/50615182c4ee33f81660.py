import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        min_deg = n - 39
        if min_deg < 0:
            min_deg = 0
        
        # For n=39, min_deg=0, empty graph works
        if min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple restarts for this n
        max_restarts = 500
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Bitmask adjacency
            adj = [0] * n
            # reach2[u] = union of N(w) for all w in N(u)
            # If x in reach2[u] and x in N(v), then C4 exists (u-w-x-v)
            reach2 = [0] * n
            degrees = [0] * n
            edges = []
            
            def get_neighbors(mask):
                neighbors = []
                m = mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    neighbors.append(w)
                    m ^= lsb
                return neighbors

            # Greedy construction with proper reach2 maintenance
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                
                # Find vertices below target degree
                low_verts = [i for i in range(n) if degrees[i] < min_deg]
                
                if not low_verts:
                    break
                
                # Pick the vertex with lowest degree (deterministic for better convergence)
                u = min(low_verts, key=lambda x: degrees[x])
                
                # Candidates: non-neighbors of u
                candidates = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                
                if not candidates:
                    break
                
                # Score candidates: prefer those with fewer conflicts (smaller |reach2[u] & adj[v]|)
                # and prefer higher-degree vertices to help raise min degree
                scored = []
                for v in candidates:
                    conflict = bin(reach2[u] & adj[v]).count('1')
                    # We want low conflict and high degree
                    score = -conflict * 10 + degrees[v]
                    scored.append((score, v))
                
                # Sort by score descending, then randomize ties
                random.shuffle(scored)
                scored.sort(key=lambda x: -x[0])
                
                # Try top candidates
                for score, v in scored[:min(20, len(scored))]:
                    if (reach2[u] & adj[v]) == 0:
                        # Save old neighbor sets BEFORE updating adj
                        old_nu_mask = adj[u]
                        old_nv_mask = adj[v]
                        
                        # Update adj
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        edges.append([u, v])
                        added_any = True
                        
                        # Properly update reach2:
                        # reach2[u] should be union of N(w) for w in N(u)
                        # After adding v to N(u): reach2[u] gains N(v)
                        # N(v) after update = old_nv_mask | (1<<u)
                        new_nv = old_nv_mask | (1 << u)
                        reach2[u] |= new_nv
                        
                        # reach2[v] gains N(u)
                        new_nu = old_nu_mask | (1 << v)
                        reach2[v] |= new_nu
                        
                        # For each w in OLD N(u) (excluding v which wasn't there):
                        # reach2[w] should gain N(v) = new_nv
                        m = old_nu_mask
                        while m:
                            lsb = m & -m
                            w = lsb.bit_length() - 1
                            reach2[w] |= new_nv
                            m ^= lsb
                            
                        # For each x in OLD N(v) (excluding u which wasn't there):
                        # reach2[x] should gain N(u) = new_nu
                        m = old_nv_mask
                        while m:
                            lsb = m & -m
                            x = lsb.bit_length() - 1
                            reach2[x] |= new_nu
                            m ^= lsb
                            
                        break
            
            # Check if we met the requirement
            if min(degrees) >= min_deg:
                return {'n': n, 'edges': edges}
        
        # If we exhausted restarts for this n, try smaller n
    
    # Fallback: return the smallest valid graph
    return {'n': 39, 'edges': []}
