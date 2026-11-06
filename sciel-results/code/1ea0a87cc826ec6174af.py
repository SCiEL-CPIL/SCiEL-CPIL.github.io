import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_result = {'n': 39, 'edges': []}
    best_n = 39

    # Try n=47, 46, 45... down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break

        min_deg_needed = n - 39
        if min_deg_needed <= 0:
            best_result = {'n': n, 'edges': []}
            best_n = n
            continue

        success = False
        # Try multiple random restarts
        # For n=46, min_deg=7. For n=47, min_deg=8.
        # A 7-regular C4-free graph on 46 vertices is possible.
        
        for attempt in range(500):
            if time.time() > deadline:
                break

            random.seed(attempt * 12345 + n * 678)
            
            # adj[i] is a bitmask of neighbors
            adj = [0] * n
            edges = []
            
            # List of all possible edges, shuffled
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            # We want to add edges to satisfy min degree.
            # Greedy: Add edge if safe.
            # To ensure min degree, we can prioritize edges involving low-degree vertices.
            # However, simple random shuffle + safety check often works for sparse graphs.
            
            # Let's use a priority queue or just multiple passes.
            # Given n is small, we can do a few passes of "try to add any safe edge".
            # But we need to ensure we don't get stuck with low degrees.
            
            # Strategy: 
            # 1. Randomly shuffle edges.
            # 2. Iterate through edges. If safe, add it.
            # 3. Check min degree.
            # 4. If min degree < needed, try to add more edges specifically for deficient nodes.
            
            # Actually, a better strategy for regular-ish graphs:
            # Try to add edges until no more safe edges can be added.
            # Then check min degree.
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # To avoid bias, we can shuffle the list of remaining edges each pass?
                # Or just iterate through all pairs.
                # Since n=47, n^2/2 = ~1100 pairs. 
                # Checking safety is fast.
                
                # We'll iterate through all potential edges.
                # To be more effective, let's pick random edges to try.
                # If we just iterate 0..n-1, we might fill up high-degree nodes first.
                
                # Let's use a list of missing edges and shuffle it.
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                random.shuffle(missing)
                
                for u, v in missing:
                    if time.time() > deadline:
                        break
                    
                    # Check if adding (u,v) is safe
                    # 1. u and v must not have >1 common neighbors
                    common_uv = adj[u] & adj[v]
                    if common_uv.bit_count() > 1:
                        continue
                    
                    # 2. For all z in N(u), if v in N(z), then N(u) and N(z) must have 0 common neighbors (before adding v)
                    # Actually, if v in N(z), then v becomes a common neighbor of u and z.
                    # So current common neighbors of u and z must be 0.
                    safe = True
                    # Iterate neighbors of u
                    zu = adj[u]
                    while zu:
                        z_bit = zu & -zu
                        z = z_bit.bit_length() - 1
                        if adj[z] & (1 << v):
                            # v is neighbor of z
                            # Check if u and z have any common neighbors
                            if (adj[u] & adj[z]).bit_count() > 0:
                                safe = False
                                break
                        zu -= z_bit
                    if not safe:
                        continue
                    
                    # 3. Similarly for neighbors of v
                    zv = adj[v]
                    while zv:
                        z_bit = zv & -zv
                        z = z_bit.bit_length() - 1
                        if adj[z] & (1 << u):
                            # u is neighbor of z
                            # Check if v and z have any common neighbors
                            if (adj[v] & adj[z]).bit_count() > 0:
                                safe = False
                                break
                        zv -= z_bit
                    if not safe:
                        continue
                    
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
                    changed = True

            # Check min degree
            min_deg = min(bin(adj[i]).count('1') for i in range(n))
            
            if min_deg >= min_deg_needed:
                # Verify C4-free (just in case)
                is_c4_free = True
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] & adj[j]).bit_count() > 1:
                            is_c4_free = False
                            break
                    if not is_c4_free:
                        break
                if is_c4_free:
                    success = True
                    best_result = {'n': n, 'edges': edges}
                    best_n = n
                    break

        if success:
            break

    return best_result
