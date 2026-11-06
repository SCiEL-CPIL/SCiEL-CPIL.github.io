import time
import random

def _creates_c4(adj_int, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj_int: list of integers representing adjacency.
    u, v: vertices to connect. Assumes not already connected.
    """
    nu = adj_int[u]
    nv = adj_int[v]
    
    # Check common neighbors: if |N(u) & N(v)| >= 2, adding (u,v) creates C4
    common = nu & nv
    if common & (common - 1):  # More than 1 bit set
        return True
        
    # Check edges between N(u) and N(v)
    # For each x in N(u), check if N(x) intersects N(v) excluding x itself
    # We iterate over the smaller set
    deg_u = bin(nu).count('1')
    deg_v = bin(nv).count('1')
    
    if deg_u <= deg_v:
        temp = nu
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            # Check if N(x) intersects N(v), excluding x
            if adj_int[x] & (nv & ~(1 << x)):
                return True
            temp ^= lsb
    else:
        temp = nv
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            if adj_int[y] & (nu & ~(1 << y)):
                return True
            temp ^= lsb
            
    return False

def _prune_c4s(adj_int, degrees, n, time_limit, start_time):
    """
    Iteratively find and remove edges that create C4s.
    A C4 exists iff there's a pair (u,v) with |N(u) ∩ N(v)| >= 2.
    """
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            return False
        changed = False
        for u in range(n):
            if time.time() - start_time > time_limit:
                return False
            for v in range(u+1, n):
                common = adj_int[u] & adj_int[v]
                if common & (common - 1):  # >= 2 bits
                    # Found C4. Remove an edge.
                    # Pick a common neighbor w
                    lsb = common & -common
                    w = lsb.bit_length() - 1
                    common_rest = common ^ lsb
                    if common_rest:
                        lsb2 = common_rest & -common_rest
                        x = lsb2.bit_length() - 1
                        # Vertices involved: u, v, w, x
                        # Edges in C4: (u,w), (w,v), (v,x), (x,u)
                        candidates = [
                            (u, w, degrees[u]),
                            (w, v, degrees[w]),
                            (v, x, degrees[v]),
                            (x, u, degrees[x])
                        ]
                        candidates.sort(key=lambda t: t[2], reverse=True)
                        a, b, _ = candidates[0]
                    else:
                        # Only 1 common neighbor found but we said >=2?
                        # This shouldn't happen. Fallback: remove (u,w)
                        a, b = u, w
                        lsb2 = 0
                        x = w  # dummy
                    
                    # Remove edge (a, b)
                    if (adj_int[a] >> b) & 1:
                        adj_int[a] &= ~(1 << b)
                        adj_int[b] &= ~(1 << a)
                        degrees[a] -= 1
                        degrees[b] -= 1
                        changed = True
                    break
            if changed:
                break
    return True

def _add_edges_to_boost(adj_int, degrees, n, min_deg_req, time_limit, start_time):
    """
    Try to add edges to boost minimum degree while maintaining C4-free property.
    """
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            return True
        
        min_idx = min(range(n), key=lambda i: degrees[i])
        if degrees[min_idx] >= min_deg_req:
            continue
            
        # Try to add an edge from min_idx to some other vertex
        candidates = [v for v in range(n) if v != min_idx and not ((adj_int[min_idx] >> v) & 1)]
        random.shuffle(candidates)
        
        for v in candidates:
            if _creates_c4(adj_int, n, min_idx, v):
                continue
            adj_int[min_idx] |= (1 << v)
            adj_int[v] |= (1 << min_idx)
            degrees[min_idx] += 1
            degrees[v] += 1
            break
    return min(degrees) >= min_deg_req

def _construct_random_init(n, min_deg_req, time_limit, start_time):
    """
    Construct C4-free graph via random dense initialization + pruning + edge boosting.
    """
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            return None
            
        # Random dense graph
        p = random.uniform(0.55, 0.80)
        adj_int = [0] * n
        degrees = [0] * n
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj_int[u] |= (1 << v)
                    adj_int[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Prune C4s
        if not _prune_c4s(adj_int, degrees, n, time_limit, start_time):
            continue
            
        # Check min degree
        if min(degrees) >= min_deg_req:
            edges = []
            for u in range(n):
                temp = adj_int[u]
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if u < v:
                        edges.append([u, v])
                    temp ^= lsb
            return {'n': n, 'edges': edges}
            
        # If close, try to boost
        if min(degrees) >= min_deg_req - 3:
            if _add_edges_to_boost(adj_int, degrees, n, min_deg_req, time_limit, start_time):
                edges = []
                for u in range(n):
                    temp = adj_int[u]
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        temp ^= lsb
                return {'n': n, 'edges': edges}
                
    return None

def _construct_greedy(n, min_deg_req, time_limit, start_time):
    """
    Construct C4-free graph via greedy edge addition from empty graph.
    """
    max_attempts = 30
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            return None
            
        adj_int = [0] * n
        degrees = [0] * n
        
        # Greedy fill
        for step in range(3000):
            if time.time() - start_time > time_limit:
                return None
                
            if min(degrees) >= min_deg_req:
                edges = []
                for u in range(n):
                    temp = adj_int[u]
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        temp ^= lsb
                return {'n': n, 'edges': edges}
                
            # Find min degree vertex
            min_deg_vertex = min(range(n), key=lambda i: degrees[i])
            
            # Find valid neighbor
            found = False
            candidates = [v for v in range(n) if v != min_deg_vertex and not ((adj_int[min_deg_vertex] >> v) & 1)]
            random.shuffle(candidates)
            
            for v in candidates:
                if not _creates_c4(adj_int, n, min_deg_vertex, v):
                    adj_int[min_deg_vertex] |= (1 << v)
                    adj_int[v] |= (1 << min_deg_vertex)
                    degrees[min_deg_vertex] += 1
                    degrees[v] += 1
                    found = True
                    break
                    
            if not found:
                # Stuck. Remove a random edge to unblock
                if not any(degrees):
                    break
                edge_found = False
                for _ in range(50):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u < v and ((adj_int[u] >> v) & 1):
                        adj_int[u] &= ~(1 << v)
                        adj_int[v] &= ~(1 << u)
                        degrees[u] -= 1
                        degrees[v] -= 1
                        edge_found = True
                        break
                if not edge_found:
                    break

    return None

def _construct_c4_free(n, min_deg_req, time_limit):
    """
    Try both random-init+prune and greedy approaches.
    """
    start_time = time.time()
    remaining = time_limit - (time.time() - start_time)
    if remaining < 0.5:
        return None
    
    # Split time between two strategies
    t_rand = remaining * 0.55
    t_greedy = remaining * 0.45
    
    # Try random init + prune first (usually more effective)
    result = _construct_random_init(n, min_deg_req, t_rand, start_time)
    if result:
        return result
    
    # Try greedy
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1:
        result = _construct_greedy(n, min_deg_req, remaining, start_time)
        if result:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n=47, 46, 45, ...
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
        remaining = deadline - (time.time() - start_time)
        if remaining < 2:
            break
            
        min_deg_req = max(1, n - 39)
        
        # Allocate time: more for higher n
        if n >= 46:
            t_limit = min(45.0, remaining - 1)
        elif n >= 44:
            t_limit = min(25.0, remaining - 1)
        else:
            t_limit = min(15.0, remaining - 1)
        
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result:
            return result
            
    return {'n': 39, 'edges': []}
