"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

N = 15
ALL_EDGES = [(u, v) for u in range(N) for v in range(u + 1, N)]
EDGE_SET = set(ALL_EDGES)
NUM_EDGES = len(ALL_EDGES)  # 105

def _edge_index(u, v):
    if u > v:
        u, v = v, u
    return u * N + v

def _check_valid(adj, deg):
    """Check if any 3-edge path has equal-degree endpoints."""
    # For each edge (u,v), for each neighbor s of u (s != v), for each neighbor t of v (t != u):
    # if s != t and deg[s] == deg[t], invalid
    for u in range(N):
        for v in adj[u]:
            if v > u:  # each edge once
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
    return True

def _compute_deg(adj):
    return [len(adj[i]) for i in range(N)]

def _try_add_edge(adj, deg, e):
    """Try adding edge e; return (success, new_adj, new_deg) or None."""
    u, v = e
    if v in adj[u]:
        return None
    # Check if adding this edge creates a violation
    # New degree: deg[u]+1, deg[v]+1
    # Need to check all 3-edge paths involving this new edge
    # Path s-u-v-t: s in adj[u] (s!=v), t in adj[v] (t!=u), s!=t, deg[s]==deg[t]
    # Path s-v-u-t: s in adj[v] (s!=u), t in adj[u] (t!=v), s!=t, deg[s]==deg[t]
    # Also paths where u or v is an internal node in other paths? No, only paths using the new edge.
    # Actually, adding edge (u,v) creates new 3-edge paths:
    # 1) s-u-v-t for s in N(u)\{v}, t in N(v)\{u}, s!=t
    # 2) s-v-u-t for s in N(v)\{u}, t in N(u)\{v}, s!=t
    # But these are the same set of paths, just direction. So check:
    for s in adj[u]:
        if s == v:
            continue
        for t in adj[v]:
            if t == u or t == s:
                continue
            if deg[s] == deg[t]:
                return None
    # Also need to check: does adding this edge make deg[u]+1 == deg[t] for some t,
    # creating a new equal-degree pair (u,t) that is at distance 3?
    # Similarly (v,t) for some t.
    # This is complex. Let's just do full check after adding.
    new_adj = [list(x) for x in adj]
    new_adj[u].append(v)
    new_adj[v].append(u)
    new_deg = list(deg)
    new_deg[u] += 1
    new_deg[v] += 1
    if _check_valid(new_adj, new_deg):
        return (True, new_adj, new_deg)
    return None

def _try_remove_edge(adj, deg, e):
    """Try removing edge e; return (success, new_adj, new_deg) or None."""
    u, v = e
    if v not in adj[u]:
        return None
    new_adj = [list(x) for x in adj]
    new_adj[u].remove(v)
    new_adj[v].remove(u)
    new_deg = list(deg)
    new_deg[u] -= 1
    new_deg[v] -= 1
    if _check_valid(new_adj, new_deg):
        return (True, new_adj, new_deg)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_edges = 0
    best_adj = None
    best_deg = None
    
    # Initialize with a random graph
    adj = [set() for _ in range(N)]
    deg = [0] * N
    edge_count = 0
    
    # Start with a moderately dense random graph
    for e in ALL_EDGES:
        if random.random() < 0.5:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            edge_count += 1
    
    # Convert to lists for speed
    adj = [list(s) for s in adj]
    
    if _check_valid(adj, deg):
        best_adj = [list(x) for x in adj]
        best_deg = list(deg)
        best_edges = edge_count
    
    # Local search
    random.seed(42)
    while time.time() < deadline:
        # Try to add random edges
        for _ in range(200):
            if time.time() > deadline:
                break
            e = random.choice(ALL_EDGES)
            if e[1] in adj[e[0]]:
                continue
            result = _try_add_edge(adj, deg, e)
            if result is not None:
                _, adj, deg = result
                edge_count += 1
                if edge_count > best_edges:
                    best_adj = [list(x) for x in adj]
                    best_deg = list(deg)
                    best_edges = edge_count
        
        # If stuck, try removing some edges to unstick
        if time.time() > deadline:
            break
        # Find a problematic edge and remove it
        # Just remove a random edge
        if edge_count > 0:
            e = random.choice(ALL_EDGES)
            if e[1] in adj[e[0]]:
                result = _try_remove_edge(adj, deg, e)
                if result is not None:
                    _, adj, deg = result
                    edge_count -= 1
        
        # Occasionally restart with a new random graph
        if random.random() < 0.05:
            adj = [set() for _ in range(N)]
            deg = [0] * N
            edge_count = 0
            for e in ALL_EDGES:
                if random.random() < 0.4:
                    u, v = e
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edge_count += 1
            adj = [list(s) for s in adj]
            if _check_valid(adj, deg):
                if edge_count > best_edges:
                    best_adj = [list(x) for x in adj]
                    best_deg = list(deg)
                    best_edges = edge_count
    
    # Return best result
    if best_adj is None:
        # Fallback: star graph
        return [[0, v] for v in range(1, N)]
    
    result = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
