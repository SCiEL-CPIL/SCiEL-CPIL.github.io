import time
import random
from itertools import combinations

def _q3_edges():
    """Return the 12 edges of Q3 as a list of (i,j) pairs for vertices 0-7."""
    edges = []
    for i in range(8):
        for j in range(i+1, 8):
            # Check if i and j differ in exactly one bit
            diff = i ^ j
            if diff and (diff & (diff - 1)) == 0:  # power of 2
                edges.append((i, j))
    return edges

Q3_EDGES = _q3_edges()

def _contains_q3(vertices, adj):
    """Check if the 8 vertices contain Q3 as a subgraph."""
    # Quick check: need at least 12 edges among these 8 vertices
    edge_count = 0
    for i in range(8):
        for j in range(i+1, 8):
            if vertices[i] in adj[vertices[j]]:
                edge_count += 1
    if edge_count < 12:
        return False
    
    # Try to find Q3 as a subgraph using backtracking
    # Assign Q3 vertex 0 to each of the 8 vertices
    for start_v in range(8):
        assign = {0: start_v}
        if _q3_backtrack(assign, 1, vertices, adj):
            return True
    
    return False

def _q3_backtrack(assign, next_v, vertices, adj):
    if next_v == 8:
        # Verify all Q3 edges
        for a, b in Q3_EDGES:
            if a in assign and b in assign:
                va = vertices[assign[a]]
                vb = vertices[assign[b]]
                if vb not in adj[va]:
                    return False
        return True
    
    for our_v in range(8):
        if our_v in assign.values():
            continue
        
        # Check compatibility with already-assigned vertices
        valid = True
        for prev in assign:
            if (prev, next_v) in Q3_EDGES or (next_v, prev) in Q3_EDGES:
                vp = vertices[assign[prev]]
                vo = vertices[our_v]
                if vo not in adj[vp]:
                    valid = False
                    break
            else:
                # Not adjacent in Q3, no constraint (subgraph, not induced)
                pass
        
        if valid:
            assign[next_v] = our_v
            if _q3_backtrack(assign, next_v + 1, vertices, adj):
                return True
            del assign[next_v]
    
    return False

def _check_add_edge(u, v, adj):
    """Check if adding edge (u,v) creates a Q3."""
    adj[u].add(v)
    adj[v].add(u)
    
    other = [x for x in range(16) if x != u and x != v]
    for subset in combinations(other, 6):
        verts = [u, v] + list(subset)
        if _contains_q3(verts, adj):
            adj[u].remove(v)
            adj[v].remove(u)
            return True
    
    adj[u].remove(v)
    adj[v].remove(u)
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_count = 39
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        adj = [set() for _ in range(16)]
        
        all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not _check_add_edge(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
        
        count = sum(len(s) for s in adj) // 2
        if count > best_count:
            best_count = count
            best_edges = sorted([[min(u,v), max(u,v)] for u in range(16) for v in adj[u] if u < v])
    
    return best_edges
