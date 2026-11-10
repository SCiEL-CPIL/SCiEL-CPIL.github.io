import itertools
import random
import time

def generate_q3_edge_sets():
    edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                edges.add((i, j))
    
    unique_sets = set()
    for perm in itertools.permutations(range(8)):
        mapped_edges = set()
        for u, v in edges:
            mapped_edges.add((perm[u], perm[v]))
        unique_sets.add(frozenset(mapped_edges))
        
    return list(unique_sets)

Q3_EDGE_SETS = generate_q3_edge_sets()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Bitmask representation
    # adj[i] is an integer with bits for neighbors
    adj = [0] * 16
    for u in range(16):
        for v in range(u+1, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    def has_edge(u, v):
        return (adj[u] >> v) & 1

    def contains_q3(verts):
        # verts is a tuple of 8 ints
        # Check degrees
        degs = [0]*8
        for i in range(8):
            vi = verts[i]
            mask = adj[vi]
            # Count bits in mask that correspond to other verts
            # Create a mask for the 8 verts
            v_mask = 0
            for j in range(8):
                if i != j:
                    v_mask |= (1 << verts[j])
            degs[i] = bin(mask & v_mask).count('1')
            
        if any(d < 3 for d in degs):
            return None
            
        for q3_edges in Q3_EDGE_SETS:
            ok = True
            for u, v in q3_edges:
                if not has_edge(verts[u], verts[v]):
                    ok = False
                    break
            if ok:
                return q3_edges
        return None

    all_8sets = list(itertools.combinations(range(16), 8))
    
    while time.time() < deadline:
        found = False
        for verts in all_8sets:
            if time.time() > deadline:
                break
            q3_edges = contains_q3(verts)
            if q3_edges:
                # Remove random edge
                u, v = random.choice(list(q3_edges))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                found = True
                # Do not break, continue to find more Q3s in this pass
                # This is more efficient than restarting
        
        if not found:
            # No Q3 found. Try to add edges.
            added = False
            missing = []
            for u in range(16):
                for v in range(u+1, 16):
                    if not has_edge(u, v):
                        missing.append((u, v))
            
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    break
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if created Q3
                # Only check 8-sets containing u and v
                created = False
                others = [x for x in range(16) if x != u and x != v]
                for o in itertools.combinations(others, 6):
                    verts = tuple([u, v] + list(o))
                    if contains_q3(verts):
                        created = True
                        break
                if created:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    added = True
            
            if not added:
                break
    
    result = []
    for u in range(16):
        for v in range(u+1, 16):
            if has_edge(u, v):
                result.append([u, v])
    return result
