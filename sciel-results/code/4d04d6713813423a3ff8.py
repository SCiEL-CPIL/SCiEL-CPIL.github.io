import time
import random
from itertools import combinations

# Precompute all 8-subsets for each edge
# Vertices 0..15
# Edge (u, v) with u < v
# Subsets must contain u and v.
# Remaining 6 from the other 14.

EDGE_SUBSETS = {}
ALL_VERTS = list(range(16))

def init_precomputation():
    for u in range(16):
        for v in range(u + 1, 16):
            other_verts = [x for x in ALL_VERTS if x != u and x != v]
            subs = []
            for combo in combinations(other_verts, 6):
                subs.append(tuple(sorted([u, v] + list(combo))))
            EDGE_SUBSETS[(u, v)] = subs

def is_q3(subset, adj):
    # subset is a tuple of 8 vertices
    # 1. Check degrees
    mask = 0
    for v in subset:
        mask |= (1 << v)
    
    for v in subset:
        deg = bin(adj[v] & mask).count('1')
        if deg != 3:
            return False
            
    # 2. Check bipartite
    # Bipartite graph with 8 vertices and 12 edges must be 3-regular.
    # A 3-regular bipartite graph on 8 vertices is a Q3 iff it is connected.
    # Actually, all 3-regular bipartite graphs on 8 vertices are Q3?
    # No, there are two non-isomorphic 3-regular bipartite graphs on 8 vertices:
    # 1. Q3 (the cube)
    # 2. The complete bipartite graph K_{4,4} minus a perfect matching? No, that's 3-regular.
    #    K_{4,4} has 16 edges. Remove 4 disjoint edges -> 12 edges.
    #    Is that a Q3?
    #    Q3 is bipartite. Parts are {000, 011, 101, 110} and {001, 010, 100, 111}.
    #    Edges in Q3:
    #    000-001, 000-010, 000-100
    #    011-001, 011-010, 011-110
    #    101-001, 101-100, 101-110
    #    111-011, 111-101, 111-110
    #    
    #    Let's check if K_{4,4} minus matching is Q3.
    #    In K_{4,4} minus matching, each vertex is connected to 3 vertices in the other part.
    #    It contains 4-cycles.
    #    Q3 contains 4-cycles.
    #    Are they isomorphic?
    #    Q3 is the unique 3-regular bipartite graph on 8 vertices that is "cubic" and "bipartite"?
    #    Actually, there is only one 3-regular bipartite graph on 8 vertices up to isomorphism?
    #    No, there are two: Q3 and the "Möbius ladder" M_8? No M_8 is not bipartite.
    #    Wait, K_{4,4} minus a perfect matching IS the Q3?
    #    Let's check.
    #    In Q3, take part A = {000, 011, 101, 110}.
    #    000 connects to 001, 010, 100.
    #    011 connects to 001, 010, 110.
    #    101 connects to 001, 100, 110.
    #    110 connects to 011, 101, 111.
    #    
    #    Let's check if 000 is connected to 111? No.
    #    In K_{4,4} minus matching, if we remove (000, 111), (011, 100), (101, 010), (110, 001)?
    #    This depends on the matching.
    #    
    #    Actually, it is a known fact that Q3 is the ONLY 3-regular bipartite graph on 8 vertices.
    #    Source: "The number of 3-regular bipartite graphs on 8 vertices is 1".
    #    So, if a graph on 8 vertices is 3-regular and bipartite, it is Q3.
    #    
    #    So we just need to check:
    #    1. 3-regular (degrees all 3).
    #    2. Bipartite.
    
    # Bipartite check
    visited = [False] * 8
    color = [-1] * 8
    # Map vertex index to local index 0..7
    vert_map = {v: i for i, v in enumerate(subset)}
    
    def bfs(start_local):
        color[start_local] = 0
        queue = [start_local]
        while queue:
            u_local = queue.pop(0)
            u_vert = subset[u_local]
            neighbors = adj[u_vert] & mask
            # Iterate neighbors in subset
            for v_local in range(8):
                v_vert = subset[v_local]
                if neighbors & (1 << v_vert):
                    if color[v_local] == -1:
                        color[v_local] = 1 - color[u_local]
                        queue.append(v_local)
                    elif color[v_local] == color[u_local]:
                        return False
        return True

    # Check connectivity and bipartiteness
    # Since it's 3-regular, if it's bipartite, it's likely connected unless it's two disjoint components.
    # But 3-regular components must have even size. Smallest is 4 (K_{2,2} minus edge? No, K_{2,2} is 2-regular).
    # 3-regular on 4 vertices? K_{2,2} is 2-regular. No 3-regular graph on 4 vertices exists (handshake: 4*3=12, even, but max deg 3 in K4 is 3. K4 is 3-regular but not bipartite).
    # So any 3-regular bipartite graph on 8 vertices is connected.
    
    if not bfs(0):
        return False
        
    # Ensure all visited (connectivity)
    for i in range(8):
        if not visited[i]: # Wait, I didn't set visited in bfs.
             pass
    # Let's just check if color[i] != -1 for all i.
    for i in range(8):
        if color[i] == -1:
            return False
            
    return True

def find_q3_containing(u, v, adj):
    key = (u, v) if u < v else (v, u)
    for subset in EDGE_SUBSETS[key]:
        if is_q3(subset, adj):
            return subset
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    init_precomputation()
    
    best_edges = 39
    best_adj = None
    
    # Seed graph: K_{3,13}
    def get_seed_adj():
        adj = [0] * 16
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        return adj

    # Random graph generator
    def get_random_adj(num_edges):
        adj = [0] * 16
        edges = []
        all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
        random.shuffle(all_pairs)
        for i in range(num_edges):
            u, v = all_pairs[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
        return adj, edges

    def remove_q3s(adj):
        # Remove edges until no Q3
        # Iterate all edges, check if it's in a Q3?
        # Simpler: Find ANY Q3, remove a random edge from it.
        changed = True
        while changed:
            changed = False
            # Find a Q3
            # Iterate all edges, check if adding it creates Q3? No, graph already has edges.
            # We need to find an existing Q3.
            # Iterate all 8-subsets? 12870.
            # Or iterate all edges (u,v) and check if (u,v) is in a Q3.
            # If we find one, remove a random edge from that Q3.
            found = False
            # Shuffle edges to randomize removal
            current_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    if adj[u] & (1 << v):
                        current_edges.append((u, v))
            
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                q3 = find_q3_containing(u, v, adj)
                if q3:
                    # Remove a random edge from q3
                    q3_edges = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            a = q3[i]
                            b = q3[j]
                            if adj[a] & (1 << b):
                                q3_edges.append((a, b))
                    
                    if q3_edges:
                        rem_u, rem_v = random.choice(q3_edges)
                        adj[rem_u] &= ~(1 << rem_v)
                        adj[rem_v] &= ~(1 << rem_u)
                        changed = True
                        found = True
                        break
            if not found:
                break
        return adj

    def count_edges(adj):
        count = 0
        for u in range(16):
            count += bin(adj[u]).count('1')
        return count // 2

    # Main Loop
    for seed in range(200):
        if time.time() > deadline:
            break
            
        # Strategy 1: Start from K_{3,13} and grow
        if seed % 2 == 0:
            adj = get_seed_adj()
        else:
            # Strategy 2: Start from random dense graph
            adj, _ = get_random_adj(50)
            adj = remove_q3s(adj)
        
        # Grow phase
        non_edges = [(i, j) for i in range(16) for j in range(i+1, 16) if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline:
                break
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            q3 = find_q3_containing(u, v, adj)
            if q3:
                # Remove a random edge from q3
                q3_edges = []
                for i in range(8):
                    for j in range(i+1, 8):
                        a = q3[i]
                        b = q3[j]
                        if adj[a] & (1 << b):
                            q3_edges.append((a, b))
                
                if q3_edges:
                    rem_u, rem_v = random.choice(q3_edges)
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                else:
                    # Should not happen
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            # Else keep edge
            
        cur_edges = count_edges(adj)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [x for x in adj]
            
    # Return edges list
    if best_adj is None:
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges

