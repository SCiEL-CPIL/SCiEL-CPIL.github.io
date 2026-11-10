import random
import itertools
import time

N = 16
NUM_EDGES = N * (N - 1) // 2
edges = []
edge_map = {}
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        edges.append((u, v))
        edge_map[(u, v)] = idx
        idx += 1

# Precompute for each edge, the list of 6-vertex subsets from the remaining 14
# that could form a Q3 with this edge.
edge_to_candidates = [[] for _ in range(NUM_EDGES)]
all_vertices = list(range(N))

for e_idx, (u, v) in enumerate(edges):
    others = [x for x in all_vertices if x != u and x != v]
    for combo in itertools.combinations(others, 6):
        v_set = [u, v] + list(combo)
        # Precompute mask for internal edges excluding (u,v)
        mask = 0
        internal_edges = []
        for i in range(len(v_set)):
            for j in range(i + 1, len(v_set)):
                a, b = v_set[i], v_set[j]
                if (a, b) == (u, v) or (b, a) == (u, v):
                    continue
                e_i = edge_map[(a, b)]
                mask |= (1 << e_i)
                internal_edges.append(e_i)
        # Store (mask, v_set_tuple, u_local_idx, v_local_idx)
        # u and v are at indices 0 and 1 in v_set
        edge_to_candidates[e_idx].append((mask, tuple(v_set)))

def is_q3_with_edge(e_idx, graph_bitset):
    """
    Returns True if adding edge e_idx to graph_bitset creates a Q3.
    Assumes e_idx is NOT in graph_bitset.
    """
    u, v = edges[e_idx]
    candidates = edge_to_candidates[e_idx]
    
    for mask, v_set in candidates:
        # 1. Count edges in graph_bitset & mask
        count = bin(graph_bitset & mask).count('1')
        if count != 11:
            continue
            
        # 2. Check Degrees
        # u and v must have degree 2 in the induced subgraph (excluding e)
        # Others must have degree 3
        
        # We need to check degrees.
        # We can do this by iterating through the 11 present edges.
        # To avoid iterating, we can precompute for each vertex in v_set
        # the mask of edges incident to it within v_set (excluding e).
        # But this is complex.
        
        # Let's just iterate. 11 edges is small.
        # Get the 11 present edge indices
        present_edges = []
        temp_mask = graph_bitset & mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            e_i = lsb.bit_length() - 1
            present_edges.append(e_i)
            temp_mask ^= lsb
            
        # Check degrees
        deg = {vert: 0 for vert in v_set}
        for e_i in present_edges:
            a, b = edges[e_i]
            deg[a] += 1
            deg[b] += 1
            
        # u is v_set[0], v is v_set[1]
        if deg[v_set[0]] != 2 or deg[v_set[1]] != 2:
            continue
            
        valid_deg = True
        for i in range(2, 8):
            if deg[v_set[i]] != 3:
                valid_deg = False
                break
        if not valid_deg:
            continue
            
        # 3. Check Bipartiteness
        # Build adjacency
        # Map global to local
        v_map = {v_set[i]: i for i in range(8)}
        adj = [[] for _ in range(8)]
        
        for e_i in present_edges:
            a, b = edges[e_i]
            la = v_map[a]
            lb = v_map[b]
            adj[la].append(lb)
            adj[lb].append(la)
            
        # Add e
        lu = 0
        lv = 1
        adj[lu].append(lv)
        adj[lv].append(lu)
        
        # BFS
        color = [-1] * 8
        is_bipartite = True
        for i in range(8):
            if color[i] == -1:
                color[i] = 0
                stack = [i]
                while stack:
                    node = stack.pop()
                    for nei in adj[node]:
                        if color[nei] == -1:
                            color[nei] = color[node] ^ 1
                            stack.append(nei)
                        elif color[nei] == color[node]:
                            is_bipartite = False
                            break
                    if not is_bipartite:
                        break
            if not is_bipartite:
                break
                
        if is_bipartite:
            return True
            
    return False

def construct():
    start_time = time.time()
    timeout = 115
    
    current_bitset = 0
    
    # Greedy Fill
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e_idx in edge_order:
        if not is_q3_with_edge(e_idx, current_bitset):
            current_bitset |= (1 << e_idx)
            
    best_bitset = current_bitset
    best_count = bin(current_bitset).count('1')
    
    no_improve = 0
    while time.time() - start_time < timeout:
        if random.random() < 0.8:
            # Try to add
            for _ in range(20):
                e_idx = random.randint(0, NUM_EDGES - 1)
                if not (current_bitset >> e_idx) & 1:
                    break
            else:
                continue
                
            if not is_q3_with_edge(e_idx, current_bitset):
                current_bitset |= (1 << e_idx)
                no_improve = 0
                if bin(current_bitset).count('1') > best_count:
                    best_count = bin(current_bitset).count('1')
                    best_bitset = current_bitset
            else:
                no_improve += 1
        else:
            # Remove
            for _ in range(20):
                e_idx = random.randint(0, NUM_EDGES - 1)
                if (current_bitset >> e_idx) & 1:
                    break
            else:
                continue
                
            current_bitset &= ~(1 << e_idx)
            no_improve += 1
            
        if no_improve > 100:
            for _ in range(3):
                e_idx = random.randint(0, NUM_EDGES - 1)
                if (current_bitset >> e_idx) & 1:
                    current_bitset &= ~(1 << e_idx)
            no_improve = 0

    result = []
    for i in range(NUM_EDGES):
        if (best_bitset >> i) & 1:
            result.append(list(edges[i]))
            
    return result
