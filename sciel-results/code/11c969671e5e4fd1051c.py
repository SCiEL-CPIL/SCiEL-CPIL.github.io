import random
import itertools
import time
import math

# Precomputation
N = 16
NUM_EDGES = N * (N - 1) // 2
edges = []
edge_map = {} # (u, v) -> index
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        edges.append((u, v))
        edge_map[(u, v)] = idx
        idx += 1

# Precompute all 8-vertex subsets and their internal edge indices
subsets = []
for combo in itertools.combinations(range(N), 8):
    combo_set = set(combo)
    internal_edges = []
    for u in combo:
        for v in combo:
            if u < v:
                internal_edges.append(edge_map[(u, v)])
    subsets.append(internal_edges)

def is_q3_subset(edge_indices, edges_bitset):
    # Check if the induced subgraph on these 8 vertices is Q3
    # 1. Count edges
    count = 0
    for e_idx in edge_indices:
        if (edges_bitset >> e_idx) & 1:
            count += 1
    if count != 12:
        return False
    
    # 2. Check 3-regular and bipartite
    # Build adjacency list for the 8 vertices
    # Map global vertex to local 0-7
    verts = [i for i in range(N) if (1 << i) & edges_bitset] # This is wrong, edges_bitset is for edges, not vertices.
    # We need the actual vertices of the subset.
    # Pass vertices as well? Or reconstruct.
    # Let's pass the vertices to the check function.
    pass

# Refactor: Precompute subset data including vertices
subset_data = []
for combo in itertools.combinations(range(N), 8):
    combo_set = set(combo)
    internal_edges = []
    for u in combo:
        for v in combo:
            if u < v:
                internal_edges.append(edge_map[(u, v)])
    subset_data.append((combo, internal_edges))

def contains_q3(edges_bitset):
    """Returns True if the graph contains a Q3. Also returns one such Q3's edge indices if found."""
    for verts, edge_indices in subset_data:
        # Count edges
        count = 0
        for e_idx in edge_indices:
            if (edges_bitset >> e_idx) & 1:
                count += 1
        if count != 12:
            continue
        
        # Check if 3-regular
        # Build degree count for the 8 vertices
        deg = [0] * 8
        # Map vertex to local index
        v_map = {v: i for i, v in enumerate(verts)}
        
        # We need to check which edges are present
        present_edges = []
        for e_idx in edge_indices:
            if (edges_bitset >> e_idx) & 1:
                present_edges.append(e_idx)
        
        if len(present_edges) != 12:
            continue
            
        for e_idx in present_edges:
            u, v = edges[e_idx]
            lu = v_map[u]
            lv = v_map[v]
            deg[lu] += 1
            deg[lv] += 1
            
        if any(d != 3 for d in deg):
            continue
            
        # Check Bipartite
        # BFS/DFS on the 8 vertices
        color = [-1] * 8
        adj = [[] for _ in range(8)]
        for e_idx in present_edges:
            u, v = edges[e_idx]
            lu = v_map[u]
            lv = v_map[v]
            adj[lu].append(lv)
            adj[lv].append(lu)
            
        # Try to color
        is_bipartite = True
        for i in range(8):
            if color[i] == -1:
                color[i] = 0
                queue = [i]
                while queue:
                    node = queue.pop(0)
                    for nei in adj[node]:
                        if color[nei] == -1:
                            color[nei] = color[node] ^ 1
                            queue.append(nei)
                        elif color[nei] == color[node]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
                    
        if is_bipartite:
            return True, present_edges
            
    return False, []

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    best_edges_bitset = 0
    best_count = 0
    
    # Initial population: Random dense graphs
    # Or start with all edges and remove
    # Let's use Simulated Annealing
    
    def evaluate(bitset):
        # Returns (is_valid, num_edges, violating_q3_edges)
        num_edges = bin(bitset).count('1')
        valid, q3_edges = contains_q3(bitset)
        return valid, num_edges, q3_edges
    
    # Start with a random graph with ~60% edges
    current_bitset = 0
    for i in range(NUM_EDGES):
        if random.random() < 0.6:
            current_bitset |= (1 << i)
            
    cur_valid, cur_count, cur_q3 = evaluate(current_bitset)
    if cur_valid:
        best_edges_bitset = current_bitset
        best_count = cur_count
        
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and time.time() - start_time < timeout:
        # Pick a random edge to flip
        e_idx = random.randint(0, NUM_EDGES - 1)
        new_bitset = current_bitset ^ (1 << e_idx)
        
        new_valid, new_count, new_q3 = evaluate(new_bitset)
        cur_valid, cur_count, cur_q3 = evaluate(current_bitset)
        
        # Score: Maximize edges. Penalty for invalidity.
        # If valid, score = count. If invalid, score = count - penalty * num_q3s
        # We don't count num_q3s in evaluate, just boolean.
        # Let's modify evaluate to return a penalty score if invalid.
        
        # Simple Hill Climbing with random restarts might be better/faster to implement correctly.
        # Let's stick to a robust local search:
        # 1. If new is valid and count > current count, accept.
        # 2. If new is invalid, reject.
        # 3. If current is invalid, we need to fix it.
        
        # Better approach: 
        # State is always valid. 
        # Start with a valid graph (e.g., K_3,13 or random sparse).
        # Try to add edges. If adding an edge creates a Q3, reject.
        # Try to remove edges and add different ones (2-opt).
        
        pass

    # Let's implement a specific Local Search:
    # 1. Start with K_3,13 (valid, 39 edges).
    # 2. Randomly try to add an edge. If it creates a Q3, find the Q3 and remove one of its edges (maybe the one we added, or another).
    # 3. Or: Try to add edge. If invalid, reject.
    # 4. To escape local optima, randomly remove an edge, then try to add different edges.
    
    # Let's restart with a cleaner Local Search implementation.
    
    def local_search(start_bitset):
        current = start_bitset
        valid, count, _ = evaluate(current)
        if not valid:
             # Fix it by removing edges from Q3s until valid
             while not valid:
                 _, q3_edges = contains_q3(current)
                 if not q3_edges: break
                 # Remove a random edge from the Q3
                 rem_idx = random.choice(q3_edges)
                 current &= ~(1 << rem_idx)
                 valid, count, _ = evaluate(current)
        
        best_local = current
        best_local_count = count
        
        no_improve = 0
        while time.time() - start_time < timeout:
            # Try to add a random edge
            if random.random() < 0.7:
                # Pick a non-existent edge
                zero_bits = [(1 << NUM_EDGES) - 1] ^ current
                if zero_bits == 0:
                    break
                # Pick random bit from zero_bits
                # Convert to list of indices? Expensive.
                # Just pick random index and check if 0
                for _ in range(10): # Try a few times to find a 0
                    e_idx = random.randint(0, NUM_EDGES - 1)
                    if not (current >> e_idx) & 1:
                        break
                else:
                    continue
                    
                new_bitset = current | (1 << e_idx)
                new_valid, new_count, new_q3 = evaluate(new_bitset)
                
                if new_valid:
                    current = new_bitset
                    count = new_count
                    no_improve = 0
                    if count > best_local_count:
                        best_local = current
                        best_local_count = count
                else:
                    no_improve += 1
            else:
                # Try to remove an edge and add another (swap)
                # Or just remove an edge to allow future additions
                one_bits = current
                if one_bits == 0:
                    break
                for _ in range(10):
                    e_idx = random.randint(0, NUM_EDGES - 1)
                    if (current >> e_idx) & 1:
                        break
                else:
                    continue
                    
                # Remove e_idx
                new_bitset = current & ~(1 << e_idx)
                # Try to add a random different edge
                for _ in range(10):
                    e_add = random.randint(0, NUM_EDGES - 1)
                    if e_add != e_idx and not (new_bitset >> e_add) & 1:
                        break
                else:
                    continue
                    
                new_bitset |= (1 << e_add)
                new_valid, new_count, _ = evaluate(new_bitset)
                
                if new_valid and new_count >= count:
                    current = new_bitset
                    count = new_count
                    no_improve = 0
                    if count > best_local_count:
                        best_local = current
                        best_local_count = count
                else:
                    no_improve += 1
                    
            if no_improve > 100:
                break
                
        return best_local, best_local_count

    # Run multiple restarts
    for i in range(20):
        # Start with K_3,13 or random
        if i == 0:
            # Seed graph
            bitset = 0
            for a in range(3):
                for b in range(3, 16):
                    bitset |= (1 << edge_map[(a, b)])
        else:
            # Random sparse
            bitset = 0
            for j in range(NUM_EDGES):
                if random.random() < 0.3:
                    bitset |= (1 << j)
                    
        best_candidate, best_cand_count = local_search(bitset)
        
        if best_cand_count > best_count:
            best_count = best_cand_count
            best_edges_bitset = best_candidate
            
        if time.time() - start_time > timeout:
            break
            
    # Convert best_bitset to list of edges
    result = []
    for i in range(NUM_EDGES):
        if (best_edges_bitset >> i) & 1:
            result.append(list(edges[i]))
            
    return result
