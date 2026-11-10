import random
import itertools
import time
import math
from collections import deque

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

# Precompute all 8-vertex subsets with their internal edge indices
subset_data = []
for combo in itertools.combinations(range(N), 8):
    internal_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            u, v = combo[i], combo[j]
            if u > v:
                u, v = v, u
            internal_edges.append(edge_map[(u, v)])
    subset_data.append((combo, internal_edges))

def contains_q3(edges_bitset):
    """Returns True if the graph contains a Q3 subgraph."""
    for verts, edge_indices in subset_data:
        # Count edges in this subset
        count = 0
        for e_idx in edge_indices:
            if (edges_bitset >> e_idx) & 1:
                count += 1
        if count != 12:
            continue
        
        # Build adjacency for the 8 vertices
        v_map = {v: i for i, v in enumerate(verts)}
        deg = [0] * 8
        adj = [[] for _ in range(8)]
        
        for e_idx in edge_indices:
            if (edges_bitset >> e_idx) & 1:
                u, v = edges[e_idx]
                lu = v_map[u]
                lv = v_map[v]
                deg[lu] += 1
                deg[lv] += 1
                adj[lu].append(lv)
                adj[lv].append(lu)
        
        # Check 3-regular
        if any(d != 3 for d in deg):
            continue
        
        # Check bipartite (Q3 is bipartite)
        color = [-1] * 8
        is_bipartite = True
        for i in range(8):
            if color[i] == -1:
                color[i] = 0
                queue = deque([i])
                while queue and is_bipartite:
                    node = queue.popleft()
                    for nei in adj[node]:
                        if color[nei] == -1:
                            color[nei] = color[node] ^ 1
                            queue.append(nei)
                        elif color[nei] == color[node]:
                            is_bipartite = False
                            break
        
        if is_bipartite:
            return True
            
    return False

def construct():
    start_time = time.time()
    timeout = 115
    
    best_bitset = 0
    best_count = 0
    
    def evaluate_add(current_bitset, e_idx):
        """Check if adding edge e_idx creates a Q3. Returns True if safe."""
        new_bitset = current_bitset | (1 << e_idx)
        return not contains_q3(new_bitset)
    
    def local_search(start_bitset, time_limit):
        current = start_bitset
        count = bin(current).count('1')
        
        # Ensure valid
        while contains_q3(current):
            # Remove a random edge
            e_idx = random.randint(0, NUM_EDGES - 1)
            if (current >> e_idx) & 1:
                current &= ~(1 << e_idx)
                count -= 1
        
        best_local = current
        best_local_count = count
        
        no_improve = 0
        while time.time() - start_time < time_limit:
            # Try to add a random missing edge
            for _ in range(20):
                e_idx = random.randint(0, NUM_EDGES - 1)
                if not (current >> e_idx) & 1:
                    if evaluate_add(current, e_idx):
                        current |= (1 << e_idx)
                        count += 1
                        no_improve = 0
                        if count > best_local_count:
                            best_local = current
                            best_local_count = count
                    else:
                        no_improve += 1
                    break
            
            # If stuck, try a swap: remove one edge, add another
            if no_improve > 50:
                # Remove a random existing edge
                for _ in range(20):
                    e_remove = random.randint(0, NUM_EDGES - 1)
                    if (current >> e_remove) & 1:
                        current &= ~(1 << e_remove)
                        count -= 1
                        break
                
                # Try to add a different edge
                for _ in range(20):
                    e_add = random.randint(0, NUM_EDGES - 1)
                    if e_add != e_remove and not (current >> e_add) & 1:
                        if evaluate_add(current, e_add):
                            current |= (1 << e_add)
                            count += 1
                            no_improve = 0
                            if count > best_local_count:
                                best_local = current
                                best_local_count = count
                        break
                
                no_improve = 0
            
            if no_improve > 200:
                break
        
        return best_local, best_local_count
    
    # Multiple restarts with different starting points
    for restart in range(30):
        if time.time() - start_time > timeout:
            break
        
        if restart == 0:
            # Start with K_3,13
            bitset = 0
            for a in range(3):
                for b in range(3, 16):
                    bitset |= (1 << edge_map[(a, b)])
        else:
            # Random sparse graph
            bitset = 0
            for i in range(NUM_EDGES):
                if random.random() < 0.25:
                    bitset |= (1 << i)
        
        remaining = timeout - (time.time() - start_time)
        if remaining < 5:
            break
        
        local_time = min(remaining * 0.3, 3.0)
        best_candidate, best_cand_count = local_search(bitset, start_time + local_time)
        
        if best_cand_count > best_count:
            best_count = best_cand_count
            best_bitset = best_candidate
    
    # Final greedy pass: try to add any edge that doesn't create Q3
    for i in range(NUM_EDGES):
        if not (best_bitset >> i) & 1:
            if evaluate_add(best_bitset, i):
                best_bitset |= (1 << i)
                best_count += 1
    
    result = []
    for i in range(NUM_EDGES):
        if (best_bitset >> i) & 1:
            result.append(list(edges[i]))
    
    return result
