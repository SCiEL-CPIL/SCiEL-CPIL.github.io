import itertools
import time
import random

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

# Precompute subset data
subset_data = []
for combo in itertools.combinations(range(N), 8):
    verts = combo
    internal_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            u = verts[i]
            v = verts[j]
            internal_edges.append(edge_map[(u, v)])
    subset_data.append((verts, internal_edges))

def contains_q3(edge_bitset):
    for verts, internal_edges in subset_data:
        count = 0
        for e_idx in internal_edges:
            if (edge_bitset >> e_idx) & 1:
                count += 1
        if count != 12:
            continue
        
        # Quick degree check
        v_map = {v: i for i, v in enumerate(verts)}
        deg = [0] * 8
        present = []
        for e_idx in internal_edges:
            if (edge_bitset >> e_idx) & 1:
                u, v = edges[e_idx]
                lu = v_map[u]
                lv = v_map[v]
                deg[lu] += 1
                deg[lv] += 1
                present.append(e_idx)
        
        if any(d != 3 for d in deg):
            continue
        
        # Bipartite check
        adj = [[] for _ in range(8)]
        for e_idx in present:
            u, v = edges[e_idx]
            lu = v_map[u]
            lv = v_map[v]
            adj[lu].append(lv)
            adj[lv].append(lu)
        
        color = [-1] * 8
        bip = True
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
                            bip = False
                            break
                    if not bip:
                        break
            if not bip:
                break
        
        if bip:
            return True, present
    
    return False, []

def construct():
    start_time = time.time()
    timeout = 115
    
    # Start with K_{3,13}
    best_bitset = 0
    for a in range(3):
        for b in range(3, 16):
            best_bitset |= (1 << edge_map[(a, b)])
    best_count = bin(best_bitset).count('1')
    
    current = best_bitset
    current_count = best_count
    
    no_improve = 0
    while time.time() - start_time < timeout:
        # Try to add a random edge
        if random.random() < 0.8:
            # Find a missing edge
            missing = []
            for i in range(NUM_EDGES):
                if not (current >> i) & 1:
                    missing.append(i)
            if not missing:
                break
            e_add = random.choice(missing)
            new_bitset = current | (1 << e_add)
            valid, _ = contains_q3(new_bitset)
            if valid:
                current = new_bitset
                current_count += 1
                no_improve = 0
                if current_count > best_count:
                    best_count = current_count
                    best_bitset = current
            else:
                no_improve += 1
        else:
            # Swap: remove one edge, add another
            present = [i for i in range(NUM_EDGES) if (current >> i) & 1]
            if not present:
                break
            e_rem = random.choice(present)
            missing = [i for i in range(NUM_EDGES) if not (current >> i) & 1]
            if not missing:
                break
            e_add = random.choice(missing)
            new_bitset = (current & ~(1 << e_rem)) | (1 << e_add)
            valid, _ = contains_q3(new_bitset)
            if valid:
                current = new_bitset
                no_improve = 0
                if current_count > best_count:
                    best_count = current_count
                    best_bitset = current
            else:
                no_improve += 1
        
        if no_improve > 500:
            # Random restart
            current = 0
            for i in range(NUM_EDGES):
                if random.random() < 0.4:
                    current |= (1 << i)
            valid, _ = contains_q3(current)
            if not valid:
                # Repair
                while not valid:
                    _, q3_edges = contains_q3(current)
                    if not q3_edges:
                        break
                    rem = random.choice(q3_edges)
                    current &= ~(1 << rem)
                    valid, _ = contains_q3(current)
            current_count = bin(current).count('1')
            no_improve = 0
    
    result = []
    for i in range(NUM_EDGES):
        if (best_bitset >> i) & 1:
            result.append(list(edges[i]))
    return result
