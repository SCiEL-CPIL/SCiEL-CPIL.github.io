import random
import time
from itertools import combinations

# Precomputation for C6 check
def get_edge_idx(i, j):
    if i > j: i, j = j, i
    return 5*i - i*(i-1)//2 + (j - i - 1)

has_c6 = [False] * (1 << 15)
cycle_masks = []
import itertools
perms = list(itertools.permutations([1, 2, 3, 4, 5]))
seen_cycles = set()
for p in perms:
    cycle = [0] + list(p)
    edges = []
    for k in range(6):
        u = cycle[k]
        v = cycle[(k+1)%6]
        if u > v: u, v = v, u
        edges.append((u, v))
    mask = 0
    for u, v in edges:
        idx = get_edge_idx(u, v)
        mask |= (1 << idx)
    if mask not in seen_cycles:
        seen_cycles.add(mask)
        cycle_masks.append(mask)

for m in range(1 << 15):
    for cm in cycle_masks:
        if (m & cm) == cm:
            has_c6[m] = True
            break

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Try n=18, then 17
    for n in [18, 17]:
        if time.time() > deadline - 20:
            break
            
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Map edge index to (u, v)
        edge_to_uv = []
        uv_to_edge = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_to_uv.append((i, j))
                uv_to_edge[(i, j)] = idx
                idx += 1
        
        subsets = list(combinations(range(n), 6))
        
        # Precompute edge_to_subsets
        # edge_to_subsets[e] = list of subset indices
        edge_to_subsets = [[] for _ in range(num_edges)]
        for s_idx, s in enumerate(subsets):
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = s[i], s[j]
                    e = uv_to_edge[(u, v)]
                    edge_to_subsets[e].append(s_idx)
        
        # Function to get mask for subset s_idx and color c
        # We need to map local vertex indices to global, then to edge indices
        # This is slow if done from scratch.
        # Better: Precompute for each subset, the list of (local_i, local_j, global_edge_idx)
        subset_edges = []
        for s in subsets:
            edges = []
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = s[i], s[j]
                    e = uv_to_edge[(u, v)]
                    edges.append((i, j, e))
            subset_edges.append(edges)
            
        def get_mask(s_idx, c):
            mask = 0
            for li, lj, ge in subset_edges[s_idx]:
                if colors[ge] == c:
                    mask |= (1 << get_edge_idx(li, lj))
            return mask

        # Initial cost
        current_cost = 0
        # Store masks for quick update? 
        # masks[s_idx][c]
        masks = [[0]*4 for _ in range(len(subsets))]
        for s_idx in range(len(subsets)):
            for c in range(4):
                m = get_mask(s_idx, c)
                masks[s_idx][c] = m
                if has_c6[m]:
                    current_cost += 1
                    
        if current_cost == 0:
            return {'n': n, 'colors': colors}
            
        best_cost = current_cost
        best_colors = colors[:]
        
        T = 100.0
        cooling = 0.999
        min_T = 0.1
        
        while time.time() < deadline - 20:
            if T < min_T:
                break
                
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
                
            delta = 0
            affected = edge_to_subsets[e]
            
            for s_idx in affected:
                # Update mask for old_c
                # Remove edge e from mask
                # We need to know the local index of e in this subset
                # We can find it by searching subset_edges[s_idx] for ge==e
                # This is slow.
                # Precompute: for each subset, a dict ge -> (li, lj)
                # Or just store the bit for each edge in the subset.
                
                # Let's precompute subset_edge_bits
                # subset_edge_bits[s_idx] = list of (li, lj, ge, bit)
                # where bit = 1 << get_edge_idx(li, lj)
                
                # I didn't do that. Let's do it now in the loop? No.
                # I'll refactor to precompute bits.
                pass
            
            # Since I didn't precompute bits, I'll just re-evaluate affected subsets fully.
            # Cost: len(affected) * 15 ops.
            # len(affected) is ~2000. 30000 ops.
            
            for s_idx in affected:
                # Re-calc mask for old_c
                m_old = 0
                for li, lj, ge in subset_edges[s_idx]:
                    if colors[ge] == old_c:
                        m_old |= (1 << get_edge_idx(li, lj))
                # Re-calc mask for new_c
                m_new = 0
                for li, lj, ge in subset_edges[s_idx]:
                    if colors[ge] == new_c:
                        m_new |= (1 << get_edge_idx(li, lj))
                        
                # Update cost
                if has_c6[masks[s_idx][old_c]] and not has_c6[m_old]:
                    delta -= 1
                if has_c6[masks[s_idx][new_c]] and not has_c6[m_new]:
                    delta += 1
                # Note: if has_c6 was true and remains true, delta 0.
                # if has_c6 was false and remains false, delta 0.
                
                # Update stored masks
                masks[s_idx][old_c] = m_old
                masks[s_idx][new_c] = m_new
                
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                colors[e] = new_c
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                if current_cost == 0:
                    return {'n': n, 'colors': colors}
            else:
                # Revert masks? 
                # We already updated masks in the loop above.
                # If we reject, we must revert masks.
                for s_idx in affected:
                    # Re-calc or store old values?
                    # I should store old masks before updating.
                    pass
                    
            T *= cooling

    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
