"""Combined program: Heuristic search for minimal edge-coloring of K16."""
import time
import random
import numpy as np

# Precomputation
N = 16
edge_indices = []
for i in range(N):
    for j in range(i + 1, N):
        edge_indices.append((i, j))

# Map (i, j) to index
idx_map = {}
for k, (i, j) in enumerate(edge_indices):
    idx_map[(i, j)] = k
    idx_map[(j, i)] = k

# Generate all K4s
k4s = []
k4_edge_ids = []
for v0 in range(N):
    for v1 in range(v0 + 1, N):
        for v2 in range(v1 + 1, N):
            for v3 in range(v2 + 1, N):
                edges = [
                    (v0, v1), (v0, v2), (v0, v3),
                    (v1, v2), (v1, v3),
                    (v2, v3)
                ]
                e_ids = [idx_map[e] for e in edges]
                k4s.append((v0, v1, v2, v3))
                k4_edge_ids.append(e_ids)

num_k4s = len(k4s)

# Map edge id to list of k4 ids containing it
edge_to_k4s = [[] for _ in range(120)]
for k4_id, e_ids in enumerate(k4_edge_ids):
    for e_id in e_ids:
        edge_to_k4s[e_id].append(k4_id)

def check_k4(colours, e_ids):
    c0, c1, c2, c3, c4, c5 = (colours[e] for e in e_ids)
    return len({c0, c1, c2, c3, c4, c5}) >= 5

def count_violations(colours, k4_edge_ids):
    violations = np.zeros(len(k4_edge_ids), dtype=bool)
    for i, e_ids in enumerate(k4_edge_ids):
        if not check_k4(colours, e_ids):
            violations[i] = True
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # Try K=13, 14, 15 in order of desirability
    # If we find a valid solution for K, we return it immediately as it's better than larger K
    for K in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        # Initialize with a random permutation-based coloring or simple mod
        # Using (i + j) % K is a good starting point
        colours = np.array([(i + j) % K for i in range(N) for j in range(i + 1, N)], dtype=int)
        
        violations = count_violations(colours, k4_edge_ids)
        current_viol_count = np.sum(violations)
        
        if current_viol_count == 0:
            # Valid found
            return colours.tolist()
            
        # Simulated Annealing
        best_colours = colours.copy()
        best_viol_count = current_viol_count
        
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.05
        
        while time.time() < deadline - 2:
            if current_viol_count == 0:
                return colours.tolist()
                
            # Pick a random violating K4
            viol_indices = np.where(violations)[0]
            if len(viol_indices) == 0:
                break
            k4_id = viol_indices[random.randint(0, len(viol_indices) - 1)]
            e_ids = k4_edge_ids[k4_id]
            
            # Pick a random edge in this K4
            e_id = e_ids[random.randint(0, 5)]
            
            # Pick a new colour
            new_c = random.randint(0, K - 1)
            old_c = colours[e_id]
            
            if new_c == old_c:
                continue
                
            # Tentative change
            colours[e_id] = new_c
            
            # Update violations for K4s containing e_id
            affected_k4s = edge_to_k4s[e_id]
            
            delta = 0
            for ak4 in affected_k4s:
                old_val = violations[ak4]
                new_val = not check_k4(colours, k4_edge_ids[ak4])
                if old_val and not new_val:
                    delta -= 1
                elif not old_val and new_val:
                    delta += 1
                violations[ak4] = new_val
                
            new_total = current_viol_count + delta
            
            if new_total < current_viol_count:
                current_viol_count = new_total
                if current_viol_count < best_viol_count:
                    best_viol_count = current_viol_count
                    best_colours = colours.copy()
            else:
                # Accept with probability exp(-delta / temp)
                if delta > 0:
                    prob = np.exp(-delta / temp)
                    if random.random() < prob:
                        current_viol_count = new_total
                    else:
                        # Revert
                        colours[e_id] = old_c
                        for ak4 in affected_k4s:
                            violations[ak4] = not check_k4(colours, k4_edge_ids[ak4])
                else:
                    current_viol_count = new_total
            
            temp *= cooling
            if temp < min_temp:
                temp = min_temp
                
        if best_viol_count == 0:
            return best_colours.tolist()
            
    # Fallback: return the 17-colour solution if we couldn't find better
    return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
