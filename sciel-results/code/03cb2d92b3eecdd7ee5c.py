import random
import time
import itertools

def precompute_q3_lookup():
    base_edges = [
        (0, 4), (0, 5), (0, 6),
        (1, 5), (1, 7), (1, 4),
        (2, 6), (2, 7), (2, 4),
        (3, 6), (3, 7), (3, 5)
    ]
    valid_masks = set()
    for perm_u in itertools.permutations([0, 1, 2, 3]):
        for perm_v in itertools.permutations([4, 5, 6, 7]):
            mask = 0
            for u_old, v_old in base_edges:
                u_new = perm_u[u_old]
                v_new = perm_v[v_old - 4] + 4
                bit_idx = u_new * 4 + (v_new - 4)
                mask |= (1 << bit_idx)
            valid_masks.add(mask)
            
    contains_q3 = [False] * 65536
    valid_mask_list = list(valid_masks)
    for m in range(65536):
        if m.bit_count() < 12:
            continue
        for vm in valid_mask_list:
            if (m & vm) == vm:
                contains_q3[m] = True
                break
    return contains_q3

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    contains_q3 = precompute_q3_lookup()
    
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
            
    neighbors = [set() for _ in range(16)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    non_edges = []
    for u in range(16):
        for v in range(u + 1, 16):
            if (u, v) not in edges:
                non_edges.append((u, v))
                
    best_edges = len(edges)
    best_edge_set = set(edges)
    
    steps = 0
    max_steps = 100000
    
    while steps < max_steps and time.time() < deadline:
        if not non_edges:
            break
            
        # Try to add a random non-edge
        idx = random.randint(0, len(non_edges) - 1)
        u, v = non_edges[idx]
        
        A = neighbors[v] - {u}
        B = neighbors[u] - {v}
        
        valid = True
        if len(A) >= 3 and len(B) >= 3:
            for U_prime in itertools.combinations(A, 3):
                for V_prime in itertools.combinations(B, 3):
                    if set(U_prime).intersection(V_prime):
                        continue
                    U_set = sorted([u] + list(U_prime))
                    V_set = sorted([v] + list(V_prime))
                    
                    mask = 0
                    for i, u_i in enumerate(U_set):
                        for j, v_j in enumerate(V_set):
                            if (u_i == u and v_j == v) or ((u_i, v_j) in edges) or ((v_j, u_i) in edges):
                                mask |= (1 << (i * 4 + j))
                                
                    if contains_q3[mask]:
                        valid = False
                        break
                if not valid:
                    break
            
        if valid:
            edges.add((u, v))
            neighbors[u].add(v)
            neighbors[v].add(u)
            non_edges.pop(idx)
            
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_edge_set = set(edges)
                
        steps += 1
        
        # If stuck, kick
        if steps % 1000 == 0 and len(non_edges) > 0:
             # Remove a random edge
             if edges:
                 rem_u, rem_v = random.choice(list(edges))
                 edges.remove((rem_u, rem_v))
                 neighbors[rem_u].remove(rem_v)
                 neighbors[rem_v].remove(rem_u)
                 non_edges.append((rem_u, rem_v))
                 
    return [[u, v] for u, v in best_edge_set]
