import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 512:
        squares.add(s * s)
        s += 1
    
    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Pre-filter: nodes where 2a is a square are invalid
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)
            
    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all
        
    n_valid = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build adjacency as bitmasks over valid node indices
    adj = [0] * n_valid
    for a in valid_nodes:
        ai = node_index[a]
        m = conflict_mask[a] & valid_mask_all
        for b in valid_nodes:
            if (m >> b) & 1:
                adj[ai] |= (1 << node_index[b])
    
    # Branch and bound for maximum independent set
    # State: candidates (bitset), current_size, best_size
    best_size = [0]
    best_set = [0]
    
    def greedy_is_size(candidates_mask):
        """Compute size of greedy independent set from candidates."""
        size = 0
        c = candidates_mask
        while c:
            # Pick node with min degree in remaining candidates
            best_deg = float('inf')
            best_node = -1
            cc = c
            while cc:
                lb = cc & (-cc)
                node = lb.bit_length() - 1
                cc ^= lb
                deg = bin(adj[node] & c).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
                    if deg == 0:
                        break
            if best_node < 0:
                break
            size += 1
            c &= ~(1 << best_node)
            c &= ~adj[best_node]
        return size
    
    def branch_and_bound(candidates_mask, current_size, depth):
        if time.time() - start_time > deadline - 2:
            return
        
        if candidates_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = candidates_mask  # placeholder, we track via set
            return
        
        # Upper bound: current_size + greedy IS size of remaining
        ub = current_size + greedy_is_size(candidates_mask)
        if ub <= best_size[0]:
            return
        
        # Pick pivot node: one with max degree in candidates
        c = candidates_mask
        best_deg = -1
        pivot = -1
        cc = c
        while cc:
            lb = cc & (-cc)
            node = lb.bit_length() - 1
            cc ^= lb
            deg = bin(adj[node] & c).count('1')
            if deg > best_deg:
                best_deg = deg
                pivot = node
        
        if pivot < 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        
        # Branch 1: include pivot
        new_candidates = candidates_mask & ~(1 << pivot) & ~adj[pivot]
        branch_and_bound(new_candidates, current_size + 1, depth + 1)
        
        # Branch 2: exclude pivot
        new_candidates = candidates_mask & ~(1 << pivot)
        branch_and_bound(new_candidates, current_size, depth + 1)
    
    # Initial solution via greedy min-degree
    def greedy_init():
        remaining = valid_mask_all
        chosen = []
        while remaining:
            min_deg = float('inf')
            min_node = -1
            cc = remaining
            while cc:
                lb = cc & (-cc)
                node = lb.bit_length() - 1
                cc ^= lb
                deg = bin(adj[node] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
                    if deg == 0:
                        break
            if min_node < 0:
                break
            chosen.append(valid_nodes[min_node])
            remaining &= ~(1 << min_node)
            remaining &= ~adj[min_node]
        return chosen
    
    init_sol = greedy_init()
    best_size[0] = len(init_sol)
    
    # Track the actual set for the best solution
    best_nodes = set(init_sol)
    
    def branch_and_bound_track(candidates_mask, current_size, chosen_mask, depth):
        if time.time() - start_time > deadline - 2:
            return
        
        if candidates_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                # Reconstruct set from chosen_mask
                nodes = []
                cm = chosen_mask
                while cm:
                    lb = cm & (-cm)
                    idx = lb.bit_length() - 1
                    cm ^= lb
                    nodes.append(valid_nodes[idx])
                best_nodes.clear()
                best_nodes.update(nodes)
            return
        
        # Upper bound
        ub = current_size + greedy_is_size(candidates_mask)
        if ub <= best_size[0]:
            return
        
        # Pick pivot node: one with max degree in candidates
        c = candidates_mask
        best_deg = -1
        pivot = -1
        cc = c
        while cc:
            lb = cc & (-cc)
            node = lb.bit_length() - 1
            cc ^= lb
            deg = bin(adj[node] & c).count('1')
            if deg > best_deg:
                best_deg = deg
                pivot = node
        
        if pivot < 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
            return
        
        # Branch 1: include pivot
        new_candidates = candidates_mask & ~(1 << pivot) & ~adj[pivot]
        new_chosen = chosen_mask | (1 << pivot)
        branch_and_bound_track(new_candidates, current_size + 1, new_chosen, depth + 1)
        
        # Branch 2: exclude pivot
        new_candidates = candidates_mask & ~(1 << pivot)
        branch_and_bound_track(new_candidates, current_size, chosen_mask, depth + 1)
    
    # Run B&B with tracking
    all_valid_mask = 0
    for i in range(n_valid):
        all_valid_mask |= (1 << i)
    
    branch_and_bound_track(all_valid_mask, 0, 0, 0)
    
    # If B&B didn't improve, use the greedy solution
    if best_nodes:
        return list(best_nodes)
    return init_sol
