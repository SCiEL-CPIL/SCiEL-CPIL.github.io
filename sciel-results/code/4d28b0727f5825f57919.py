import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)
    
    # Precompute all edge indices
    edge_set = set(all_edges)
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_edges(masks):
        return sum(popcount(m) for m in masks) // 2
    
    def get_violation(masks):
        """Find one K3,3 violation or return None.
        A K3,3 exists iff there exist 3 vertices {a,b,c} with >=3 common neighbors.
        Equivalently, for some pair (i,j) with |N(i)∩N(j)| >= 3, 
        there exists k such that |N(k) ∩ N(i) ∩ N(j)| >= 3.
        """
        for i in range(n):
            mi = masks[i]
            for j in range(i+1, n):
                common_ij = mi & masks[j]
                if popcount(common_ij) < 3:
                    continue
                # Check each k: does N(k) share >=3 with common_ij?
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if popcount(common_ij & masks[k]) >= 3:
                        return (i, j, k, common_ij)
        return None
    
    def is_valid(masks):
        return get_violation(masks) is None
    
    def try_add_edge(masks, u, v):
        """Try to add edge (u,v), return True if valid."""
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        if is_valid(masks):
            return True
        else:
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            return False
    
    def try_remove_edge(masks, u, v):
        """Remove edge (u,v)."""
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
    
    def greedy_fill(masks):
        """Greedily add all possible edges that maintain validity."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)):
                        if try_add_edge(masks, i, j):
                            improved = True
        return masks
    
    # Simulated annealing
    best_edges = 0
    best_masks = [0] * n
    
    # Multiple restarts with different strategies
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Strategy 1: Random dense graph
        # Strategy 2: Start from a structured graph (e.g., based on Petersen-like structure)
        # Strategy 3: Random sparse then fill
        
        if restart % 3 == 0:
            # Dense random
            p = 0.75 + 0.15 * (restart % 3)
            masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        elif restart % 3 == 1:
            # Structured: bipartite-like with some cross edges
            # Split into two groups of 10
            masks = [0] * n
            # Full bipartite between {0..9} and {10..19}
            for i in range(10):
                for j in range(10, 20):
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
            # Add some intra-group edges
            for i in range(10):
                for j in range(i+1, 10):
                    if random.random() < 0.3:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
            for i in range(10, 20):
                for j in range(i+1, 20):
                    if random.random() < 0.3:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        else:
            # Sparse random then greedy fill
            p = 0.4 + 0.1 * (restart % 3)
            masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        
        # Repair: remove edges until valid
        max_repair = 300
        for _ in range(max_repair):
            if time.time() > deadline - 10:
                break
            viol = get_violation(masks)
            if viol is None:
                break
            i, j, k, common_mask = viol
            # Get 3 common neighbors
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            
            # Remove a random edge from the K3,3
            u = random.choice([i, j, k])
            v = random.choice(neighbors)
            try_remove_edge(masks, u, v)
        
        if not is_valid(masks):
            continue
        
        # Greedy fill
        greedy_fill(masks)
        
        current_edges = count_edges(masks)
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
        
        # Simulated annealing phase
        T = 1.5
        T_min = 0.005
        cooling_rate = 0.9997
        max_sa_steps = 80000
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Random move: either add an edge, remove an edge, or swap
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random missing edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                old_state = (masks[u], masks[v])
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        masks[u] = old_state[0]
                        masks[v] = old_state[1]
                else:
                    masks[u] = old_state[0]
                    masks[v] = old_state[1]
                    
            elif move_type < 0.60:
                # Remove a random existing edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                old_state = (masks[u], masks[v])
                try_remove_edge(masks, u, v)
                
                new_edges = count_edges(masks)
                delta = new_edges - current_edges
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
                else:
                    masks[u] = old_state[0]
                    masks[v] = old_state[1]
                    
            else:
                # Swap: remove one edge, add another
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                    
                old_state = (masks[u1], masks[v1], masks[u2], masks[v2])
                try_remove_edge(masks, u1, v1)
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                
                if is_valid(masks):
                    new_edges = count_edges(masks)
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        masks[u1] = old_state[0]
                        masks[v1] = old_state[1]
                        masks[u2] = old_state[2]
                        masks[v2] = old_state[3]
                else:
                    masks[u1] = old_state[0]
                    masks[v1] = old_state[1]
                    masks[u2] = old_state[2]
                    masks[v2] = old_state[3]
            
            T *= cooling_rate
        
        # Final greedy fill
        greedy_fill(masks)
        current_edges = count_edges(masks)
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
    
    # Convert best_masks to edge list
    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
