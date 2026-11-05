import numpy as np
import time
import random

def priority(adjacency):
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    deadline = time.time() + 100  # Safety margin
    
    # Precompute neighbor masks
    nbr_masks = [0] * n
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        neighbors = np.where(adjacency[i])[0]
        nbrs[i] = neighbors.tolist()
        mask = 0
        for j in neighbors:
            mask |= (1 << j)
        nbr_masks[i] = mask
        
    def eval_order(order):
        blocked = 0
        count = 0
        for v in order:
            bit = 1 << v
            if not (blocked & bit):
                count += 1
                blocked |= nbr_masks[v]
                blocked |= bit
        return count

    # Initial Order: Greedy Min-Degree
    current_remaining = list(range(n))
    current_deg = [len(nbrs[i]) for i in range(n)]
    order = []
    
    while current_remaining:
        min_v = None
        min_d = float('inf')
        for v in current_remaining:
            if current_deg[v] < min_d:
                min_d = current_deg[v]
                min_v = v
        
        order.append(min_v)
        current_remaining.remove(min_v)
        for u in nbrs[min_v]:
            if u in current_remaining:
                current_deg[u] -= 1
                
    best_order = order[:]
    best_count = eval_order(best_order)
    
    current_order = best_order[:]
    current_count = best_count
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_count = eval_order(current_order)
                
                if new_count > current_count:
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_order = current_order[:]
                    improved = True
                    break
                else:
                    current_order[i], current_order[j] = current_order[j], current_order[i]
            
            if improved:
                break
            if time.time() > deadline:
                break
        
        if not improved:
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    current_count = eval_order(current_order)
            else:
                break

    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores
