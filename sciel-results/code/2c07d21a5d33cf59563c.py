import random
import time

def construct():
    n = 16
    
    def has_k44(adj):
        # Check if any 4 vertices have >= 4 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        common = adj[i] & adj[j] & adj[k] & adj[l]
                        if common.bit_count() >= 4:
                            return True
        return False

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    # Initialize
    adj = [0] * n
    # Start with a dense random graph
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    current_edges = count_edges(adj)
    valid = not has_k44(adj)
    
    best_adj = adj[:]
    best_edges = current_edges if valid else -1
    
    start_time = time.time()
    timeout = 115.0
    
    T = 10.0
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j: continue
        
        # Flip
        if adj[i] & (1 << j):
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
            delta_e = -1
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            delta_e = 1
            
        new_edges = current_edges + delta_e
        new_valid = not has_k44(adj)
        
        # Objective: Validity first, then Edges
        # Score: -1000 * (1 if invalid else 0) + edges
        
        old_score = (-1000 if not valid else 0) + current_edges
        new_score = (-1000 if not new_valid else 0) + new_edges
        
        if new_score > old_score:
            valid = new_valid
            current_edges = new_edges
            if new_valid and new_edges > best_edges:
                best_adj = adj[:]
                best_edges = new_edges
        else:
            # Accept with probability
            delta = old_score - new_score
            if delta == 0:
                accept = True
            else:
                accept = random.random() < pow(2.71828, -delta / T)
            
            if accept:
                valid = new_valid
                current_edges = new_edges
            else:
                # Revert
                if adj[i] & (1 << j):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    delta_e = -1
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    delta_e = 1
                # Note: current_edges and valid remain unchanged as we reverted
                # But we need to be careful with the state.
                # The state is adj. We reverted adj.
                # current_edges and valid correspond to the REVERTED state?
                # No, current_edges and valid were from BEFORE the flip.
                # So if we revert, the state is back to before.
                # So we do nothing to current_edges/valid.
                
        T *= cooling
        if T < 0.1: T = 0.1

    if best_edges < 0:
        # Fallback
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
