import itertools
import random
import time

N = 16
M = N * N
# Precompute coordinates
coords = [(i, j) for i in range(N) for j in range(N)]
# Map point index to coords
# Precompute squared distances
dist = [[0] * M for _ in range(M)]
for i in range(M):
    x1, y1 = coords[i]
    for j in range(M):
        x2, y2 = coords[j]
        dist[i][j] = (x1 - x2) ** 2 + (y1 - y2) ** 2

def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

def can_add(p, S):
    """Check if point p can be added to set S without forming an isosceles triangle."""
    # S is a list of indices
    # If len(S) < 2, always can add
    if len(S) < 2:
        return True
    
    # Optimization: Check against all pairs in S
    # For each a in S:
    #   d_pa = dist[p][a]
    #   For each b in S, b != a:
    #     if dist[p][b] == d_pa or dist[a][b] == d_pa or dist[a][b] == dist[p][b]:
    #        return False
    
    # A slightly faster way:
    # For each a in S:
    #   d_pa = dist[p][a]
    #   # Check if any b in S has dist[p][b] == d_pa
    #   # Check if any b in S has dist[a][b] == d_pa
    #   # Check if any b in S has dist[a][b] == dist[p][b]
    
    # Let's just iterate pairs. |S| is likely small (< 30). |S|^2 is < 900.
    # M is 256.
    # Total checks per add: 256 * 900 = 230k.
    # Greedy: 256 adds. 256 * 230k = 59M ops.
    # This is fast in Python? 59M simple ops might take ~5-10s.
    # We need to be careful.
    
    # We can prune:
    # If p is in S, return False (should not happen)
    
    for a in S:
        d_pa = dist[p][a]
        for b in S:
            if b == a:
                continue
            d_pb = dist[p][b]
            d_ab = dist[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return False
    return True

def greedy_maximal(order, S_initial=None):
    """
    Greedily add points from 'order' to S_initial to make it maximal.
    order: list of indices to consider
    S_initial: list of indices already in the set
    Returns the maximal set (list of indices)
    """
    if S_initial is None:
        S = []
    else:
        S = list(S_initial)
        
    # Create a set for fast lookup? Not needed for can_add as it takes list.
    # But can_add iterates S.
    
    for p in order:
        if p in S:
            continue
        if can_add(p, S):
            S.append(p)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Random seed for reproducibility in evaluation?
    # random.seed(42) # Maybe not, we want diversity? 
    # The evaluator runs this once. We want the best result in 300s.
    # Using a fixed seed makes it deterministic.
    # Using random.seed() makes it non-deterministic.
    # Let's use a fixed seed to ensure the "best" found is consistent?
    # Or better, just let it run.
    
    # We will do multiple restarts.
    # Each restart:
    # 1. Random order
    # 2. Greedy maximal
    # 3. Refine (local search)
    
    while time.time() < deadline:
        # 1. Generate random order
        order = list(range(M))
        random.shuffle(order)
        
        # 2. Greedy Maximal
        S = greedy_maximal(order)
        
        # Check time
        if time.time() > deadline - 10:
            break
            
        # 3. Refine
        # Try to remove points and re-fill
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle S to try removing different points
            S_copy = list(S)
            random.shuffle(S_copy)
            
            for p in S_copy:
                if time.time() > deadline - 5:
                    break
                    
                # Remove p
                S_temp = [x for x in S if x != p]
                
                # Try to extend S_temp to a maximal set
                # Use a random order for the extension to explore different maximal sets
                # But we should prioritize points that were in S? Or just random?
                # Let's use a random order for all points not in S_temp.
                # Actually, greedy_maximal takes an 'order'.
                # We want to see if a different maximal extension is smaller.
                
                # Strategy: 
                # 1. Try to add points from the original S (except p) first? No, they are in S_temp.
                # 2. Try to add points from the complement.
                
                # Let's just run greedy_maximal with a random order.
                # But to be efficient, we can limit the number of points we try to add?
                # No, we need it to be maximal.
                
                # Optimization: 
                # If we just remove p, the set is "almost" maximal.
                # Maybe we can try to add points one by one?
                
                # Let's use a randomized greedy fill.
                # To speed up, we can use the same 'order' as the initial greedy?
                # Or a new random one?
                # A new random one explores more.
                
                order_fill = list(range(M))
                random.shuffle(order_fill)
                
                S_new = greedy_maximal(order_fill, S_temp)
                
                if len(S_new) < len(S):
                    S = S_new
                    improved = True
                    break # Restart the while loop with the new S
        
        # Update best
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
        # If we found a very small set, we can stop?
        # Theoretical minimum is unknown, but likely small.
        
    # Convert indices to coordinates
    if best_set is None:
        # Fallback: return empty or a valid set?
        # Should not happen.
        return []
        
    return [coords[i] for i in best_set]

# EVOLVE-BLOCK-END
