import time
import random
from itertools import combinations

def is_2_colorable_brute(n, edges):
    # edges is a list of tuples
    for mask in range(1 << n):
        valid = True
        for e in edges:
            first_bit = (mask >> e[0]) & 1
            for v in e[1:]:
                if ((mask >> v) & 1) != first_bit:
                    break
            else:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    edges = set(all_edges)
    
    # We want to minimize the number of edges.
    # Greedy removal:
    # 1. Shuffle edges
    # 2. For each edge, try to remove it.
    # 3. If the hypergraph is still non-2-colorable, keep it removed.
    # 4. Repeat passes if time allows.
    
    # Optimization: 
    # The brute force check is the bottleneck.
    # We can cache results? No, edges change.
    
    # Let's try to be smart about which edges to remove.
    # Edges that are "redundant" are more likely to be removed.
    
    edge_list = list(edges)
    random.shuffle(edge_list)
    
    passes = 0
    while time.time() < deadline:
        passes += 1
        progress = False
        # Reshuffle each pass to avoid bias
        random.shuffle(edge_list)
        
        for e in edge_list:
            if time.time() > deadline:
                break
                
            if e not in edges:
                continue
                
            # Try removing e
            edges.remove(e)
            
            # Check if still non-2-colorable
            # is_2_colorable_brute returns True if 2-colorable.
            # We want it to be NOT 2-colorable, so is_2_colorable_brute should return False.
            if not is_2_colorable_brute(n, list(edges)):
                # Still non-2-colorable, so e was redundant. Keep it removed.
                progress = True
            else:
                # Became 2-colorable, so e was critical. Put it back.
                edges.add(e)
            
        # If no progress, we might be stuck. 
        # But with random shuffling, we might find new redundancies.
        # If no progress in a full pass, we can break or continue a few more times.
        if not progress:
            # Try a few more random shuffles
            for _ in range(5):
                if time.time() > deadline:
                    break
                random.shuffle(edge_list)
                progress2 = False
                for e in edge_list:
                    if time.time() > deadline:
                        break
                    if e not in edges:
                        continue
                    edges.remove(e)
                    if not is_2_colorable_brute(n, list(edges)):
                        progress2 = True
                    else:
                        edges.add(e)
                if not progress2:
                    break
            if not progress2:
                break

    return {'n': n, 'edges': [list(e) for e in edges]}
