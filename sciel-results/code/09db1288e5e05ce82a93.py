import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    
    # Generate all edges
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                
    def can_add(u, v, selected):
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            mask = 1 << j
            w1 = u ^ mask
            w2 = v ^ mask
            
            e1 = (min(u, w1), max(u, w1))
            e2 = (min(v, w2), max(v, w2))
            e3 = (min(w1, w2), max(w1, w2))
            
            if e1 in selected and e2 in selected and e3 in selected:
                return False
        return True

    best_set = set()
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Phase 1: Randomized Greedy
    # Run until we have used 80% of the time or found a very good solution?
    # Just run until time is up, then do local search if time remains.
    
    while time.time() < deadline - 10.0:
        random.shuffle(edges)
        current_set = set()
        for u, v in edges:
            if can_add(u, v, current_set):
                current_set.add((u, v))
        
        if len(current_set) > len(best_set):
            best_set = current_set

    # Phase 2: Local Search
    # Try to improve best_set by removing one edge and refilling
    # We do this for a few rounds or until time is up
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the edges we are considering for removal
        edges_to_try = list(best_set)
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            if time.time() > deadline:
                break
                
            old_set = set(best_set)
            old_len = len(old_set)
            
            # Remove e
            best_set.remove(e)
            
            # Refill
            # We can use a fixed order or shuffled. Shuffled might escape local minima better.
            # But for a single refill, fixed order is faster and consistent.
            # Let's use the global 'edges' list order (unshuffled) for determinism in refill?
            # Or shuffle a copy?
            # Let's just iterate through 'edges'.
            for ue, ve in edges:
                if (ue, ve) in best_set:
                    continue
                if can_add(ue, ve, best_set):
                    best_set.add((ue, ve))
            
            if len(best_set) > old_len:
                improved = True
            else:
                best_set = old_set

    return [[u, v] for u, v in best_set]
