import random
import time

def construct():
    n = 20
    best_edges = set()
    deadline = time.time() + 110.0
    
    def has_k33_in_add(u, v, edges):
        # Check if adding (u, v) creates a K3,3
        # u and v must be in opposite partitions
        # Need 2 neighbors of u and 2 neighbors of v that form a K2,2
        nu = [x for x in range(n) if (min(u, x), max(u, x)) in edges]
        nv = [x for x in range(n) if (min(v, x), max(v, x)) in edges]
        
        # If |nu| < 2 or |nv| < 2, no K3,3 can be formed with u,v in opposite partitions
        if len(nu) < 2 or len(nv) < 2:
            return False
            
        # Check for K2,2 between nu and nv
        # For each pair in nu, count common neighbors in nv
        for i in range(len(nu)):
            for j in range(i+1, len(nu)):
                a1 = nu[i]
                a2 = nu[j]
                # Find common neighbors of a1 and a2 in nv
                common = 0
                for w in nv:
                    if (min(a1, w), max(a1, w)) in edges and (min(a2, w), max(a2, w)) in edges:
                        common += 1
                        if common >= 2:
                            return True
        return False

    def try_greedy():
        edges = set()
        candidates = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(candidates)
        for u, v in candidates:
            if not has_k33_in_add(u, v, edges):
                edges.add((u, v))
        return edges

    # Run multiple restarts
    while time.time() < deadline:
        current_edges = try_greedy()
        if len(current_edges) > len(best_edges):
            best_edges = current_edges
            # Optional: break if we hit a theoretical upper bound?
            # ex(20, K3,3) is likely around 80-90.
            # If we reach 90, we can stop.
            if len(best_edges) > 90:
                break
                
    return [[u, v] for u, v in best_edges]
