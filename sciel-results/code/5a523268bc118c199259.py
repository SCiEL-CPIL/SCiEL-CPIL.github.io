import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 18
    
    def to_bitmask(adj_list):
        return adj_list
    
    def is_triangle_free(bitmask, i, j, k):
        # Check if vertices i,j,k form a triangle in G (all 3 edges present)
        return (bitmask[i] >> j & 1) and (bitmask[i] >> k & 1) and (bitmask[j] >> k & 1)
    
    def is_k4_free(bitmask):
        # Check no K4 exists
        for a in range(n):
            for b in range(a+1, n):
                if not (bitmask[a] >> b & 1):
                    continue
                for c in range(b+1, n):
                    if not ((bitmask[a] >> c & 1) and (bitmask[b] >> c & 1)):
                        continue
                    for d in range(c+1, n):
                        if ((bitmask[a] >> d & 1) and (bitmask[b] >> d & 1) and (bitmask[c] >> d & 1)):
                            return False
        return True
    
    def max_triangle_free_size(bitmask):
        # Find the largest set S such that G[S] is triangle-free
        # G[S] triangle-free means no 3 vertices in S with all 3 edges present in G
        # We check from large to small
        from itertools import combinations
        
        # Try size 8, 7, 6, 5, 4, 3, 2
        for size in range(8, 1, -1):
            for subset in combinations(range(n), size):
                s = list(subset)
                # Check if any triangle exists in G[s]
                has_triangle = False
                for i in range(size):
                    for j in range(i+1, size):
                        if (bitmask[s[i]] >> s[j]) & 1:
                            for k in range(j+1, size):
                                if (bitmask[s[i]] >> s[k]) & 1 and (bitmask[s[j]] >> s[k]) & 1:
                                    has_triangle = True
                                    break
                            if has_triangle:
                                break
                    if has_triangle:
                        break
                if not has_triangle:
                    return size
        return 2  # Any 2 vertices form a triangle-free subgraph
    
    def remove_k4s(bitmask):
        # Remove edges to eliminate all K4s
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if not (bitmask[a] >> b & 1):
                        continue
                    for c in range(b+1, n):
                        if not ((bitmask[a] >> c & 1) and (bitmask[b] >> c & 1)):
                            continue
                        for d in range(c+1, n):
                            if ((bitmask[a] >> d & 1) and (bitmask[b] >> d & 1) and (bitmask[c] >> d & 1)):
                                # Found K4: a,b,c,d - remove edge (a,d) to break it
                                bitmask[a] ^= (1 << d)
                                bitmask[d] ^= (1 << a)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return bitmask
    
    def make_edges_from_bitmask(bitmask):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (bitmask[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Strategy 1: Start with a structured graph and do local search
    # Use a circulant-like construction as starting point
    
    # Initialize with a moderately dense random graph
    best_score = 18
    best_bitmask = [0] * n
    
    # Try multiple starting points
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        bitmask = [0] * n
        p = 0.4 + 0.3 * random.random()  # density between 0.4 and 0.7
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    bitmask[i] |= (1 << j)
                    bitmask[j] |= (1 << i)
        
        # Make K4-free
        bitmask = remove_k4s(bitmask)
        
        current_score = max_triangle_free_size(bitmask)
        
        if current_score < best_score:
            best_score = current_score
            best_bitmask = bitmask[:]
            if best_score <= 5:
                break
        
        # Local search: try single-edge flips
        if time.time() > deadline:
            break
            
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding an edge
            for i in range(n):
                for j in range(i+1, n):
                    if (bitmask[i] >> j) & 1:
                        continue
                    # Add edge (i,j)
                    bitmask[i] |= (1 << j)
                    bitmask[j] |= (1 << i)
                    
                    # Check if still K4-free
                    if is_k4_free(bitmask):
                        new_score = max_triangle_free_size(bitmask)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_bitmask = bitmask[:]
                            improved = True
                        else:
                            # Revert
                            bitmask[i] ^= (1 << j)
                            bitmask[j] ^= (1 << i)
                    else:
                        # Revert
                        bitmask[i] ^= (1 << j)
                        bitmask[j] ^= (1 << i)
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try removing an edge
                for i in range(n):
                    for j in range(i+1, n):
                        if not ((bitmask[i] >> j) & 1):
                            continue
                        # Remove edge (i,j)
                        bitmask[i] ^= (1 << j)
                        bitmask[j] ^= (1 << i)
                        
                        new_score = max_triangle_free_size(bitmask)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_bitmask = bitmask[:]
                            improved = True
                        else:
                            # Revert
                            bitmask[i] |= (1 << j)
                            bitmask[j] |= (1 << i)
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
    
    # Convert best_bitmask to edge list
    return make_edges_from_bitmask(best_bitmask)

# Test locally
if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Edges: {len(edges)}")
    # Verify K4-free and compute score
    bitmask = [0]*18
    for e in edges:
        bitmask[e[0]] |= 1 << e[1]
        bitmask[e[1]] |= 1 << e[0]
    print(f"K4-free: {is_k4_free(bitmask)}")
    print(f"Score: {max_triangle_free_size(bitmask)}")
