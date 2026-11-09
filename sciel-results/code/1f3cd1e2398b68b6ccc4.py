import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Try n=18, then 17, etc.
    for n in range(18, 0, -1):
        if time.time() > deadline:
            break
        # Run search for n
        result = search_coloring(n, deadline)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback: return a known valid small coloring if all else fails
    # The seed program returns n=9. We should at least return that.
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def search_coloring(n, deadline):
    total_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # neighbors[c][u] is a bitset of neighbors of u in color c
    neighbors = [[0] * n for _ in range(4)]
    colors = [None] * total_edges
    
    def has_c6(c, u, v):
        # Check if there is a simple path of length 5 from u to v in graph c
        if u == v: return False
        start_visited = 1 << u
        stack = [(u, 0, start_visited)]
        
        while stack:
            node, depth, visited = stack.pop()
            if depth == 5:
                if node == v:
                    return True
                continue
            
            nbs = neighbors[c][node] & ~visited
            
            if depth == 4:
                if (nbs >> v) & 1:
                    return True
                continue
            
            temp = nbs
            while temp:
                lsb = temp & -temp
                nei = lsb.bit_length() - 1
                temp ^= lsb
                new_visited = visited | (1 << nei)
                stack.append((nei, depth + 1, new_visited))
        
        return False

    def backtrack(idx):
        if time.time() > deadline:
            return None
        
        if idx == total_edges:
            return colors[:]
        
        u, v = edges[idx]
        
        # Symmetry breaking for first few edges
        if idx == 0 and n > 1:
            candidates = [0]
        elif idx == 1 and n > 2:
            candidates = [1]
        elif idx == 2 and n > 3:
            candidates = [2]
        elif idx == 3 and n > 4:
            candidates = [3]
        else:
            candidates = [0, 1, 2, 3]
            random.shuffle(candidates)
            
        for c in candidates:
            if not has_c6(c, u, v):
                # Assign
                colors[idx] = c
                neighbors[c][u] |= (1 << v)
                neighbors[c][v] |= (1 << u)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                # Unassign
                colors[idx] = None
                neighbors[c][u] &= ~(1 << v)
                neighbors[c][v] &= ~(1 << u)
        
        return None

    # Try multiple random restarts if the first one fails
    # But backtrack is exhaustive. If it returns None, it explored the whole space (within time).
    # If it times out, it returns None.
    # We can try again with a different random seed to see if we get lucky with a different path?
    # No, the order of edges is fixed. Only the color order is shuffled.
    # If it times out, it's likely stuck in a hard region.
    # Restarting might help if the "hard region" is due to a bad early choice.
    
    # Let's do 5 restarts, each with a time slice?
    # No, just one run. If it fails, it fails.
    
    return backtrack(0)
