import time
import random
import sys

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute max colors for Gallai colorings of K_n
    # M[n] = max colors in Gallai coloring of K_n
    M = [0] * (N + 1)
    M[0] = 0
    M[1] = 0
    M[2] = 1
    M[3] = 3
    
    for n in range(4, N + 1):
        max_val = 0
        # Try all splits a, b, c such that a+b+c=n
        # We can have 2 non-empty parts (one c=0) or 3 non-empty parts
        for a in range(1, n):
            for b in range(1, n - a + 1):
                c = n - a - b
                if c < 0:
                    continue
                # Case 1: 3 non-empty parts
                if a > 0 and b > 0 and c > 0:
                    val = M[a] + M[b] + M[c] + 3
                # Case 2: 2 non-empty parts (c=0 implies b is the rest, effectively a, b)
                # But my loop covers c=0 if I allow it.
                # If c=0, the split is effectively a, b.
                # The number of added colors is 1.
                elif c == 0 and a > 0 and b > 0:
                    val = M[a] + M[b] + 1
                else:
                    continue # a or b is 0, handled by symmetry or skipped
                
                if val > max_val:
                    max_val = val
        M[n] = max_val

    # Store the optimal split for each n to reconstruct the coloring
    # split[n] = (a, b, c)
    splits = [None] * (N + 1)
    for n in range(4, N + 1):
        best_val = -1
        best_split = None
        for a in range(1, n):
            for b in range(1, n - a + 1):
                c = n - a - b
                if c < 0:
                    continue
                if a > 0 and b > 0 and c > 0:
                    val = M[a] + M[b] + M[c] + 3
                elif c == 0 and a > 0 and b > 0:
                    val = M[a] + M[b] + 1
                else:
                    continue
                
                if val > best_val:
                    best_val = val
                    best_split = (a, b, c)
        splits[n] = best_split

    # Edge index helper
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return (N-1)*i - i*(i-1)//2 + (j-i-1)

    # Build the Gallai coloring
    # Returns a list of colors for edges in the order of lexicographic pairs of the original vertices
    # We use a dict to map (u, v) -> color, then flatten
    colors_dict = {}
    next_color = 0

    def build_gallai(vertices):
        nonlocal next_color
        n = len(vertices)
        if n <= 1:
            return
        if n == 2:
            u, v = vertices[0], vertices[1]
            colors_dict[(u, v)] = next_color
            next_color += 1
            return
        if n == 3:
            u, v, w = vertices[0], vertices[1], vertices[2]
            colors_dict[(u, v)] = next_color
            colors_dict[(u, w)] = next_color + 1
            colors_dict[(v, w)] = next_color + 2
            next_color += 3
            return
        
        a, b, c = splits[n]
        V1 = vertices[:a]
        V2 = vertices[a:a+b]
        V3 = vertices[a+b:]
        
        # Assign inter-set colors
        if c > 0:
            c12 = next_color
            c13 = next_color + 1
            c23 = next_color + 2
            next_color += 3
            
            # Edges between V1 and V2
            for u in V1:
                for v in V2:
                    colors_dict[(u, v)] = c12
                    
            # Edges between V1 and V3
            for u in V1:
                for v in V3:
                    colors_dict[(u, v)] = c13
                    
            # Edges between V2 and V3
            for u in V2:
                for v in V3:
                    colors_dict[(u, v)] = c23
        else:
            # c == 0, so effectively split into V1, V2
            c12 = next_color
            next_color += 1
            
            # Edges between V1 and V2
            for u in V1:
                for v in V2:
                    colors_dict[(u, v)] = c12
            
            # V3 is empty, no edges to assign for c13, c23

        # Recurse
        build_gallai(V1)
        build_gallai(V2)
        if c > 0:
            build_gallai(V3)

    vertices = list(range(N))
    build_gallai(vertices)
    
    # Flatten to list
    result = [0] * (N*(N-1)//2)
    for i in range(N-1):
        for j in range(i+1, N):
            idx = edge_idx(i, j)
            # Ensure key exists
            if (i, j) in colors_dict:
                result[idx] = colors_dict[(i, j)]
            elif (j, i) in colors_dict:
                result[idx] = colors_dict[(j, i)]
            else:
                # Should not happen for complete graph
                result[idx] = 0
                
    return result

if __name__ == "__main__":
    c = construct()
    print(len(set(c)))
