N = 16

# Precompute optimal splits and max colors
# f[n] = max colors for n vertices
# split[n] = tuple of sizes for the optimal split (2 or 3 parts)
f = [0] * (N + 1)
split = [()] * (N + 1)

for n in range(1, N + 1):
    if n == 1:
        f[n] = 0
        split[n] = (1,)
        continue
    
    best_val = -1
    best_split = ()
    
    # Try binary split
    for i in range(1, n):
        j = n - i
        val = 1 + f[i] + f[j]
        if val > best_val:
            best_val = val
            best_split = (i, j)
            
    # Try ternary split
    for i in range(1, n):
        for j in range(1, n - i):
            k = n - i - j
            if k < 1:
                continue
            val = 3 + f[i] + f[j] + f[k]
            if val > best_val:
                best_val = val
                best_split = (i, j, k)
                
    f[n] = best_val
    split[n] = best_split

def construct():
    # Map edge (i, j) to color
    edge_colors = {}
    
    def assign(vertices, offset):
        n = len(vertices)
        if n == 1:
            return 0 # 0 colors used
        
        parts = []
        current = 0
        for size in split[n]:
            parts.append(vertices[current:current + size])
            current += size
            
        num_parts = len(parts)
        
        # Assign colors to edges between parts
        # Colors used: num_parts * (num_parts - 1) / 2
        # We assign them sequentially starting from offset
        color_idx = offset
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                c = color_idx
                color_idx += 1
                for u in parts[i]:
                    for v in parts[j]:
                        if u < v:
                            edge_colors[(u, v)] = c
                        else:
                            edge_colors[(v, u)] = c
        
        # Recurse on parts
        # The offset for the next part must skip the colors used in the previous part's subtree
        # Number of colors used in a subtree of size s is f[s]
        current_offset = color_idx
        for p in parts:
            used = assign(p, current_offset)
            current_offset += used
            
        return num_parts * (num_parts - 1) // 2 + sum(f[len(p)] for p in parts)

    # Initial call
    vertices = list(range(N))
    assign(vertices, 0)
    
    # Construct the output list
    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(edge_colors[(i, j)])
            
    return result

