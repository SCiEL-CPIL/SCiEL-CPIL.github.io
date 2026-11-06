import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    # Precompute edge endpoints
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            
    # Map edge index to (u, v)
    edge_to_uv = {idx: edges[idx] for idx in range(NUM_EDGES)}
    
    # Precompute constraints for each edge
    # For each edge e=(u,v), and for each pair (x,y) from remaining vertices,
    # we have 2 cycles involving e.
    # Cycle 1: u-v-x-y-u -> edges: e, (v,x), (x,y), (y,u)
    # Cycle 2: u-v-y-x-u -> edges: e, (v,y), (y,x), (x,u)
    
    constraints = [[] for _ in range(NUM_EDGES)]
    for e_idx in range(NUM_EDGES):
        u, v = edges[e_idx]
        others = [k for k in range(N) if k != u and k != v]
        for x, y in itertools.combinations(others, 2):
            # Cycle 1
            e1 = edge_idx(v, x)
            e2 = edge_idx(x, y)
            e3 = edge_idx(y, u)
            constraints[e_idx].append((e1, e2, e3))
            # Cycle 2
            e1 = edge_idx(v, y)
            e2 = edge_idx(y, x) # same as (x,y)
            e3 = edge_idx(x, u)
            constraints[e_idx].append((e1, e2, e3))

    colors = [0] * NUM_EDGES
    current_distinct = 1
    best_colors = colors[:]
    best_distinct = 1
    
    # Local Search
    # To speed up distinct count, maintain a frequency array
    freq = [0] * 120
    freq[0] = NUM_EDGES
    
    def count_distinct():
        return sum(1 for c in freq if c > 0)

    # Optimization: Instead of recounting every time, update incrementally
    # But 120 is small, recounting is fast.
    
    for step in range(200000):
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
            
        # Check validity
        valid = True
        for c1, c2, c3 in constraints[e_idx]:
            # Cycle edges: e_idx, c1, c2, c3
            # Colors: new_color, colors[c1], colors[c2], colors[c3]
            # Check if 4 distinct
            # We can use a set, but for speed:
            if len({new_color, colors[c1], colors[c2], colors[c3]}) == 4:
                valid = False
                break
                
        if not valid:
            continue
            
        # Update distinct count
        # Decrement old_color freq
        freq[old_color] -= 1
        # Increment new_color freq
        freq[new_color] += 1
        
        new_distinct = count_distinct()
        
        # Acceptance criterion
        if new_distinct > current_distinct:
            # Always accept
            colors[e_idx] = new_color
            current_distinct = new_distinct
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = colors[:]
        elif new_distinct == current_distinct:
            # Accept with some probability to explore
            if random.random() < 0.5:
                colors[e_idx] = new_color
            else:
                # Revert
                colors[e_idx] = old_color
                freq[old_color] += 1
                freq[new_color] -= 1
        else:
            # Revert
            colors[e_idx] = old_color
            freq[old_color] += 1
            freq[new_color] -= 1

    return best_colors
