import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 256

    # Build conflict graph: edge (a,b) if a*b divisible by a+b
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Greedy independent set for initial lower bound
    def greedy_independent_set(cand):
        result = 0
        remaining = cand
        while remaining:
            best_v = -1
            best_d = N + 1
            rem = remaining
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                d = (adj[v] & remaining).bit_count()
                if d < best_d:
                    best_d = d
                    best_v = v
                rem &= rem - 1
            result |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return result

    full_mask = (1 << N) - 1
    init_set = greedy_independent_set(full_mask)
    best_size = init_set.bit_count()
    best_set = init_set

    # Tomita's algorithm: greedy coloring for upper bound
    # Colors vertices such that vertices with same color are not adjacent
    # The number of colors is an upper bound on the size of MIS in the subgraph
    def greedy_color(cand):
        """
        Returns list of color classes (bitmasks) for vertices in cand.
        Vertices in the same color class are pairwise non-adjacent.
        The number of color classes is an upper bound on MIS size.
        """
        colors = []
        remaining = cand
        
        while remaining:
            color_class = 0
            # Find all vertices that can share this color (independent set)
            # Greedy: pick vertices in order, skip if adjacent to any in current class
            rem = remaining
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                # Check if v is adjacent to any vertex already in color_class
                if (adj[v] & color_class) == 0:
                    color_class |= lsb
                rem &= rem - 1
            
            if color_class == 0:
                break
            colors.append(color_class)
            remaining &= ~color_class
        
        return colors

    # B&B with Tomita's coloring bound
    def bb(cand, current_set, current_size):
        nonlocal best_set, best_size
        
        if time.time() > DEADLINE:
            return
            
        if cand == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Get coloring and upper bound
        color_classes = greedy_color(cand)
        num_colors = len(color_classes)
        
        # Upper bound: current_size + num_colors
        if current_size + num_colors <= best_size:
            return
        
        # Branch on vertex with highest color (last color class)
        # Process vertices in reverse color order for better pruning
        # Pick a vertex from the last (highest) color class
        last_class = color_classes[-1]
        
        # Iterate through vertices in last_class, branch on each
        rem = last_class
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem &= rem - 1
            
            # Branch 1: Include v
            new_cand = cand & ~(adj[v] | (1 << v))
            bb(new_cand, current_set | lsb, current_size + 1)
            
            # Branch 2: Exclude v (remove from candidate set)
            # Update cand to exclude v for subsequent branches
            cand &= ~lsb

    bb(full_mask, 0, 0)
    
    # Convert bitmask to list
    result = []
    m = best_set
    while m:
        lsb = m & (-m)
        result.append(lsb.bit_length())
        m &= m - 1
        
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    s = construct()
    print(len(s))
