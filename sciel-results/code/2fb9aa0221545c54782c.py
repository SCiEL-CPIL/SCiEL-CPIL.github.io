"""Solve: max subset of [1,256] with (a*b)%(a+b)!=0 for all distinct a,b."""
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Build conflict graph: edge (i,j) if (i+1)*(j+1) % (i+j+2) == 0
    # Vertices are 0..255 representing numbers 1..256
    conflicts = [0] * N  # bitset of conflicts for each vertex
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Greedy coloring for upper bound (Tomita-style)
    # Returns list of colors where colors[i] = color class index
    # The number of colors is an upper bound on independent set size
    
    def greedy_color(vertices, remaining_mask):
        """Color the vertices in order, return (num_colors, color_of_vertex list)"""
        n = len(vertices)
        if n == 0:
            return 0, []
        
        colors = [0] * n
        color_classes = []  # list of sets of vertex indices
        
        for idx, v in enumerate(vertices):
            bit = 1 << v
            # Find smallest color class where v has no conflicts with any member
            placed = False
            for ci, cls in enumerate(color_classes):
                # Check if v conflicts with any in this class
                conflict_with_class = False
                for u in cls:
                    if (conflicts[v] >> u) & 1:
                        conflict_with_class = True
                        break
                if not conflict_with_class:
                    cls.append(v)
                    colors[idx] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[idx] = len(color_classes) - 1
        
        return len(color_classes), colors
    
    # Faster coloring using bitsets and degree ordering
    def color_bound(remaining_mask):
        """Return upper bound on independent set size in remaining_mask using greedy coloring."""
        # Extract vertices from mask
        vertices = []
        m = remaining_mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            vertices.append(v)
            m ^= low
        
        if not vertices:
            return 0
        
        # Sort by degree in remaining_mask descending for tighter bound
        def get_deg(v):
            return bin(conflicts[v] & remaining_mask).count('1')
        
        vertices.sort(key=get_deg, reverse=True)
        
        # Greedy coloring: assign each vertex to first available color
        color_classes = []  # list of bitsets
        
        for v in vertices:
            placed = False
            for ci, cls_mask in enumerate(color_classes):
                # v can go in color ci if no conflict with any vertex in cls_mask
                if (conflicts[v] & cls_mask) == 0:
                    color_classes[ci] = cls_mask | (1 << v)
                    placed = True
                    break
            if not placed:
                color_classes.append(1 << v)
        
        return len(color_classes)
    
    # Branch and bound for maximum independent set
    best_size = [0]
    best_set = [set()]
    
    def branch_and_bound(remaining_mask, current_set, current_size):
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current_size + color_bound
        ub = current_size + color_bound(remaining_mask)
        if ub <= best_size[0]:
            return
        
        # Select vertex with max degree in remaining subgraph
        best_v = -1
        best_deg = -1
        m = remaining_mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            deg = bin(conflicts[v] & remaining_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            m ^= low
        
        if best_v < 0:
            return
        
        # Branch 1: include best_v
        new_set = current_set + [best_v + 1]
        # Remove best_v and its neighbors from remaining
        new_remaining = remaining_mask & ~(1 << best_v) & ~conflicts[best_v]
        branch_and_bound(new_remaining, new_set, current_size + 1)
        
        # Branch 2: exclude best_v
        new_remaining2 = remaining_mask & ~(1 << best_v)
        branch_and_bound(new_remaining2, current_set, current_size)
    
    # Initial greedy solution: Min-Degree heuristic
    available_mask = (1 << N) - 1
    greedy_set = set()
    
    while available_mask:
        if time.time() > deadline:
            break
            
        # Find vertex with minimum degree in available subgraph
        min_v = -1
        min_deg = N + 1
        m = available_mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            deg = bin(conflicts[v] & available_mask).count('1')
            if deg < min_deg:
                min_deg = deg
                min_v = v
            m ^= low
            
        if min_v < 0:
            break
            
        greedy_set.add(min_v + 1)
        # Remove min_v and its neighbors
        available_mask &= ~(1 << min_v)
        available_mask &= ~conflicts[min_v]
        
    best_size[0] = len(greedy_set)
    best_set[0] = greedy_set
    
    # Now run branch and bound to try to improve
    full_mask = (1 << N) - 1
    branch_and_bound(full_mask, [], 0)
    
    if time.time() > deadline:
        pass  # already handled in recursion
    
    result = sorted(best_set[0])
    if not result:
        result = list(range(1, 257, 2))
    
    return result

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Size: {len(result)}, Time: {t1-t0:.2f}s")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if (a*b) % (a+b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
