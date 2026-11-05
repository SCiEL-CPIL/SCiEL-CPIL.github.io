"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(n):
    if n <= 1:
        return "1"
    
    def standard_construct(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n < 7:
        return standard_construct(n)
    
    # For n=7, use heuristic optimization on the top level
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate standard S6
    s6_str = standard_construct(6)
    s6 = [int(c) for c in s6_str]
    
    # Get the order of permutations of {1..6} as they appear in S6
    perms_in_s6, seen = [], set()
    for i in range(len(s6) - 5):
        window = tuple(s6[i:i+6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_in_s6.append(window)
            
    N = len(perms_in_s6)
    perms = perms_in_s6
    
    def get_overlap(u, v):
        max_ov = 5
        for o in range(max_ov, -1, -1):
            if o == 0 or u[-o:] == v[:o]:
                return o
        return 0
    
    # Precompute overlap matrix
    overlap_matrix = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                overlap_matrix[i][j] = get_overlap(perms[i], perms[j])
    
    best_order = None
    best_cost = float('inf')
    
    # Try a few starting points for Nearest Neighbor
    start_indices = [0, N//2, N-1, N//4, 3*N//4]
    
    for start_idx in start_indices:
        visited = [False] * N
        current = start_idx
        visited[current] = True
        order = [current]
        current_cost = 0
        
        for _ in range(N - 1):
            min_cost = float('inf')
            next_node = -1
            for j in range(N):
                if not visited[j]:
                    c = 6 - overlap_matrix[current][j]
                    if c < min_cost:
                        min_cost = c
                        next_node = j
            
            if next_node == -1:
                break
            
            current_cost += min_cost
            visited[next_node] = True
            order.append(next_node)
            current = next_node
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_order = order[:]
            
        if time.time() > deadline:
            break
    
    if best_order is None:
        return standard_construct(n)
    
    # Optional: Local Search (2-opt) to improve the order
    # We have time, so try to improve best_order
    improved = True
    iterations = 0
    max_iterations = 50 # Limit iterations to avoid timeout
    
    while improved and time.time() < deadline and iterations < max_iterations:
        improved = False
        iterations += 1
        # Try random swaps or 2-opt
        # 2-opt: reverse a segment
        # Cost change for reversing a segment [i, j] in a path
        # Old edges: (i-1, i), (j, j+1)
        # New edges: (i-1, j), (i, j+1)
        # We need to check if the new cost is lower.
        
        # Simple random swap local search is safer and easier to implement correctly
        # Let's do a few random 2-opts
        import random
        for _ in range(1000):
            if time.time() > deadline:
                break
            i = random.randint(1, N-2)
            j = random.randint(i+1, N-1)
            
            # Calculate cost delta
            # Old cost parts:
            # Edge (i-1, i) and (j, j+1)
            # Note: indices in order list are 0-based
            # Let's use indices in the order list
            # Segment is order[i-1] to order[j+1]?
            # Let's reverse order[i:j+1]
            
            # Old edges involved:
            # (order[i-1], order[i]) and (order[j], order[j+1])
            # New edges involved:
            # (order[i-1], order[j]) and (order[i], order[j+1])
            
            # Be careful with boundaries
            if i == 0:
                # Start of path
                # Old: (order[0], order[1])
                # New: (order[j], order[1]) -- No, reversing [0..j]
                # New start is order[j].
                # Old edge (0,1) is removed.
                # New edge (j, 1) is added? No.
                # If we reverse [0..j], the new sequence starts with order[j], then order[j-1]...
                # The edge from start to next is (order[j], order[j-1]).
                # The edge to the rest is (order[0], order[j+1]).
                # This is complex. Let's stick to internal swaps or simple 2-opt on internal segments.
                continue
            
            if j == N-1:
                # End of path
                # Old: (order[i-1], order[i]) and (order[N-2], order[N-1])
                # New: (order[i-1], order[N-1]) and (order[i], order[N-2])
                pass # Simplified: skip boundaries for now to avoid bugs
            
            # Internal 2-opt
            # Old edges: (order[i-1], order[i]) and (order[j], order[j+1])
            # New edges: (order[i-1], order[j]) and (order[i], order[j+1])
            
            old_cost = (6 - overlap_matrix[order[i-1]][order[i]]) + (6 - overlap_matrix[order[j]][order[j+1]])
            new_cost = (6 - overlap_matrix[order[i-1]][order[j]]) + (6 - overlap_matrix[order[i]][order[j+1]])
            
            if new_cost < old_cost:
                # Perform swap
                order[i:j+1] = reversed(order[i:j+1])
                best_cost += new_cost - old_cost
                improved = True
                
    # Construct S7 using best_order
    word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [7] + list(p)
        overlap = 0
        max_ov = min(len(word), len(block))
        for o in range(max_ov, -1, -1):
            if o == 0 or word[len(word) - o:] == block[:o]:
                overlap = o
                break
        word += block[overlap:]
        
    result = "".join(map(str, word))
    
    # Compare with standard length
    std_len = len(standard_construct(n))
    if len(result) < std_len:
        return result
    else:
        return standard_construct(n)

# EVOLVE-BLOCK-END
