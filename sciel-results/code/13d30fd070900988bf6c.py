import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_edges = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s
    k4s = list(itertools.combinations(range(N), 4))
    num_k4s = len(k4s)
    
    # Map each edge to the list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, k4 in enumerate(k4s):
        # Generate 6 edges from the 4 vertices
        v0, v1, v2, v3 = k4
        k4_edges = [
            (v0, v1), (v0, v2), (v0, v3),
            (v1, v2), (v1, v3),
            (v2, v3)
        ]
        for e in k4_edges:
            e_idx = edge_index[e]
            edge_to_k4s[e_idx].append(k4_idx)
            
    # Initialize colors with the 17-color solution
    # colors[i] is the color of edge i
    colors = [(i + j) % 17 for i, j in edges]
    
    # Function to count distinct colors in a K4 given current colors
    # k4_edges_indices: list of 6 edge indices
    def count_distinct(k4_idx):
        k4 = k4s[k4_idx]
        v0, v1, v2, v3 = k4
        e_indices = [
            edge_index[(v0, v1)], edge_index[(v0, v2)], edge_index[(v0, v3)],
            edge_index[(v1, v2)], edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        c_set = set(colors[i] for i in e_indices)
        return len(c_set)

    # Initial state check
    violations = 0
    for k4_idx in range(num_k4s):
        if count_distinct(k4_idx) < 5:
            violations += 1
            
    # We know the initial state is valid (violations=0) and uses 17 colors.
    best_colors = colors[:]
    best_distinct = len(set(colors))
    best_violations = violations
    
    # SA Parameters
    T = 1.0
    T_min = 0.001
    alpha = 0.9995
    steps = 0
    
    # We want to minimize distinct colors.
    # Cost = distinct_colors + 1000 * violations
    
    def get_cost():
        return len(set(colors)) + 1000 * violations

    current_cost = get_cost()
    
    while time.time() < deadline:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Pick a new color
        # We can try colors from 0 to best_distinct (or a bit more to allow exploration)
        # Let's allow 0 to 20
        new_color = random.randint(0, 20)
        
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        # Only K4s containing this edge are affected
        delta_violations = 0
        for k4_idx in edge_to_k4s[e_idx]:
            old_distinct = count_distinct(k4_idx)
            # Temporarily change color
            colors[e_idx] = new_color
            new_distinct = count_distinct(k4_idx)
            # Revert for now, we'll commit if accepted
            colors[e_idx] = old_color
            
            old_bad = 1 if old_distinct < 5 else 0
            new_bad = 1 if new_distinct < 5 else 0
            delta_violations += new_bad - old_bad
            
        # Calculate delta distinct colors
        # This is tricky to calculate exactly without committing.
        # Let's just commit, calculate cost, and revert if rejected.
        # Since num_k4s is small, this is fast enough.
        
        # Commit change
        colors[e_idx] = new_color
        
        # Recalculate violations for affected K4s
        new_violations = violations
        for k4_idx in edge_to_k4s[e_idx]:
            d = count_distinct(k4_idx)
            if d < 5:
                # We need to know if it was bad before.
                # We don't have the old state easily here without storing it.
                # Let's just recount all violations? No, too slow.
                # Let's do the delta calculation properly.
                pass
        
        # Better approach: Store old distinct counts for affected K4s.
        
        # Let's restart the loop logic with a cleaner delta calculation.
        # Actually, given 120s, we can afford to be slightly less optimized if correct.
        # But let's stick to the delta method.
        
        # Re-implementing the move logic inside the loop for clarity and correctness
        # (The previous block was a draft)
        
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Pick new color
        # Bias towards existing colors or lower indices?
        # Let's pick from 0..max_color_seen
        max_c = max(colors)
        new_color = random.randint(0, max_c + 1)
        
        if new_color == old_color:
            continue
            
        # Identify affected K4s
        affected_k4s = edge_to_k4s[e_idx]
        
        # Store old states
        old_states = {}
        for k4_idx in affected_k4s:
            old_states[k4_idx] = count_distinct(k4_idx)
            
        # Apply change
        colors[e_idx] = new_color
        
        # Calculate new states and delta violations
        delta_violations = 0
        for k4_idx in affected_k4s:
            new_distinct = count_distinct(k4_idx)
            old_distinct = old_states[k4_idx]
            
            old_bad = 1 if old_distinct < 5 else 0
            new_bad = 1 if new_distinct < 5 else 0
            delta_violations += new_bad - old_bad
            
        new_violations = violations + delta_violations
        
        # Calculate delta distinct colors
        # We need to know if new_color was already present in the global palette
        # and if old_color becomes absent.
        # This requires counting color frequencies.
        # Let's maintain a frequency array.
        
        # For simplicity in this draft, let's just compute cost directly.
        # It might be slower but 1820 K4s * 91 edges/K4 is manageable?
        # No, computing full cost is O(E) for set creation. 120 edges.
        # 120 operations per step. 100k steps = 12M ops. Very fast.
        
        new_distinct_count = len(set(colors))
        new_cost = new_distinct_count + 1000 * new_violations
        
        if new_cost < current_cost or random.random() < pow(2.71828, -(new_cost - current_cost) / T):
            # Accept
            violations = new_violations
            current_cost = new_cost
            
            # Update best
            if new_violations == 0 and new_distinct_count < best_distinct:
                best_distinct = new_distinct_count
                best_colors = colors[:]
                best_violations = new_violations
                
            # Update temperature
            T = max(T * alpha, T_min)
        else:
            # Reject
            colors[e_idx] = old_color
            
        steps += 1
        
    # Return the best valid solution found
    if best_violations == 0:
        return best_colors
    else:
        # Fallback to initial valid solution
        return [(i + j) % 17 for i, j in edges]
