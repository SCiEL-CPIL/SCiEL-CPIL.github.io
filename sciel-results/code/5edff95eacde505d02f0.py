import time
import sys

def get_edge_index(i, j):
    if i > j: i, j = j, i
    return 15*i - i*(i-1)//2 + (j - i - 1)

# Precompute K4s and their edge indices
k4s = []
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                idxs = tuple(sorted([
                    get_edge_index(a, b),
                    get_edge_index(a, c),
                    get_edge_index(a, d),
                    get_edge_index(b, c),
                    get_edge_index(b, d),
                    get_edge_index(c, d)
                ]))
                k4s.append(idxs)

# Map edge index to list of K4 indices containing it
edge_to_k4s = [[] for _ in range(120)]
for k_idx, k4 in enumerate(k4s):
    for e_idx in k4:
        edge_to_k4s[e_idx].append(k_idx)

def is_feasible(coloring, k4_idx, num_colors):
    """
    Check if the K4 at k4_idx is still feasible given current coloring.
    Returns True if it is possible to assign colors to remaining edges
    such that at least 5 distinct colors are used in the K4.
    """
    k4 = k4s[k4_idx]
    used_colors = set()
    uncolored_count = 0
    
    for e_idx in k4:
        c = coloring[e_idx]
        if c == -1:
            uncolored_count += 1
        else:
            used_colors.add(c)
            
    current_distinct = len(used_colors)
    if current_distinct >= 5:
        return True
        
    # Need at least (5 - current_distinct) new distinct colors from uncolored edges.
    # Max new colors possible is uncolored_count (if all are distinct and not in used_colors).
    if uncolored_count >= (5 - current_distinct):
        return True
    else:
        return False

def solve(num_colors):
    coloring = [-1] * 120
    # Order edges by index for determinism
    edge_order = list(range(120))
    
    # Heuristic: Maybe process edges that appear in many K4s first?
    # All appear in 15 K4s.
    # Let's just use index order.
    
    def backtrack(edge_idx):
        if edge_idx == 120:
            return True
            
        e_idx = edge_order[edge_idx]
        
        # Try colors
        # Heuristic: Try colors that are less frequent globally? Or just 0..K-1
        for c in range(num_colors):
            coloring[e_idx] = c
            
            # Check forward constraints for all K4s containing e_idx
            valid = True
            for k_idx in edge_to_k4s[e_idx]:
                if not is_feasible(coloring, k_idx, num_colors):
                    valid = False
                    break
                    
            if valid:
                if backtrack(edge_idx + 1):
                    return True
                    
            coloring[e_idx] = -1
            
        return False

    return backtrack(0), coloring

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Try 14 colors
    for k in [14, 15]:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Use a timeout wrapper? Python doesn't have easy thread timeouts.
        # We can add a check in the recursion.
        
        # To prevent stack overflow or infinite loops, we rely on the speed of FC.
        # If it's too slow, we might need to add a node limit.
        
        # Let's add a node counter to the backtrack function to limit search.
        # We'll create a closure or pass a state object.
        
        state = {'nodes': 0}
        
        def backtrack_limited(edge_idx):
            state['nodes'] += 1
            if state['nodes'] > 500000: # Limit nodes to avoid hanging
                return False
            if time.time() - start_time > time_limit - 2:
                return False

            if edge_idx == 120:
                return True
                
            e_idx = list(range(120))[edge_idx] # Precompute this outside
            
            for c in range(k):
                coloring[e_idx] = c
                
                valid = True
                for k_idx in edge_to_k4s[e_idx]:
                    if not is_feasible(coloring, k_idx, k):
                        valid = False
                        break
                        
                if valid:
                    if backtrack_limited(edge_idx + 1):
                        return True
                        
                coloring[e_idx] = -1
                
            return False

        coloring = [-1] * 120
        # Precompute edge order outside to avoid list creation in recursion?
        # edge_order is static.
        
        # Re-defining backtrack with access to coloring and k
        # The above nested function captures 'coloring' and 'k' and 'start_time'
        
        # We need to reset coloring for each k attempt
        coloring = [-1] * 120
        edge_order = list(range(120))
        
        def bt(edge_idx):
            state['nodes'] += 1
            if state['nodes'] > 200000:
                return False
            if time.time() - start_time > time_limit - 2:
                return False

            if edge_idx == 120:
                return True
                
            e_idx = edge_order[edge_idx]
            
            # Heuristic: Randomize color order? 
            # For determinism and speed, maybe try 0,1,2...
            # But symmetry breaking: 
            # We can fix the first edge to color 0.
            # We can fix the second edge to color 0 or 1.
            # This reduces search space significantly.
            
            # Simple symmetry breaking:
            # Edge 0 (0,1) must be 0.
            # Edge 1 (0,2) must be 0 or 1.
            # ...
            
            # Let's implement basic symmetry breaking.
            # If edge_idx == 0: only try color 0
            # If edge_idx == 1: try colors 0, 1 (but 0 is symmetric to 0, 1 is new)
            # Actually, just fixing the first few edges helps.
            
            if edge_idx == 0:
                colors_to_try = [0]
            elif edge_idx == 1:
                colors_to_try = [0, 1]
            elif edge_idx == 2:
                # Edge (0,3). Can be 0, 1, 2
                colors_to_try = [0, 1, 2]
            else:
                colors_to_try = list(range(k))
                
            for c in colors_to_try:
                coloring[e_idx] = c
                
                valid = True
                for k_idx in edge_to_k4s[e_idx]:
                    if not is_feasible(coloring, k_idx, k):
                        valid = False
                        break
                        
                if valid:
                    if bt(edge_idx + 1):
                        return True
                        
                coloring[e_idx] = -1
                
            return False

        if bt(0):
            # Verify solution
            # Check all K4s
            for k_idx in range(len(k4s)):
                c_set = set(coloring[e] for e in k4s[k_idx])
                if len(c_set) < 5:
                    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
            return coloring

    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
