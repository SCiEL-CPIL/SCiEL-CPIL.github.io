"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

def construct():
    """
    Find a maximum cardinality subset of [1, 256] such that a+b is never a perfect square
    for any a <= b in the set (including a=b).
    
    We model this as a maximum independent set problem on a conflict graph where:
    - Vertices: numbers 1..256
    - Edge between u and v if u+v is a perfect square (including u==v, i.e., 2u is a square)
    
    We use branch-and-bound with bitset-based operations and greedy coloring upper bounds.
    """
    N = 256
    
    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Identify which numbers can even be in the set:
    # Number x (1..256) is "self-valid" if 2*x is NOT a perfect square.
    
    self_valid = []
    for x in range(1, N + 1):
        if 2 * x not in squares:
            self_valid.append(x)
    
    m = len(self_valid)
    
    # Build adjacency bitsets: adj[i] is a bitmask where bit j is set if i and j conflict
    adj = [0] * m
    for i in range(m):
        a = self_valid[i]
        mask = 0
        for j in range(m):
            if i == j:
                continue
            b = self_valid[j]
            if a + b in squares:
                mask |= (1 << j)
        adj[i] = mask
        
    # Precompute global degrees for faster sorting in color_bound
    global_deg = [0] * m
    for i in range(m):
        global_deg[i] = adj[i].bit_count()

    # Full mask of all vertices
    full_mask = (1 << m) - 1
    
    # Greedy initial solution to get a good lower bound
    def greedy_solution(candidates_mask):
        """Greedy: repeatedly pick vertex with min degree, add to solution, remove it and neighbors."""
        sol = []
        remaining = candidates_mask
        while remaining:
            # Pick vertex with lowest degree in remaining subgraph
            best_v = -1
            best_deg = m + 1
            for v in range(m):
                if (remaining >> v) & 1:
                    deg = (remaining & adj[v]).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            sol.append(best_v)
            # Remove best_v and all its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return sol
    
    initial_sol = greedy_solution(full_mask)
    
    # Local search to improve the initial solution
    def local_search(sol):
        """
        Try to improve the solution by swapping or adding vertices.
        Returns a better solution.
        """
        sol_set = set(sol)
        improved = True
        best_size = len(sol)
        best_set = list(sol)
        
        while improved:
            improved = False
            # Try to add a vertex not in sol
            for v in range(m):
                if v in sol_set:
                    continue
                # Check conflicts with sol
                conflicts = [u for u in sol_set if (adj[v] >> u) & 1]
                if not conflicts:
                    # Can add v
                    new_sol = list(sol_set) + [v]
                    # Try to improve further? No, just add and continue search
                    sol_set.add(v)
                    improved = True
                    break # Restart loop to try more additions
            if improved:
                continue
                
            # Try to swap: remove one conflict, add v
            # This is complex. Let's just do a simple "move" heuristic?
            # Actually, just adding is a good start.
            # Let's try a 1-swap: remove u, add v.
            # If |conflicts| == 1, we can remove that one and add v. Size stays same.
            # If |conflicts| == 0, size increases.
            # We want to increase size.
            
            # If no additions possible, try to see if we can replace a vertex to allow a better configuration?
            # For now, simple greedy + add is enough for a strong start.
            
            # To be more effective, let's try all v not in sol:
            # If adding v requires removing k vertices from sol, and k < 1, we add.
            # If k == 1, size doesn't change.
            # We only care about increasing size for the bound.
            
            # Let's refine: 
            # If we can't add any single vertex, the solution is "locally maximal".
            # This is a good lower bound.
            
            break # No more simple additions possible
            
        return list(sol_set)

    improved_sol = local_search(initial_sol)
    best_size = len(improved_sol)
    best_set = list(improved_sol)
    
    # Greedy coloring bound on subgraph induced by mask
    def color_bound(mask):
        """
        Compute a greedy coloring upper bound on the maximum independent set size
        for the subgraph induced by 'mask'.
        """
        if mask == 0:
            return 0
        
        # Get vertices in mask
        vertices = []
        temp = mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            vertices.append(v)
            temp ^= lsb
            
        if not vertices:
            return 0
            
        # Sort by global degree descending (proxy for subgraph degree)
        vertices.sort(key=lambda v: -global_deg[v])
        
        colors = {}
        color_num = 0
        for v in vertices:
            neighbor_colors = set()
            # Check neighbors in the subgraph that are already colored
            # We only need to check neighbors that are in 'vertices' and have been processed
            # Since we process in sorted order, we can just check all processed neighbors
            # To be fast, we check all u in vertices where u != v and adj[v] has u
            # But we only care if u is colored.
            
            # Optimization: Only check u if u is in the mask (it is) and u != v
            # And u is a neighbor.
            # We can iterate through the vertices list again? No.
            # We can iterate through the bits of adj[v] & mask?
            
            # Let's iterate through the colored vertices.
            # Since we are in a loop, we can maintain a list of colored vertices.
            # But simpler: just check all u in vertices.
            for u in vertices:
                if u == v:
                    continue
                if (adj[v] >> u) & 1:
                    if u in colors:
                        neighbor_colors.add(colors[u])
            
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_num:
                color_num = c + 1
        
        return color_num
    
    # Branch and bound using bitsets
    def branch_and_bound(candidates_mask, current_size, current_set):
        nonlocal best_set, best_size
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound: current_size + popcount(candidates_mask)
        if current_size + candidates_mask.bit_count() <= best_size:
            return
        
        # Get coloring bound
        cb = color_bound(candidates_mask)
        if current_size + cb <= best_size:
            return
        
        # Choose a vertex to branch on: pick one with highest degree in candidates
        best_v = -1
        best_deg = -1
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (candidates_mask & adj[v]).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        # Remove best_v and all its neighbors from candidates
        new_mask_incl = candidates_mask & ~(1 << best_v) & ~adj[best_v]
        current_set.append(best_v)
        branch_and_bound(new_mask_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: exclude best_v
        new_mask_excl = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_mask_excl, current_size, current_set)
    
    branch_and_bound(full_mask, 0, [])
    
    # Convert back to original numbers
    result = [self_valid[i] for i in best_set]
    
    # Ensure nonempty
    if not result:
        for x in range(1, N + 1):
            if 2 * x not in squares:
                return [x]
        return [1]
    
    return result

# EVOLVE-BLOCK-END
