import math
import time

def construct():
    deadline = time.time() + 85  # safety margin under 90s kill
    
    n = 256
    
    # Precompute gcd relationships
    # conflict[i][j] = True if gcd(i,j) > 1
    # We'll build the graph of conflicts
    
    # First, let's determine which numbers are "useful"
    # A number a conflicts with b if gcd(a,b) > 1
    # Two numbers are coprime if they share no prime factors
    
    # Build prime factor sets for each number
    factor_sets = [set() for _ in range(n + 1)]
    for num in range(2, n + 1):
        x = num
        d = 2
        while d * d <= x:
            if x % d == 0:
                factor_sets[num].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            factor_sets[num].add(x)
    
    # Build adjacency: for each number, which other numbers conflict with it
    # conflict_mask[i] is a set of numbers that conflict with i
    conflict_mask = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                conflict_mask[i].add(j)
                conflict_mask[j].add(i)
    
    # For maximum weight independent set, we want to select vertices
    # such that no two selected are adjacent (i.e., no two conflict)
    # This is equivalent to: selected set must be pairwise coprime
    
    # Branch and bound with greedy coloring upper bound
    
    # Order vertices by weight (value) descending for better pruning
    # Vertex i has weight i
    
    # We'll use a recursive branch and bound
    # State: available vertices, current sum, current selection
    
    # For the coloring bound: color the subgraph induced by available vertices
    # The number of colors gives an upper bound on the size of an independent set
    # But we want weighted, so we use a weighted coloring bound
    
    # Actually, for maximum weight independent set, a common approach:
    # Sort available vertices by weight descending
    # Use a greedy coloring to get an upper bound
    
    # Let's implement a standard branch and bound for MWIS
    
    # Pre-sort vertices by weight descending
    vertices = list(range(1, n + 1))
    vertices.sort(key=lambda v: -v)
    
    # For each vertex, precompute its non-neighbors (compatible vertices)
    # compatible[i] = set of vertices j where gcd(i,j)==1 and j != i
    compatible = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and math.gcd(i, j) == 1:
                compatible[i].add(j)
    
    # Branch and bound
    best_sum = 0
    best_set = set()
    
    def bound(available, current_sum):
        """
        Compute an upper bound on the total sum we can achieve.
        Use greedy coloring on the conflict graph induced by available vertices.
        """
        if not available:
            return current_sum
        
        # Greedy coloring: assign colors to vertices in available
        # Vertices with same color don't conflict with each other
        # Upper bound: for each color class, we can pick at most one vertex
        # So upper bound = sum of max weight in each color class + current_sum
        
        # To make the bound tighter, sort available by weight descending
        avail_sorted = sorted(available, key=lambda v: -v)
        
        # Greedy coloring
        color_of = {}
        next_color = 0
        for v in avail_sorted:
            # Find the smallest color not used by any neighbor of v in available
            used_colors = set()
            for u in conflict_mask[v]:
                if u in available and u in color_of:
                    used_colors.add(color_of[u])
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        
        # Upper bound: sum of max weight per color class
        color_max = {}
        for v, c in color_of.items():
            if c not in color_max or v > color_max[c]:
                color_max[c] = v
        
        return current_sum + sum(color_max.values())
    
    def branch_and_bound(available, current_sum, current_set):
        nonlocal best_sum, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if not available:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Prune if upper bound doesn't beat best
        ub = bound(available, current_sum)
        if ub <= best_sum:
            return
        
        # Pick a vertex to branch on
        # Choose the vertex with highest weight that is in available
        v = max(available, key=lambda x: x)
        
        # Option 1: include v
        new_available = available - conflict_mask[v] - {v}
        new_set = current_set | {v}
        branch_and_bound(new_available, current_sum + v, new_set)
        
        # Option 2: exclude v
        new_available = available - {v}
        branch_and_bound(new_available, current_sum, current_set)
    
    # Initial available set
    initial_available = set(range(1, n + 1))
    branch_and_bound(initial_available, 0, set())
    
    # Verify the result
    result = list(best_set)
    result.sort()
    
    # Safety check
    if not result:
        result = [1]
    
    # Verify pairwise coprime
    def verify(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    if not verify(result):
        # Fallback: just use 1
        result = [1]
    
    return result
