"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    N = 256
    numbers = list(range(1, N + 1))
    n = len(numbers)
    
    # Precompute GCD matrix and conflict graph
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # conflict[i][j] = True if numbers[i] and numbers[j] are NOT coprime
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(numbers[i], numbers[j]) != 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Weights are the numbers themselves
    weights = numbers
    
    # Greedy coloring for upper bound
    def greedy_color(vertices):
        """Color vertices such that adjacent vertices have different colors.
        Returns list of color assignments (0-indexed colors)."""
        colors = [-1] * n
        color_count = 0
        # Order by degree descending for better coloring
        order = sorted(vertices, key=lambda v: sum(conflict[v][u] for u in vertices if u != v), reverse=True)
        for v in order:
            used_colors = set()
            for u in vertices:
                if u != v and conflict[v][u] and colors[u] != -1:
                    used_colors.add(colors[u])
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            color_count = max(color_count, c + 1)
        return colors, color_count
    
    # Branch and bound for Maximum Weight Independent Set
    best_sum = [0]
    best_set = [set()]
    
    def bb_solve(remaining, current_sum, current_set, depth=0):
        """
        remaining: list of indices that can still be added
        current_sum: sum of weights in current_set
        current_set: set of indices in current solution
        """
        if time.time() > deadline:
            return
        
        if not remaining:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current_sum + sum of all remaining weights
        # But we can do better with coloring
        if len(remaining) <= 1:
            # Check if we can add the last vertex
            if len(remaining) == 1:
                v = remaining[0]
                # Check if v conflicts with any in current_set
                can_add = True
                for u in current_set:
                    if conflict[v][u]:
                        can_add = False
                        break
                if can_add:
                    new_sum = current_sum + weights[v]
                    if new_sum > best_sum[0]:
                        best_sum[0] = new_sum
                        best_set[0] = set(current_set | {v})
                else:
                    if current_sum > best_sum[0]:
                        best_sum[0] = current_sum
                        best_set[0] = set(current_set)
            else:
                if current_sum > best_sum[0]:
                    best_sum[0] = current_sum
                    best_set[0] = set(current_set)
            return
        
        # Greedy coloring upper bound
        colors, num_colors = greedy_color(remaining)
        # Upper bound: for each color class, we can take at most one vertex
        # So upper bound = current_sum + sum over color classes of max weight in that class
        color_max = [0] * num_colors
        for v in remaining:
            c = colors[v]
            color_max[c] = max(color_max[c], weights[v])
        upper_bound = current_sum + sum(color_max)
        
        if upper_bound <= best_sum[0]:
            return
        
        # Choose vertex to branch on: pick vertex with highest weight among remaining
        # that has conflicts with other remaining vertices
        v = max(remaining, key=lambda x: weights[x])
        
        # Option 1: Include v
        # Find all vertices in remaining that conflict with v
        conflicting = [u for u in remaining if u != v and conflict[v][u]]
        new_remaining = [u for u in remaining if u != v and u not in set(conflicting)]
        
        # Check if v conflicts with any in current_set
        can_add_v = True
        for u in current_set:
            if conflict[v][u]:
                can_add_v = False
                break
        
        if can_add_v:
            bb_solve(new_remaining, current_sum + weights[v], current_set | {v}, depth + 1)
        
        # Option 2: Exclude v
        bb_solve([u for u in remaining if u != v], current_sum, current_set, depth + 1)
    
    # Initial solution: greedy - pick numbers in order, adding if no conflict
    greedy_set = set()
    greedy_sum = 0
    # Sort by weight descending for better greedy
    for i in sorted(range(n), key=lambda x: -weights[x]):
        can_add = True
        for u in greedy_set:
            if conflict[i][u]:
                can_add = False
                break
        if can_add:
            greedy_set.add(i)
            greedy_sum += weights[i]
    
    best_sum[0] = greedy_sum
    best_set[0] = set(greedy_set)
    
    # Run branch and bound
    all_indices = list(range(n))
    bb_solve(all_indices, 0, set())
    
    # Convert to list of Python ints
    result = [weights[i] for i in best_set[0]]
    result.sort()
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
