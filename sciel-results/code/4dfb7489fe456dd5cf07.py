import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 256
    
    # Precompute conflict graph: two numbers conflict if they can form a 3-AP
    # with any third number in [1, n]
    # A 3-AP is (a, b, c) where b-a = c-b, i.e., a+c = 2b
    # So x and y conflict if there exists z in [1,n] such that {x,y,z} forms a 3-AP
    # This means: 2x = y+z, or 2y = x+z, or 2z = x+y for some z
    
    # Build adjacency: conflict[x][y] = True if x,y conflict
    # Two numbers x,y conflict if:
    # - There exists z such that 2x = y+z (x is middle) -> z = 2x-y, need 1<=z<=n, z!=x, z!=y
    # - There exists z such that 2y = x+z (y is middle) -> z = 2y-x, need 1<=z<=n, z!=x, z!=y  
    # - There exists z such that 2z = x+y (z is middle) -> z = (x+y)/2, need z integer, 1<=z<=n, z!=x, z!=y
    
    conflict = [[False] * (n+1) for _ in range(n+1)]
    for x in range(1, n+1):
        for y in range(x+1, n+1):
            # Check if x,y can be in a 3-AP with some z
            # Case 1: x is middle, z = 2x - y
            z = 2*x - y
            if 1 <= z <= n and z != x and z != y:
                conflict[x][y] = True
            # Case 2: y is middle, z = 2y - x
            z = 2*y - x
            if 1 <= z <= n and z != x and z != y:
                conflict[x][y] = True
            # Case 3: z is middle, z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= n and z != x and z != y:
                    conflict[x][y] = True
            conflict[y][x] = conflict[x][y]
    
    # Now solve Maximum Weight Independent Set
    # Weight of vertex x is 1/x
    # We want to find independent set S maximizing sum_{x in S} 1/x
    
    # Use branch and bound with good ordering
    # Sort vertices by weight descending (small numbers first since 1/x is larger)
    # But also consider degree - high degree vertices are more constraining
    
    # For MWIS, a good heuristic is to process vertices in order of decreasing weight/degree ratio
    # Or simply by decreasing weight
    
    # Let's use a recursive branch and bound approach
    # State: current set S, candidate set C (vertices not yet decided)
    # Upper bound: sum of weights of all candidates that can be added
    
    # Precompute for each vertex its neighbors
    neighbors = [set() for _ in range(n+1)]
    for x in range(1, n+1):
        for y in range(x+1, n+1):
            if conflict[x][y]:
                neighbors[x].add(y)
                neighbors[y].add(x)
    
    # Weight of each vertex
    weight = [0.0] * (n+1)
    for x in range(1, n+1):
        weight[x] = 1.0 / x
    
    # Branch and bound
    # We'll use a recursive approach with memoization on the candidate set (as a bitmask)
    # But 256 bits is too large for direct memoization. Instead, we'll use a different approach.
    
    # Alternative: Use a greedy + local search approach on the conflict graph
    # 1. Greedy: add vertices in order of decreasing weight, if no conflict with already added
    # 2. Local search: try swaps to improve
    
    # But the directive says to use branch-and-bound for MWIS. Let me implement a proper B&B.
    
    # For B&B on MWIS, we can use the following:
    # - Order vertices by some criterion (e.g., decreasing weight)
    # - At each step, decide to include or exclude the next vertex
    # - Upper bound: current weight + sum of weights of all remaining vertices that don't conflict with included ones
    
    # This is exponential in worst case, but with good ordering and pruning, it can work for n=256
    # if the graph is sparse enough.
    
    # Let's try a different approach: since n=256 is moderate, we can use a MILP-style approach
    # or a more sophisticated combinatorial algorithm.
    
    # Actually, let's use a practical approach:
    # 1. Start with a good initial solution (greedy)
    # 2. Use local search (swap-based) to improve
    # 3. The conflict graph formulation makes the validity check O(1) per pair
    
    def can_add(x, S):
        """Check if x can be added to S without creating a conflict"""
        for y in S:
            if conflict[x][y]:
                return False
        return True
    
    def get_score(S):
        return sum(weight[x] for x in S)
    
    # Greedy initialization
    def greedy_construct():
        selected = set()
        # Process vertices in order of decreasing weight (i.e., increasing x)
        for x in range(1, n+1):
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    best_set = greedy_construct()
    best_score = get_score(best_set)
    
    # Local search: try swaps
    while time.time() < deadline:
        improved = False
        current_set = set(best_set)
        current_score = best_score
        
        # Try all possible swaps: remove u from S, add v not in S
        # Gain = weight[v] - weight[u] = 1/v - 1/u
        # We want v < u for positive gain (since 1/v > 1/u when v < u)
        
        S_list = sorted(current_set, key=lambda x: x)  # increasing x, so decreasing weight
        NotS_list = sorted([x for x in range(1, n+1) if x not in current_set], key=lambda x: x)
        
        found_improvement = False
        best_swap_gain = 0
        best_swap_u = None
        best_swap_v = None
        
        # Try swaps where v < u (potential gain)
        for u in S_list:
            if found_improvement:
                break
            for v in NotS_list:
                if v >= u:
                    break  # No more potential gains for this u
                # Check if (S \ {u}) ∪ {v} is valid
                # v must not conflict with any element in S \ {u}
                valid = True
                for y in current_set:
                    if y == u:
                        continue
                    if conflict[v][y]:
                        valid = False
                        break
                if valid:
                    gain = weight[v] - weight[u]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_u = u
                        best_swap_v = v
                        found_improvement = True
        
        if found_improvement:
            best_set.remove(best_swap_u)
            best_set.add(best_swap_v)
            best_score += best_swap_gain
            improved = True
        else:
            # Try additions
            added = False
            for v in NotS_list:
                if can_add(v, best_set):
                    best_set.add(v)
                    best_score += weight[v]
                    added = True
                    break
            if not added:
                # Try removals to enable better swaps
                # Or just break
                break
    
    return sorted(best_set)
