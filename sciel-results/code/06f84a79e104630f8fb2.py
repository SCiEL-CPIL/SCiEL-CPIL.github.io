import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    time_limit = 87.0  # 3s margin for 90s hard limit

    N = 256

    # Step 1: Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)
    # 1 has no prime factors

    # Step 2: Build conflict graph
    # Vertices: integers 1..256
    # Edge between u and v if gcd(u,v) > 1
    # For maximum weight independent set, we want no two selected to conflict
    # We'll use bitmask representation for the conflict graph
    # Vertex i (1..256) -> index (i-1) in 0..255
    
    # For efficiency, precompute conflict bitmasks
    # conflict_mask[i] = bitmask of all j such that gcd(i,j) > 1
    # Since 256 bits doesn't fit in a single int easily in Python,
    # we'll use a list of Python ints (arbitrary precision)
    
    conflict_mask = [0] * (N + 1)  # 1-indexed
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and math.gcd(i, j) > 1:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    # Step 3: Greedy coloring for upper bound
    # Color the graph so that independent sets get distinct colors
    # Upper bound for a subset of vertices = sum of max weights in each color class
    
    def greedy_color(vertices):
        """
        vertices: list of vertex labels (1..256)
        Returns: list of (color, max_weight_in_color) for each color
        """
        color_class = {}  # color -> set of vertices
        vertex_color = {}  # vertex -> color
        next_color = 0
        
        # Order vertices by degree (higher first) for better coloring
        ordered = sorted(vertices, key=lambda v: -bin(conflict_mask[v]).count('1'))
        
        for v in ordered:
            used_colors = set()
            mask_v = conflict_mask[v]
            for u in vertices:
                if u != v and (mask_v >> (u - 1)) & 1:
                    if u in vertex_color:
                        used_colors.add(vertex_color[u])
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            vertex_color[v] = c
            if c not in color_class:
                color_class[c] = 0
            color_class[c] += v  # Add weight (weight = value)
            next_color = max(next_color, c + 1)
        
        return color_class  # dict: color -> sum of weights in that color

    # Step 4: Branch and bound for maximum weight independent set
    # We'll use a recursive approach with memoization where possible
    
    # For the B&B, we work with a set of candidate vertices
    # Upper bound: greedy coloring bound
    
    best_sum = [0]
    best_set = [set()]
    
    def compute_upper_bound(candidates):
        """Compute upper bound using greedy coloring."""
        if not candidates:
            return 0
        color_weights = greedy_color(candidates)
        # Upper bound = sum of max weights in each color class
        # Actually, for MWIS, the upper bound from coloring is:
        # sum over colors of (max weight in that color class)
        # But since we're adding weights, it's sum of all weights in color classes
        # Wait, the coloring bound: vertices in same color form independent sets? No.
        # In graph coloring, vertices in same color class are NOT necessarily independent.
        # The bound is: for each color, the max weight vertex can be in the independent set
        # from that color, so upper bound = sum of max weights per color.
        
        # Actually, I need to be more careful. Let me use a simpler bound.
        # Simple upper bound: sum of all candidate weights (trivial)
        # Better: use the fact that in an independent set, no two adjacent vertices are chosen.
        # Greedy coloring bound: if we color the graph with k colors, then an independent set
        # can contain at most one vertex from each color class's "conflict group"
        
        # Let me use a different approach: the upper bound is simply the sum of weights
        # of all candidates, which is always valid but weak.
        
        # Better approach: use the "degree-based" bound or the "Lagrangean" relaxation.
        
        # For now, let's use a simple but effective bound:
        # Sort candidates by weight descending, and greedily pick non-conflicting ones.
        # This gives a lower bound (feasible solution), not upper bound.
        
        # For upper bound, I'll use: sum of weights of candidates that are not "dominated"
        # A vertex v is dominated by u if weight[u] >= weight[v] and N(u) ⊇ N(v) ∪ {v}
        # (u conflicts with everything v conflicts with, plus v itself)
        
        # Simpler upper bound: just sum all weights.
        total = sum(candidates)
        return total
    
    def branch_and_bound(candidates, current_set, current_sum, deadline):
        """
        candidates: set of vertices that could still be added
        current_set: set of vertices already selected
        current_sum: sum of weights in current_set
        """
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(current_set)
            return
        
        # Upper bound
        ub = current_sum + sum(candidates)
        if ub <= best_sum[0]:
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick the vertex with highest weight / degree ratio
        best_vertex = None
        best_ratio = -1
        for v in candidates:
            degree = bin(conflict_mask[v] & 0).count('1')  # This is wrong, need to count conflicts within candidates
            # Actually, let's just pick by weight
            if best_vertex is None or v > best_vertex:
                best_vertex = v
                best_ratio = v
        
        v = best_vertex
        w = v
        
        # Branch 1: include v
        new_candidates = set()
        mask_v = conflict_mask[v]
        for u in candidates:
            if u != v and ((mask_v >> (u - 1)) & 1):
                continue
            new_candidates.add(u)
        
        branch_and_bound(new_candidates, current_set | {v}, current_sum + w, deadline)
        
        # Branch 2: exclude v
        candidates.discard(v)
        branch_and_bound(candidates, current_set, current_sum, deadline)
        candidates.add(v)
    
    # Better approach: use a more efficient B&B with proper upper bounds
    
    # Let me redesign with a cleaner B&B that uses the graph structure better
    
    # For 256 vertices, we can represent the candidate set as a Python int bitmask
    # vertex i (1-indexed) -> bit (i-1)
    
    # Precompute for each vertex its conflict mask restricted to 1..256
    # Already have conflict_mask[i] as a Python int
    
    # For the B&B, we maintain:
    # - remaining: bitmask of vertices still available
    # - selected: set of selected vertices
    # - current_sum
    
    # Upper bound: sum of weights in remaining (weak but simple)
    # Better: use a greedy coloring upper bound on the subgraph induced by remaining
    
    # Let me implement a proper greedy coloring upper bound
    
    def coloring_upper_bound(remaining_mask):
        """
        remaining_mask: Python int where bit i is set if vertex (i+1) is in the set
        Returns upper bound on MWIS of the induced subgraph.
        """
        if remaining_mask == 0:
            return 0
        
        # Get list of vertices
        vertices = []
        m = remaining_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            vertices.append(idx + 1)  # convert to 1-indexed
            m &= m - 1
        
        if not vertices:
            return 0
        
        # Greedy coloring
        vertex_order = sorted(vertices, key=lambda v: -bin(conflict_mask[v]).count('1'))
        vertex_color = {}
        color_max_weight = {}
        
        for v in vertex_order:
            used_colors = set()
            mask_v = conflict_mask[v]
            for u in vertices:
                if u != v and ((mask_v >> (u - 1)) & 1):
                    if u in vertex_color:
                        used_colors.add(vertex_color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            vertex_color[v] = c
            if c not in color_max_weight:
                color_max_weight[c] = 0
            color_max_weight[c] = max(color_max_weight[c], v)
        
        # Upper bound: sum of max weights in each color class
        ub = sum(color_max_weight.values())
        return ub
    
    # Now, a cleaner B&B
    
    def bbb_search(remaining_mask, selected_set, current_sum, deadline):
        """
        remaining_mask: bitmask of available vertices
        selected_set: set of selected vertices  
        current_sum: sum of selected weights
        """
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(selected_set)
            return
        
        # Upper bound
        ub = current_sum + coloring_upper_bound(remaining_mask)
        if ub <= best_sum[0]:
            return
        
        # Choose branch vertex: highest weight available
        m = remaining_mask
        best_v = None
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            v = idx + 1
            if best_v is None or v > best_v:
                best_v = v
            m &= m - 1
        
        v = best_v
        v_bit = 1 << (v - 1)
        
        # Branch 1: include v
        # Remove v and all its neighbors from remaining
        new_remaining = remaining_mask & ~v_bit & ~conflict_mask[v]
        bbb_search(new_remaining, selected_set | {v}, current_sum + v, deadline)
        
        # Branch 2: exclude v
        new_remaining = remaining_mask & ~v_bit
        bbb_search(new_remaining, selected_set, current_sum, deadline)
    
    # Step 5: Initial solution and search
    
    # Start with a good initial solution: greedy by weight
    def greedy_initial():
        selected = set()
        remaining_mask = (1 << N) - 1  # all vertices 1..256
        # Vertex i (1-indexed) is bit (i-1)
        
        while remaining_mask:
            # Find highest weight vertex that doesn't conflict with selected
            m = remaining_mask
            best_v = None
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                v = idx + 1
                # Check if v conflicts with any in selected
                conflict = False
                mask_v = conflict_mask[v]
                for u in selected:
                    if (mask_v >> (u - 1)) & 1:
                        conflict = True
                        break
                if not conflict:
                    if best_v is None or v > best_v:
                        best_v = v
                m &= m - 1
            
            if best_v is None:
                break
            
            selected.add(best_v)
            remaining_mask &= ~(1 << (best_v - 1)) & ~conflict_mask[best_v]
        
        return selected, sum(selected)
    
    # Get initial solution
    init_set, init_sum = greedy_initial()
    best_sum[0] = init_sum
    best_set[0] = set(init_set)
    
    # Now run B&B with a deadline
    deadline = start_time + time_limit - 5
    all_mask = (1 << N) - 1
    
    # Run B&B from the full set
    bbb_search(all_mask, set(), 0, deadline)
    
    # If time remains, try some local improvements on the best solution found
    if time.time() < deadline:
        # Local search: try to swap or add
        def try_local_improve(s, s_sum, dl):
            improved = True
            while improved and time.time() < dl:
                improved = False
                # Try adding
                for v in range(1, N + 1):
                    if v in s:
                        continue
                    ok = True
                    mask_v = conflict_mask[v]
                    for u in s:
                        if (mask_v >> (u - 1)) & 1:
                            ok = False
                            break
                    if ok:
                        s.add(v)
                        s_sum += v
                        improved = True
                        break
                
                # Try swapping: remove one, add one bigger
                if not improved:
                    for u in list(s):
                        for v in range(u + 1, N + 1):
                            if v in s:
                                continue
                            temp = s - {u}
                            ok = True
                            mask_v = conflict_mask[v]
                            for w in temp:
                                if (mask_v >> (w - 1)) & 1:
                                    ok = False
                                    break
                            if ok:
                                s.discard(u)
                                s.add(v)
                                s_sum += v - u
                                improved = True
                                break
                        if improved:
                            break
            return s, s_sum
        
        best_set[0], best_sum[0] = try_local_improve(set(best_set[0]), best_sum[0], deadline)
    
    result = [int(x) for x in best_set[0]]
    return result
