import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    # 1. Build Graph
    # squares up to 2*256 = 512
    squares = [i*i for i in range(1, 23)] # 1^2 to 22^2 = 484
    
    # adjacency list
    # adj[i] is a set of neighbors of i
    # Also, we need to handle diagonal: 2*i is square => i cannot be in set?
    # Wait, if a=b, a+b=2a. If 2a is square, then a cannot be in S.
    # So we can remove such vertices from consideration entirely.
    
    valid_vertices = []
    for i in range(1, N+1):
        if 2*i not in squares:
            valid_vertices.append(i)
            
    # Map valid vertices to indices 0..M-1
    M = len(valid_vertices)
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # Build adjacency bitmask for each valid vertex
    # adj_mask[i] has bits set for neighbors j
    adj_mask = [0] * M
    
    for i, u in enumerate(valid_vertices):
        for k in squares:
            v = k - u
            if 1 <= v <= N and v in v_to_idx:
                j = v_to_idx[v]
                if j > i: # add edge only once to symmetric mask? No, mask is per node.
                    # We want adj_mask[i] to contain j.
                    # And adj_mask[j] to contain i.
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
                    
    # Note: The loop above adds edges for all squares.
    # Since we iterate all u and all squares, we cover all pairs.
    # We should ensure we don't add self-loops (u==v), which is handled by 2u not in squares check?
    # If 2u is square, u is not in valid_vertices. So no self-loop.
    
    # 2. Greedy Initial Solution
    # Order vertices by degree (ascending) to pick isolated/low-degree ones first?
    # Or descending to eliminate conflicts fast?
    # Standard greedy for MIS: pick vertex, remove it and neighbors.
    # To maximize size, picking vertices with low degree in the remaining graph is often good.
    
    def greedy_mis(adj_masks, active_mask):
        # active_mask: bits set for vertices available
        selected = []
        current_active = active_mask
        while current_active:
            # Find vertex with minimum degree in current_active subgraph
            min_deg = float('inf')
            best_v = -1
            # Iterate through bits in current_active
            temp = current_active
            while temp:
                # Get lowest set bit
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                # Count neighbors in current_active
                deg = (adj_masks[v] & current_active).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp -= lsb
            
            if best_v == -1:
                break
                
            selected.append(best_v)
            # Remove best_v and its neighbors from current_active
            # Neighbors of best_v are adj_masks[best_v]
            # We also remove best_v itself
            to_remove = adj_masks[best_v] | (1 << best_v)
            current_active &= ~to_remove
            
        return selected

    initial_selected = greedy_mis(adj_mask, (1 << M) - 1)
    best_size = len(initial_selected)
    best_set = set(initial_selected)
    
    # 3. Branch and Bound
    # We will use a recursive search.
    # State: (candidates_mask, current_count)
    # Upper bound: current_count + greedy_coloring_bound(candidates_mask)
    
    def greedy_color_bound(candidates_mask):
        # Returns an upper bound on the size of MIS in the subgraph induced by candidates_mask
        # Using greedy coloring.
        # The number of colors needed is an upper bound on the independence number?
        # No, the independence number is <= chromatic number.
        # So if we color the graph, the number of colors is an upper bound on the size of the largest independent set?
        # Wait. In a colored graph, an independent set can have at most one vertex of each color?
        # No, an independent set can have multiple vertices of the same color if they are not adjacent?
        # Definition of coloring: adjacent vertices have different colors.
        # So an independent set (no adjacent vertices) CAN have multiple vertices of the same color.
        # Example: Path 1-2-3. Colors: 1(R), 2(G), 3(R).
        # Independent set {1,3} has size 2. Colors used: R, G. Number of colors 2.
        # Independent set {2} has size 1.
        # Max Independent Set size is 2. Chromatic number is 2.
        # Is it always true that $\alpha(G) \le \chi(G)$?
        # Yes. Because if you have an independent set $I$, no two vertices in $I$ are adjacent.
        # In a proper coloring, all vertices in $I$ must have distinct colors?
        # No! If $u, v \in I$, they are not adjacent. They CAN have the same color.
        # So $\alpha(G)$ is NOT bounded by $\chi(G)$ in the sense of "number of colors used".
        # Actually, $\alpha(G) \ge \chi(G)$ is false. $\chi(G) \ge \alpha(G)$ is false.
        # Counter example: Triangle. $\alpha=1, \chi=3$.
        # Example: $K_{n,n}$. $\alpha=n, \chi=2$.
        # So coloring bound is NOT an upper bound on $\alpha$.
        # I was confused. The bound is: $\alpha(G) \le \text{something}$.
        # A common upper bound is the size of the maximum clique? No, $\alpha \le n - \omega$? No.
        # $\alpha(G) \cdot \omega(G) \le n$. So $\alpha(G) \le n / \omega(G)$.
        # But finding max clique is hard.
        
        # Let's use a different bound.
        # Simple bound: number of vertices in candidates.
        # Better bound: Sum of (1 / (degree + 1))? No.
        
        # Let's stick to a simple pruning:
        # If current_count + popcount(candidates) <= best_size: prune.
        # This is weak.
        
        # Let's try to compute a better bound.
        # One heuristic bound: 
        # Sort vertices by degree.
        # But for exact B&B on N=256, we need a strong bound.
        
        # Alternative: Use the fact that the graph is sparse.
        # Maybe a simple B&B with "popcount" bound is enough if the greedy solution is close to optimal?
        # The seed score is 1.0 (size 86).
        # Greedy might find ~90-100.
        # If the gap is small, popcount bound works.
        
        # Let's implement a slightly stronger bound:
        # Compute a maximal matching? No.
        # Let's just use popcount and see if it's fast enough.
        # If not, we can add a coloring-based bound for the *complement*? No.
        
        # Actually, there is a bound: 
        # $\alpha(G) \le \sum_{v \in V} \frac{1}{d(v)+1}$ is not generally true.
        
        # Let's try to improve the greedy to get a very high lower bound.
        # If the lower bound is high, the popcount pruning is effective.
        
        return candidates_mask.bit_count()

    # To make B&B faster, we should order the branching.
    # Pick a vertex with high degree in the candidate set to branch on?
    # Or low degree?
    # If we pick a vertex $v$, we branch on:
    # 1. Include $v$: new candidates = candidates \ (N(v) \cup {v})
    # 2. Exclude $v$: new candidates = candidates \ {v}
    
    # Including $v$ usually reduces the candidate set more (removes neighbors).
    # Excluding $v$ keeps more candidates.
    # So we should explore "Include" first to find a good solution quickly?
    # Or "Exclude" to prune?
    # Standard: Explore the branch that is more likely to contain the optimum.
    # Usually, including high-degree vertices is "risky" (removes many options), so excluding them might be better for finding large sets?
    # Actually, if a vertex has high degree, it conflicts with many, so it's "hard" to include.
    # Let's branch on the vertex with the HIGHEST degree in the candidate subgraph.
    # Why? Because it has the most impact.
    
    best_set_global = set(initial_selected)
    
    def search(candidates_mask, current_count, current_set):
        nonlocal best_size, best_set_global
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_count > best_size:
                best_size = current_count
                best_set_global = set(current_set)
            return
            
        # Prune
        if current_count + candidates_mask.bit_count() <= best_size:
            return
            
        # Find vertex with highest degree in candidates_mask
        # To speed up, we can just pick the first one? No, heuristic is important.
        # Let's find max degree.
        max_deg = -1
        best_v = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj_mask[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp -= lsb
            
        if best_v == -1:
             return

        # Branch 1: Include best_v
        # New candidates: remove best_v and its neighbors
        new_candidates_incl = candidates_mask & ~(adj_mask[best_v] | (1 << best_v))
        current_set.append(best_v)
        search(new_candidates_incl, current_count + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_candidates_excl = candidates_mask & ~(1 << best_v)
        search(new_candidates_excl, current_count, current_set)
        
    # Start search
    # Use a list for current_set to allow backtracking
    search((1 << M) - 1, 0, [])
    
    # Map back to original values
    result = [valid_vertices[i] for i in best_set_global]
    
    # Verify
    # Just in case
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a <= b:
                if (a+b) in squares:
                    # Invalid
                    return list(range(1, 257, 3)) # Fallback
    return result
