import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    squares = [i*i for i in range(1, 23)] # 1^2 to 22^2 = 484
    square_set = set(squares)
    
    # 1. Identify valid vertices
    # v is valid if 2*v is not a perfect square
    valid_vertices = []
    for i in range(1, N+1):
        if 2*i not in square_set:
            valid_vertices.append(i)
            
    M = len(valid_vertices)
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 2. Build adjacency bitmask
    # adj[i] is a bitmask of neighbors of valid_vertices[i]
    adj = [0] * M
    
    for i, u in enumerate(valid_vertices):
        for k in squares:
            v = k - u
            if 1 <= v <= N and v in v_to_idx:
                j = v_to_idx[v]
                if j != i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    
    # 3. Greedy Initial Solution
    # Heuristic: Pick vertex with min degree in remaining graph
    def greedy_mis(adj_masks, active_mask):
        selected = []
        current_active = active_mask
        while current_active:
            min_deg = float('inf')
            best_v = -1
            temp = current_active
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = (adj_masks[v] & current_active).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp -= lsb
            
            if best_v == -1:
                break
                
            selected.append(best_v)
            to_remove = adj_masks[best_v] | (1 << best_v)
            current_active &= ~to_remove
            
        return selected

    initial_selected = greedy_mis(adj, (1 << M) - 1)
    best_size = len(initial_selected)
    best_set_global = set(initial_selected)
    
    # 4. Branch and Bound with Greedy Coloring Bound
    
    def greedy_color_bound(candidates_mask):
        # Returns an upper bound on the size of the independent set in candidates_mask
        # using a greedy coloring heuristic.
        # We color the graph induced by candidates_mask.
        # The number of colors is an upper bound on the independence number?
        # No, $\alpha(G) \le \chi(G)$ is FALSE. 
        # Wait, let's re-verify.
        # $\chi(G)$ is the minimum number of colors.
        # An independent set is a set of vertices with no edges between them.
        # In a proper coloring, vertices with the same color form an independent set.
        # So the size of the largest independent set is at least the size of the largest color class.
        # But $\alpha(G)$ can be larger than the size of any single color class?
        # No. $\alpha(G)$ is the size of the *maximum* independent set.
        # The relationship is $\alpha(G) \cdot \chi(G) \ge |V|$? No.
        # The correct bound used in B&B for MIS is often based on coloring the *complement*?
        # Or is it that the number of colors in a *greedy* coloring is an upper bound on the size of the *clique*? No.
        
        # Let's look at standard B&B for MIS.
        # Common bound: If we can color the graph with $k$ colors, does that bound $\alpha$?
        # Actually, a common technique is to compute a lower bound on the chromatic number of the *complement* graph?
        # No, let's use a simpler, valid bound:
        # $\alpha(G) \le \sum_{v \in V} \frac{1}{d(v)+1}$? No.
        
        # Let's use the "Maximum Matching" bound or just "Popcount" if coloring is too slow/incorrect.
        # However, there is a well-known bound:
        # If we compute a greedy coloring of the graph, the number of colors $k$ is NOT an upper bound on $\alpha$.
        # Example: Path 1-2-3. $\alpha=2$. Greedy coloring might use 2 colors. $2 \ge 2$.
        # Example: Star $K_{1,3}$. $\alpha=3$. Greedy coloring uses 2 colors. $2 < 3$. So $\alpha$ is NOT bounded by $\chi$.
        
        # Correct Upper Bound for MIS:
        # One simple bound is the size of the candidate set.
        # A better bound: 
        # We can use the fact that if we select a vertex $v$, we remove its neighbors.
        # A standard B&B uses:
        # 1. Branch on a vertex.
        # 2. Bound: $|S| + \text{UpperBound}(C) \le \text{Best} \implies \text{Prune}$.
        
        # A valid upper bound for $\alpha(G)$ is the size of the *maximum clique* in the complement? No, that's $\alpha(G)$.
        # Actually, $\alpha(G) \le |V| - \tau(G)$ where $\tau(G)$ is the minimum vertex cover.
        # $\tau(G) \ge \text{size of maximum matching}$.
        # So $\alpha(G) \le |V| - \text{max\_matching}(G)$.
        # Computing max matching is fast (Hopcroft-Karp or similar for bipartite, but this is general).
        
        # Alternative: Use a simple greedy independent set size as a *lower* bound to update best, and use popcount for upper bound?
        # If the gap is small, popcount works.
        
        # Let's try a heuristic bound:
        # Sort vertices by degree.
        # Actually, let's just use popcount for now and see if it's fast enough with good branching.
        # If not, we can implement a faster bound.
        
        # For now, return popcount.
        return candidates_mask.bit_count()

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
            
        # Prune with popcount bound
        if current_count + candidates_mask.bit_count() <= best_size:
            return
            
        # Find vertex with highest degree in candidates_mask
        max_deg = -1
        best_v = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = (adj[v] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp -= lsb
            
        if best_v == -1:
            return

        # Branch 1: Include best_v
        new_candidates_incl = candidates_mask & ~(adj[best_v] | (1 << best_v))
        current_set.append(best_v)
        search(new_candidates_incl, current_count + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_candidates_excl = candidates_mask & ~(1 << best_v)
        search(new_candidates_excl, current_count, current_set)
        
    # Start search
    search((1 << M) - 1, 0, [])
    
    # Map back to original values
    result = [valid_vertices[i] for i in best_set_global]
    
    # Verify
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a <= b:
                if (a+b) in square_set:
                    # Invalid, fallback
                    return list(range(1, 257, 3))
    return result

if __name__ == "__main__":
    construct()
