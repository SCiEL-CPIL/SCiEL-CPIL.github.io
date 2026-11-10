import itertools
import random
import math
import time

def construct():
    """
    Return 120 integer colours for edges of K16 in lexicographic order.
    Uses a local search (Simulated Annealing) to find a valid coloring 
    with the minimum number of colors (trying 13, 14, 15, ...).
    """
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices
    # edge_id[i][j] = index in the flat array
    edge_id = [[-1] * n for _ in range(n)]
    edge_u = [0] * num_edges
    edge_v = [0] * num_edges
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_id[i][j] = idx
            edge_id[j][i] = idx
            edge_u[idx] = i
            edge_v[idx] = j
            idx += 1
            
    # Precompute K4s and their edge lists
    # k4_list[k] = list of 6 edge indices
    k4_list = []
    # Also map edge index to list of k4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    k4_idx = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [
            edge_id[a][b], edge_id[a][c], edge_id[a][d],
            edge_id[b][c], edge_id[b][d],
            edge_id[c][d]
        ]
        k4_list.append(edges)
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
        k4_idx += 1
        
    num_k4s = len(k4_list)
    
    def count_distinct_colors(edge_indices, colors):
        cols = set()
        for e in edge_indices:
            cols.add(colors[e])
        return len(cols)
        
    def is_valid(colors, C):
        # Check if all colors are in 0..C-1 and all K4s have >= 5 colors
        # This is for final verification
        if any(c >= C for c in colors):
            return False
        for edges in k4_list:
            if count_distinct_colors(edges, colors) < 5:
                return False
        return True
        
    def local_search(C, time_limit):
        """
        Try to find a valid coloring with C colors.
        Returns the coloring if found, else None.
        """
        # Initialize randomly
        colors = [random.randint(0, C - 1) for _ in range(num_edges)]
        
        # Initial cost: number of bad K4s
        bad_k4s = [0] * num_k4s
        total_violations = 0
        for k in range(num_k4s):
            if count_distinct_colors(k4_list[k], colors) < 5:
                bad_k4s[k] = 1
                total_violations += 1
                
        if total_violations == 0:
            return colors
            
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and (time.time() - start_time) < time_limit:
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            old_c = colors[e]
            
            # Calculate cost change
            delta_violations = 0
            affected_k4s = edge_to_k4s[e]
            
            # Temporarily change color to calculate new state? 
            # Or just calculate delta.
            # Let's pick a new color
            new_c = random.randint(0, C - 1)
            if new_c == old_c:
                continue
                
            # For each affected K4, check if status changes
            # We need to know the current set of colors for each K4 to update efficiently?
            # Storing the set for each K4 is memory heavy (1820 sets of 6 ints). 
            # But recalculating from scratch for 91 K4s is fast (91 * 6 ops).
            
            # Let's just recalculate the status of affected K4s with new_c
            # We need to be careful: we are evaluating a hypothetical change.
            
            # Optimization: Store the color set for each K4? 
            # Let's just recalculate.
            
            new_bad_count = 0
            # We need to know which K4s were bad before.
            # Let's iterate affected K4s.
            
            # To avoid modifying `colors` until we accept, we can 
            # just calculate the new distinct count.
            
            # However, we need to update `total_violations` and `bad_k4s` only if accepted.
            # So we calculate the potential new total_violations.
            
            potential_delta = 0
            for k in affected_k4s:
                edges = k4_list[k]
                # Get current colors, replace e with new_c
                # Since e is in edges, we can just build the set
                # We need the current colors of the other 5 edges
                # This is O(6) per K4.
                
                # To speed up, we can maintain a count of colors per K4?
                # No, distinct count is hard to update incrementally without counts.
                # Let's just recalculate.
                
                # Optimization: If we store `k4_colors[k]` as a tuple or frozenset, 
                # updating is easier. But let's stick to simple recalculation first.
                
                # Actually, we can just check:
                # Old status: bad_k4s[k]
                # New status: count_distinct with new_c
                
                # To make it faster, let's precompute the colors of the other 5 edges?
                # No, just loop.
                
                # For simplicity and correctness in this step, we just recalculate.
                # 91 K4s * 6 edges = 546 operations. Very fast.
                
                # But we do this for every step.
                # 120s / 546 ops is huge number of steps.
                
                # Let's implement the recalculation.
                pass # Placeholder
            
            # Actually, let's implement the delta calculation properly.
            # We'll create a helper that takes the edge, old_c, new_c and returns delta.
            
            # For each affected K4 k:
            # old_bad = bad_k4s[k]
            # new_colors_set = set(colors[e'] for e' in edges if e' != e)
            # new_colors_set.add(new_c)
            # new_distinct = len(new_colors_set)
            # new_bad = 1 if new_distinct < 5 else 0
            # delta += new_bad - old_bad
            
            # This requires iterating the 5 other edges.
            
            delta_violations = 0
            for k in affected_k4s:
                edges = k4_list[k]
                # Build set of colors for other 5 edges
                other_colors = set()
                for e2 in edges:
                    if e2 != e:
                        other_colors.add(colors[e2])
                
                old_bad = bad_k4s[k]
                
                # Check new status
                # The new set is other_colors U {new_c}
                # We don't need to build the full set if we just check length
                # len(other_colors U {new_c})
                # If new_c in other_colors, len is len(other_colors)
                # Else len(other_colors) + 1
                
                if new_c in other_colors:
                    new_distinct = len(other_colors)
                else:
                    new_distinct = len(other_colors) + 1
                    
                new_bad = 1 if new_distinct < 5 else 0
                
                if new_bad != old_bad:
                    delta_violations += (new_bad - old_bad)
                    
            # Acceptance
            if delta_violations < 0:
                # Always accept improvement
                accept = True
            elif delta_violations == 0:
                # Accept with some probability to escape plateaus
                accept = random.random() < 0.5
            else:
                # Accept with probability exp(-delta / T)
                accept = random.random() < math.exp(-delta_violations / T)
                
            if accept:
                colors[e] = new_c
                # Update bad_k4s and total_violations
                for k in affected_k4s:
                    edges = k4_list[k]
                    other_colors = set()
                    for e2 in edges:
                        if e2 != e:
                            other_colors.add(colors[e2])
                            
                    if new_c in other_colors:
                        new_distinct = len(other_colors)
                    else:
                        new_distinct = len(other_colors) + 1
                        
                    new_bad = 1 if new_distinct < 5 else 0
                    if new_bad != bad_k4s[k]:
                        total_violations += (new_bad - bad_k4s[k])
                        bad_k4s[k] = new_bad
                        
                if total_violations == 0:
                    return colors
                    
            T *= cooling
            
        return None

    # Try C from 13 to 17
    # Time budget: 20s for 13, 20s for 14, 20s for 15, 20s for 16, 20s for 17?
    # Total 100s. Leave 20s for overhead/fallback.
    # Actually, if we find 13, we stop.
    
    # Random seed for reproducibility? 
    # The task doesn't specify, but deterministic is better for evaluation stability if possible.
    # However, local search is stochastic. 
    # We can fix the seed for the random module.
    random.seed(42)
    
    for C in [13, 14, 15, 16]:
        # Time limit for each C
        # If we spend too much on 13 and fail, we might not have time for 15.
        # Let's give 13: 20s, 14: 20s, 15: 40s, 16: 20s.
        time_budget = 20
        if C == 15:
            time_budget = 40
        if C == 16:
            time_budget = 20
            
        result = local_search(C, time_budget)
        if result:
            # Final verification
            if is_valid(result, C):
                return result
            else:
                # Should not happen, but just in case
                continue
                
    # Fallback to 17 colors (Seed)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
