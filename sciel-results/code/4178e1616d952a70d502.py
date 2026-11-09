import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        
        # Build adjacency list using bitmasks for fast operations
        # adj[i] is an integer where bit j is set if i and j are connected
        adj = [0] * n
        
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                # Edge if vi + vj in V, or 2*vi == vj, or 2*vj == vi
                if (vi + vj) in v_set or 2 * vi == vj or 2 * vj == vi:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Bitmask of all vertices
        all_mask = (1 << n) - 1
        
        # Helper to check if a set is independent (sum-free)
        def is_independent(mask):
            # A set is independent if no edge has both endpoints in the mask
            # Efficient check: for each v in mask, if (adj[v] & mask) != 0, not independent
            temp = mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                if adj[v] & mask:
                    return False
                temp &= temp - 1
            return True

        # Multi-Start Randomized Greedy
        # Strategy: Repeatedly build a maximal independent set using a randomized greedy rule.
        # Rule: At each step, pick a vertex from the remaining candidates.
        # Candidates are vertices that do not conflict with the current set.
        # Selection heuristic: 
        #   - With some probability, pick the vertex with minimum degree in the remaining subgraph.
        #   - Otherwise, pick a random vertex from the candidates.
        #   - Or, pick based on a random score = degree + noise.
        
        best_mask = 0
        best_size = 0
        
        # We will run multiple iterations until time budget is exhausted
        # Each iteration is a full greedy pass.
        
        while time.time() - start_time < start_time + time_budget:
            current_mask = 0
            rem = all_mask
            
            # We will greedily add vertices.
            # To make it randomized, we maintain a list of available vertices and pick one.
            # For efficiency with bitmasks, we can iterate through bits, but that's O(n) per pick.
            # Total O(n^2) per run. For n=240, that's ~50k ops, very fast.
            
            # To add randomness, we can shuffle the order of vertices we consider, 
            # or use a priority queue with randomization.
            # Simpler: 
            # 1. Compute degrees in the current remaining graph.
            # 2. Pick a vertex v with probability proportional to exp(-degree / T) or similar.
            #    Or just pick min-degree with some probability, else random.
            
            # Let's use a "randomized min-degree" approach.
            # At each step:
            # - Find all valid vertices (those in rem that don't conflict with current_mask? 
            #   Actually, if we maintain rem as "vertices not yet decided and not excluded by neighbors of chosen vertices", 
            #   then any vertex in rem is a valid candidate to add to the independent set.
            #   Wait, standard greedy MIS:
            #   Start with S = empty, V_rem = V.
            #   While V_rem not empty:
            #       Pick v from V_rem.
            #       Add v to S.
            #       Remove v and its neighbors from V_rem.
            
            # This produces a maximal independent set.
            # The quality depends on the order of picking.
            
            # Let's implement the picking with a randomized heuristic.
            # We can compute the degree of each vertex in the subgraph induced by V_rem.
            # deg_rem(v) = (adj[v] & V_rem).bit_count()
            
            # Heuristic: Pick v that minimizes deg_rem(v) + random_noise.
            # This explores different local structures.
            
            noise_scale = 1.0 # Can tune this
            
            while rem:
                if time.time() - start_time > DEADLINE - 2:
                    break
                
                # Find best candidate
                best_v = -1
                best_score = float('inf')
                
                temp = rem
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    deg = (adj[v] & rem).bit_count()
                    score = deg + random.random() * noise_scale
                    if score < best_score:
                        best_score = score
                        best_v = v
                    temp &= temp - 1
                
                if best_v == -1:
                    break
                
                # Add best_v to current set
                current_mask |= (1 << best_v)
                
                # Remove best_v and its neighbors from rem
                rem &= ~((1 << best_v) | adj[best_v])
            
            # End of one greedy run
            current_size = current_mask.bit_count()
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        
        # Optional: Local Search Refinement (2-opt style swaps)
        # If we have time left, try to improve best_mask.
        # A simple improvement: try to add a vertex if it doesn't conflict.
        # Or try to swap: remove one vertex, add two? 
        # Given the time, let's stick to the multi-start greedy which is very robust.
        
        # Convert mask to values
        best_values = [values[i] for i in range(n) if (best_mask >> i) & 1]
        return best_values

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        
        if remaining < 5:
            # Fallback: just take all odds or something simple
            vals = instance(size, seed)
            res = [x for x in vals if x % 2 != 0]
            if not is_sum_free(res):
                res = [x for x in vals if x != 0]
            results.append(res)
            continue

        # Allocate time proportionally to size, but ensure at least some time for each
        # n=60, 120, 240.
        # Let's give 20% to first, 30% to second, 50% to third? 
        # Or just split remaining time roughly equally, adjusting for complexity.
        # Complexity is O(n^2 * iterations). 
        # Let's just use the remaining time budget for each, subtracting what's used.
        
        # Actually, the loop inside solve_instance checks the global deadline.
        # So we can just pass the remaining time as the budget.
        
        vals = instance(size, seed)
        res = solve_instance(vals, remaining)
        
        if not is_sum_free(res):
            # Fallback
            res = [x for x in vals if x % 2 != 0]
        
        results.append(res)

    return results
