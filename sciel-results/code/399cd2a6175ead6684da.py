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

    def solve_instance(values, budget):
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
        
        # Greedy Initialization: Min-Degree Greedy
        def greedy_init():
            rem = all_mask
            is_mask = 0
            # Simple greedy: pick vertex with min degree in remaining set
            while rem:
                if time.time() - start_time > DEADLINE - 10:
                    break
                # Find min degree vertex
                min_v = -1
                min_deg = n + 1
                temp_rem = rem
                while temp_rem:
                    lsb = temp_rem & (-temp_rem)
                    v = lsb.bit_length() - 1
                    deg = (adj[v] & rem).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                    temp_rem &= temp_rem - 1
                
                if min_v == -1:
                    break
                
                is_mask |= (1 << min_v)
                # Remove min_v and its neighbors from rem
                rem &= ~( (1 << min_v) | adj[min_v] )
                
            return is_mask

        best_mask = greedy_init()
        best_size = best_mask.bit_count()
        
        # Branch and Bound for MIS
        # We use a recursive search with bitsets.
        # Upper bound: Greedy coloring of the remaining subgraph.
        # We implement a simplified coloring bound.
        
        def color_bound(mask):
            # Greedy coloring to get an upper bound on MIS size
            # The number of colors is an upper bound on the size of MIS
            # (since MIS is an independent set, it can use at most 1 vertex per color? 
            # No, wait. The chromatic number is a lower bound on Clique, but for Independent Set?
            # Actually, if we color the graph, any independent set can contain at most one vertex from each color?
            # No, that's for Clique. For Independent Set, vertices in the same color are non-adjacent, 
            # so they CAN be in the same Independent Set.
            # Wait, the standard bound for MIS using coloring is:
            # If we color the graph with k colors, the MIS size is at most k? No.
            # The MIS size is at least n/k? No.
            # Let's recall: The size of the Maximum Independent Set is related to the chromatic number?
            # Actually, a common bound is based on the "greedy coloring" of the subgraph.
            # If we partition vertices into independent sets (colors), the MIS can take at most 1 vertex from each color?
            # No, if they are in the same color, they are NOT adjacent, so they CAN be together.
            # So coloring gives a LOWER bound on MIS? No.
            
            # Let's use the standard Tomita bound:
            # Greedily color the subgraph. The number of colors is an upper bound on the size of the Maximum Clique.
            # Since MIS(G) = MaxClique(complement(G)), and we are working on G directly...
            # Wait, the standard algorithm for MIS uses the complement graph for coloring?
            # Or does it use the property that if we color G, the MIS size is bounded by something?
            
            # Actually, the standard "branch and bound for MIS" often works on the graph G.
            # The bound is usually derived from a greedy coloring of the *remaining* vertices.
            # If we color the remaining subgraph with c colors, the MIS can contain at most c vertices?
            # Let's check. If we have a coloring, vertices of the same color are non-adjacent.
            # So an independent set can contain MULTIPLE vertices of the same color.
            # So the number of colors is NOT an upper bound on MIS.
            
            # Correction: The number of colors is an upper bound on the size of the Maximum CLIQUE in the COMPLEMENT graph?
            # Let's stick to a simpler bound: 
            # If we greedily select vertices for coloring, the number of colors used is an upper bound on the size of the Maximum Independent Set?
            # No, that's false. Consider a triangle (K3). MIS=1. Colors=3. 1 <= 3.
            # Consider a path of 3 (1-2-3). MIS=2. Colors=2. 2 <= 2.
            # Consider a cycle of 4. MIS=2. Colors=2.
            # It seems the number of colors is an upper bound?
            # Proof: Each color class is an independent set. The union of color classes is the vertex set.
            # Wait, if I pick an independent set S, can I pick more than 1 vertex from a color class?
            # Yes, because vertices in the same color class are non-adjacent.
            # So the number of colors is NOT an upper bound on MIS.
            
            # The correct bound for MIS using coloring is:
            # If we color the graph, the size of the MIS is at most the number of colors? NO.
            
            # Let's use a different bound: 
            # The "greedy coloring" bound is actually used for Maximum Clique.
            # For Maximum Independent Set, we can use the same logic on the complement graph?
            # Or simply: The size of MIS is at most the number of vertices.
            # A better bound: If we greedily pick vertices to form a maximal independent set, its size is a LOWER bound.
            
            # Let's use a simple bound: The size of the maximum independent set in a subgraph is at most the number of vertices in that subgraph.
            # To prune, we need a tighter upper bound.
            
            # Alternative: Use the fact that if the current best is B, and the remaining vertices count is R, 
            # we can only prune if R + current_size <= B.
            # This is weak.
            
            # Let's try a different approach: 
            # Since n is small (<=240), we can use a simple recursive search with memoization? No, state space is 2^n.
            
            # Let's stick to the standard "Branch and Bound" for MIS which uses a greedy coloring of the *subgraph* to bound the clique number of the complement?
            # Actually, there is a well-known bound: 
            # If we greedily color the remaining graph, the number of colors is an upper bound on the size of the Maximum Independent Set?
            # Let's re-verify.
            # Counter-example: Two disjoint triangles. G = K3 U K3.
            # MIS = 2 (one from each triangle).
            # Coloring: Each triangle needs 3 colors. Total colors = 6.
            # 2 <= 6. Holds.
            # Path P4: 1-2-3-4.
            # MIS = 2 (1,3 or 1,4 or 2,4).
            # Coloring: 2 colors (1,3 same; 2,4 same).
            # 2 <= 2. Holds.
            
            # It appears that the number of colors in a proper coloring is an upper bound on the size of the Maximum Independent Set?
            # Wait. If I have a complete graph K_n, MIS=1. Colors=n. 1 <= n.
            # If I have an empty graph (no edges), MIS=n. Colors=1. n <= 1? FALSE.
            # So the number of colors is NOT an upper bound on MIS.
            
            # The number of colors is a LOWER bound on the size of the Maximum Clique? No.
            # The number of colors is an UPPER bound on the size of the Maximum Clique.
            # Since MIS(G) = MaxClique(complement(G)), if we color G, it doesn't directly bound MIS(G).
            # If we color complement(G), the number of colors bounds MIS(G).
            
            # So, to get an upper bound on MIS(G), we should color the COMPLEMENT of G.
            # But building the complement is expensive.
            
            # Alternative simple bound: 
            # Greedily find a maximal independent set in the remaining subgraph. Let its size be g.
            # Then MIS(remaining) >= g. This is a lower bound.
            
            # For an upper bound, we can use: 
            # MIS(remaining) <= number of vertices in remaining.
            
            # Given the time constraints and the fact that n is small, a simple recursive search with "include/exclude" and basic pruning (current_size + remaining_count <= best) might be sufficient, especially with a good initial solution from greedy.
            
            return mask.bit_count() # Weak bound, but safe

        # Better Bound: 
        # We can use the fact that if we have a set of vertices, the MIS size is at most the number of vertices.
        # To make it tighter, we can try to find a small vertex cover? 
        # Or just use the simple count.
        
        # Let's implement a simple DFS with pruning.
        
        def search(mask, current_size, current_mask):
            nonlocal best_mask, best_size
            
            if time.time() - start_time > DEADLINE - 5:
                return
            
            if mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            # Pruning: If current_size + max_possible <= best_size
            # max_possible is at most mask.bit_count()
            if current_size + mask.bit_count() <= best_size:
                return
            
            # Pick a vertex to branch on.
            # Heuristic: Pick vertex with max degree in the remaining subgraph?
            # Or min degree?
            # For MIS, picking a vertex with high degree might be good for pruning?
            # Let's pick the first available vertex for simplicity, or use a heuristic.
            
            # Find a vertex v in mask
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            
            # Option 1: Include v
            # New mask: mask without v and its neighbors
            new_mask_include = mask & ~( (1 << v) | adj[v] )
            search(new_mask_include, current_size + 1, current_mask | (1 << v))
            
            # Option 2: Exclude v
            new_mask_exclude = mask & ~(1 << v)
            search(new_mask_exclude, current_size, current_mask)
            
        # To make the search effective, we need a better ordering and bound.
        # Given the complexity, let's rely on the Greedy result and maybe do a few local improvements (SA) if time permits, 
        # or just return the greedy result if B&B is too slow.
        
        # Let's try a limited depth B&B or just use the Greedy result which is often close to optimal for these graphs.
        # For n=240, B&B might explode. 
        # Let's stick to the Greedy + SA approach from the previous version but optimized with bitmasks for speed.
        
        # Optimized SA with bitmasks
        S_mask = best_mask
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        lambda_penalty = 100
        
        # Precompute neighbor counts for fast edge counting?
        # Edge count in S = sum( (adj[v] & S).bit_count() for v in S ) / 2
        
        def get_edges(S):
            count = 0
            temp = S
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                count += (adj[v] & S).bit_count()
                temp &= temp - 1
            return count // 2
        
        current_edges = get_edges(S_mask)
        current_score = S_mask.bit_count() - lambda_penalty * current_edges
        
        # Use random module for SA
        while T > T_min and time.time() - start_time < DEADLINE - 5:
            # Pick random vertex
            v = random.randint(0, n - 1)
            v_bit = 1 << v
            
            if S_mask & v_bit:
                # Remove v
                new_S = S_mask & ~v_bit
                edges_removed = (adj[v] & S_mask).bit_count()
                new_edges = current_edges - edges_removed
                new_score = new_S.bit_count() - lambda_penalty * new_edges
            else:
                # Add v
                new_S = S_mask | v_bit
                edges_added = (adj[v] & S_mask).bit_count()
                new_edges = current_edges + edges_added
                new_score = new_S.bit_count() - lambda_penalty * new_edges
            
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                S_mask = new_S
                current_edges = new_edges
                current_score = new_score
                
                if current_edges == 0 and S_mask.bit_count() > best_size:
                    best_size = S_mask.bit_count()
                    best_mask = S_mask
            
            T *= cooling
        
        # Convert mask to values
        best_values = [values[i] for i in range(n) if (best_mask >> i) & 1]
        return best_values

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            vals = instance(size, seed)
            res = [x for x in vals if x % 2 != 0]
            if not is_sum_free(res):
                res = [x for x in vals if x != 0]
            results.append(res)
            continue

        budget = min(remaining - 5, 10 + 50 * (size / 240))
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]
        
        results.append(res)

    return results
