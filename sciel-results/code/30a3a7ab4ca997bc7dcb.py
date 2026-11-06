import time
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish by 85s.
    deadline = 85.0
    
    n = 512
    
    # 1. Precompute conflict graph
    # vertices are 1..n
    # We use 0-based indexing for bitmasks: vertex i corresponds to bit (i-1)
    # adj[i] is a bitmask of vertices that conflict with i
    adj = [0] * (n + 1)
    
    # Generate all 4-term APs (a, a+d, a+2d, a+3d)
    # If a, b, c, d form a 4-AP, then any pair among them cannot both be in the set?
    # No, the condition is "no nonconstant k=4 term arithmetic progression".
    # This means we cannot have ALL 4 elements.
    # Wait, the problem says "containing no nonconstant k=4 term arithmetic progression".
    # Usually, "k-AP free" means no subset of size k forms an AP.
    # So if S contains {a, a+d, a+2d, a+3d}, it is invalid.
    # It is NOT invalid if S contains {a, a+d, a+2d} (that's a 3-AP).
    # It is NOT invalid if S contains {a, a+d, a+3d}.
    # So the "conflict" is not pairwise. It's a hypergraph constraint.
    # A 4-AP is a hyperedge of size 4. We want a subset of vertices that contains NO hyperedge.
    # This is the "Maximum Independent Set in a 4-uniform Hypergraph" or "Maximum 4-AP-free subset".
    # Standard MIS algorithms work on graphs (2-uniform).
    # Can we reduce this to a graph problem?
    # No, not directly.
    # However, the prompt suggests: "Model the problem as a Maximum Independent Set (MIS) in a conflict graph where vertices are integers 1–512 and edges connect numbers that would form a 4-AP if both were present".
    # This interpretation in the prompt is technically incorrect for "4-AP free".
    # If edges connected numbers that "would form a 4-AP if both were present", that implies a pairwise conflict.
    # But a 4-AP requires 4 numbers.
    # Let's re-read carefully: "containing no nonconstant k=4 term arithmetic progression".
    # This means the set S does NOT contain any 4 distinct elements in AP.
    # This is a 4-uniform constraint.
    
    # The prompt's suggestion "edges connect numbers that would form a 4-AP if both were present" is confusing.
    # Perhaps it means: if we select two numbers, do they force a 4-AP? No.
    # Perhaps it implies a different definition?
    # "Large four-term-progression-free integer sets"
    # Standard definition: S has no 4-term AP.
    
    # If I strictly follow the "MIS in a conflict graph" strategy, I must define what the edges are.
    # If I define edges as "pairs that are part of SOME 4-AP", that's not right because selecting both doesn't create a 4-AP by itself.
    # Selecting 4 of them does.
    
    # However, for n=512, the density of 4-APs is high.
    # Let's look at the "inspiration" scores.
    # Score 1.0720 used "multi-start randomized greedy + vectorized bitset local search".
    # The current program (0.9280) uses "Greedy + Local Search".
    # The directive asks for "MIS in a conflict graph ... branch-and-bound".
    
    # If I implement a true MIS solver, I need a graph.
    # Is there a way to map this to a graph?
    # If I create a graph where an edge exists between x and y if they are in the SAME 4-AP?
    # Then an Independent Set in this graph would be a set where no two elements are in the same 4-AP.
    # This is a STRONGER condition than "no 4-AP".
    # A set with no 4-AP can contain elements that are in the same 4-AP (e.g., 3 of them).
    # So MIS on this graph gives a lower bound, potentially much smaller.
    
    # Let's check the size.
    # If I take the Cantor set (digits 0,1 base 3), it is 3-AP free.
    # A 3-AP free set is automatically 4-AP free.
    # In the "same 4-AP" graph, does the Cantor set have edges?
    # If 3 elements are in a 4-AP, they are connected in the "same 4-AP" graph.
    # The Cantor set has NO 3-APs. So no 3 elements are in a 4-AP.
    # So no 2 elements are in a 4-AP?
    # If a, b are in the Cantor set, can they be in a 4-AP with c, d?
    # Yes, e.g., 1, 3, 5, 7. 1, 3 are in Cantor?
    # 1 = 1 (base 3) -> OK
    # 3 = 10 (base 3) -> OK
    # 5 = 12 (base 3) -> NO
    # 7 = 21 (base 3) -> NO
    # So 1, 3, 5, 7 is not in Cantor.
    # But are 1 and 3 in a 4-AP with ANY other numbers?
    # 1, 3, 5, 7. 5,7 not in Cantor.
    # 1, 3, 7, 11? (d=2, 6, 10, 14).
    # It is possible that two Cantor numbers are in a 4-AP.
    # If so, the MIS on the "same 4-AP" graph would reject them.
    # This would yield a very small set.
    
    # Given the prompt's specific instruction to use MIS on a conflict graph, and the ambiguity of "edges connect numbers that would form a 4-AP if both were present", I suspect the prompt implies a simplification or a specific interpretation.
    # However, "would form a 4-AP if both were present" is logically impossible for just 2 numbers.
    # Maybe it means: "edges connect numbers that CANNOT both be present"?
    # If so, the constraint must be pairwise.
    # But the task is "4-AP free".
    
    # Let's assume the standard interpretation: Maximize size of S subset [1, 512] such that S contains no 4-term AP.
    # The "MIS" strategy is likely a red herring or a misinterpretation in the prompt's "Strategy to try" section, OR it refers to a reduction I'm missing.
    # Actually, there is a known reduction for "no k-AP" to MIS? No.
    
    # Let's look at the "Strategy to try" again:
    # "Model the problem as a Maximum Independent Set (MIS) in a conflict graph where vertices are integers 1–512 and edges connect numbers that would form a 4-AP if both were present; solve this using a branch-and-bound algorithm with bitset-based coloring for upper-bound pruning"
    
    # If I follow this literally, I must define the graph.
    # If I define edges as "x and y are in some 4-AP", the resulting Independent Set is "no two elements are in any 4-AP".
    # This is equivalent to saying: For any 4-AP, S contains at most 1 element from it.
    # This is a much stricter constraint than "S contains at most 3 elements from any 4-AP".
    # The size of such a set will be significantly smaller.
    # For n=512, the number of 4-APs is roughly n^2/6.
    # Each element is in ~n/6 APs.
    # The graph is dense.
    # An Independent Set in a dense graph is small.
    # This will likely score poorly compared to the current 0.928 (which gets ~170-200 elements).
    # A 3-AP free set (Cantor) has size ~ 512^0.63 ~ 100.
    # The "at most 1 per 4-AP" set will be smaller than the 3-AP free set?
    # If a set is 3-AP free, it has no 3 elements in AP.
    # Does it have at most 1 element in any 4-AP?
    # Yes, because a 4-AP contains 3-APs (first 3, last 3, etc.).
    # If a set has 2 elements in a 4-AP, say a, a+d.
    # It doesn't necessarily have a 3-AP.
    # Example: 1, 3 in 1,3,5,7.
    # If S = {1, 3}, it has no 3-AP.
    # So the "at most 1 per 4-AP" constraint allows sets that are not 3-AP free?
    # No, if S has 1, 3, it's fine.
    # But if S has 1, 3, 5, it has a 3-AP (1,3,5).
    # So "at most 1 per 4-AP" implies "no 3 elements in a 4-AP".
    # Since any 3 elements in a 4-AP form a 3-AP, this implies "3-AP free".
    # Wait.
    # If S has 1, 3, 5, they form a 3-AP. They are part of the 4-AP 1,3,5,7.
    # So if the constraint is "at most 1 element from any 4-AP", then S cannot have 1,3,5.
    # So S is 3-AP free.
    # Thus, the "MIS on same-4-AP graph" solution is a subset of 3-AP free sets.
    # The best 3-AP free set (Cantor) has size ~100.
    # The current program gets ~170.
    # So this strategy will likely score < 1.0.
    
    # However, the directive says: "A first try in a new direction may score worse; that is acceptable."
    # And "Make a STRUCTURAL change".
    
    # So I will implement the MIS strategy as described, interpreting the graph as "x and y are in the same 4-AP".
    
    # Precompute adj masks
    # For each pair (x, y) with x < y, check if they are in a 4-AP.
    # x, y are in a 4-AP if there exist a, d such that the AP is {a, a+d, a+2d, a+3d} and x,y are in it.
    # This is O(n^3) or O(n^2 * log n).
    # For n=512, n^2 = 262k. Checking each pair against all APs is too slow.
    # Better: Iterate over all 4-APs. For each 4-AP (a,b,c,d), add edges between all pairs in {a,b,c,d}.
    # Number of 4-APs is ~ n^2/6 ~ 43k.
    # Each AP has 6 pairs.
    # Total edges to add ~ 260k.
    # This is fast.
    
    adj = [0] * (n + 1)
    
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Add edges between all pairs in {a, b, c, e}
            # a-b, a-c, a-e, b-c, b-e, c-e
            
            adj[a] |= (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            adj[b] |= (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            adj[c] |= (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            adj[e] |= (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            
    # Now solve MIS on this graph.
    # Vertices: 1..n
    # We want a subset S such that for any x, y in S, there is NO edge between x and y.
    # i.e., (adj[x] & (1 << (y-1))) == 0.
    
    # Branch and Bound with Bitset Coloring.
    # Standard algorithm: Tomita's MCS (Maximum Clique) algorithm can be adapted for MIS by finding Max Clique in the complement graph?
    # Or just Max Independent Set directly.
    # MIS in G = Max Clique in G_complement.
    # G_complement is dense (since G is sparse? No, G has ~260k edges out of ~130k possible? No, n=512, total pairs ~130k.
    # Wait. Total pairs in 512 vertices is 512*511/2 = 130,816.
    # Edges added: 43k APs * 6 pairs = 258k.
    # But many pairs are in multiple APs.
    # Let's estimate density.
    # Most pairs are NOT in a 4-AP.
    # Probability that two random numbers are in a 4-AP is low.
    # So G is SPARSE.
    # G_complement is DENSE.
    # Finding Max Clique in a Dense graph is hard?
    # Actually, MCS is very fast on Dense graphs because the candidate set shrinks quickly.
    # Finding Max Independent Set in a Sparse graph is equivalent to Max Clique in Dense graph.
    # This is a good direction.
    
    # Let's implement Max Independent Set directly on G.
    # Algorithm:
    # 1. Sort vertices by degree (descending) for better pruning?
    # 2. Recursive branch:
    #    - Pick a vertex v.
    #    - Branch 1: Include v. Remove v and its neighbors from candidate set.
    #    - Branch 2: Exclude v. Remove v from candidate set.
    #    - Bound: If |current_set| + |candidate_set| <= best_size, prune.
    
    # To speed this up, we use bitsets for the candidate set.
    # N = 512. Bitset fits in 8 64-bit words or one Python int.
    # Python int is arbitrary precision, so it's very fast for bitwise ops.
    
    # We need a coloring bound to improve on simple cardinality bound.
    # For MIS, we can color the candidate subgraph.
    # If we color the candidate graph with k colors, the MIS size is at most k.
    # So if current_size + k <= best_size, prune.
    # Coloring a sparse graph is fast.
    
    # Implementation details:
    # - Use Python int for bitsets.
    # - Precompute adj as list of ints.
    # - Function: mis(candidates, current_set)
    #   - If candidates == 0: update best
    #   - Bound: if len(current_set) + popcount(candidates) <= best_size: return
    #   - Bound: if len(current_set) + color_bound(candidates) <= best_size: return
    #   - Pick a vertex v from candidates (e.g., lowest index, or highest degree in candidate subgraph).
    #   - Branch Include:
    #       new_candidates = candidates & ~adj[v] & ~(1 << (v-1))
    #       mis(new_candidates, current_set + [v])
    #   - Branch Exclude:
    #       new_candidates = candidates & ~(1 << (v-1))
    #       mis(new_candidates, current_set)
    
    # To make it efficient, we need a good coloring bound.
    # Greedy coloring of the candidate subgraph.
    
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy coloring of a subgraph defined by bitmask `cand`
    # Returns the number of colors used.
    def color_bound(cand):
        if cand == 0:
            return 0
            
        colors = 0
        remaining = cand
        while remaining:
            colors += 1
            # Greedy: pick a vertex, add it to current color class, remove it and its neighbors from remaining
            # To maximize pruning, we want the coloring to be "tight".
            # Standard greedy:
            # 1. Pick a vertex v from remaining.
            # 2. Add v to current color class.
            # 3. Remove v and all its neighbors in `remaining` from `remaining`.
            #    Wait, this is for Independent Set coloring?
            #    In a graph, a color class is an Independent Set.
            #    So we pick v, then we CANNOT pick any neighbor of v in the same color class.
            #    So we remove v and N(v) from the pool of vertices available for this color class?
            #    No.
            #    Algorithm for coloring:
            #    1. Start with empty color class C.
            #    2. While there are vertices in `remaining`:
            #         Pick a vertex v.
            #         Add v to C.
            #         Remove v from `remaining`.
            #         Remove all neighbors of v from `remaining`? NO.
            #         Neighbors of v CANNOT be in C.
            #         So we remove v and N(v) from the set of vertices we consider for C?
            #         Yes. The vertices that CAN be in C are those not adjacent to any vertex in C.
            #         So if we pick v, the next vertex we pick for C must not be adjacent to v.
            #         So we remove v and N(v) from the candidate pool for this color class.
            #    3. Once no more vertices can be added to C, move to next color.
            
            color_class_candidates = remaining
            while color_class_candidates:
                # Pick a vertex. To be fast, pick the lowest set bit.
                # Or pick the vertex with highest degree?
                # Lowest set bit is fast.
                lsb = color_class_candidates & (-color_class_candidates)
                v_bit = lsb
                v_idx = v_bit.bit_length() - 1
                
                # Add v to color class
                # Remove v and its neighbors from color_class_candidates
                # Neighbors of v in the ORIGINAL graph, but restricted to `remaining`?
                # Actually, we just need to ensure no two vertices in C are adjacent.
                # So if we add v, we must remove N(v) from the pool of vertices we can still add to C.
                # N(v) is adj[v].
                # So new_pool = color_class_candidates & ~adj[v] & ~v_bit
                
                color_class_candidates &= ~adj[v_idx+1] # v_idx is 0-based, adj is 1-based
                color_class_candidates &= ~v_bit
                
            # After this, `remaining` should be updated to remove all vertices in C.
            # How to get all vertices in C?
            # We lost track.
            # Better approach:
            # C = 0
            # pool = remaining
            # while pool:
            #   pick v
            #   C |= v_bit
            #   pool &= ~adj[v] & ~v_bit
            # remaining &= ~C
            
            # Let's re-implement color_bound properly.
            
            C = 0
            pool = remaining
            while pool:
                lsb = pool & (-pool)
                v_bit = lsb
                v_idx = v_bit.bit_length() - 1
                C |= v_bit
                pool &= ~adj[v_idx+1]
                pool &= ~v_bit
            remaining &= ~C
            
        return colors

    best_size = 0
    best_set = 0
    
    # Initial candidate set: all vertices
    init_cand = (1 << n) - 2  # bits 1 to n set. Bit 0 is 0.
    # Actually, vertices are 1..n.
    # Bit i-1 corresponds to vertex i.
    # So bits 0 to n-1 correspond to vertices 1 to n.
    # init_cand = (1 << n) - 1
    
    def solve(cand, cur_size, cur_set):
        nonlocal best_size, best_set
        
        if cand == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_set = cur_set
            return
            
        # Bound 1: Cardinality
        if cur_size + popcount(cand) <= best_size:
            return
            
        # Bound 2: Coloring
        # Coloring is expensive. Use it sparingly or only when cardinality bound is weak?
        # For now, use it always to ensure strong pruning.
        # If it's too slow, we'll see.
        cb = color_bound(cand)
        if cur_size + cb <= best_size:
            return
            
        # Pick a vertex
        # Heuristic: Pick the vertex with the highest degree in the candidate subgraph?
        # Or just the lowest index for simplicity?
        # Lowest index is fast.
        lsb = cand & (-cand)
        v_bit = lsb
        v_idx = v_bit.bit_length() # 1-based index
        
        # Branch 1: Include v
        # New candidates: cand without v and without N(v)
        new_cand = cand & ~adj[v_idx] & ~v_bit
        solve(new_cand, cur_size + 1, cur_set | v_bit)
        
        # Branch 2: Exclude v
        # New candidates: cand without v
        new_cand = cand & ~v_bit
        solve(new_cand, cur_size, cur_set)
        
    # To improve performance, we can order vertices.
    # But recursive call with Python ints is okay for n=512 if pruning is good.
    # However, depth can be 512. Recursion limit?
    # Python default is 1000. 512 is safe.
    
    # One issue: The graph is sparse, so MIS is large.
    # The coloring bound might be weak if the graph is sparse?
    # No, for sparse graphs, the chromatic number is low, so the bound is tight (small).
    # Wait. If chromatic number is low, the bound `cur_size + cb` is small, so we prune LESS.
    # We want a LARGE bound to prune LESS? No, we want a SMALL bound to prune MORE.
    # If the true MIS size is 100, and cb is 10, and cur_size is 0, 0+10 <= best_size (100) is FALSE. So we don't prune.
    # If cb is 200, 0+200 <= 100 is FALSE.
    # We prune if cur_size + cb <= best_size.
    # So we want cb to be an UPPER bound on the MIS size of the subgraph.
    # The chromatic number is an UPPER bound on the MIS size?
    # No. The chromatic number is an UPPER bound on the size of a CLIQUE? No.
    # In a graph, the size of the Maximum Independent Set is at most the number of colors?
    # Yes! Because each color class is an Independent Set.
    # So if we color the graph with k colors, the MIS size is at most k?
    # NO.
    # If we color with k colors, we have k Independent Sets.
    # The MIS is the largest of these? No, the MIS is a subset of one color class?
    # No. The MIS is an Independent Set. It is contained in ONE color class?
    # No. An Independent Set can contain vertices of DIFFERENT colors?
    # No! By definition of coloring, vertices of the same color are NOT adjacent.
    # So each color class is an Independent Set.
    # The MIS is the largest Independent Set.
    # Is the MIS contained in a single color class?
    # No. The MIS is a subset of the vertices.
    # If v and u are in the MIS, they are not adjacent.
    # In a valid coloring, non-adjacent vertices CAN have the same color or different colors.
    # So the MIS is NOT necessarily contained in one color class.
    # Therefore, the chromatic number is NOT an upper bound on the MIS size.
    # Example: A matching of 3 edges. Chromatic number 2. MIS size 3.
    # So `color_bound` as implemented (number of colors) is NOT a valid upper bound for MIS.
    
    # Correction:
    # The number of colors is an upper bound on the size of the MAXIMUM CLIQUE? No.
    # The chromatic number is an upper bound on the size of the MAXIMUM INDEPENDENT SET? NO.
    # The size of the Maximum Independent Set is at most the number of vertices.
    # A common bound for MIS is based on the degree or greedy coloring of the COMPLEMENT?
    
    # For Max Clique, the number of colors in the candidate graph is an upper bound on the clique size.
    # Since MIS in G = Max Clique in G_complement.
    # So we should color G_complement.
    # But G is sparse, so G_complement is dense.
    # Coloring a dense graph uses many colors.
    # So the bound will be large (weak).
    
    # Alternative bound for MIS in G:
    # Use the fact that if we pick a vertex v, the rest of the MIS must be in N(v) complement?
    
    # Let's stick to the simple cardinality bound first.
    # If it's too slow, we can add a better bound.
    # Given the time limit, and n=512, even a bad B&B might be slow.
    # But the graph is very sparse.
    # The MIS is likely large (close to n).
    # If the MIS is large, the search tree is small?
    # If we include v, we remove N(v).
    # If N(v) is small, we remove few vertices.
    # So the tree stays wide.
    
    # This might time out.
    # Let's add a timeout check inside the recursion.
    
    def solve_timed(cand, cur_size, cur_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if cand == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_set = cur_set
            return
            
        if cur_size + popcount(cand) <= best_size:
            return
            
        lsb = cand & (-cand)
        v_bit = lsb
        v_idx = v_bit.bit_length()
        
        # Include v
        new_cand = cand & ~adj[v_idx] & ~v_bit
        solve_timed(new_cand, cur_size + 1, cur_set | v_bit)
        
        # Exclude v
        new_cand = cand & ~v_bit
        solve_timed(new_cand, cur_size, cur_set)

    # Initialize
    init_cand = (1 << n) - 1
    solve_timed(init_cand, 0, 0)
    
    # Convert best_set bitmask to list
    result = [i for i in range(1, n + 1) if (best_set & (1 << (i-1)))]
    
    # Final Verification
    S_set = set(result)
    valid = True
    for a in S_set:
        for d in range(1, (n - a) // 3 + 1):
            if (a+d) in S_set and (a+2*d) in S_set and (a+3*d) in S_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Greedy
        S_mask = 0
        for x in range(1, n+1):
            # Check if x can be added
            # We need to check 4-APs containing x.
            # This is O(n^2) per x, total O(n^3) = 134M. Too slow for Python.
            # Use the conflict masks from the original program?
            # Let's just return an empty set or a small valid set if B&B fails.
            # A single element is always valid.
            return [1]
            
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")

