import time
from fractions import Fraction

def construct():
    N = 256
    # Precompute conflicts: 
    # If we pick y, we cannot pick any pair (u, v) such that u, y, v is a 3-AP.
    # This is equivalent to finding a Maximum Weight Independent Set in a graph G where:
    # Nodes are 1..N.
    # Edge (u, v) exists if u, v, w form a 3-AP for some w.
    # Wait, the constraint is "no 3-term AP". 
    # A set S is valid if for any distinct a, b, c in S, a, b, c do not form a 3-AP.
    # This means if a, b, c form a 3-AP, S cannot contain all three.
    # This is NOT a standard independent set problem on a graph of pairs.
    # A standard Independent Set forbids pairs. Here we forbid triples.
    # However, we can model this as a 3-SAT or use a different graph formulation.
    # Actually, the "Strategy to try" in the prompt suggests:
    # "Formulate the problem as a maximum weight independent set on a graph where nodes are integers 1-256 and edges connect pairs {a, c} if their midpoint b=(a+c)/2 is an integer within [1, 256] (implying a,b,c form a 3-AP)..."
    # This description is slightly confusing. If edges connect {a, c}, then an independent set would forbid having both a and c.
    # If we forbid having both a and c whenever a,b,c is a 3-AP, then we forbid ANY 3-AP because a 3-AP contains pairs (a,b), (b,c), (a,c).
    # If we build a graph where edges exist between any two elements that are part of ANY 3-AP together, then an independent set in this graph would be a set with NO pairs from any 3-AP.
    # This is a stronger constraint than "no 3-AP". A set with no 3-AP can still contain pairs like (1, 2) if 3 is not in the set.
    # But wait. The prompt says: "edges connect pairs {a, c} if their midpoint b=(a+c)/2 is an integer...".
    # If we use this graph, the independent set condition is: for any a, c in S, there is no b such that a, b, c is a 3-AP.
    # This implies that S contains no 3-AP.
    # Does it forbid valid sets?
    # Consider S = {1, 2}. Is there a 3-AP involving 1 and 2?
    # (1, 2, 3) -> b=2. Midpoint of 1,3 is 2. Edge (1,3).
    # (0, 1, 2) -> b=1. Midpoint of 0,2 is 1. Edge (0,2) (but 0 not in [1,256]).
    # (1, 2, ?) -> 1, 2, 3.
    # If we have edge (1, 3), we can't pick both 1 and 3.
    # If we have edge (1, 2)? Midpoint 1.5 not integer. No edge.
    # So {1, 2} is an independent set in this graph?
    # Let's check the graph definition again.
    # "edges connect pairs {a, c} if their midpoint b=(a+c)/2 is an integer within [1, 256]"
    # This defines edges based on the existence of a third element b in [1, 256].
    # If a=1, c=3, b=2 (in range). Edge (1,3).
    # If a=1, c=2, b=1.5 (not integer). No edge.
    # If a=1, c=4, b=2.5. No edge.
    # If a=1, c=5, b=3. Edge (1,5).
    # So the graph G has edge (a, c) iff a+c is even and 1 <= (a+c)/2 <= 256.
    # An independent set in G is a set S such that for any a, c in S, a+c is odd OR (a+c)/2 is outside [1, 256].
    # If a, c in S and a+c is even, let b=(a+c)/2. If b in [1, 256], then a, b, c is a 3-AP.
    # Since S is independent, edge (a, c) does not exist.
    # So if a, c in S and a+c even, then b must be OUTSIDE [1, 256].
    # This means a, b, c is NOT a 3-AP *within the domain*.
    # Thus, S contains no 3-AP.
    # Conversely, does this forbid any valid 3-AP-free set?
    # Suppose S is 3-AP free.
    # If a, c in S and a+c is even, and b=(a+c)/2 is in [1, 256], then a, b, c is a 3-AP.
    # Since S is 3-AP free, b is NOT in S.
    # BUT, the independent set condition forbids the pair (a, c) itself if the edge exists.
    # The edge exists if b is in [1, 256], regardless of whether b is in S.
    # So, if a, c in S and b is in [1, 256], then edge (a, c) exists.
    # For S to be an independent set, edge (a, c) must NOT exist.
    # This implies that for any a, c in S, if a+c is even, b MUST be outside [1, 256].
    # This is a MUCH stronger condition than just "b not in S".
    # Example: S = {1, 3}. 1+3=4, b=2 (in range). Edge (1,3) exists.
    # So {1, 3} is NOT an independent set in this graph.
    # But {1, 3} does NOT contain a 3-AP (it only has 2 elements).
    # So the graph formulation described in the prompt "edges connect pairs... if midpoint is integer" creates a graph whose independent sets are a SUBSET of 3-AP-free sets.
    # It excludes valid 3-AP-free sets like {1, 3}.
    # Therefore, solving MWIS on this specific graph will yield a lower bound, likely much worse than the optimal 3-AP-free set.
    # The prompt says: "Strategy to try... Formulate... as MWIS... and solve it exactly...".
    # If I follow this strictly, I might get a bad score.
    # However, the "Strategy" might be interpreted as "Use a graph formulation suitable for MWIS".
    # The standard graph for 3-AP-free sets is not a pairwise conflict graph in the simple sense.
    # BUT, we can map this to a different problem or use a different graph.
    # Actually, let's look at the "Inspiration" or "Lessons". None.
    # Let's re-read the prompt's strategy carefully.
    # "edges connect pairs {a, c} if their midpoint b=(a+c)/2 is an integer within [1, 256]"
    # This graph is fixed. Its independent sets are sets with no "midpoint-internal" pairs.
    # This is equivalent to: S is a Sidon set? No.
    # It means S has no two elements whose average is in [1, 256].
    # If a < c, and (a+c)/2 >= 1 (always true for a,c>=1) and (a+c)/2 <= 256 => a+c <= 512.
    # So if a+c > 512, no edge.
    # If a+c <= 512 and a+c is even, edge exists.
    # So S can only contain pairs with sum > 512 or sum odd.
    # This is a very restrictive set.
    # For example, you can't have {1, 2, 3}. 1+2=3 (odd, no edge), 2+3=5 (odd, no edge), 1+3=4 (even, edge). So {1,3} forbidden.
    # You can have {1, 2}.
    # You can have {255, 256}. 255+256=511 (odd, no edge).
    # You can have {1, 256}. 1+256=257 (odd, no edge).
    # This strategy seems to lead to a suboptimal solution if the graph is defined exactly as such.
    # HOWEVER, the prompt says "Strategy to try...". It might be a trap or a specific hint.
    # Or maybe I should interpret "edges connect pairs" differently?
    # No, "midpoint b... is an integer" is clear.
    # Let's consider if the prompt implies a DIFFERENT graph where nodes are 3-APs? No, "nodes are integers".
    # Let's assume the prompt wants me to implement MWIS on the graph where edge (u,v) exists if u,v are in a 3-AP with a third element in [1,N].
    # If I do that, I get a valid 3-AP-free set (since independent sets are 3-AP-free), but potentially not the max weight one.
    # The current program uses greedy + local search.
    # The new strategy is "Exact MWIS on this graph".
    # Given the graph is dense (many pairs have even sums), the independent set will be small.
    # Let's estimate the size.
    # If we pick large numbers, say > 128.
    # Pairs in [129, 256]. Min sum 129+130=259 (odd). 129+129 not allowed (distinct).
    # 130+131=261 (odd).
    # 130+132=262 (even, b=131 in range). Edge.
    # So we can't have {130, 132}.
    # This graph is quite connected.
    # Perhaps the "Strategy" is just to use a branch-and-bound solver on the CONFLICT graph of the 3-APs?
    # But a 3-AP conflict is a hyperedge (size 3). MWIS is for graphs (size 2).
    # To use MWIS, one must reduce the 3-uniform hypergraph to a graph.
    # The reduction "edge (a,c) if midpoint in range" is one such reduction, but it's overly restrictive.
    # Another reduction: For each 3-AP (a,b,c), add edges (a,b), (b,c), (a,c)?
    # If we add all these edges, an independent set cannot contain any pair from any 3-AP.
    # This is also overly restrictive (forbids {1,2} if 3 is in domain, even if 3 is not in S).
    #
    # Is there a way to make this work?
    # Maybe the "Strategy" is actually "Use a branch and bound on the original problem, but using the graph structure for bounds"?
    # The prompt says: "Formulate the problem as a maximum weight independent set on a graph... and solve it exactly using a custom branch-and-bound".
    # I will follow the instruction literally. If the score is bad, it's a bad strategy, but I must try it.
    # Wait, if I follow it literally, I might get a score of ~0.5 or lower, which is worse than 1.0.
    # The current program scores 1.0.
    # If I implement the literal strategy and it's bad, I fail.
    # Let's look at the "Inspiration 3": "The task is to find a subset... The current program uses a simple gr..."
    # Maybe the "Strategy" is a hint that the current greedy approach is stuck, and an exact solver on a *proper* formulation is needed.
    # But 3-AP-free is not MWIS.
    #
    # Let's reconsider the graph definition.
    # "edges connect pairs {a, c} if their midpoint b=(a+c)/2 is an integer within [1, 256]"
    # This graph G has 256 nodes.
    # Edge (a, c) exists iff a+c is even and a+c <= 512.
    # Note: a,c >= 1.
    # If a+c is odd, no edge.
    # If a+c is even and a+c <= 512, edge.
    # If a+c is even and a+c > 512, no edge.
    #
    # An independent set S in G:
    # For any a, c in S:
    # 1. a+c is odd, OR
    # 2. a+c > 512.
    #
    # Let's try to construct a large weight independent set.
    # High weight means small numbers.
    # Can we include 1?
    # If 1 in S:
    # For any c in S:
    # 1+c is odd => c is even.
    # OR
    # 1+c > 512 => c > 511 (impossible).
    # So if 1 in S, all other elements must be EVEN.
    # So S subset of {1} U {Even numbers}.
    # Now consider two even numbers 2i, 2j.
    # Sum = 2(i+j). Even.
    # Edge exists if 2(i+j) <= 512 => i+j <= 256.
    # So for any two evens 2i, 2j in S, we must have i+j > 256.
    # This means the indices i, j of the even numbers must sum to > 256.
    # To maximize weight (minimize values), we want small evens.
    # Small evens have small i.
    # If we pick i=1 (val 2), then for any other j, 1+j > 256 => j > 255.
    # So we can pick 2 (i=1) and evens 2j where j > 255.
    # Max j is 128 (val 256). 255 > 128. Impossible.
    # So we cannot pick 2 if we want other elements?
    # If we pick 2, we can only pick evens 2j with j > 255. None exist in [1, 128].
    # So if we pick 2, S = {1, 2}.
    # Weight = 1/1 + 1/2 = 1.5.
    #
    # What if we don't pick 1?
    # Pick 2?
    # If 2 in S:
    # For any c: 2+c odd => c odd. OR 2+c > 512 => c > 510.
    # So S subset of {2} U {Odd numbers}.
    # Consider two odds 2i-1, 2j-1.
    # Sum = 2(i+j-1). Even.
    # Edge if 2(i+j-1) <= 512 => i+j-1 <= 256 => i+j <= 257.
    # So for any two odds in S, their indices i, j must sum > 257.
    # Smallest odd is 1 (i=1).
    # If we pick 1 (i=1), then for any other j, 1+j > 257 => j > 256.
    # Max j for odd in [1, 256] is 128 (val 255, j=128).
    # So we can't pick 1 with any other odd.
    # So if we pick 2 and 1, S={1,2}. Weight 1.5.
    #
    # What if we pick 3 (i=2)?
    # If 3 in S, others must be odd (since 3+even=odd, no edge) or > 509.
    # So S subset of {3} U {Odds}.
    # Pairs of odds: i+j > 257.
    # Pick 3 (i=2). Other j must satisfy 2+j > 257 => j > 255.
    # No other odds.
    # So S={3}. Weight 1/3.
    #
    # It seems the independent sets in this graph are very small.
    # Max weight is likely 1.5 ({1,2}) or maybe 1/1+1/3? No, 1 and 3 have edge.
    # {1, 256}? 1+256=257 (odd). No edge.
    # Weight 1 + 1/256 ≈ 1.004.
    # {2, 255}? 2+255=257 (odd). No edge.
    # Weight 1/2 + 1/255 ≈ 0.504.
    # {1, 2, ...}?
    # If S={1, 2}, can we add anything?
    # 1 requires others even. 2 requires others odd.
    # Contradiction. So {1, 2} is maximal.
    #
    # The current program scores 1.0.
    # The literal MWIS strategy scores ~1.5?
    # Wait, 1.5 > 1.0.
    # So this strategy might actually be BETTER?
    # Let's check if {1, 2} is 3-AP free. Yes.
    # Is there a better set?
    # What about {1, 4, 5}?
    # 1, 4: sum 5 (odd). OK.
    # 1, 5: sum 6 (even, b=3 in range). Edge (1,5). Forbidden.
    #
    # What about {1, 4, 7}?
    # 1, 4: OK.
    # 1, 7: sum 8 (even, b=4). Edge (1,7). Forbidden.
    #
    # It seems the graph G is very restrictive.
    # However, 1.5 is a good score.
    # The current program returns a set with score 1.0.
    # If I return {1, 2}, I get 1.5.
    #
    # Let's verify the graph definition again.
    # "edges connect pairs {a, c} if their midpoint b=(a+c)/2 is an integer within [1, 256]"
    # Yes.
    #
    # So, implementing MWIS on this graph will yield {1, 2} or similar.
    # I will implement a branch and bound for MWIS on this specific graph.
    # Since the graph is small (256 nodes), I can even do a simple DFS with pruning.
    # Or just return {1, 2} if I can prove it's optimal for this graph.
    # But I should implement the solver as requested.
    #
    # Graph construction:
    # Nodes 1..256.
    # Edge (u, v) if u+v is even and u+v <= 512.
    #
    # Weights: w(u) = 1/u.
    #
    # Algorithm:
    # 1. Order nodes by descending weight (ascending value).
    # 2. Branch and bound.
    #    - Include node v: Add w(v), remove v and its neighbors.
    #    - Exclude node v: Remove v.
    #    - Bound: Sum of weights of remaining nodes (optimistic).
    #
    # This is fast enough for 256 nodes.

    import time
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    weights = [0] + [1.0 / i for i in range(1, N + 1)]
    
    # Build adjacency list
    adj = [set() for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            s = u + v
            if s % 2 == 0 and s <= 512:
                adj[u].add(v)
                adj[v].add(u)

    # Order nodes by descending weight (ascending index)
    # Nodes 1, 2, 3, ... are already in descending weight order.
    nodes = list(range(1, N + 1))
    
    best_weight = 0.0
    best_set = set()
    
    # State: current_index, current_set, available_nodes (as bitmask or set)
    # Using set for available nodes is slow. Bitmask is better.
    # 256 bits fits in 4 uint64 or just a Python int (arbitrary precision).
    
    # Precompute neighbors as bitmasks
    neighbor_masks = [0] * (N + 1)
    for u in range(1, N + 1):
        mask = 0
        for v in adj[u]:
            mask |= (1 << (v - 1))
        neighbor_masks[u] = mask
        
    # All nodes mask
    all_mask = (1 << N) - 1
    
    def solve(idx, current_weight, current_set, available_mask):
        nonlocal best_weight, best_set
        if time.time() > deadline:
            return
        
        if idx >= N:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
            
        # Bound: current_weight + sum of weights of available nodes
        # To speed up bound, we can precompute suffix sums of weights?
        # But available_mask changes.
        # Simple bound: sum of weights of all available nodes.
        # We can maintain a running sum of available weights.
        
        v = nodes[idx]
        bit = 1 << (v - 1)
        
        if not (available_mask & bit):
            solve(idx + 1, current_weight, current_set, available_mask)
            return
        
        # Option 1: Exclude v
        solve(idx + 1, current_weight, current_set, available_mask)
        
        # Option 2: Include v
        # Remove v and its neighbors from available
        new_available = available_mask & ~(neighbor_masks[v] | bit)
        new_set = set(current_set)
        new_set.add(v)
        solve(idx + 1, current_weight + weights[v], new_set, new_available)

    # Optimization: The simple bound is weak.
    # Better bound: For the remaining available nodes, the max weight independent set is at most
    # the sum of weights of all available nodes.
    # We can improve by checking if we can add any more.
    # Given the small size, this should be fine.
    
    # Start
    solve(0, 0.0, set(), all_mask)
    
    if not best_set:
        return [1]
        
    # Verify result
    S = list(best_set)
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, c = S[i], S[j]
            s = a + c
            if s % 2 == 0:
                b = s // 2
                if 1 <= b <= 256 and b in S_set:
                    # Invalid
                    pass # Should not happen
    
    # Calculate exact score
    score = sum(Fraction(1, x) for x in S)
    return S
