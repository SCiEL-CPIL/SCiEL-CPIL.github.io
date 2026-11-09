def construct():
    import time
    from math import gcd, lcm
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for 60s limit

    # Candidates are 2..256
    # Index 0 corresponds to value 2, Index 1 to 3, ..., Index 254 to 256
    N = 255
    values = list(range(2, 257))
    weights = [1.0 / v for v in values]
    
    # Conflict graph: edge (i, j) exists if lcm(values[i], values[j]) <= 256
    # We want Maximum Weight Independent Set in this graph.
    # This is equivalent to Maximum Weight Clique in the Complement Graph.
    
    # Build adjacency matrix for the COMPPLEMENT graph (Compatibility graph)
    # adj[i][j] is True if values[i] and values[j] are COMPATIBLE (lcm > 256)
    # Note: A node is always compatible with itself in the context of clique expansion, 
    # but we only consider distinct pairs for the condition. 
    # In Bron-Kerbosch, we look for cliques in the compatibility graph.
    
    adj = [[False] * N for _ in range(N)]
    for i in range(N):
        vi = values[i]
        for j in range(i + 1, N):
            vj = values[j]
            # Check compatibility
            # lcm(vi, vj) > 256
            # To avoid large numbers, we can check if vi * vj / gcd(vi, vj) > 256
            g = gcd(vi, vj)
            # lcm = (vi // g) * vj
            # We want (vi // g) * vj > 256
            # To avoid overflow (though Python handles big ints, it's slower), 
            # we can check vi // g > 256 // vj
            # But careful with integer division.
            # Simpler: just compute lcm. N=255 is small.
            if lcm(vi, vj) > 256:
                adj[i][j] = True
                adj[j][i] = True
    
    # Weights are 1/x. We want to maximize sum of weights.
    # Sort nodes by weight descending to help branching?
    # Bron-Kerbosch with pivot doesn't strictly require sorted input, 
    # but processing high-weight nodes first helps find a good incumbent early.
    
    # We will use a recursive Bron-Kerbosch with pivot.
    # To improve pruning, we can use a coloring bound on the *cardinality* 
    # but that doesn't bound the *weight* directly unless weights are uniform.
    # However, we can use the sum of weights of the remaining candidates as a bound.
    # A tighter bound: 
    # If we color the remaining subgraph, the max weight of a clique in a color class 
    # is the max weight in that class? No, a clique can have at most one vertex from each color class.
    # So the max weight of a clique in the remaining graph is <= sum of (max weight in each color class).
    # This is a valid upper bound!
    
    # Let's implement greedy coloring on the remaining candidates P.
    # The colors will be assigned such that no two adjacent vertices in the *compatibility* graph 
    # (which are non-adjacent in the conflict graph, i.e., they form an independent set in conflict, 
    # wait. We are finding a CLIQUE in the compatibility graph.
    # A clique in compatibility graph is an independent set in conflict graph.
    # Greedy coloring is usually done on the graph itself to bound the clique size.
    # If we color the compatibility graph, a clique can have at most one vertex of each color?
    # No, in a colored graph, a clique can contain multiple vertices of the same color if they are not adjacent?
    # No, standard graph coloring: adjacent vertices have different colors.
    # In a clique, all vertices are adjacent. So all vertices in a clique must have distinct colors.
    # Therefore, the size of a clique <= number of colors.
    # For weights: The weight of a clique <= sum of the maximum weight vertex in each color class?
    # Yes, because the clique picks at most one vertex from each color class (since all picked vertices must have distinct colors).
    # So, UpperBound = sum_{c in colors} (max_{v in P, color(v)=c} weight(v)).
    
    # This bound is much tighter than sum of all weights.
    
    best_sum = 0.0
    best_clique = []
    
    # To speed up coloring, we can pre-sort or just do it on the fly.
    # N=255 is small enough.
    
    def color_bound(P):
        """
        P is a list of node indices.
        Returns an upper bound on the max weight of a clique in the subgraph induced by P.
        Uses greedy coloring on the compatibility graph induced by P.
        """
        if not P:
            return 0.0
        
        # Greedy coloring
        # We want to assign colors 1, 2, ... to vertices in P.
        # Adjacent vertices in compatibility graph must have different colors.
        # We process vertices in some order. To get a tight bound (low number of colors, or low max weights per color),
        # we should process vertices with high weights first? 
        # Actually, the bound is sum of max weights per color.
        # If we assign high-weight vertices to distinct colors, the sum of max weights will be high.
        # We want a LOWER bound on the colors? No, we want a valid UPPER bound on the clique weight.
        # The formula is: Sum over colors of (Max weight in that color).
        # To make this bound as TIGHT (small) as possible, we want to group high-weight vertices into the SAME color if possible?
        # But they can only be in the same color if they are NOT adjacent in the compatibility graph.
        # If they are not adjacent, they can't be in the same clique anyway.
        # So grouping non-adjacent high-weight vertices into the same color is GOOD for the bound.
        # Standard greedy coloring (first-fit) tends to use few colors.
        
        # Let's try a simple greedy coloring.
        # Order P by weight descending? Or ascending?
        # If we process high weight first, we assign them colors 1, 2, 3...
        # Then lower weights might fit into existing colors.
        # This seems reasonable.
        
        sorted_P = sorted(P, key=lambda x: weights[x], reverse=True)
        
        color_classes = [] # list of lists of nodes
        # To speed up, we can check against existing classes.
        # N is small, O(N^2) is fine.
        
        for v in sorted_P:
            placed = False
            # Try to put v in an existing color class
            # v can be in class c if v is not adjacent to any node in class c
            for c_idx, class_nodes in enumerate(color_classes):
                compatible_with_all = True
                for u in class_nodes:
                    if adj[v][u]:
                        compatible_with_all = False
                        break
                if compatible_with_all:
                    class_nodes.append(v)
                    placed = True
                    break
            
            if not placed:
                color_classes.append([v])
        
        # Calculate bound
        bound = 0.0
        for class_nodes in color_classes:
            # Max weight in this class
            max_w = max(weights[v] for v in class_nodes)
            bound += max_w
            
        return bound

    def bron_kerbosch(R, P, X):
        nonlocal best_sum, best_clique
        
        # Pruning check
        if not P:
            # Check if current R is better
            # R is a clique in compatibility graph
            current_sum = sum(weights[v] for v in R)
            if current_sum > best_sum:
                best_sum = current_sum
                best_clique = R[:]
            return
        
        # Calculate upper bound for this branch
        # If current_sum + bound(P) <= best_sum, prune
        # Note: R is already fixed. We only add from P.
        # We need to pass R's sum to avoid recalculating? 
        # For simplicity, we calculate bound on P.
        # If sum(R) + bound(P) <= best_sum, return.
        
        # To avoid expensive coloring if the bound is clearly not going to work, 
        # we can use a simple sum of weights in P first.
        # sum_w_P = sum(weights[v] for v in P)
        # if current_sum + sum_w_P <= best_sum: return
        
        # Let's use the coloring bound.
        # Optimization: If P is small, coloring is fast.
        bound_val = color_bound(P)
        
        # We need current_sum. 
        # Passing R and calculating sum every time is O(N). 
        # We can pass current_sum as an argument.
        
        # Let's refactor to pass current_sum.
        # But the signature is fixed by the recursive call structure.
        # I'll modify the inner function to accept current_sum.
        
        # Wait, I defined bron_kerbosch above. I need to redefine it or use a wrapper.
        # Let's rewrite the recursive function properly.
        
        pass # Placeholder, will replace below

    # Redefine with current_sum
    def bk(R, P, X, current_sum):
        nonlocal best_sum, best_clique
        
        if not P:
            if current_sum > best_sum:
                best_sum = current_sum
                best_clique = R[:]
            return
        
        # Prune
        # Simple bound: sum of weights in P
        # Tighter bound: coloring bound
        # We'll use coloring bound if P is not too large, otherwise simple sum.
        # Actually coloring is O(N^2) which is fine for N=255.
        
        bound_val = color_bound(P)
        
        if current_sum + bound_val <= best_sum:
            return
        
        # Pivot selection
        # Choose u from P U X that maximizes |P \cap N(u)|
        # In compatibility graph, N(u) are neighbors.
        # We want to minimize the number of vertices we need to process.
        # We iterate over P \ N(u).
        
        # Select pivot u
        # To speed up, we can just pick a random or first one if N is small?
        # Standard is max degree.
        
        max_deg = -1
        pivot = -1
        for u in P:
            # Count neighbors in P
            deg = 0
            for v in P:
                if u != v and adj[u][v]:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                pivot = u
        
        # If P is empty, handled above.
        # If X is not empty, we should consider P U X for pivot?
        # Standard Bron-Kerbosch uses P U X.
        # But for Maximum Clique, using P U X is correct.
        
        # However, implementing pivot with P U X is slightly more complex.
        # Given N=255, even without pivot, it might be okay?
        # No, without pivot it can be exponential.
        # Let's implement pivot properly.
        
        # Candidates for pivot are P U X.
        # But we only branch on P \ N(pivot).
        
        # To simplify and ensure correctness, let's use a standard implementation.
        # Since Python is slow, we need to be careful.
        
        # Let's try a simpler pivot: just pick a node from P with max degree in P.
        # This is less optimal but simpler.
        
        if pivot == -1:
            return # Should not happen if P is not empty

        # Branch on P \ N(pivot)
        # We iterate through vertices in P that are NOT neighbors of pivot.
        # Note: If pivot is in P, N(pivot) does not include pivot itself in standard graph def, 
        # but in BK, we usually exclude the pivot's neighbors.
        # The set to iterate is P \ N(pivot).
        
        # To improve order, we process high weight vertices first.
        # Sort P \ N(pivot) by weight descending.
        
        candidates = [v for v in P if not adj[pivot][v]]
        # Note: adj[pivot][pivot] is False, so if pivot in P, it is in candidates.
        # This is correct.
        
        # Sort by weight descending
        candidates.sort(key=lambda x: weights[x], reverse=True)
        
        for v in candidates:
            # Recurse
            # New R = R U {v}
            # New P = P \cap N(v)
            # New X = X \cap N(v)
            
            new_R = R + [v]
            new_P = [u for u in P if adj[v][u]]
            new_X = [u for u in X if adj[v][u]]
            
            # Check time
            if time.time() > deadline:
                return
            
            bk(new_R, new_P, new_X, current_sum + weights[v])
            
            # Move v to X
            P.remove(v)
            X.append(v)
            
            # Check time
            if time.time() > deadline:
                return

    # Initial call
    # R = []
    # P = all nodes
    # X = []
    # current_sum = 0
    
    # To help find a good incumbent early, we can run a greedy heuristic first.
    # Greedy: Pick highest weight, remove conflicts, repeat.
    P_init = list(range(N))
    R_greedy = []
    sum_greedy = 0.0
    P_work = P_init[:]
    while P_work:
        # Pick max weight
        v = max(P_work, key=lambda x: weights[x])
        R_greedy.append(v)
        sum_greedy += weights[v]
        # Remove v and its non-neighbors (conflicts) from P_work
        # Keep only neighbors of v
        P_work = [u for u in P_work if u != v and adj[v][u]]
        
    best_sum = sum_greedy
    best_clique = R_greedy[:]

    # Now run exact search
    # We need to ensure we don't spend too much time.
    # The coloring bound is expensive. 
    # For N=255, it might be okay.
    
    # One optimization: If the best_sum found by greedy is very high, 
    # the bound will prune many branches.
    
    bk([], list(range(N)), [], 0.0)
    
    # Convert indices back to values
    result = [values[i] for i in best_clique]
    
    # Verify result
    # Check non-empty
    if not result:
        return [2] # Fallback
        
    # Check constraints
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                # Invalid, fallback to greedy
                return [values[i] for i in R_greedy]
                
    return result
