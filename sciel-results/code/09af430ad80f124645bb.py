import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 15
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We want to maximize edges. 
    # Strategy: Try to include as many edges as possible.
    # We can model this as finding a subset of edges.
    # However, exact search over 2^105 is impossible.
    
    # Alternative Structural Approach:
    # The constraint is: For all s, t, if deg(s) == deg(t), then dist(s, t) != 3.
    # Note: dist(s, t) is the shortest path distance. If dist(s, t) < 3 (i.e., 1 or 2), 
    # does a 3-edge path exist?
    # The problem statement says: "Reject if any simple three-edge path has endpoints of equal degree".
    # A "simple three-edge path" is s-u-v-t with distinct vertices s,u,v,t and edges (s,u), (u,v), (v,t).
    # This implies s and t are at distance exactly 3 in the subgraph formed by that path, 
    # but s and t might be connected by a shorter path in the full graph.
    # So the condition is: There does NOT exist a walk s-u-v-t of length 3 with distinct vertices 
    # such that deg(s) == deg(t).
    # Wait, "simple path" usually means no repeated vertices. 
    # So s, u, v, t are all distinct.
    # And the edges (s,u), (u,v), (v,t) must exist.
    
    # Let's use a randomized local search with a better cost function and simulated annealing,
    # but with a crucial optimization: efficient violation checking.
    
    # Actually, let's try a different angle: 
    # If we can ensure that all vertices have DISTINCT degrees, the constraint is trivially satisfied.
    # Is it possible to have 15 distinct degrees in a graph of 15 vertices?
    # Degrees must be in {0, ..., 14}.
    # Sum of degrees must be even.
    # Sum of 0..14 = 105 (odd). So we cannot have exactly the set {0, 1, ..., 14}.
    # But we can have 15 distinct degrees if we skip one and add another? No, max degree is 14.
    # So we MUST have at least one pair of vertices with the same degree (Pigeonhole Principle).
    # In fact, since sum is even, we can't have 0,1,...,14.
    # We can have e.g. 0,1,2,3,4,5,6,7,8,9,10,11,12,13,13 (sum = 104, even).
    # Or 1,2,3,4,5,6,7,8,9,10,11,12,13,14,14 (sum = 125? No. 1+..+14 = 105. 105+14=119 odd. Wait.)
    # Let's check parity.
    # If we have degrees d_1..d_15. Sum(d_i) = 2E.
    # If we want distinct degrees, we need 15 distinct integers in [0, 14].
    # The only set of 15 distinct integers in [0, 14] is {0, 1, ..., 14}.
    # Sum = 105. Odd. Impossible.
    # Therefore, at least two vertices must share a degree.
    
    # So the problem is: Construct a graph on 15 vertices maximizing edges such that
    # for any pair (s, t) with deg(s) == deg(t), there is no simple path of length 3 between them.
    
    # Let's use a heuristic search:
    # 1. Start with a random dense graph.
    # 2. Calculate degrees.
    # 3. Identify "bad pairs" (s, t) where deg(s) == deg(t).
    # 4. For each bad pair, check if a 3-path exists.
    # 5. If yes, remove an edge from that path.
    # 6. Repeat.
    
    # To improve on the previous approach (which scored 0.5, likely due to getting stuck or being too slow/inefficient),
    # we will use Simulated Annealing with a robust violation counter.
    
    def check_violations(edges):
        """
        Returns the number of violating 3-paths.
        A violation is a simple path s-u-v-t such that deg(s) == deg(t).
        """
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        violations = 0
        # Iterate over all potential endpoints s, t
        # To be efficient, group by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            # For each pair in this group, check for 3-path
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    # Check if there is a simple path s-u-v-t
                    # s is connected to u, u to v, v to t.
                    # s, u, v, t distinct.
                    # u in adj[s], v in adj[u] (v != s, v != t), t in adj[v] (t != s, t != u)
                    # Actually, just check existence of ANY such path.
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                # We found a path s-u-v-t
                                # s, u, v, t are distinct?
                                # u != s (adj)
                                # u != t (checked)
                                # v != s (checked)
                                # v != t (checked)
                                # s != t (group check)
                                # So yes, simple path.
                                found = True
                                break
                        if found:
                            break
                    if found:
                        violations += 1
        return violations

    def get_violating_paths(edges):
        """
        Returns a list of tuples (s, u, v, t) representing violating paths.
        """
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    s = vertices[i]
                    t = vertices[j]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t:
                                continue
                            if t in adj[v]:
                                paths.append((s, u, v, t))
        return paths

    # Simulated Annealing
    # Start with a complete graph? No, too many violations.
    # Start with a random graph with ~50% edges?
    
    best_edges = None
    best_violations = float('inf')
    best_score = -1
    
    # We will try multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Let's start with a random subset of edges, biased towards higher density
        # Or start with complete and remove?
        # Starting with complete is risky because violation count is huge.
        # Let's start with a random graph with probability p=0.7
        current_edges = set()
        for u, v in all_edges:
            if random.random() < 0.7:
                current_edges.add((u, v))
        
        # Calculate initial violations
        current_violations = check_violations(current_edges)
        if current_violations == 0:
            # Found a valid graph immediately? Unlikely with random.
            # But if so, try to add more edges.
            pass
        
        T = 10.0
        T_min = 0.1
        alpha = 0.999
        
        # Local search loop
        steps = 0
        max_steps = 5000
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            if current_violations == 0:
                # Valid graph. Try to add an edge to increase density.
                # Pick a random non-edge
                non_edges = [e for e in all_edges if e not in current_edges]
                if not non_edges:
                    break # Complete graph and valid? Impossible.
                e = random.choice(non_edges)
                new_edges = set(current_edges)
                new_edges.add(e)
                new_violations = check_violations(new_edges)
                if new_violations == 0:
                    current_edges = new_edges
                    current_violations = 0
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_violations - 0 # We want to stay at 0, but adding edge breaks it
                    # Actually, if we are at 0 violations, we only accept if new is 0.
                    # Otherwise we might get stuck.
                    # Let's allow temporary violations if we are far from optimal?
                    # No, if we have a valid graph, we want to keep it valid if possible.
                    # If adding an edge creates violations, we can still accept it if T is high,
                    # hoping to remove them later.
                    if random.random() < pow(2.718, -delta / T):
                        current_edges = new_edges
                        current_violations = new_violations
            else:
                # We have violations. Try to remove an edge involved in a violation.
                paths = get_violating_paths(current_edges)
                if not paths:
                    # Inconsistent state? Should not happen if check_violations > 0
                    break
                
                # Pick a random violating path
                s, u, v, t = random.choice(paths)
                # Pick a random edge from this path to remove
                candidate_edges = [(s, u), (u, v), (v, t)]
                # Normalize edges
                cand_set = set()
                for a, b in candidate_edges:
                    if a > b: a, b = b, a
                    cand_set.add((a, b))
                
                e = random.choice(list(cand_set))
                
                if e in current_edges:
                    new_edges = set(current_edges)
                    new_edges.discard(e)
                    new_violations = check_violations(new_edges)
                    
                    delta = new_violations - current_violations
                    # We want to decrease violations, so delta should be negative.
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_edges = new_edges
                        current_violations = new_violations
            
            # Update temperature
            T *= alpha
            
            # Track best
            if current_violations == 0:
                score = len(current_edges)
                if score > best_score:
                    best_score = score
                    best_edges = list(current_edges)
        
        # End of restart loop
        if time.time() > deadline:
            break
    
    # If no valid graph found, return empty or best effort?
    # The task says "Reject if...". So we must return a valid graph.
    # If we didn't find one, we need a fallback.
    # A star graph is valid?
    # Center 0, leaves 1..14.
    # Deg(0) = 14. Deg(leaves) = 1.
    # Pairs with same degree: all leaves.
    # Path between two leaves s, t: s-0-t is length 2.
    # Is there a path of length 3? s-0-x-t? No, x would have to be connected to 0 and t.
    # In a star, leaves are only connected to 0. So no path of length 3 between leaves.
    # So a star graph is valid.
    # Edges: 14.
    
    if best_edges is None:
        # Fallback: Star graph
        best_edges = [[0, i] for i in range(1, n)]
    else:
        # Ensure types are correct
        best_edges = [[int(u), int(v)] for u, v in best_edges]
        
        # Final validation check just in case
        if check_violations(best_edges) != 0:
             # If our SA produced an invalid result due to bugs, fallback to star
             best_edges = [[0, i] for i in range(1, n)]
    
    return best_edges
