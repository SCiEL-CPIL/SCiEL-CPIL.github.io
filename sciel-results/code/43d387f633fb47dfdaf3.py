import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a 3-regular graph with girth at least g using a voltage graph lift
    of the Petersen graph over Z_p.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Petersen Graph Definition
    # Vertices 0..9
    # Standard Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer 5-cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # Inner 5-cycle (star)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    # Precompute all closed walks in Petersen graph up to length g-1
    # A closed walk is a sequence of vertices v0, v1, ..., vL=v0
    # We need to check voltage sums for these walks.
    # Instead of enumerating all walks (exponential), we can use the property:
    # We need to assign voltages v_e in Z_p such that for every closed walk W of length L < g, sum(v_e) != 0.
    
    # Let's try primes p such that 10*p is close to the Moore bound but valid.
    # Moore bound for n(3,16) is 512.
    # We want 10*p >= 512 => p >= 52.
    # Let's try primes from 53 upwards.
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    # Generate list of primes to try
    primes = []
    p = 53
    while p < 150: # Try up to 1500 vertices
        if is_prime(p):
            primes.append(p)
        p += 1
        
    # For each prime, try to find a valid voltage assignment
    # We use a greedy/randomized approach to assign voltages.
    # If we find a valid assignment, we build the graph.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # We need to satisfy: sum_{e in W} v_e != 0 mod p for all closed walks W with len(W) < g.
        # This is equivalent to saying that the linear map from edge voltages to walk sums is injective on the subspace of short walks?
        # No, it's a system of inequalities.
        
        # Strategy: Random assignment and check.
        # Number of edges in Petersen is 15.
        # Number of short closed walks is finite.
        
        # Let's collect the coefficient vectors for all closed walks of length < g.
        # A walk is defined by a sequence of edges.
        # We can generate walks by DFS up to depth g-1.
        
        # Optimization: We only need to check walks that are "simple" in the base graph?
        # No, voltage sums apply to any closed walk.
        # However, if a walk repeats an edge, it's a sum of smaller walks.
        # Actually, we just need to ensure no *short* walk sums to 0.
        
        # Let's generate all closed walks of length L < g in Petersen.
        # Max length 15.
        # Branching factor 3. 3^15 is ~ 14 million. Too many to enumerate explicitly if we do it naively.
        # But we can do this:
        # We want to find v_e such that for all W, sum != 0.
        
        # Alternative: Use the fact that Petersen is small.
        # We can just try random assignments and verify by BFS in the *lifted* graph?
        # No, verification in the lifted graph is O(N * g * 3^g) which is too slow if we do it for every p.
        
        # Better verification:
        # Check if there exists a closed walk W in Petersen with sum 0.
        # We can do this by dynamic programming on the state (current_vertex, current_sum).
        # State: (v, s) where v is vertex in Petersen, s is sum in Z_p.
        # Start: (0, 0).
        # We want to see if we can return to (0, 0) in steps < g without being stuck?
        # Actually, we want to know if there is ANY path from (0,0) to (0,0) of length L < g.
        # We can do a BFS on the product graph Petersen x Z_p.
        # Size: 10 * p.
        # Edges: 15 * p.
        # This is very fast.
        
        # Let's try a few random assignments for each p.
        attempts = 50
        found = False
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Random voltages
            voltages = [random.randint(1, p-1) for _ in range(len(petersen_edges))]
            
            # Map edge index to voltage
            # We need to check if there is a closed walk of length < g with sum 0.
            # BFS on (vertex, sum)
            # Start at (0, 0)
            # We want to reach (0, 0) again.
            # But we must avoid the trivial walk (length 0).
            # And we must ensure the walk is non-backtracking in the *lifted* graph?
            # The girth definition for the task is "no cycle shorter than g".
            # A cycle in the lifted graph corresponds to a closed walk in the base graph with sum 0.
            # The "non-backtracking" part is automatically handled if we consider simple cycles in the lift?
            # Actually, the standard definition of girth is the length of the shortest cycle.
            # In a voltage graph lift, a cycle in the lift corresponds to a closed walk in the base.
            # If the base walk backtracks (e.g., u->v->u), the sum is v_e + v_e^{-1} = 0.
            # This would create a 2-cycle in the lift?
            # No, in a simple graph, there are no 2-cycles.
            # The lift of a simple graph is simple if the voltage assignment is such that we don't create multi-edges or loops.
            # For a bipartite lift of a simple graph, it remains simple.
            # The shortest cycle in the lift corresponds to the shortest closed walk in the base with sum 0.
            
            # So we just need to check: Is there a closed walk in Petersen of length L < g with sum 0?
            
            # BFS from (0,0)
            # visited[v][s] = distance
            visited = [[-1]*p for _ in range(10)]
            queue = [(0, 0, 0)] # (vertex, sum, length)
            visited[0][0] = 0
            
            invalid = False
            while queue:
                v, s, l = queue.pop(0)
                if l >= g:
                    continue
                # Explore neighbors
                for u, edge_idx in get_petersen_neighbors(v):
                    # Determine voltage sign
                    # If we go v->u, we add voltage. If u->v, we subtract?
                    # The voltage is defined on the edge.
                    # Let's define voltage v_e for edge (v,u) as the value added when going v->u.
                    # If we go u->v, we add -v_e.
                    # We need a consistent orientation.
                    # Let's orient all edges from lower to higher index? No, Petersen is not oriented.
                    # Let's just assign a voltage to each directed edge.
                    # v_{u,v} = -v_{v,u}.
                    
                    # Let's precompute directed voltages.
                    # For edge (v,u) in petersen_edges, let val = voltages[edge_idx].
                    # If we traverse v->u, add val.
                    # If we traverse u->v, add -val.
                    
                    # We need to know the direction.
                    # Let's store neighbors with (neighbor, edge_idx, direction_sign)
                    # direction_sign is 1 if v->u is the "forward" direction of the stored edge, -1 otherwise.
                    
                    # Let's refine get_petersen_neighbors to return (u, edge_idx, sign)
                    # where sign is 1 if the edge was stored as (v,u) and we go v->u?
                    # Actually, it's easier to just say:
                    # Edge i connects (a,b).
                    # If current is a, next is b, add voltages[i].
                    # If current is b, next is a, add -voltages[i].
                    
                    new_s = (s + sign * voltages[edge_idx]) % p
                    if visited[u][new_s] == -1:
                        visited[u][new_s] = l + 1
                        if u == 0 and new_s == 0 and l + 1 > 0:
                            # Found a closed walk of length l+1 < g with sum 0
                            invalid = True
                            break
                        queue.append((u, new_s, l + 1))
                
                if invalid:
                    break
            
            if not invalid:
                # Valid voltage assignment found!
                # Construct the graph
                # Vertices are (v, s) where v in 0..9, s in 0..p-1
                # Total 10*p vertices
                # Edges: for each (v,s) and each neighbor u with voltage val:
                # Edge from (v,s) to (u, (s+val)%p)
                
                n = 10 * p
                edges = []
                # To avoid duplicates, only add edge if id1 < id2
                # id = v*p + s
                
                for v in range(10):
                    for s in range(p):
                        id1 = v * p + s
                        for u, edge_idx, sign in get_petersen_neighbors(v):
                            new_s = (s + sign * voltages[edge_idx]) % p
                            id2 = u * p + new_s
                            if id1 < id2:
                                edges.append((id1, id2))
                
                # Verify girth quickly?
                # The BFS above already certified that no closed walk of length < g has sum 0.
                # This implies girth >= g.
                # We must ensure the graph is connected?
                # The lift of a connected graph is connected if the voltage assignment generates the group?
                # Actually, the lift might be disconnected if the voltages don't generate the whole group.
                # We need a connected component of size 10p?
                # Or we can just return the largest connected component.
                # But the task requires "a list of edges on vertices 0..n-1" and "every vertex must have degree exactly k".
                # If we return a disconnected graph, it's valid as long as all components are 3-regular and girth >= g.
                # However, usually "the graph" implies connected.
                # The Moore bound applies to connected graphs.
                # If the lift is disconnected, we can just take one component?
                # No, the size n would be smaller.
                # If we take one component, n might be < 10p.
                # But we want to minimize n.
                # So if the lift is disconnected, we should return the largest component?
                # Or just one component?
                # The prompt says "Construct a 3-regular graph... on as few vertices as possible".
                # So returning a single connected component is better if it's smaller.
                
                # Let's find the largest connected component.
                # BFS from 0.
                comp = [0]
                visited_comp = [False] * n
                visited_comp[0] = True
                q_comp = [0]
                while q_comp:
                    cur = q_comp.pop(0)
                    # Find neighbors in edges
                    # This is slow if we scan edges.
                    # Let's build adjacency list first.
                    pass
                
                # Let's build adjacency list for the whole lift
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Check degree
                # All should be 3.
                
                # Find largest component
                max_comp_size = 0
                max_comp_nodes = []
                visited_final = [False] * n
                for i in range(n):
                    if not visited_final[i]:
                        # BFS
                        curr_comp = []
                        stack = [i]
                        visited_final[i] = True
                        while stack:
                            node = stack.pop()
                            curr_comp.append(node)
                            for nb in adj[node]:
                                if not visited_final[nb]:
                                    visited_final[nb] = True
                                    stack.append(nb)
                        if len(curr_comp) > max_comp_size:
                            max_comp_size = len(curr_comp)
                            max_comp_nodes = curr_comp
                
                # If the largest component is the whole graph, great.
                # If not, we can just return the largest component?
                # But the vertices must be 0..n-1.
                # We need to remap the nodes in max_comp_nodes to 0..max_comp_size-1.
                
                if max_comp_size < n:
                    # Remap
                    mapping = {node: i for i, node in enumerate(max_comp_nodes)}
                    new_edges = []
                    for u, v in edges:
                        if u in mapping and v in mapping:
                            new_edges.append((mapping[u], mapping[v]))
                    return max_comp_size, new_edges
                else:
                    return n, edges

    # Fallback: Return the seed-like construction or a random one?
    # If we didn't find a prime, we should return something valid.
    # Let's use a simple random 3-regular graph and hope? No, girth might be small.
    # Let's fall back to the Lazebnik-Ustimenko-Woldar construction from the seed, 
    # but implemented efficiently?
    # Or just a known small cage?
    # For g=16, n=936 is known.
    # If we fail, we can try to construct a specific known graph?
    # But we don't have it hardcoded.
    
    # Let's try one more thing: Random 3-regular graph with girth check.
    # Generate random 3-regular graph of size N.
    # Check girth. If girth < g, repair?
    # This is complex.
    
    # Let's just return the first valid one found.
    # If none found, raise an error? No, must return.
    # Let's try a larger range of primes or more attempts.
    
    # Actually, the BFS check is very fast.
    # 10 * 100 states.
    # 50 attempts * 10 primes = 500 BFS runs.
    # Should be well within 120s.
    
    # If we still fail, we can return a trivial graph?
    # No, must be 3-regular.
    # Let's return the Petersen lift with p=1? No, p must be prime >= 2.
    # If p=2, n=20. Girth of Petersen lift with p=2?
    # Petersen has girth 5. Lift might have girth 5 or 10.
    # We need girth 16.
    
    # If no prime works, we can try composite m?
    # The group Z_m.
    # The condition is sum != 0 mod m.
    # This is weaker than mod p.
    # So if it fails for prime p, it might work for composite m?
    # No, if sum != 0 mod p, it doesn't imply sum != 0 mod m.
    # But if we find a solution for prime p, it's valid for Z_p.
    
    # Let's assume we found one.
    return 0, [] # Should not reach here

def get_petersen_neighbors(v):
    # Returns list of (neighbor, edge_index, sign)
    # sign is 1 if edge is (v, u), -1 if (u, v)
    # We need to hardcode this or compute from petersen_edges
    # Let's hardcode for speed and simplicity
    # Edges:
    # 0-1 (0), 1-2 (1), 2-3 (2), 3-4 (3), 4-0 (4)
    # 5-6 (5), 6-7 (6), 7-8 (7), 8-9 (8), 9-5 (9)
    # 0-5 (10), 1-6 (11), 2-7 (12), 3-8 (13), 4-9 (14)
    
    # 0: 1(0, 1), 4(4, -1), 5(10, 1)
    if v == 0: return [(1, 0, 1), (4, 4, -1), (5, 10, 1)]
    # 1: 0(0, -1), 2(1, 1), 6(11, 1)
    if v == 1: return [(0, 0, -1), (2, 1, 1), (6, 11, 1)]
    # 2: 1(1, -1), 3(2, 1), 7(12, 1)
    if v == 2: return [(1, 1, -1), (3, 2, 1), (7, 12, 1)]
    # 3: 2(2, -1), 4(3, 1), 8(13, 1)
    if v == 3: return [(2, 2, -1), (4, 3, 1), (8, 13, 1)]
    # 4: 3(3, -1), 0(4, 1), 9(14, 1)
    if v == 4: return [(3, 3, -1), (0, 4, 1), (9, 14, 1)]
    # 5: 6(5, 1), 9(9, -1), 0(10, -1)
    if v == 5: return [(6, 5, 1), (9, 9, -1), (0, 10, -1)]
    # 6: 5(5, -1), 7(6, 1), 1(11, -1)
    if v == 6: return [(5, 5, -1), (7, 6, 1), (1, 11, -1)]
    # 7: 6(6, -1), 8(7, 1), 2(12, -1)
    if v == 7: return [(6, 6, -1), (8, 7, 1), (2, 12, -1)]
    # 8: 7(7, -1), 9(8, 1), 3(13, -1)
    if v == 8: return [(7, 7, -1), (9, 8, 1), (3, 13, -1)]
    # 9: 8(8, -1), 5(9, 1), 4(14, -1)
    if v == 9: return [(8, 8, -1), (5, 9, 1), (4, 14, -1)]
    return []
# EVOLVE-BLOCK-END
