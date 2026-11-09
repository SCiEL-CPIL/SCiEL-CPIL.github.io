import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 85.0  # 5 seconds margin
    
    # Fallback: Two disjoint K_{3,3}
    # Cycle lengths: 4, 6. Value: 2.
    def get_fallback():
        edges = []
        p1 = [0, 1, 2]
        p2 = [3, 4, 5]
        for u in p1:
            for v in p2:
                edges.append([int(u), int(v)])
        p3 = [6, 7, 8]
        p4 = [9, 10, 11]
        for u in p3:
            for v in p4:
                edges.append([int(u), int(v)])
        return edges

    # Try to find a graph with Value 1 (only 4-cycles)
    # Search space: 3-regular bipartite graphs on 12 vertices
    # Partitions: A={0..5}, B={6..11}
    
    best_edges = None
    best_value = 3 # Higher than fallback's 2
    
    # Random search
    # We generate random 3-regular bipartite graphs
    # A simple way: for each u in A, pick 3 distinct neighbors in B.
    # This doesn't guarantee degree 3 for B.
    # Better: use a random permutation or matching based approach.
    
    # Since 12 vertices is small, we can iterate through random permutations
    # or just random edge sets and check degrees.
    
    # More efficient: 
    # 1. Create a list of all possible edges between A and B.
    # 2. Randomly select edges such that each vertex in A has degree 3 and each in B has degree 3.
    # This is equivalent to finding a 3-regular bipartite graph.
    
    # We can simulate this by taking 3 random permutations of B and combining them,
    # but we must avoid duplicate edges.
    
    # Let's try a simple random walk / greedy approach.
    # Initialize adjacency lists.
    
    A = list(range(6))
    B = list(range(6, 12))
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Generate a random 3-regular bipartite graph
        # Method:
        # 1. For each u in A, assign 3 random neighbors from B.
        # 2. Check if degrees in B are all 3.
        # 3. If yes, check cycles.
        
        # To increase success rate of degree 3 in B:
        # We can use the configuration model or just retry.
        # Given the small size, retrying is fine.
        
        deg_A = {u: 0 for u in A}
        deg_B = {v: 0 for v in B}
        edges = []
        adj = {u: set() for u in A}
        adj.update({v: set() for v in B})
        
        # Randomly assign neighbors for A
        # We need to ensure B gets degree 3.
        # This is a bipartite matching problem with capacity 3.
        # We can just randomly pick 3 neighbors for each A node,
        # and then check if B nodes have degree 3.
        
        valid = True
        for u in A:
            neighbors = random.sample(B, 3)
            for v in neighbors:
                if deg_B[v] >= 3:
                    valid = False
                    break
                # Add edge
                edges.append([u, v])
                adj[u].add(v)
                adj[v].add(u)
                deg_B[v] += 1
            if not valid:
                break
            deg_A[u] = 3
            
        if not valid:
            continue
            
        # Check if all B have degree 3
        if any(deg_B[v] != 3 for v in B):
            continue
            
        # Check if graph is simple (it is by construction)
        # Check triangle-free (bipartite by construction)
        
        # Enumerate simple cycles to find distinct lengths
        # This is expensive. 
        # Optimization: 
        # If we find a 6-cycle, value >= 2 (since 4-cycles exist in any 3-reg bip graph? 
        # Actually, does every 3-reg bip graph have a 4-cycle?
        # No. But if it has a 4-cycle and a 6-cycle, value >= 2.
        # If it has only 4-cycles, value = 1.
        # If it has no 4-cycles, min length >= 6.
        
        # To check for 4-cycles:
        # Two vertices in A share 2 neighbors in B.
        has_4_cycle = False
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                common = adj[A[i]] & adj[A[j]]
                if len(common) >= 2:
                    has_4_cycle = True
                    break
            if has_4_cycle:
                break
        
        # To check for 6-cycles:
        # This is harder.
        # However, if we assume Value 1 is rare, we can just check if has_4_cycle is False.
        # If no 4-cycles, then min length >= 6.
        # If min length is 6, and no other lengths, Value=1.
        
        # Let's just run a BFS/DFS to find all cycle lengths for this small graph.
        # 12 vertices is small enough.
        
        cycle_lengths = set()
        
        # Simple DFS for cycles
        # This is a bit slow in Python, but 12 vertices is small.
        # We can use networkx if available? No, standard library only.
        # We can implement a simple cycle enumeration.
        
        # For speed, we can just check if 4-cycle exists.
        # If yes, we check if 6-cycle exists.
        # If 4 and 6, value=2.
        # If 4 only, value=1.
        # If no 4, we check 6, 8, etc.
        
        # Given the high probability of 6-cycles in 3-reg bip graphs,
        # we might not find Value 1.
        
        # Let's do a quick check for 6-cycles if 4-cycles exist.
        if has_4_cycle:
            # Check for 6-cycles
            # A 6-cycle involves 3 vertices in A and 3 in B.
            # u1-v1-u2-v2-u3-v3-u1
            # This requires u1-u2 share v1? No.
            # u1 connected to v1, v3.
            # u2 connected to v1, v2.
            # u3 connected to v2, v3.
            
            found_6 = False
            # Iterate over triples of A
            for i in range(len(A)):
                for j in range(i+1, len(A)):
                    for k in range(j+1, len(A)):
                        u1, u2, u3 = A[i], A[j], A[k]
                        # Check if there is a 6-cycle through these
                        # This is equivalent to checking if the induced subgraph on these 3 A-nodes
                        # and their neighbors contains a cycle of length 6.
                        # Specifically, we need a permutation of neighbors.
                        
                        # Get neighbors
                        n1 = adj[u1]
                        n2 = adj[u2]
                        n3 = adj[u3]
                        
                        # We need v1 in n1 & n2, v2 in n2 & n3, v3 in n3 & n1
                        # And v1, v2, v3 distinct.
                        
                        common_12 = n1 & n2
                        common_23 = n2 & n3
                        common_31 = n3 & n1
                        
                        if len(common_12) > 0 and len(common_23) > 0 and len(common_31) > 0:
                            # Try to find distinct v1, v2, v3
                            for v1 in common_12:
                                for v2 in common_23:
                                    if v1 == v2: continue
                                    for v3 in common_31:
                                        if v3 == v1 or v3 == v2: continue
                                        # Check if v1, v2, v3 form the cycle
                                        # u1-v1-u2-v2-u3-v3-u1
                                        # Edges:
                                        # u1-v1 (yes, v1 in n1)
                                        # v1-u2 (yes, v1 in n2)
                                        # u2-v2 (yes, v2 in n2)
                                        # v2-u3 (yes, v2 in n3)
                                        # u3-v3 (yes, v3 in n3)
                                        # v3-u1 (yes, v3 in n1)
                                        found_6 = True
                                        break
                                    if found_6: break
                                if found_6: break
                            if found_6: break
                        if found_6: break
                    if found_6: break
                if found_6: break
            if found_6:
                # Has 4 and 6. Value >= 2.
                # If we find one with Value 1, it must have no 6-cycles.
                # So if has_4_cycle and not found_6, Value might be 1.
                # But there could be 8-cycles.
                # For now, if no 6-cycle, it's a candidate for Value 1.
                # We need to verify no other cycles.
                # Given time, we assume if no 6-cycle, it's a strong candidate.
                # Let's mark it as best if we haven't found a better one.
                # Actually, we need to be sure.
                # But 4-cycles are the only ones if no 6-cycles?
                # In a 3-reg bip graph, if no 6-cycles, are there 8-cycles?
                # Possibly.
                # But Value 1 requires ALL cycles same length.
                # If we have 4 and 8, Value=2.
                
                # To be safe, we only accept if we are sure.
                # Given the difficulty, we might just return the fallback.
                
                # However, if we find a graph with NO 4-cycles, min length >= 6.
                # If min length is 6, and no 8, Value=1?
                
                # Let's just stick to the fallback if we don't find a clear Value 1.
                pass
            else:
                # No 6-cycles.
                # Check for 8-cycles?
                # If no 4, no 6, maybe 8?
                # If only 4-cycles, Value=1.
                # Let's assume this is a candidate.
                # We need to verify it has cycles.
                # 3-reg bip graph always has cycles.
                
                # To be rigorous, we should check all cycles.
                # But for now, let's store it.
                best_edges = edges
                best_value = 1 # Optimistic
                
                # If we find one, we can break?
                # No, there might be a better one? No, 1 is min.
                # So we can return immediately if we verify it.
                # Verification of "only 4-cycles" is hard.
                # But if we assume the search space is small, we might just return the fallback.
                
                # Given the risk of returning an invalid graph (if it has 8-cycles),
                # and the fact that Value 2 is already achieved by fallback,
                # it's safer to return the fallback unless we are 100% sure.
                
                # Let's just return the fallback to be safe.
                # The probability of finding a 3-reg bip graph with only 4-cycles is low.
                break

    # If no better graph found, return fallback
    return get_fallback()
