import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290

    # 1. Initialize with Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = set()
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.add((i, j))
    
    n = q
    # Adjacency matrix/list for fast lookup
    # Use a set of edges for O(1) lookup
    # Or a list of sets
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
        
    # Precompute K4s and Ind4s
    K4s = []
    Ind4s = []
    
    # Function to update K4s and Ind4s is complex.
    # Given time constraints, we might just recompute them if n is small enough?
    # No, recomputing for n=40 is expensive.
    
    # Alternative:
    # Just use the validity check on the whole graph.
    # But we need to find a valid neighbor set for the new vertex.
    
    # Let's try a simpler heuristic for adding a vertex:
    # Randomly assign neighbors.
    # Check if the resulting graph is valid.
    # If not, try again.
    # This is "randomized restarts".
    
    # To check validity of the whole graph, we need to check for K5 and Ind5.
    # We can do this using bitsets.
    
    def is_valid(n, adj):
        # Check for K5
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    for k in range(j+1, n):
                        if k in adj[i] and k in adj[j]:
                            for l in range(k+1, n):
                                if l in adj[i] and l in adj[j] and l in adj[k]:
                                    for m in range(l+1, n):
                                        if m in adj[i] and m in adj[j] and m in adj[k] and m in adj[l]:
                                            return False
        # Check for Ind5
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    for k in range(j+1, n):
                        if k not in adj[i] and k not in adj[j]:
                            for l in range(k+1, n):
                                if l not in adj[i] and l not in adj[j] and l not in adj[k]:
                                    for m in range(l+1, n):
                                        if m not in adj[i] and m not in adj[j] and m not in adj[k] and m not in adj[l]:
                                            return False
        return True

    # This is O(n^5). For n=40, 40^5 = 100M. Too slow in Python.
    
    # We need a faster validity check.
    # Use bitsets.
    
    def is_valid_bitset(n, adj_bits):
        # adj_bits[i] is an integer
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    # i-j is edge
                    common = adj_bits[i] & adj_bits[j]
                    # Remove i and j from common? No, common contains neighbors of both.
                    # We need to find k, l, m in common such that they form a clique.
                    # This is still complex.
                    
        return True # Placeholder
        
    # Given the difficulty, let's stick to the known Paley(37) and try a very simple extension.
    # If extension fails, return 37.
    
    # Try to add vertices 37, 38, ...
    for new_v in range(n, 43):
        if time.time() > deadline:
            break
            
        success = False
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Randomly choose neighbors
            neighbors = set()
            for i in range(n):
                if random.random() < 0.5:
                    neighbors.add(i)
                    
            # Check if adding new_v with these neighbors creates K5 or Ind5
            # K5 involving new_v: N(new_v) contains K4 in G
            # Ind5 involving new_v: Non-N(new_v) contains Ind4 in G
            
            # We don't have K4s/Ind4s precomputed.
            # Let's just check the whole graph validity using a slower method if n is small.
            # For n=38, O(n^5) is 38^5 = 80M. Still too slow.
            
            # We need to precompute K4s and Ind4s for the current graph.
            
            if not K4s and not Ind4s:
                # Compute K4s and Ind4s for current n
                # This is O(n^4)
                for i in range(n):
                    for j in range(i+1, n):
                        if j in adj[i]:
                            for k in range(j+1, n):
                                if k in adj[i] and k in adj[j]:
                                    for l in range(k+1, n):
                                        if l in adj[i] and l in adj[j] and l in adj[k]:
                                            K4s.append(frozenset([i,j,k,l]))
                for i in range(n):
                    for j in range(i+1, n):
                        if j not in adj[i]:
                            for k in range(j+1, n):
                                if k not in adj[i] and k not in adj[j]:
                                    for l in range(k+1, n):
                                        if l not in adj[i] and l not in adj[j] and l not in adj[k]:
                                            Ind4s.append(frozenset([i,j,k,l]))
            
            valid = True
            for K in K4s:
                if K.issubset(neighbors):
                    valid = False
                    break
            if valid:
                for I in Ind4s:
                    if I.issubset(set(range(n)) - neighbors):
                        valid = False
                        break
                        
            if valid:
                # Success
                adj.append(set(neighbors))
                for i in neighbors:
                    adj[i].add(n)
                n += 1
                success = True
                # Update K4s and Ind4s?
                # This is complex. Let's just recompute if we add another vertex.
                # Or just clear them and recompute next time.
                K4s = []
                Ind4s = []
                break
                
        if not success:
            break
            
    # Convert adj to edges list
    final_edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                final_edges.append((i, j))
                
    return n, final_edges
