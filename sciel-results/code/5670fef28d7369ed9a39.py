import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 285

    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # Adjacency matrix using Python sets for fast lookup
    # adj[i] = set of neighbors of vertex i
    adj = [set() for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i].add(j)
                adj[j].add(i)
    
    n = q
    
    # Function to compute all K4s (cliques of size 4) and Ind4s (independent sets of size 4)
    # Returns (k4s, ind4s) where each is a list of frozensets of 4 vertices
    def get_k4s_and_ind4s(n, adj):
        k4s = []
        ind4s = []
        
        # Find all K4s
        for combo in combinations(range(n), 4):
            i, j, k, l = combo
            if (j in adj[i] and k in adj[i] and l in adj[i] and
                k in adj[j] and l in adj[j] and
                l in adj[k]):
                k4s.append(frozenset(combo))
        
        # Find all Ind4s
        for combo in combinations(range(n), 4):
            i, j, k, l = combo
            if (j not in adj[i] and k not in adj[i] and l not in adj[i] and
                k not in adj[j] and l not in adj[j] and
                l not in adj[k]):
                ind4s.append(frozenset(combo))
        
        return k4s, ind4s
    
    # Precompute K4s and Ind4s for initial graph
    k4s, ind4s = get_k4s_and_ind4s(n, adj)
    
    # Try to add vertices
    for new_v in range(n, 43):
        if time.time() > deadline - 30:  # Reserve time for final conversion
            break
            
        success = False
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
                
            # Randomly choose neighbors for new vertex
            neighbors = set()
            for i in range(n):
                if random.random() < 0.5:
                    neighbors.add(i)
            
            # Check if adding new_v with these neighbors creates K5 or Ind5
            # K5 involving new_v: N(new_v) contains a K4 in G
            # Ind5 involving new_v: Non-N(new_v) contains an Ind4 in G
            
            non_neighbors = set(range(n)) - neighbors
            
            valid = True
            for K in k4s:
                if K.issubset(neighbors):
                    valid = False
                    break
            
            if valid:
                for I in ind4s:
                    if I.issubset(non_neighbors):
                        valid = False
                        break
                        
            if valid:
                # Success! Add the vertex
                adj.append(set(neighbors))
                for i in neighbors:
                    adj[i].add(n)
                n += 1
                success = True
                # Recompute K4s and Ind4s
                k4s, ind4s = get_k4s_and_ind4s(n, adj)
                break
                
        if not success:
            break
    
    # Convert adj to edges list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    return n, edges
