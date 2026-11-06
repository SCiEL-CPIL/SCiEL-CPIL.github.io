"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 280  # Safety margin
    
    # 1. Initialize with the Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = set()
    adj = [set() for _ in range(q)]
    
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
                
    n = q
    
    # Helper to check if a set of vertices is a clique or independent set
    def is_clique(vertices, adj):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True
        
    def is_independent(vertices, adj):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    return False
        return True

    # Helper to get all K4s and Ind4s in the current graph
    def get_k4s_and_ind4s(n, adj):
        k4s = []
        ind4s = []
        for combo in itertools.combinations(range(n), 4):
            if is_clique(combo, adj):
                k4s.append(combo)
            if is_independent(combo, adj):
                ind4s.append(combo)
        return k4s, ind4s

    # Attempt to add vertices one by one, up to 46
    for target_n in range(n + 1, 47):
        if time.time() - start_time > time_limit:
            break
            
        # We want to add vertex 'target_n'
        # The new vertex is index 'target_n'
        # We need to determine its connections to 0..target_n-1
        
        # Get constraints from current graph
        k4s, ind4s = get_k4s_and_ind4s(n, adj)
        
        # If there are no K4s or Ind4s, any connection works?
        # Actually, if no K4s, we can't form K5 by adding 1 vertex.
        # If no Ind4s, we can't form Ind5 by adding 1 vertex.
        
        found_valid_connection = False
        max_attempts = 50000 # Heuristic limit for local search per vertex
        
        # Local search for the neighborhood of the new vertex
        # State: a set of neighbors for the new vertex
        # We represent the neighborhood as a boolean array of length n
        # 1 if connected, 0 if not
        
        # Initial random guess
        neighbors = [random.choice([0, 1]) for _ in range(n)]
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Check constraints
            valid = True
            
            # Constraint 1: No K5
            # A K5 is formed if the new vertex connects to all 4 vertices of a K4.
            for k4 in k4s:
                if all(neighbors[v] == 1 for v in k4):
                    valid = False
                    # Flip a random edge in this K4 to break the clique
                    v_to_flip = random.choice(k4)
                    neighbors[v_to_flip] ^= 1
                    break
            
            if not valid:
                continue
                
            # Constraint 2: No Ind5
            # An Ind5 is formed if the new vertex connects to NONE of the 4 vertices of an Ind4.
            for ind4 in ind4s:
                if all(neighbors[v] == 0 for v in ind4):
                    valid = False
                    # Flip a random edge in this Ind4 to break the independence
                    v_to_flip = random.choice(ind4)
                    neighbors[v_to_flip] ^= 1
                    break
            
            if not valid:
                continue
                
            # If we passed both checks, we have a valid neighborhood
            found_valid_connection = True
            break
            
        if found_valid_connection:
            # Add the new vertex
            new_v = n
            adj.append(set())
            for i in range(n):
                if neighbors[i] == 1:
                    adj[i].add(new_v)
                    adj[new_v].add(i)
                    edges.add((i, new_v))
            n += 1
        else:
            # Failed to add this vertex, try next target? 
            # Or just stop? 
            # Usually if you can't add one, you might not be able to add more 
            # with this specific graph structure without restructuring.
            # But maybe a different random seed would work? 
            # Given time constraints, we break.
            break
            
    # Convert edges set to list of tuples (i, j) with i < j
    final_edges = [(min(u, v), max(u, v)) for u, v in edges]
    
    return n, final_edges
# EVOLVE-BLOCK-END
