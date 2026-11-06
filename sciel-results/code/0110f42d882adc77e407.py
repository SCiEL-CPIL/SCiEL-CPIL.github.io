import random
import time
import itertools

def construct():
    start_time = time.time()
    time_limit = 290  # Safety margin for 300s limit
    
    # We try to find a graph of size N, starting from high to low.
    # The seed is 37. We know 43 is possible. We try up to 46.
    best_n = 37
    best_edges = None
    
    # Initialize with the Paley graph edges for 37 vertices as a fallback or start
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    p_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_edges = p_edges
    
    # Try to find larger graphs
    for target_n in range(46, 37, -1):
        if time.time() - start_time > time_limit:
            break
        found = search_for_graph(target_n, time_limit - (time.time() - start_time))
        if found:
            n, edges = found
            best_n = n
            best_edges = edges
            # Since we go from high to low, the first one found is the best.
            # But we might want to check if we can go higher? 
            # The loop goes 46, 45... so if we find 46, we break.
            # If we fail 46, we try 45, etc.
            # If we find one, it's the max for this run. 
            # We can break immediately if we find one, as we are checking descending N.
            break
            
    return best_n, best_edges

def search_for_graph(n, time_budget):
    start_time = time.time()
    random.seed(42) # For reproducibility within the attempt? Or random each time? 
    # Let's use random seed per attempt to explore different spaces.
    
    # We will try multiple restarts
    max_restarts = 1000
    restarts = 0
    
    while restarts < max_restarts:
        if time.time() - start_time > time_budget:
            break
        
        # Start with 0 vertices? Or 1?
        # Let's start with 0.
        vertices = 0
        edges = set()
        adj = [set() for _ in range(n)] # Pre-allocate for max n, but only use [0..vertices)
        
        # We build vertex by vertex
        success = True
        for v in range(n):
            if time.time() - start_time > time_budget:
                success = False
                break
            
            if v == 0:
                # First vertex, no edges
                continue
                
            # Get current K4s and Independent 4-sets
            # Optimization: Only check subsets of existing vertices
            current_k4s = []
            current_ind4s = []
            
            # Find K4s in [0, v)
            # This is O(v^4). For v=40, 40^4 = 2.5M. Too slow if done every step.
            # We need a faster way or incremental update.
            # However, n is small. 
            # Let's optimize: 
            # A K4 is a set of 4 vertices with all edges present.
            # An Ind4 is a set of 4 vertices with no edges present.
            
            # Instead of finding all K4s/Ind4s from scratch, we can check constraints on the fly for the new vertex.
            # But we need to know if a subset A contains a K4.
            # We can precompute for the current graph:
            # For every subset of 4 vertices, is it a K4? Is it an Ind4?
            
            # Let's maintain lists of K4s and Ind4s incrementally.
            # When adding vertex v-1, we generated new K4s/Ind4s involving v-1.
            
            # Let's restart the logic:
            # We maintain `k4s` as a list of tuples of 4 vertices.
            # We maintain `ind4s` as a list of tuples of 4 vertices.
            
            # For the new vertex v, we need to pick a neighbor set A.
            # Constraints:
            # 1. No K4 in A.
            # 2. No Ind4 in V \ A.
            
            # We can sample A.
            # To make sampling efficient, we can use a heuristic.
            # Let's try random A.
            
            found_valid_a = False
            attempts = 0
            max_attempts = 5000
            
            # Pre-compute K4s and Ind4s for current graph [0, v)
            # This is the expensive part.
            # We can cache these? No, graph changes.
            # But we can compute them once per vertex addition.
            
            k4_list = []
            ind4_list = []
            
            # Iterate over all 4-subsets of [0, v)
            # This is C(v, 4). For v=40, C(40,4) ~ 91k. This is fast.
            for comb in itertools.combinations(range(v), 4):
                is_k4 = True
                is_ind4 = True
                for i in range(4):
                    for j in range(i+1, 4):
                        u, w = comb[i], comb[j]
                        if (u, w) in edges or (w, u) in edges:
                            is_ind4 = False
                        else:
                            is_k4 = False
                if is_k4:
                    k4_list.append(comb)
                if is_ind4:
                    ind4_list.append(comb)
            
            # Now find A
            while attempts < max_attempts:
                attempts += 1
                # Random subset A of [0, v)
                # We can bias probability? 
                # Let's just pick random 0/1 for each edge.
                a_set = set()
                for u in range(v):
                    if random.random() < 0.5:
                        a_set.add(u)
                
                # Check constraints
                valid = True
                for k4 in k4_list:
                    if all(x in a_set for x in k4):
                        valid = False
                        break
                if not valid:
                    continue
                
                for ind4 in ind4_list:
                    if all(x not in a_set for x in ind4):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Valid A found
                found_valid_a = True
                # Add edges
                for u in a_set:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
                break
            
            if not found_valid_a:
                success = False
                break
        
        if success:
            # Convert edges set to list of tuples
            edge_list = list(edges)
            return n, edge_list
            
        restarts += 1
        
    return None
