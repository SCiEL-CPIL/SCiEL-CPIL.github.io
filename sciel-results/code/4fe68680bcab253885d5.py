import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    def is_55_free(n, edges):
        """Check if graph with n vertices and given edges is (5,5)-free."""
        if n < 5:
            return True
        
        adj = [[False] * n for _ in range(n)]
        for i, j in edges:
            adj[i][j] = True
            adj[j][i] = True
        
        # Check for clique of size 5
        for clique in combinations(range(n), 5):
            all_connected = True
            for a, b in combinations(clique, 2):
                if not adj[a][b]:
                    all_connected = False
                    break
            if all_connected:
                return False
        
        # Check for independent set of size 5
        for indep in combinations(range(n), 5):
            all_independent = True
            for a, b in combinations(indep, 2):
                if adj[a][b]:
                    all_independent = False
                    break
            if all_independent:
                return False
        
        return True
    
    def paley_graph(q):
        """Generate Paley graph on q vertices."""
        if q < 2:
            return q, []
        residues = {(x * x) % q for x in range(1, q)}
        residues.discard(0)
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges
    
    # Start with Paley graph on 37 vertices
    q37, edges37 = paley_graph(37)
    if not is_55_free(q37, edges37):
        # Fallback: return empty or minimal valid graph
        return 0, []
    
    best_n = q37
    best_edges = edges37
    
    # Try to extend using constraint-based vertex addition
    n = q37
    adj = [[False] * n for _ in range(n)]
    for i, j in edges:
        adj[i][j] = True
        adj[j][i] = True
    
    def get_k4s_and_indep_k4s(n, adj):
        """Get all K4s and independent K4s as frozensets for fast lookup."""
        k4s = []
        indep_k4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i+1, 4):
                    if adj[combo[i]][combo[j]]:
                        is_indep = False
                    else:
                        is_clique = False
            if is_clique:
                k4s.append(frozenset(combo))
            if is_indep:
                indep_k4s.append(frozenset(combo))
        return k4s, indep_k4s
    
    def find_valid_connection(n, adj, k4s, indep_k4s, max_attempts=500):
        """
        Find a valid connection pattern for a new vertex.
        Returns a set of vertices to connect to, or None if not found.
        Uses backtracking with constraint propagation.
        """
        # Precompute for each vertex, which K4s it belongs to and which indep K4s
        k4_membership = {v: [] for v in range(n)}
        indep_k4_membership = {v: [] for v in range(n)}
        
        for idx, k4 in enumerate(k4s):
            for v in k4:
                k4_membership[v].append(idx)
        
        for idx, ik4 in enumerate(indep_k4s):
            for v in ik4:
                indep_k4_membership[v].append(idx)
        
        # Use iterative deepening with random restarts
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random initial assignment biased by density
            density = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j]) / (n*(n-1)/2) if n > 1 else 0.5
            conn = [random.random() < density for _ in range(n)]
            
            # Local search to fix violations
            for _ in range(2000):
                if time.time() > deadline - 5:
                    return None
                
                # Check for violations
                violated_k4 = None
                violated_indep_k4 = None
                
                for k4 in k4s:
                    if all(conn[v] for v in k4):
                        violated_k4 = k4
                        break
                
                if violated_k4 is None:
                    for ik4 in indep_k4s:
                        if not any(conn[v] for v in ik4):
                            violated_indep_k4 = ik4
                            break
                
                if violated_k4 is None and violated_indep_k4 is None:
                    return set(i for i in range(n) if conn[i])
                
                # Fix violation
                if violated_k4:
                    # Remove connection to one vertex in the K4
                    v = random.choice(list(violated_k4))
                    conn[v] = False
                elif violated_indep_k4:
                    # Add connection to one vertex in the indep K4
                    v = random.choice(list(violated_indep_k4))
                    conn[v] = True
        
        return None
    
    # Try to add vertices one by one
    for target_n in range(38, 47):
        if time.time() > deadline - 10:
            break
        
        k4s, indep_k4s = get_k4s_and_indep_k4s(n, adj)
        
        # Try multiple random restarts to find a valid connection
        found = False
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            conn = find_valid_connection(n, adj, k4s, indep_k4s, max_attempts=50)
            if conn is not None:
                # Add the new vertex
                for i in conn:
                    adj[n][i] = True
                    adj[i][n] = True
                    edges.append((n, i))
                n += 1
                best_n = n
                best_edges = edges.copy()
                found = True
                break
        
        if not found:
            break
    
    # Verify the final result
    if not is_55_free(best_n, best_edges):
        # Return the known valid 37-vertex graph
        return q37, edges37
    
    return best_n, best_edges
