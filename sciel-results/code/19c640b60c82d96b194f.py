import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin for evaluator overhead (300s limit)
    
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    def paley_graph(n):
        """Generate Paley graph on n vertices (n must be prime, n ≡ 1 mod 4)."""
        if n < 2:
            return []
        if not is_prime(n) or n % 4 != 1:
            return None
        
        # Quadratic residues mod n
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    edges.append((i, j))
        return edges
    
    def build_adjacency(n, edges):
        """Build adjacency sets from edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def has_k5(n, adj):
        """Check if graph contains a K5. Returns True if found, False otherwise."""
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if combo[j] not in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    
    def has_independent_5(n, adj):
        """Check if graph contains an independent set of size 5. Returns True if found, False otherwise."""
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    
    def is_valid(n, adj):
        """Check if graph is (5,5)-free: no K5 and no independent 5-set."""
        if has_k5(n, adj):
            return False
        if has_independent_5(n, adj):
            return False
        return True
    
    def find_k5(n, adj):
        """Find a K5 if it exists. Returns the tuple or None."""
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if combo[j] not in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return combo
        return None
    
    def find_independent_5(n, adj):
        """Find an independent 5-set if it exists. Returns the tuple or None."""
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return combo
        return None
    
    # Start with the best known valid Paley graph
    best_n = 37
    best_edges = paley_graph(37)
    best_adj = build_adjacency(37, best_edges)
    
    # Verify the seed is valid
    if not is_valid(37, best_adj):
        # Fallback: should not happen, but handle it
        return 37, best_edges
    
    # Try to extend to larger graphs using local search
    # Strategy: Start from Paley 37, try to add vertices and fix violations
    
    current_n = 37
    current_edges = list(best_edges)
    current_adj = [set(s) for s in best_adj]
    
    # Helper to add a vertex with random connections
    def try_add_vertex(n, edges, adj, target_n):
        """Try to add a vertex with random connections and return (success, new_edges, new_adj)."""
        new_edges = list(edges)
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        
        new_vertex = n
        for i in range(n):
            if random.random() < 0.5:
                new_edges.append((i, new_vertex))
                new_adj[i].add(new_vertex)
                new_adj[new_vertex].add(i)
        
        return new_edges, new_adj
    
    # Helper to check if adding a vertex with specific connections is valid
    def check_add_vertex_valid(n, adj, connections):
        """Check if adding a vertex with given connections creates no K5 or independent 5-set."""
        # connections is a set of vertices connected to the new vertex
        new_vertex = n
        new_adj = [set(s) for s in adj]
        new_adj.append(set())
        
        for i in connections:
            new_adj[i].add(new_vertex)
            new_adj[new_vertex].add(i)
        
        # Check for K5 involving the new vertex
        # A K5 involving the new vertex requires 4 other vertices that form a K4 and are all connected to new_vertex
        for combo in combinations(range(n), 4):
            if all(v in connections for v in combo):
                # Check if combo forms a K4
                ok = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        if combo[j] not in adj[combo[i]]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return False  # Creates K5
        
        # Check for independent 5-set involving the new vertex
        # An independent 5-set involving the new vertex requires 4 other vertices that form an independent set and are NOT connected to new_vertex
        non_connections = set(range(n)) - connections
        for combo in combinations(non_connections, 4):
            # Check if combo forms an independent set
            ok = True
            for i in range(4):
                for j in range(i + 1, 4):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                    if not ok:
                        break
            if ok:
                return False  # Creates independent 5-set
        
        return True
    
    # Try to extend the graph
    for target_n in range(38, 47):  # Try up to 46 vertices
        if time.time() > deadline - 15:
            break
        
        found = False
        max_trials = 2000
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            # Generate random connections for the new vertex
            connections = set()
            for i in range(current_n):
                if random.random() < 0.5:
                    connections.add(i)
            
            # Check if this is valid
            if check_add_vertex_valid(current_n, current_adj, connections):
                # Add the vertex
                new_edges = list(current_edges)
                new_adj = [set(s) for s in current_adj]
                new_adj.append(set())
                
                for i in connections:
                    new_edges.append((i, current_n))
                    new_adj[i].add(current_n)
                    new_adj[current_n].add(i)
                
                current_n = target_n
                current_edges = new_edges
                current_adj = new_adj
                found = True
                break
        
        if not found:
            # Try local search: start from current graph and modify edges to fix violations
            # This is a fallback for when simple addition fails
            break
    
    # Local search phase: try to improve the current graph
    if time.time() < deadline - 20:
        # Run local search to try to find a valid larger graph
        search_budget = deadline - time.time() - 20
        
        # Try to find a valid graph with more vertices using simulated annealing-like approach
        for target_n in range(current_n + 1, 47):
            if time.time() > deadline - 15:
                break
            
            found = False
            max_trials = 5000
            
            for trial in range(max_trials):
                if time.time() > deadline - 15:
                    break
                
                # Start from the best known graph and try to extend
                # Use a different strategy: random restart with bias
                connections = set()
                # Bias towards connections that avoid creating violations
                for i in range(37):
                    # Compute a score for connecting to vertex i
                    # Count K4s in the neighborhood that would become K5
                    k4_count = 0
                    indep4_count = 0
                    
                    # For efficiency, just use random with some heuristic
                    if random.random() < 0.5:
                        connections.add(i)
                
                if check_add_vertex_valid(37, best_adj, connections):
                    # Build the new graph
                    new_edges = list(best_edges)
                    new_adj = [set(s) for s in best_adj]
                    new_adj.append(set())
                    
                    for i in connections:
                        new_edges.append((i, 37))
                        new_adj[i].add(37)
                        new_adj[37].add(i)
                    
                    # Now try to add more vertices
                    temp_n = 38
                    temp_edges = new_edges
                    temp_adj = new_adj
                    
                    for next_n in range(39, target_n + 1):
                        next_found = False
                        for sub_trial in range(100):
                            if time.time() > deadline - 15:
                                break
                            
                            sub_connections = set()
                            for i in range(temp_n):
                                if random.random() < 0.5:
                                    sub_connections.add(i)
                            
                            if check_add_vertex_valid(temp_n, temp_adj, sub_connections):
                                # Add vertex
                                new_temp_edges = list(temp_edges)
                                new_temp_adj = [set(s) for s in temp_adj]
                                new_temp_adj.append(set())
                                
                                for i in sub_connections:
                                    new_temp_edges.append((i, temp_n))
                                    new_temp_adj[i].add(temp_n)
                                    new_temp_adj[temp_n].add(i)
                                
                                temp_n = next_n
                                temp_edges = new_temp_edges
                                temp_adj = new_temp_adj
                                next_found = True
                                break
                        
                        if not next_found:
                            break
                    
                    if temp_n == target_n:
                        current_n = target_n
                        current_edges = temp_edges
                        current_adj = temp_adj
                        found = True
                        break
            
            if not found:
                break
    
    # Final verification
    if is_valid(current_n, current_adj):
        return current_n, current_edges
    else:
        # Return the seed if current is invalid
        return 37, best_edges
