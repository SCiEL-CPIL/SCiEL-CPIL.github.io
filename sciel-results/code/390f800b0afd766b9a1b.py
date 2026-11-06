import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    def is_triangle_free(adj):
        # adj is a list of sets
        for i in range(len(adj)):
            for j in adj[i]:
                if i > j: continue
                # check if k is connected to both i and j
                # This is O(n^3) if naive. 
                # Better: for each edge (i,j), check if there is k in adj[i] & adj[j]
                pass
        # Actually, we maintain triangle-free property by construction in the search loop.
        return True

    def has_triangle(adj, n):
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    # check common neighbors
                    # if adj[i] & adj[j] is not empty, triangle exists
                    # but adj[i] contains j? No, adj is undirected.
                    # Let's use bitsets for speed? 
                    # n <= 40, so int bitsets are perfect.
                    pass
        return False

    # Use bitsets for adjacency
    # adj[i] is an int representing neighbors of i
    
    def create_random_graph(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def remove_triangles(adj, n):
        # Repeatedly find a triangle and remove an edge
        # To be fast, we can just scan all triples? O(n^3) = 40^3 = 64000. Fast enough.
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        # Check if k is neighbor of both
                        common = adj[i] & adj[j]
                        # Exclude i and j from common? 
                        # adj[i] does not contain i. adj[j] does not contain j.
                        # But adj[i] contains j. So common contains j? No, j is not in adj[j].
                        # So common are vertices connected to both i and j.
                        if common:
                            # Pick a random k from common
                            # Find lowest set bit
                            k = (common & -common).bit_length() - 1
                            # Remove edge (i, k) or (j, k) or (i, j)
                            # Random choice
                            choice = random.randint(0, 2)
                            if choice == 0:
                                # remove (i, j)
                                adj[i] &= ~(1 << j)
                                adj[j] &= ~(1 << i)
                            elif choice == 1:
                                # remove (i, k)
                                adj[i] &= ~(1 << k)
                                adj[k] &= ~(1 << i)
                            else:
                                # remove (j, k)
                                adj[j] &= ~(1 << k)
                                adj[k] &= ~(1 << j)
                            changed = True
                            break
                if changed:
                    break
            if changed:
                continue
            # If no triangle found in this pass, we are triangle free
            break
        return adj

    def find_independent_set(adj, n, target_size):
        # Returns a list of vertices forming an independent set of size target_size
        # or None if not found
        # Use backtracking
        # Order vertices by degree? 
        # We want to find a set of size 10.
        
        def backtrack(current_set, available):
            if len(current_set) == target_size:
                return current_set[:]
            if not available:
                return None
            # Pruning: if len(current_set) + popcount(available) < target_size, return None
            if len(current_set) + bin(available).count('1') < target_size:
                return None
            
            # Pick a vertex from available
            # Heuristic: pick one with low degree in the induced subgraph?
            # Or just pick the first one.
            # To improve, we can pick a vertex that has fewest neighbors in available.
            # This is O(n^2) per step.
            
            # Simple approach: pick random or first
            # Let's pick the one with minimum degree in available
            min_deg = n
            best_v = -1
            v = available
            while v:
                u = (v & -v).bit_length() - 1
                v &= v - 1
                # degree of u in available
                deg = bin(adj[u] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = u
            
            if best_v == -1:
                return None
                
            # Branch 1: include best_v
            new_available = available & ~(1 << best_v) & ~adj[best_v]
            res = backtrack(current_set + [best_v], new_available)
            if res:
                return res
            
            # Branch 2: exclude best_v
            new_available = available & ~(1 << best_v)
            res = backtrack(current_set, new_available)
            if res:
                return res
            
            return None
        
        initial_available = (1 << n) - 1
        return backtrack([], initial_available)

    def add_edge_to_break_independent_set(adj, n, indep_set):
        # Try to add an edge between two vertices in indep_set
        # such that no triangle is formed
        for i in range(len(indep_set)):
            for j in range(i+1, len(indep_set)):
                u = indep_set[i]
                v = indep_set[j]
                # Check if adding (u,v) creates a triangle
                # Triangle exists if there is w connected to both u and v
                # i.e., adj[u] & adj[v] is not empty
                if (adj[u] & adj[v]) == 0:
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    return True
        return False

    best_n = 0
    best_graph = None
    
    # Try n=40
    for target_n in [40, 39]:
        if time.time() > deadline:
            break
            
        attempts = 0
        while time.time() < deadline and attempts < 500:
            attempts += 1
            adj = create_random_graph(target_n)
            adj = remove_triangles(adj, target_n)
            
            # Check independent set
            indep = find_independent_set(adj, target_n, 10)
            
            if indep is None:
                # Valid graph
                if target_n > best_n:
                    best_n = target_n
                    # Convert to edge list
                    edges = []
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_graph = {'n': target_n, 'edges': edges}
                break # Found a valid one for this n, move to next n or done
            
            # If indep found, try to fix
            # Try a few times to add edges
            fixed = False
            for _ in range(10):
                if add_edge_to_break_independent_set(adj, target_n, indep):
                    # Re-check triangle freeness? 
                    # We ensured no triangle with the added edge.
                    # But we might have created a triangle with other edges? No, we only added one edge.
                    # So it remains triangle free.
                    # Now check independent set again
                    indep = find_independent_set(adj, target_n, 10)
                    if indep is None:
                        if target_n > best_n:
                            best_n = target_n
                            edges = []
                            for i in range(target_n):
                                for j in range(i+1, target_n):
                                    if (adj[i] >> j) & 1:
                                        edges.append([i, j])
                            best_graph = {'n': target_n, 'edges': edges}
                        fixed = True
                        break
                    # If still has indep set, loop again to try another edge
                else:
                    break # Couldn't add any edge
            
            if fixed:
                break

    if best_graph:
        return best_graph
    
    # Fallback to seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
