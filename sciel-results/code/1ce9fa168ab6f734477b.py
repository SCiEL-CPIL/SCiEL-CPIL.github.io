import random
import time

def construct():
    # Try n=47 first, then 46
    for n in [47, 46]:
        min_deg_required = n - 39
        
        # Try multiple random instances with different densities
        for trial in range(200):
            random.seed(12345 + trial)
            
            # We want a graph with min degree >= n-39.
            # Let's start with a random graph with probability p.
            # For n=46, min deg 7. Average degree ~ 0.3*(45) = 13.5.
            # For n=47, min deg 8. Average degree ~ 0.3*(46) = 13.8.
            # Let's vary p between 0.2 and 0.4.
            p = 0.2 + (trial % 10) * 0.02
            
            adj = [set() for _ in range(n)]
            
            # Initialize random graph
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        adj[u].add(v)
                        adj[v].add(u)
                        
            # Step 1: Remove C4s
            # A C4 is a-u-b-v-a. 
            # For any pair of non-adjacent vertices u, v, they must have at most 1 common neighbor.
            # If they have 2+ common neighbors, say a, b, then u-a-v-b-u is a C4.
            # We can remove edges to break these.
            
            # Efficient C4 removal:
            # Iterate until no C4s found.
            # Finding all C4s is O(n^4) or O(n^3) with bitsets.
            # n=47 is small.
            
            def has_c4(adj):
                for u in range(n):
                    for v in range(u+1, n):
                        if v not in adj[u]:
                            common = adj[u].intersection(adj[v])
                            if len(common) >= 2:
                                return True, u, v, common
                return False, -1, -1, None

            # Remove C4s by deleting edges
            # To preserve degree, we should delete edges from the "common neighbors" side?
            # If u and v share neighbors a and b, the C4 is u-a-v-b-u.
            # We can remove edge (u,a) or (v,a) or (u,b) or (v,b).
            # Removing (u,a) reduces deg(u) and deg(a).
            # We want to keep degrees high.
            # Let's just remove a random edge from the cycle.
            
            max_removals = 1000
            for _ in range(max_removals):
                found, u, v, common = has_c4(adj)
                if not found:
                    break
                # common has at least 2 vertices
                a = random.choice(list(common))
                # The C4 is u-a-v-...-u. 
                # The edges involved in the C4 involving a are (u,a) and (v,a).
                # Actually, any two common neighbors a,b form u-a-v-b-u.
                # Removing (u,a) breaks this specific C4, but might leave others.
                # Let's remove (u,a)
                adj[u].remove(a)
                adj[a].remove(u)
                
            # Check if C4-free
            if has_c4(adj)[0]:
                continue
                
            # Step 2: Check Min Degree
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg >= min_deg_required:
                # Success
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            # Step 3: Try to add edges to increase min degree
            # If we are below min degree, we need to add edges.
            # But adding edges might create C4s.
            # We can try to add edges that don't create C4s.
            
            success_add = True
            for _ in range(1000):
                min_deg = min(len(adj[i]) for i in range(n))
                if min_deg >= min_deg_required:
                    break
                
                low_vertices = [i for i in range(n) if len(adj[i]) < min_deg_required]
                if not low_vertices:
                    break
                    
                u = random.choice(low_vertices)
                
                # Find candidates v for u
                candidates = []
                for v in range(n):
                    if v == u or v in adj[u]:
                        continue
                    # Check if adding (u,v) creates a C4
                    # This happens if |N(u) \cap N(v)| >= 2
                    inter = adj[u].intersection(adj[v])
                    if len(inter) < 2:
                        candidates.append(v)
                        
                if not candidates:
                    # Cannot add any edge to u without creating C4
                    # This trial is likely to fail or need backtracking
                    success_add = False
                    break
                    
                # Choose a candidate
                # Prefer one that also has low degree? Or random?
                # Let's pick one that minimizes the increase in "C4 potential" for others?
                # Just random for now.
                v = random.choice(candidates)
                adj[u].add(v)
                adj[v].add(u)
                
            if not success_add:
                continue
                
            # Final Verification
            is_c4_free = True
            for a in range(n):
                for c in range(a+1, n):
                    if c not in adj[a]:
                        if len(adj[a].intersection(adj[c])) >= 2:
                            is_c4_free = False
                            break
                if not is_c4_free:
                    break
            
            if is_c4_free:
                min_deg_final = min(len(adj[i]) for i in range(n))
                if min_deg_final >= min_deg_required:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                    
    # Fallback: Return a valid small graph
    # For n=39, min deg 0 is required. Empty graph works.
    return {'n': 39, 'edges': []}
