"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 15
    
    # Target degrees: sum must be even, and we want them as spread out as possible
    # to minimize the chance of equal-degree endpoints on any 3-edge path
    # Degrees: 1,1,2,3,4,5,6,7,8,9,10,11,12,13,14 (sum=106, even)
    target_degrees = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # We need to assign these degrees to vertices and find a graph realizing them
    # Strategy: greedy edge addition with backtracking
    
    def is_valid_path_constraint(adj, n):
        """Check that no simple 3-edge path s-u-v-t has deg(s) == deg(t)"""
        deg = [sum(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def try_construct(target_degrees, n, seed):
        """Try to construct a graph with given target degrees using greedy + backtracking"""
        random.seed(seed)
        
        # Shuffle vertex assignment of target degrees
        perm = list(range(n))
        random.shuffle(perm)
        assigned_degrees = [0] * n
        for i, v in enumerate(perm):
            assigned_degrees[v] = target_degrees[i]
        
        remaining = list(assigned_degrees)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Greedy: try to add edges between vertices with highest remaining degree
        # Use a priority-based approach
        for step in range(100):
            # Find pair (u,v) where u < v, not yet connected, and both have remaining > 0
            # Prefer pairs where adding the edge keeps feasibility
            best_pair = None
            best_score = -1
            
            # Try all pairs, score by how well it helps
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u] and remaining[u] > 0 and remaining[v] > 0:
                        # Score: prefer connecting high-degree vertices
                        score = remaining[u] + remaining[v]
                        candidates.append((score, u, v))
            
            if not candidates:
                break
            
            # Sort by score descending, try top candidates
            candidates.sort(reverse=True)
            
            added = False
            for score, u, v in candidates[:20]:  # Try top 20
                # Check if adding edge keeps feasibility
                # After adding, remaining[u] -= 1, remaining[v] -= 1
                # Need: for all w, remaining[w] <= n-1 and sum(remaining) is even
                new_rem = remaining[:]
                new_rem[u] -= 1
                new_rem[v] -= 1
                
                # Feasibility: no vertex has negative remaining
                if any(r < 0 for r in new_rem):
                    continue
                
                # Sum must be even
                if sum(new_rem) % 2 != 0:
                    continue
                
                # Max degree constraint: remaining[w] <= (n-1) - (number of already connected neighbors that are still possible)
                # Simpler: just check remaining[w] <= n - 1 - len(adj[w]) + (1 if w==u or w==v else 0)
                # Actually just check remaining[w] <= n - 1 - len(adj[w]) for w not in {u,v}
                # For u,v: remaining[u]-1 <= n-1-len(adj[u]) is always true if remaining[u] <= n-len(adj[u])
                
                feasible = True
                for w in range(n):
                    max_possible = n - 1 - len(adj[w])
                    if w == u:
                        max_possible = n - 1 - len(adj[w])  # adj[u] doesn't include v yet
                        if new_rem[w] > max_possible:
                            feasible = False
                            break
                    elif w == v:
                        max_possible = n - 1 - len(adj[w])
                        if new_rem[w] > max_possible:
                            feasible = False
                            break
                    else:
                        if new_rem[w] > max_possible:
                            feasible = False
                            break
                
                if not feasible:
                    continue
                
                # Accept this edge
                adj[u].add(v)
                adj[v].add(u)
                remaining[u] -= 1
                remaining[v] -= 1
                edges.append([u, v])
                added = True
                break
            
            if not added:
                break
            
            # Check if all remaining are zero
            if all(r == 0 for r in remaining):
                break
        
        # Check if we achieved the target
        if not all(r == 0 for r in remaining):
            return None
        
        # Verify the path constraint
        if not is_valid_path_constraint(adj, n):
            return None
        
        return edges
    
    # Try multiple seeds
    best_edges = None
    best_count = -1
    
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        edges = try_construct(target_degrees, n, seed)
        if edges is not None:
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
                # If we got a good solution, we can stop early
                if best_count >= 50:  # 106/2 = 53 max, so 50+ is good
                    break
    
    if best_edges is not None:
        # Ensure proper types
        result = []
        for e in best_edges:
            u, v = e
            result.append([int(u), int(v)])
        return result
    
    # Fallback: return a star (always valid since all endpoints have degree 1)
    return [[0, v] for v in range(1, n)]

# EVOLVE-BLOCK-END
