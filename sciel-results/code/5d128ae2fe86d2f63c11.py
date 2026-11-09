"""New approach: Greedy Topological Minor Search via Disjoint Path Packing."""
# EVOLVE-BLOCK-START
import itertools
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = int(P['n'])
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Helper: Find a simple path from src to dst in the graph,
    # avoiding 'blocked' vertices as interiors.
    # src and dst are allowed.
    def find_path(src, dst, blocked):
        if src == dst:
            return [src]
        visited = {src}
        queue = [[src]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in adj[node]:
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor in visited:
                    continue
                if neighbor in blocked:
                    continue
                visited.add(neighbor)
                queue.append(path + [neighbor])
        return None

    # Heuristic: Score vertices by degree
    degrees = [len(adj[i]) for i in range(n)]
    # Sort vertices by degree descending
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We will try to find the largest k such that K_k is a topological minor.
    # We search k from high to low.
    # Candidate pool: Top 12 vertices by degree.
    pool_size = min(n, 12)
    pool = sorted_vertices[:pool_size]
    
    best_result = None
    
    # Try k from pool_size down to 2
    for k in range(pool_size, 1, -1):
        # Generate combinations of k vertices from the pool
        # To limit computation, we might not check ALL combinations if pool is large,
        # but 12 choose 6 is ~900, which is fine.
        for comb in itertools.combinations(pool, k):
            branches = list(comb)
            # Try to find disjoint paths for all pairs in branches
            # We try a few random orderings of the pairs to avoid getting stuck
            # in a bad greedy choice.
            
            pairs = list(itertools.combinations(branches, 2))
            found_solution = False
            paths_dict = {}
            
            # Try up to 5 random orderings of the pairs
            for attempt in range(5):
                order = pairs[:]
                random.shuffle(order)
                
                used_interiors = set()
                success = True
                current_paths = {}
                
                for u, v in order:
                    # Blocked vertices: all branches (except u,v) + used interiors
                    blocked = set(branches) - {u, v}
                    blocked.update(used_interiors)
                    
                    path = find_path(u, v, blocked)
                    if path is None:
                        success = False
                        break
                    
                    # Update used interiors
                    interiors = path[1:-1]
                    # Check if interiors are disjoint from previous ones
                    # They are by construction of 'blocked', but let's be safe
                    if any(x in used_interiors for x in interiors):
                        success = False
                        break
                        
                    used_interiors.update(interiors)
                    current_paths[(u, v)] = path
                    current_paths[(v, u)] = path # Store for both directions? Task says unordered pair.
                
                if success:
                    found_solution = True
                    paths_dict = current_paths
                    break
            
            if found_solution:
                # Construct the return value
                # paths must be a list of paths for each unordered pair
                result_paths = []
                for u, v in pairs:
                    result_paths.append(paths_dict[(u, v)])
                
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': result_paths
                }
                break # Found a solution for this k, which is the max possible for this pool
        
        if best_result:
            break # Found max k in this pool, no need to check smaller k
    
    if best_result:
        return best_result

    # Fallback: If no topological minor found in the pool (unlikely for k>=2),
    # try any edge.
    if edges:
        u, v = edges[0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    else:
        # Should not happen for n>=2
        return {
            'branches': [0, 1],
            'paths': [[0, 1]]
        }
# EVOLVE-BLOCK-END
