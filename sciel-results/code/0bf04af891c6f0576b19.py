import time
import random

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for the evaluator's own checks/overhead
    deadline = start_time + 110.0 

    def find_K4(n, adj):
        """
        Finds a K4 in the graph. Returns a list of 4 vertices or None.
        adj is a 2D list of booleans.
        """
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    continue
                # Find common neighbors of i and j with index > j
                common = []
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        common.append(k)
                
                # If there are at least 2 common neighbors, check if they are connected
                if len(common) >= 2:
                    for idx_k in range(len(common)):
                        k = common[idx_k]
                        for idx_l in range(idx_k + 1, len(common)):
                            l = common[idx_l]
                            if adj[k][l]:
                                return [i, j, k, l]
        return None

    def find_indep_6(n, adj):
        """
        Finds an independent set of size 6. Returns a list of 6 vertices or None.
        Uses backtracking with pruning.
        """
        def backtrack(candidates, current_set):
            if len(current_set) == 6:
                return current_set[:]
            
            # Pruning: not enough candidates left
            if len(candidates) < 6 - len(current_set):
                return None
            
            # Heuristic: Order candidates by degree in the subgraph induced by candidates?
            # For simplicity, we iterate in index order, but we can shuffle if needed.
            # Given n=35, simple order is likely fine.
            
            for i in range(len(candidates)):
                v = candidates[i]
                
                # Check if v is non-adjacent to all in current_set
                is_indep = True
                for u in current_set:
                    if adj[v][u]:
                        is_indep = False
                        break
                if not is_indep:
                    continue
                
                # Recurse: new candidates are those after i that are non-adjacent to v
                new_candidates = []
                for j in range(i + 1, len(candidates)):
                    w = candidates[j]
                    if not adj[v][w]:
                        new_candidates.append(w)
                
                res = backtrack(new_candidates, current_set + [v])
                if res:
                    return res
            return None
        
        return backtrack(list(range(n)), [])

    def local_search(n, max_time):
        """
        Attempts to construct a valid graph on n vertices using local search.
        Returns dict if successful, None otherwise.
        """
        # Initialize random graph
        # p=0.5 is a good starting point for Ramsey graphs
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        current_time = time.time()
        steps = 0
        max_steps = 50000 # Safety cap on iterations
        
        while current_time < max_time and steps < max_steps:
            # Check for K4
            k4 = find_K4(n, adj)
            if k4:
                # Remove a random edge from the K4
                # Edges in K4 (v0,v1,v2,v3): (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                edge_idx = random.randrange(6)
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                u, v = pairs[edge_idx]
                adj[k4[u]][k4[v]] = False
                adj[k4[v]][k4[u]] = False
                steps += 1
                current_time = time.time()
                continue
            
            # Check for Independent Set of 6
            indep6 = find_indep_6(n, adj)
            if indep6:
                # Add a random edge within the independent set
                # 15 possible edges
                edge_idx = random.randrange(15)
                pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
                u, v = pairs[edge_idx]
                adj[indep6[u]][indep6[v]] = True
                adj[indep6[v]][indep6[u]] = True
                steps += 1
                current_time = time.time()
                continue
            
            # If neither K4 nor Indep 6 is found, the graph is valid
            # (Because find_K4 and find_indep_6 are complete searches)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
            # Note: The loop above returns immediately if valid.
            # If we reach here, it means we didn't return, which shouldn't happen 
            # if finders are correct. But just in case, we can add a random flip.
            # Actually, the structure above is:
            # if k4: fix, continue
            # if indep: fix, continue
            # else: return valid
            
        return None

    # Try to construct graphs for n = 35 down to 17
    best_result = None
    
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for each n
        # 35 is the target, give it more chances
        max_restarts = 30 if n >= 30 else 10
        
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = local_search(n, deadline)
            if result:
                best_result = result
                break # Found a valid graph for this n, move to next smaller n? 
                # No, we want the MAX n. If we found valid for n=35, we return it immediately.
                # If we failed n=35, we try n=34.
                
        if best_result and best_result['n'] == n:
            return best_result

    # Fallback: Return the seed graph (Paley 17) if no better graph was found
    # This ensures we always return a valid graph
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
