import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree edge lists and adjacency
    tree_edges = [trees[i] for i in range(n)]
    tree_adj = []
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for u, v in tree_edges[i]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # All possible edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Precompute degree of each vertex in each tree
    tree_deg = []
    for i in range(n):
        k = i + 1
        deg = [0] * k
        for u, v in tree_edges[i]:
            deg[u] += 1
            deg[v] += 1
        tree_deg.append(deg)
    
    # Find a good ordering of trees to place (most constrained first)
    # Use a heuristic: trees with higher max degree first, then by size
    def tree_priority(t_idx):
        k = t_idx + 1
        max_deg = max(tree_deg[t_idx]) if k > 1 else 0
        return (-max_deg, -k, t_idx)
    
    order = sorted(range(n), key=tree_priority)
    
    def find_embedding(t_idx, used_edges, host_map, adj, deadline_check):
        """
        DFS to find embedding of tree t_idx into available edges.
        host_map: dict {tree_vertex: host_vertex} for vertices already placed
        Returns True if found, modifies host_map in place.
        deadline_check: callable that returns True if we should stop
        """
        k = t_idx + 1
        if k == 1:
            return True
        
        if deadline_check():
            return False
        
        # Find unplaced tree vertices
        unplaced = [v for v in range(k) if v not in host_map]
        if not unplaced:
            return True
        
        # Pick the most constrained unplaced vertex
        # (one with most placed neighbors, tie-break by degree in tree)
        best_v = None
        best_key = (-1, -1)
        for v in unplaced:
            placed_neighbors = [host_map[nb] for nb in adj[v] if nb in host_map]
            constraint = len(placed_neighbors)
            deg = tree_deg[t_idx][v]
            key = (constraint, deg)
            if key > best_key:
                best_key = key
                best_v = v
        
        if best_v is None:
            best_v = unplaced[0]
        
        # Determine candidate host vertices for best_v
        placed_neighbors = [host_map[nb] for nb in adj[best_v] if nb in host_map]
        if not placed_neighbors:
            candidates = list(range(n))
        else:
            candidates = set(range(n))
            for ph in placed_neighbors:
                valid = set()
                for v in range(n):
                    if v == ph:
                        continue
                    e = (min(ph, v), max(ph, v))
                    if e not in used_edges:
                        valid.add(v)
                candidates &= valid
        
        # Filter out vertices already used by this tree
        used_in_tree = set(host_map.values())
        candidates = [v for v in candidates if v not in used_in_tree]
        
        # Order candidates: prefer vertices with more available edges (more flexibility)
        # or less used edges (to preserve options)
        random.shuffle(candidates)
        
        for hv in candidates:
            host_map[best_v] = hv
            if find_embedding(t_idx, used_edges, host_map, adj, deadline_check):
                return True
            del host_map[best_v]
        
        return False
    
    def try_pack(order, deadline_check):
        """
        Attempt to pack all trees in given order using backtracking.
        Returns solution list or None.
        """
        used_edges = set()
        solution = [None] * n
        tree_used_edges = [set() for _ in range(n)]
        
        def backtrack(idx):
            if idx == len(order):
                return True
            
            if deadline_check():
                return False
            
            t_idx = order[idx]
            k = t_idx + 1
            
            if k == 1:
                # T1: assign a random vertex
                solution[t_idx] = [random.randint(0, n - 1)]
                if backtrack(idx + 1):
                    return True
                solution[t_idx] = None
                return False
            
            # Try to find embedding with multiple random roots
            adj = tree_adj[t_idx]
            
            # Try different random roots and orderings
            for attempt in range(20):
                if deadline_check():
                    return False
                
                # Choose a random root
                root = random.randint(0, k - 1)
                root_host = random.randint(0, n - 1)
                host_map = {root: root_host}
                
                if find_embedding(t_idx, used_edges, host_map, adj, deadline_check):
                    # Success! Commit
                    emb = [host_map[i] for i in range(k)]
                    solution[t_idx] = emb
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
                        tree_used_edges[t_idx].add(e)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Backtrack: undo this tree
                    for e in tree_used_edges[t_idx]:
                        used_edges.discard(e)
                    tree_used_edges[t_idx] = set()
                    solution[t_idx] = None
            
            return False
        
        if backtrack(0):
            return solution
        return None
    
    def verify_solution(sol):
        """Verify that the solution is valid and compute its value."""
        used = set()
        total = 0
        for i in range(n):
            if sol[i] is not None:
                k = i + 1
                if len(sol[i]) != k:
                    return None
                if len(set(sol[i])) != k:
                    return None
                for v in sol[i]:
                    if not (0 <= v < n):
                        return None
                for u, v in tree_edges[i]:
                    hu, hv = sol[i][u], sol[i][v]
                    if hu == hv:
                        return None
                    e = (min(hu, hv), max(hu, hv))
                    if e in used:
                        return None
                    used.add(e)
                    total += 1
        return total
    
    best_solution = None
    best_value = 0
    
    # Try full backtracking search with the prioritized order
    def deadline_check():
        return time.time() > deadline
    
    sol = try_pack(order, deadline_check)
    if sol is not None:
        val = verify_solution(sol)
        if val is not None:
            return sol
    
    # If full backtracking failed, try greedy with restarts
    def try_greedy(order, deadline_check):
        used_edges = set()
        solution = [None] * n
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [random.randint(0, n - 1)]
                continue
            
            adj = tree_adj[t_idx]
            found = False
            
            for attempt in range(30):
                if deadline_check():
                    break
                root = random.randint(0, k - 1)
                root_host = random.randint(0, n - 1)
                host_map = {root: root_host}
                if find_embedding(t_idx, used_edges, host_map, adj, deadline_check):
                    emb = [host_map[i] for i in range(k)]
                    solution[t_idx] = emb
                    for u, v in tree_edges[t_idx]:
                        hu, hv = emb[u], emb[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            solution[t_idx] = None
                            found = False
                            break
                        used_edges.add(e)
                    found = True
                    break
            
            if not found:
                pass
        
        val = verify_solution(solution)
        if val is None:
            return solution, 0
        return solution, val
    
    # Do greedy restarts with different random seeds
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        sol, val = try_greedy(order, deadline_check)
        if val > best_value:
            best_value = val
            best_solution = sol
        if val == 190:
            return sol
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    fallback = [None] * n
    fallback[0] = [0]
    return fallback

if __name__ == "__main__":
    pass
