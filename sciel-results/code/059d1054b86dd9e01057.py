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
    
    # Precompute for each tree, the list of edges it uses
    # and degree sequences to guide search
    
    def find_embedding_backtrack(t_idx, used_edges, host_map, adj):
        """
        DFS to find embedding of tree t_idx into available edges.
        host_map: dict {tree_vertex: host_vertex} for vertices already placed
        Returns True if found, modifies host_map in place.
        Uses MRV heuristic.
        """
        k = t_idx + 1
        if k == 1:
            return True
        
        # Find unplaced tree vertices
        unplaced = [v for v in range(k) if v not in host_map]
        if not unplaced:
            return True
        
        # MRV: Pick the unplaced vertex with the fewest valid host candidates
        best_v = None
        best_cands = None
        min_cands = float('inf')
        
        for v in unplaced:
            cands = set(range(n))
            # Filter based on placed neighbors
            for nb in adj[v]:
                if nb in host_map:
                    ph = host_map[nb]
                    valid = set()
                    for hv in range(n):
                        if hv == ph:
                            continue
                        e = (min(ph, hv), max(ph, hv))
                        if e not in used_edges:
                            valid.add(hv)
                    cands &= valid
            
            # Filter out vertices already used by this tree
            used_in_tree = set(host_map.values())
            cands -= used_in_tree
            
            if len(cands) < min_cands:
                min_cands = len(cands)
                best_v = v
                best_cands = cands
                
                if min_cands == 0:
                    return False
                if min_cands == 1:
                    break # Can't get better
            
        if best_v is None:
            return False
            
        candidates = list(best_cands)
        random.shuffle(candidates)
        
        for hv in candidates:
            host_map[best_v] = hv
            if find_embedding_backtrack(t_idx, used_edges, host_map, adj):
                return True
            del host_map[best_v]
        
        return False
    
    def try_full_pack():
        """
        Attempt to pack all trees using backtracking over tree placements.
        Returns solution list or None.
        """
        used_edges = set()
        solution = [None] * n
        # Track edges used by each tree for backtracking
        tree_used_edges = [set() for _ in range(n)]
        
        # Order: largest trees first
        order = list(range(n - 1, -1, -1))
        
        def backtrack(idx):
            if idx == len(order):
                return True
            
            t_idx = order[idx]
            k = t_idx + 1
            
            if k == 1:
                # T1: assign a random vertex
                solution[t_idx] = [random.randint(0, n - 1)]
                if backtrack(idx + 1):
                    return True
                solution[t_idx] = None
                return False
            
            # Try to find embedding
            adj = tree_adj[t_idx]
            host_map = {}
            
            # Try multiple random roots
            for _ in range(50):  # Try 50 different random orderings
                if time.time() > deadline:
                    return False
                
                # Randomly order vertices for DFS
                # Actually, let's just shuffle the candidate order in find_embedding_backtrack
                # by using a random root
                root = random.randint(0, k - 1)
                host_map = {root: random.randint(0, n - 1)}
                
                # Try to complete the embedding
                if find_embedding_backtrack(t_idx, used_edges, host_map, adj):
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
    
    best_solution = None
    best_value = 0
    
    # Try full backtracking search
    sol = try_full_pack()
    if sol is not None:
        # Verify
        used = set()
        total = 0
        valid = True
        for i in range(n):
            if sol[i] is not None:
                k = i + 1
                if len(sol[i]) != k:
                    valid = False
                    break
                if len(set(sol[i])) != k:
                    valid = False
                    break
                for u, v in tree_edges[i]:
                    hu, hv = sol[i][u], sol[i][v]
                    if hu == hv:
                        valid = False
                        break
                    e = (min(hu, hv), max(hu, hv))
                    if e in used:
                        valid = False
                        break
                    used.add(e)
                    total += 1
                if not valid:
                    break
        if valid:
            return sol
    
    # If full backtracking failed or timed out, do greedy with restarts
    def try_greedy():
        used_edges = set()
        solution = [None] * n
        order = list(range(n - 1, -1, -1))
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                solution[t_idx] = [random.randint(0, n - 1)]
                continue
            
            adj = tree_adj[t_idx]
            host_map = {}
            found = False
            
            for attempt in range(100):
                if time.time() > deadline:
                    break
                root = random.randint(0, k - 1)
                host_map = {root: random.randint(0, n - 1)}
                if find_embedding_backtrack(t_idx, used_edges, host_map, adj):
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
                # Try to still get some edges? No, just skip
                pass
        
        # Calculate value
        value = 0
        for i in range(n):
            if solution[i] is not None:
                value += len(tree_edges[i])
        
        return solution, value
    
    # Do greedy restarts
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        sol, val = try_greedy()
        if val > best_value:
            best_value = val
            best_solution = sol
        if val == 190:
            # Verify
            used = set()
            valid = True
            for i in range(n):
                if sol[i] is not None:
                    for u, v in tree_edges[i]:
                        hu, hv = sol[i][u], sol[i][v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used:
                            valid = False
                            break
                        used.add(e)
                if not valid:
                    break
            if valid:
                return sol
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    fallback = [None] * n
    fallback[0] = [0]
    return fallback

if __name__ == "__main__":
    pass
