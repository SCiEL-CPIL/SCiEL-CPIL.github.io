"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _get_cycle_lengths(edges, n=12):
    """
    Find all distinct simple cycle lengths in an undirected graph.
    Returns a set of integers representing the lengths.
    """
    if not edges:
        return set()
        
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        if u >= n or v >= n or u < 0 or v < 0:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_set:
            continue
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
        
    cycle_lengths = set()
    
    # To find all simple cycles, we can use DFS with a constraint that 
    # we only close a cycle if the start node is the smallest index in the cycle.
    # This avoids counting the same cycle multiple times with different starts.
    
    def dfs(start, current, visited, path_len):
        for neighbor in adj[current]:
            if neighbor == start and path_len >= 4:
                cycle_lengths.add(path_len)
            elif neighbor > start and neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, visited, path_len + 1)
                visited.remove(neighbor)
                
    for start in range(n):
        visited = {start}
        dfs(start, start, visited, 1)
        
    return cycle_lengths

def _is_valid(edges, n=12):
    """Check if edges form a valid graph for the task."""
    if len(edges) == 0:
        return False
        
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
        
    # Check min degree >= 3
    for i in range(n):
        if len(adj[i]) < 3:
            return False
            
    # Check triangle-free (girth >= 4)
    # A triangle exists if for any edge (u,v), u and v share a common neighbor.
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = set(adj[u]).intersection(adj[v])
                if common:
                    return False
                    
    # Check has a cycle
    # If min degree >= 3, it must have a cycle. 
    # But let's be sure it's not a tree (impossible with deg>=3 for N>2).
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Strategy:
    # 1. Try to find a graph with 1 distinct cycle length (Raw 1 -> Score 5.0).
    # 2. Fall back to two disjoint K_3,3 components (Raw 2 -> Score 2.5).
    
    # Known best: Two K_3,3
    def make_two_k33():
        edges = []
        # Component 1: 0-2 (A), 3-5 (B)
        for u in [0, 1, 2]:
            for v in [3, 4, 5]:
                edges.append([u, v])
        # Component 2: 6-8 (A), 9-11 (B)
        for u in [6, 7, 8]:
            for v in [9, 10, 11]:
                edges.append([u, v])
        return edges
        
    best_edges = make_two_k33()
    best_score = len(_get_cycle_lengths(best_edges)) # Expected 2
    
    # Attempt to find a better graph (Raw 1)
    # This is a heuristic search for cubic bipartite graphs on 12 vertices.
    # We use the configuration model to generate random 3-regular bipartite graphs.
    
    random.seed(42)
    
    def random_cubic_bipartite():
        # Part A: 0-5, Part B: 6-11
        # Each vertex in A has degree 3, each in B has degree 3.
        # Total edges = 18.
        a_list = [i for i in range(6) for _ in range(3)]
        b_list = [i for i in range(6, 12) for _ in range(3)]
        
        # Shuffle B
        random.shuffle(b_list)
        
        edges = []
        edge_set = set()
        valid = True
        for i, a in enumerate(a_list):
            b = b_list[i]
            key = (min(a, b), max(a, b))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append([a, b])
            
        if not valid or len(edge_set) != 18:
            return None
            
        if _is_valid(edges):
            return edges
        return None

    # Try a limited number of random graphs
    # If we find one with 1 cycle length, we return it.
    # Given the theoretical constraints, this is unlikely, but we try.
    trials = 0
    max_trials = 5000
    
    while time.time() < deadline - 10 and trials < max_trials:
        edges = random_cubic_bipartite()
        if edges:
            lengths = _get_cycle_lengths(edges)
            if len(lengths) < best_score:
                best_score = len(lengths)
                best_edges = edges
                if best_score == 1:
                    break
        trials += 1
        
    # Ensure return type is correct
    result = [[int(u), int(v)] for u, v in best_edges]
    
    return result

# EVOLVE-BLOCK-END
