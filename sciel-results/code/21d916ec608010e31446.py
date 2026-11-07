import time
import random
import sys

P = {'n': 15}

def construct():
    n = P['n']
    deadline = time.time() + 55.0
    
    # Target: Maximize edges.
    # We know 53 edges is possible with degrees {14, 13, ..., 2, 1, 1}.
    # Let's try to find a valid graph with 53 or 54 edges.
    
    best_edges = []
    best_count = 0
    
    def is_valid(edges):
        if not edges: return True
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all paths s-u-v-t
        # For each edge (u,v), for each s in N(u), for each t in N(v)
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Attempt 1: Construct specific degree sequence
    # Degrees: 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1
    target_deg = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    
    def construct_greedy(target_deg):
        deg = target_deg[:]
        adj = [set() for _ in range(n)]
        
        # Sort vertices by target degree descending
        # But we need to be careful.
        # Havel-Hakimi style:
        # Pick vertex with highest deg, connect to next highest.
        
        # Simple greedy:
        # 1. Sort vertices by remaining degree.
        # 2. Connect top vertex to others with highest remaining degree.
        
        remaining = target_deg[:]
        for _ in range(n):
            if time.time() > deadline: break
            # Find vertex with max remaining degree
            max_idx = -1
            max_deg = -1
            for i in range(n):
                if remaining[i] > max_deg:
                    max_deg = remaining[i]
                    max_idx = i
            
            if max_deg < 0:
                break
                
            # Connect max_idx to max_deg other vertices with highest remaining degree
            # Exclude max_idx itself
            candidates = [i for i in range(n) if i != max_idx and remaining[i] > 0]
            # Sort candidates by remaining degree descending
            candidates.sort(key=lambda x: remaining[x], reverse=True)
            
            for i in candidates[:max_deg]:
                adj[max_idx].add(i)
                adj[i].add(max_idx)
                remaining[max_idx] -= 1
                remaining[i] -= 1
            remaining[max_idx] = 0
            
        # Check if we achieved the target degrees
        actual_deg = [len(adj[i]) for i in range(n)]
        if sorted(actual_deg, reverse=True) == sorted(target_deg, reverse=True):
            edges = get_edges_from_adj(adj)
            if is_valid(edges):
                return edges
        return []

    edges_53 = construct_greedy(target_deg)
    if edges_53:
        best_count = len(edges_53)
        best_edges = edges_53

    # Attempt 2: Local search to improve or find 54 edges
    # Start from a dense random graph or the 53-edge graph
    if best_edges:
        current_edges = list(best_edges)
    else:
        # Fallback: Random dense
        current_edges = [(i, j) for i in range(n) for j in range(i+1, n) if random.random() < 0.6]
        
    current_valid = is_valid(current_edges)
    
    # Local Search
    # Try to add edges if valid, or swap edges to maintain validity and increase count
    # Actually, just try to add edges to the current set.
    # If adding makes it invalid, try to remove some other edge to fix it?
    # Simpler: Simulated Annealing on the edge set.
    
    # Let's just try to add all possible edges to the 53-edge graph and see if any work
    if best_count >= 53:
        current_set = set(current_edges)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for u, v in all_pairs:
            if time.time() > deadline: break
            if (u, v) in current_set: continue
            if (v, u) in current_set: continue
            
            new_edges = current_edges + [(u, v)]
            if is_valid(new_edges):
                current_edges = new_edges
                current_set.add((u, v))
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = current_edges[:]
                    
    # If we are stuck, do some random restarts with local search
    for _ in range(100):
        if time.time() > deadline: break
        
        # Randomize
        rand_edges = [(i, j) for i in range(n) for j in range(i+1, n) if random.random() < 0.5]
        
        # Greedy add
        for u, v in rand_edges:
            new_edges = rand_edges + [(u, v)]
            if is_valid(new_edges):
                rand_edges = new_edges
                
        if len(rand_edges) > best_count:
            best_count = len(rand_edges)
            best_edges = rand_edges[:]

    # Final check
    if not is_valid(best_edges):
        # Fallback to empty or known valid small graph
        return []
        
    return [list(e) for e in best_edges]

if __name__ == '__main__':
    construct()
