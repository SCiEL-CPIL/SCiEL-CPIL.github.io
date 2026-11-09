import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    target_edges = 65
    
    # Initial graph: K_8,8 + edge (0,1)
    # Vertices 0-7 in A, 8-15 in B
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    def get_edges_list(edge_set):
        return [[u, v] for u, v in edge_set if u < v]

    def compute_score(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        max_sum = 0
        # Check all triangles
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = compute_score(edges)
    best_score = current_score
    best_edges = set(edges)
    
    # Local Search
    # Moves: Add edge, Remove edge, Swap (Add one, Remove one)
    # Constraint: Exactly 65 edges.
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        # Try swaps
        random.shuffle(all_possible_edges)
        for e_add in all_possible_edges:
            if e_add in edges:
                continue
            # Try removing an edge to keep count 65
            # Heuristic: remove edge incident to high degree vertices in triangles?
            # Just random remove for now
            random.shuffle(list(edges))
            for e_rem in list(edges):
                if e_rem == e_add:
                    continue
                new_edges = set(edges)
                new_edges.add(e_add)
                new_edges.remove(e_rem)
                
                # Quick check: does this create a triangle with high sum?
                # Just compute score
                new_score = compute_score(new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(edges)
                    break # Restart loop with new edges
                elif new_score == current_score:
                    # Accept with some probability? Or just accept to explore
                    if random.random() < 0.5:
                        edges = new_edges
                        # Score unchanged, but structure changed
                        break
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            
        if best_score <= 24:
            break

    return {'n': n, 'edges': get_edges_list(best_edges)}

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(len(res['edges']))
