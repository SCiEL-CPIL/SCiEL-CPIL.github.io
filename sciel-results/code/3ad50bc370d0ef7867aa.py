import time
import random
import math

N = 20
C = 20
DEADLINE = 145.0
START = time.time()

def elapsed():
    return time.time() - START

def construct():
    # Start with complete graph K20
    # Color edge (i, j) with (i + j) % 20
    # This is the "slope" of points (i, i^2)
    
    # adj[i][j] = color, -1 if no edge
    adj = [[-1] * N for _ in range(N)]
    edges = []
    
    for i in range(N):
        for j in range(i + 1, N):
            c = (i + j) % C
            adj[i][j] = c
            adj[j][i] = c
            edges.append([i, j, c])
            
    # Remove edges to ensure all 4-cycles are rainbow
    # A 4-cycle is a,b,c,d with edges (a,b), (b,c), (c,d), (d,a)
    # Colors: (a+b)%20, (b+c)%20, (c+d)%20, (d+a)%20
    # We need all 4 distinct.
    
    # We iterate and remove edges that are part of a non-rainbow 4-cycle.
    # To be efficient, we can just check all 4-cycles and mark edges for removal.
    # But removing one edge might fix multiple cycles.
    # Greedy: Find a bad cycle, remove an edge from it.
    
    # Precompute all 4-cycles? 20 choose 4 = 4845.
    # For each set of 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles (up to rotation/reversal).
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    
    bad_edges_to_remove = set()
    
    # We need to check validity dynamically as we remove edges.
    # However, removing an edge only *improves* validity (fewer cycles).
    # So we can identify all "bad" 4-cycles in the initial K20 coloring,
    # and then pick edges to remove to cover all these bad cycles.
    # This is a Set Cover problem, which is NP-hard, but N=20 is small.
    # Or simpler: Greedy removal.
    
    # Let's just iterate through all 4-cycles.
    # If a 4-cycle is bad, remove one of its edges.
    # Which one? The one that is in the most other bad cycles?
    # Or just the first one?
    
    # Let's count how many bad cycles each edge is in.
    # Then remove the "most bad" edges first.
    
    cycle_edges_count = {e: 0 for e in edges} # Use tuple (u,v) as key
    
    # 1. Identify all bad 4-cycles in the initial complete graph
    bad_cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Check 3 distinct 4-cycles on {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    if adj[a][b] != -1 and adj[b][c] != -1 and adj[c][d] != -1 and adj[d][a] != -1:
                        cols = [adj[a][b], adj[b][c], adj[c][d], adj[d][a]]
                        if len(set(cols)) < 4:
                            bad_cycles.append([(a,b), (b,c), (c,d), (d,a)])
                            
                    # Cycle 2: a-b-d-c-a
                    if adj[a][b] != -1 and adj[b][d] != -1 and adj[d][c] != -1 and adj[c][a] != -1:
                        cols = [adj[a][b], adj[b][d], adj[d][c], adj[c][a]]
                        if len(set(cols)) < 4:
                            bad_cycles.append([(a,b), (b,d), (d,c), (c,a)])
                            
                    # Cycle 3: a-c-b-d-a
                    if adj[a][c] != -1 and adj[c][b] != -1 and adj[b][d] != -1 and adj[d][a] != -1:
                        cols = [adj[a][c], adj[c][b], adj[b][d], adj[d][a]]
                        if len(set(cols)) < 4:
                            bad_cycles.append([(a,c), (c,b), (b,d), (d,a)])
                            
    # Count occurrences of each edge in bad cycles
    edge_bad_count = {}
    for cycle in bad_cycles:
        for e in cycle:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            edge_bad_count[key] = edge_bad_count.get(key, 0) + 1
            
    # Sort edges by bad count descending
    sorted_edges = sorted(edge_bad_count.items(), key=lambda x: x[1], reverse=True)
    
    removed = set()
    # Greedily remove edges
    for e, count in sorted_edges:
        if elapsed() > DEADLINE:
            break
        # If this edge is still present and part of a bad cycle, remove it
        # We don't need to re-check all cycles, just remove it.
        # But we must ensure we don't remove an edge that doesn't exist anymore (already removed).
        if e not in removed:
            # Check if removing this edge actually helps?
            # Yes, it breaks all cycles containing it.
            removed.add(e)
            
    # Construct final edge list
    final_edges = []
    for u, v, c in edges:
        key = (min(u, v), max(u, v))
        if key not in removed:
            final_edges.append([u, v, c])
            
    # Optional: Try to add back removed edges if possible?
    # Or try to recolor?
    # Given the density, just returning the remaining graph is a strong baseline.
    # K20 has 190 edges. Removing a few is a huge score.
    
    return final_edges

def is_rainbow(col_edges):
    # Fast verification for the final result
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
        
    for a in range(N):
        for b in range(a+1, N):
            for c_ in range(b+1, N):
                for d in range(c_+1, N):
                    # Check 3 cycles
                    if adj[a][b] >= 0 and adj[b][c_] >= 0 and adj[c_][d] >= 0 and adj[d][a] >= 0:
                        if len({adj[a][b], adj[b][c_], adj[c_][d], adj[d][a]}) < 4:
                            return False
                    if adj[a][b] >= 0 and adj[b][d] >= 0 and adj[d][c_] >= 0 and adj[c_][a] >= 0:
                        if len({adj[a][b], adj[b][d], adj[d][c_], adj[c_][a]}) < 4:
                            return False
                    if adj[a][c_] >= 0 and adj[c_][b] >= 0 and adj[b][d] >= 0 and adj[d][a] >= 0:
                        if len({adj[a][c_], adj[c_][b], adj[b][d], adj[d][a]}) < 4:
                            return False
    return True
