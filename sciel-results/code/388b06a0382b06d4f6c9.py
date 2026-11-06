"""Optimized construction for fewest C4s on 31 vertices, 91 edges."""
import time
import numpy as np

N = 31
M = 91
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def count_c4(edges):
    """Count the number of 4-cycles (subgraphs) in the graph defined by edges."""
    n = N
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
            total += common * (common - 1) // 2
    return total


def construct():
    start_time = time.time()
    
    # Construct the polarity graph of PG(2,5).
    # Points are 1D subspaces of F_5^3.
    p = 5
    points = []
    # x = 1
    for y in range(p):
        for z in range(p):
            points.append((1, y, z))
    # x = 0, y = 1
    for z in range(p):
        points.append((0, 1, z))
    # x = 0, y = 0, z = 1
    points.append((0, 0, 1))
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a non-degenerate symmetric polarity for PG(2,5).
    # B(u, v) = u0*v0 + u1*v1 + u2*v2 + u0*v1 + u1*v0
    # Matrix: [[1, 1, 0], [1, 1, 0], [0, 0, 1]] -> Det = 0. Degenerate.
    # Use: B(u, v) = u0*v0 + u1*v2 + u2*v1 + u1*v0? No.
    # Standard non-degenerate: [[1, 0, 0], [0, 1, 1], [0, 1, 1]]? Det 0.
    # Use: [[1, 1, 0], [1, 0, 0], [0, 0, 1]]? Det = -1. Good.
    # B(u, v) = u0*v0 + u0*v1 + u1*v0 + u2*v2
    
    def perp(a, b):
        return (a[0]*b[0] + a[0]*b[1] + a[1]*b[0] + a[2]*b[2]) % p == 0
    
    # Build the polarity graph
    edges_set = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if perp(points[i], points[j]):
                edges_set.add((points[i], points[j]))
    
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    polarity_edges = []
    for u_pt, v_pt in edges_set:
        u_idx = point_to_idx[u_pt]
        v_idx = point_to_idx[v_pt]
        polarity_edges.append((u_idx, v_idx))
    
    current_edges = [list(e) for e in polarity_edges]
    
    # If we have fewer than M edges, add the best ones.
    if len(current_edges) < M:
        adj = [[False] * N for _ in range(N)]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        current_edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in current_edges)
        
        # Find all non-edges
        candidates = []
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in current_edge_set:
                    candidates.append((i, j))
        
        # Sort by common neighbors (fewer common neighbors = fewer new C4s)
        def c4_increase(edge):
            u, v = edge
            return sum(1 for w in range(N) if adj[u][w] and adj[v][w])
        
        candidates.sort(key=c4_increase)
        
        # Add edges until we reach M
        for _ in range(M - len(current_edges)):
            if not candidates:
                break
            # Pick the best candidate
            best_edge = candidates.pop(0)
            u, v = best_edge
            current_edges.append([u, v])
            current_edge_set.add((u, v))
            adj[u][v] = True
            adj[v][u] = True
            
            # Remove candidates involving u or v from the list to avoid re-evaluating them (optional optimization)
            # But since we sorted once, and adding an edge increases common neighbors for others,
            # the order might change. For 91 edges, re-sorting or just taking the first few is fine.
            # Given the small number of additions (likely 1 or a few), simple pop is okay.
            # However, the "best" edge for the 2nd addition might not be the 2nd best of the 1st.
            # For robustness, we could re-evaluate, but 1 edge addition is likely sufficient if start is 90.
            
    # If we have more than M edges (should not happen with correct polarity count), remove worst.
    elif len(current_edges) > M:
        # This branch is unlikely to be triggered if polarity graph has 90 edges.
        pass

    # Final check
    if len(current_edges) != M:
        # Fallback
        all_pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
        current_edges = all_pairs[:M]

    return current_edges


if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
    print(f"Number of C4s: {count_c4(edges)}")
