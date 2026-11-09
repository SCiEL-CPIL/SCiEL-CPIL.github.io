import time
import random
from itertools import combinations

N = 18

def is_k4_free(adj):
    """Check if graph with adjacency matrix adj is K4-free."""
    for a in range(N):
        for b in range(a+1, N):
            if not adj[a][b]:
                continue
            for c in range(b+1, N):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, N):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def max_triangle_free_induced(adj):
    """Find the size of the largest triangle-free induced subgraph."""
    # Branch and bound
    # A set S is triangle-free if no three vertices in S form a triangle.
    # We build S incrementally.
    
    # Precompute: for each pair (i,j) with edge, the set of common neighbors
    # If we include i and j (adjacent), we cannot include any common neighbor.
    
    best = [0]
    
    def branch(candidates, included_count, forbidden):
        """
        candidates: set of vertices that can still be added
        included_count: number of vertices in current triangle-free set
        forbidden: set of vertices that cannot be added (would create triangle)
        """
        if not candidates:
            if included_count > best[0]:
                best[0] = included_count
            return
        
        # Upper bound: included_count + len(candidates)
        if included_count + len(candidates) <= best[0]:
            return
        
        # Try to add vertices
        # Pick a vertex to branch on
        v = next(iter(candidates))
        
        # Option 1: include v
        new_forbidden = set(forbidden)
        new_candidates = set(candidates)
        new_candidates.discard(v)
        
        # If we include v, any vertex that forms a triangle with v and another included vertex is forbidden
        # But we need to track which vertices are included... 
        # Actually, the standard approach: a set S is triangle-free iff for every edge (u,w) in S, there's no common neighbor in S.
        # Alternative: S is triangle-free iff S is an independent set in the "triangle graph" where... no.
        
        # Better approach: S is triangle-free means no 3 vertices in S are pairwise adjacent.
        # When we add v to S, v must not form a triangle with any two existing vertices in S.
        # But we're not tracking S explicitly, just the count and forbidden.
        
        # Let me restructure: track the actual set S.
        pass
    
    # Simpler: use a different B&B
    # State: (S, candidates) where S is the current triangle-free set, candidates are vertices not yet decided
    # A vertex v can be added to S iff for all u,w in S with u~w, v is not adjacent to both u and w.
    
    def branch2(S, candidates):
        if not candidates:
            if len(S) > best[0]:
                best[0] = len(S)
            return
        if len(S) + len(candidates) <= best[0]:
            return
        
        # Pick vertex with fewest options to branch
        v = min(candidates, key=lambda x: sum(adj[x][y] for y in S))
        
        # Option 1: include v (if it doesn't create a triangle)
        can_include = True
        for u in S:
            for w in S:
                if u < w and adj[u][w] and adj[v][u] and adj[v][w]:
                    can_include = False
                    break
            if not can_include:
                break
        
        if can_include:
            S.append(v)
            cands2 = list(candidates)
            cands2.remove(v)
            branch2(S, cands2)
            S.pop()
        
        # Option 2: exclude v
        cands3 = list(candidates)
        cands3.remove(v)
        branch2(S, cands3)
    
    branch2([], list(range(N)))
    return best[0]

def construct():
    deadline = time.time() + 100  # 100s budget, 20s safety margin
    
    # Start with K6,6,6
    adj = [[0]*N for _ in range(N)]
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = 1
                    adj[b][a] = 1
    
    best_score = max_triangle_free_induced(adj)
    best_adj = [row[:] for row in adj]
    
    # Local search
    random.seed(42)
    t = time.time()
    
    while t < deadline:
        # Try random edge flips
        improved = False
        for _ in range(200):
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v:
                continue
            
            old_val = adj[u][v]
            adj[u][v] = 1 - old_val
            adj[v][u] = 1 - old_val
            
            if not is_k4_free(adj):
                adj[u][v] = old_val
                adj[v][u] = old_val
                continue
            
            score = max_triangle_free_induced(adj)
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in adj]
                improved = True
            else:
                adj[u][v] = old_val
                adj[v][u] = old_val
        
        t = time.time()
        if not improved and t > deadline - 30:
            break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
