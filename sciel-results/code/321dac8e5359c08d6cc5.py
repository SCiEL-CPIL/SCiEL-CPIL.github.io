"""Rewritten program: randomized greedy + local search for K2,3-free graphs on 20 vertices."""
import time
import random
import sys

N = 20
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    all_edges_set = set(all_edges)
    
    best_edges = []
    best_count = 0
    best_adj = None
    best_codeg = None
    
    def time_left():
        return deadline - time.time()
    
    def make_graph():
        """Return (adj, codeg, edges) for a given edge list."""
        adj = [set() for _ in range(N)]
        codeg = [[0] * N for _ in range(N)]
        edges = []
        for u, v in all_edges:
            if (u, v) in set(map(tuple, edges)) or (v, u) in set(map(tuple, edges)):
                continue
        # Actually, let's just build from scratch properly
        adj = [set() for _ in range(N)]
        codeg = [[0] * N for _ in range(N)]
        return adj, codeg, []
    
    def add_edge(adj, codeg, edges, u, v):
        """Add edge (u,v) and update adj/codeg. Returns True if valid."""
        # Check: for every common neighbor w of u and v, codeg[w][?] must stay <= 2
        # Actually, the constraint is: no two vertices have 3 common neighbors.
        # Adding edge (u,v): for any w that is neighbor of both u and v,
        # w becomes a common neighbor of... wait.
        # The K2,3-free condition: for any pair {a,b}, |N(a) ∩ N(b)| <= 2.
        # When we add edge (u,v), this changes N(u) and N(v).
        # For any w in N(u) (after adding v), w and u now share... 
        # Let me think more carefully.
        
        # The key insight: adding edge (u,v) means:
        # - v is added to N(u), u is added to N(v)
        # - For any w in N(u) (before adding), the pair (w,v) now has u as a common neighbor
        #   (since w~u and u~v after adding). So codeg[w][v] increases by 1.
        # - For any w in N(v) (before adding), the pair (w,u) now has v as a common neighbor
        #   So codeg[w][u] increases by 1.
        # - Also, the pair (u,v) itself: their common neighbors are N(u)∩N(v) (before adding).
        #   But u and v are now adjacent, so we need |N(u)∩N(v)| <= 2 where N includes the new edge?
        #   No - common neighbors don't include u or v themselves. N(u)∩N(v) before adding is what matters
        #   for the pair (u,v), but we also need to check that after adding, for pairs involving u or v,
        #   the common neighbor counts don't exceed 2.
        
        # Let me reconsider. The codeg[a][b] = |N(a) ∩ N(b)| for a != b.
        # When we add edge (u,v):
        # 1. For the pair (u,v): codeg[u][v] = |N(u) ∩ N(v)|. Before adding, N(u) and N(v) don't include
        #    each other. After adding, N(u) includes v and N(v) includes u, but common neighbors are
        #    vertices w != u,v with w~u and w~v. So codeg[u][v] = |N_old(u) ∩ N_old(v)|.
        #    This must be <= 2.
        # 2. For any w in N_old(u): w and v. Before, N(w)∩N(v). After, since u~v now and w~u, 
        #    u is in N(w)∩N(v). So codeg[w][v] increases by 1. Must stay <= 2.
        # 3. For any w in N_old(v): similarly, codeg[w][u] increases by 1. Must stay <= 2.
        # 4. No other pairs are affected.
        
        # Check condition 1:
        common_uv = len(adj[u] & adj[v])
        if common_uv > 2:
            return False
        
        # Check condition 2: for each w in adj[u], codeg[w][v] + 1 <= 2, i.e., codeg[w][v] <= 1
        for w in adj[u]:
            if codeg[w][v] >= 2:
                return False
        
        # Check condition 3: for each w in adj[v], codeg[w][u] + 1 <= 2, i.e., codeg[w][u] <= 1
        for w in adj[v]:
            if codeg[w][u] >= 2:
                return False
        
        # All checks passed, add the edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
        # Update codeg
        # For pair (u,v): codeg[u][v] = codeg[v][u] = common_uv
        codeg[u][v] = common_uv
        codeg[v][u] = common_uv
        
        # For each w in adj[u] (now includes v), update codeg[w][v] += 1
        for w in list(adj[u]):
            if w != v:  # don't update codeg[v][v]
                codeg[w][v] += 1
                codeg[v][w] += 1
        
        # For each w in adj[v] (now includes u), update codeg[w][u] += 1
        for w in list(adj[v]):
            if w != u:
                codeg[w][u] += 1
                codeg[u][w] += 1
        
        return True
    
    def remove_edge(adj, codeg, edges, u, v):
        """Remove edge (u,v) and update adj/codeg."""
        # Reverse of add_edge
        # First, decrement codeg for the affected pairs
        for w in list(adj[u]):
            if w != v:
                codeg[w][v] -= 1
                codeg[v][w] -= 1
        
        for w in list(adj[v]):
            if w != u:
                codeg[w][u] -= 1
                codeg[u][w] -= 1
        
        codeg[u][v] = 0
        codeg[v][u] = 0
        
        adj[u].discard(v)
        adj[v].discard(u)
        edges.remove((u, v))
    
    def local_search(adj, codeg, edges, max_iterations=1000):
        """Try to add more edges via local search. Returns improved edge list."""
        current_edges = list(edges)
        current_adj = [set(s) for s in adj]
        current_codeg = [row[:] for row in codeg]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time_left() > 5:
            improved = False
            iterations += 1
            # Try to add each missing edge
            for u, v in all_edges:
                if (u, v) in set(current_edges) or (v, u) in set(current_edges):
                    continue
                # Quick check
                if u in current_adj[v] or v in current_adj[u]:
                    continue
                common = len(current_adj[u] & current_adj[v])
                if common > 2:
                    continue
                valid = True
                for w in current_adj[u]:
                    if current_codeg[w][v] >= 2:
                        valid = False
                        break
                if valid:
                    for w in current_adj[v]:
                        if current_codeg[w][u] >= 2:
                            valid = False
                            break
                if valid:
                    # Add the edge
                    add_edge(current_adj, current_codeg, current_edges, u, v)
                    improved = True
                    break  # Restart scanning after a change
        
        return current_edges, current_adj, current_codeg
    
    # Randomized greedy with local search
    restarts = 0
    while time_left() > 10:
        restarts += 1
        
        # Shuffle edges for randomized order
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        adj = [set() for _ in range(N)]
        codeg = [[0] * N for _ in range(N)]
        edges = []
        
        # Greedy phase
        for u, v in shuffled:
            if u in adj[v]:
                continue
            if add_edge(adj, codeg, edges, u, v):
                pass
        
        greedy_count = len(edges)
        
        # Local search phase
        edges2, adj2, codeg2 = local_search(adj, codeg, edges)
        
        if len(edges2) > best_count:
            best_count = len(edges2)
            best_edges = [(u, v) for u, v in edges2]
            best_adj = [set(s) for s in adj2]
            best_codeg = [row[:] for row in codeg2]
        
        # If we've hit a high count, we can stop early
        if best_count >= 58:
            break
    
    # Convert to list of [u, v] format
    result = [[u, v] for u, v in best_edges]
    return result


# Ensure we can run as a script for testing
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
