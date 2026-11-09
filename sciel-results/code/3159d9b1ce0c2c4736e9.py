"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
import sys
from functools import lru_cache

N = 24

# Generate the pinned edge list with random.Random(614)
def _build_edge_order():
    rng = random.Random(614)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    rng.shuffle(edges)
    return edges

EDGE_ORDER = _build_edge_order()
EDGE_INDEX = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(u, v)] = idx
    EDGE_INDEX[(v, u)] = idx

def _edge_score(u, v):
    return EDGE_INDEX[(u, v)]

def construct():
    # Strategy:
    # 1. Get a good initial upper bound T_max using a heuristic.
    # 2. Binary search for the minimum T such that a Hamiltonian cycle exists in G_T.
    # 3. Use an optimized DFS with heuristics and time limits for the cycle check.

    # Heuristic to find a good initial cycle
    def greedy_cycle_improved():
        best_cycle = None
        best_max_idx = float('inf')
        
        # Try multiple random orderings of vertices or restarts?
        # For N=24, simple greedy is fast. Let's try greedy from every start node?
        # No, the cycle is closed, so start node doesn't matter for the set of edges, 
        # but the path construction might.
        
        # Let's try a greedy approach that minimizes the max edge at each step?
        # Actually, minimizing the current edge index is a good proxy.
        
        for start in range(N):
            visited = [False] * N
            cycle = [start]
            visited[start] = True
            
            valid = True
            for _ in range(N - 1):
                last = cycle[-1]
                best_next = -1
                best_idx = float('inf')
                for v in range(N):
                    if not visited[v]:
                        idx = _edge_score(last, v)
                        if idx < best_idx:
                            best_idx = idx
                            best_next = v
                
                if best_next == -1:
                    valid = False
                    break
                
                cycle.append(best_next)
                visited[best_next] = True
            
            if valid and _edge_score(cycle[-1], cycle[0]) < float('inf'):
                max_idx = max(_edge_score(cycle[i], cycle[i+1]) for i in range(N-1))
                max_idx = max(max_idx, _edge_score(cycle[-1], cycle[0]))
                if max_idx < best_max_idx:
                    best_max_idx = max_idx
                    best_cycle = cycle[:]
        
        if best_cycle:
            return best_cycle, best_max_idx
        else:
            # Fallback
            cycle = list(range(N))
            max_idx = max(_edge_score(cycle[i], cycle[i+1]) for i in range(N-1))
            max_idx = max(max_idx, _edge_score(cycle[-1], cycle[0]))
            return cycle, max_idx

    best_cycle, best_max_idx = greedy_cycle_improved()
    
    # Binary search setup
    lo = N
    hi = best_max_idx
    
    sys.setrecursionlimit(10000)
    
    # Pre-compute all edge indices for speed?
    # We already have EDGE_INDEX.
    
    def find_cycle_with_max_idx(max_allowed):
        """Find a Hamiltonian cycle using only edges with index <= max_allowed."""
        # Build adjacency list for edges with index <= max_allowed
        # Sort neighbors by edge index to guide DFS
        adj = [[] for _ in range(N)]
        for u in range(N):
            for v in range(u + 1, N):
                if _edge_score(u, v) <= max_allowed:
                    adj[u].append((v, _edge_score(u, v)))
                    adj[v].append((u, _edge_score(u, v)))
        
        # Sort adjacency lists by edge index
        for u in range(N):
            adj[u].sort(key=lambda x: x[1])
            
        # Check necessary condition: min degree >= 2
        for u in range(N):
            if len(adj[u]) < 2:
                return None

        visited = [False] * N
        path = [0]
        visited[0] = True
        
        # Time limit for DFS
        start_time = time.time()
        time_limit = 5.0 # 5 seconds per check
        
        def dfs():
            nonlocal start_time, time_limit
            
            if time.time() - start_time > time_limit:
                return False
            
            if len(path) == N:
                # Check if we can close the cycle
                last = path[-1]
                for v, idx in adj[last]:
                    if v == 0 and idx <= max_allowed:
                        return True
                return False
            
            last = path[-1]
            
            # Heuristic: Check for forced moves
            # If an unvisited vertex has only 1 unvisited neighbor, we must go there?
            # This is complex to check efficiently in every step.
            # Simpler heuristic: Try neighbors in order of increasing edge index.
            
            for v, idx in adj[last]:
                if not visited[v]:
                    # Don't visit 0 until the end
                    if v == 0 and len(path) < N:
                        continue
                    
                    # Additional pruning: 
                    # If v is not 0, and there is an unvisited vertex w 
                    # that has no unvisited neighbors other than v (and v is not the last one), 
                    # then we are stuck.
                    # This is the "degree 1" check.
                    
                    visited[v] = True
                    path.append(v)
                    
                    if dfs():
                        return True
                    
                    path.pop()
                    visited[v] = False
            
            return False
        
        if dfs():
            return path[:]
        return None

    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        cycle = find_cycle_with_max_idx(mid)
        if cycle is not None:
            hi = mid
            best_cycle = cycle
            best_max_idx = mid
        else:
            lo = mid + 1
            
    return best_cycle

# EVOLVE-BLOCK-END
