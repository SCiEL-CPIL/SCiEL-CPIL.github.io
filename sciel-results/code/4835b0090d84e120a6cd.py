"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency structure
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Convert to sorted lists for faster iteration
adj_list = [sorted(s) for s in adj]

# Find connected components
visited = [False] * N
components = []
for i in range(N):
    if not visited[i]:
        comp = []
        stack = [i]
        visited[i] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj_list[node]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(comp)

# Sort components by size descending
components.sort(key=len, reverse=True)

# For small components (size <= 20), use exact DP over subsets to find longest path
def longest_path_exact(comp):
    """Find the longest simple path in a small component using DP over subsets."""
    n = len(comp)
    if n <= 1:
        return comp[:]
    
    # Map vertices to 0..n-1
    idx_map = {v: i for i, v in enumerate(comp)}
    # Build local adjacency
    local_adj = [set() for _ in range(n)]
    for u in comp:
        for v in adj_list[u]:
            if v in idx_map:
                local_adj[idx_map[u]].add(idx_map[v])
    
    # DP: dp[mask][i] = True if there's a path visiting exactly the vertices in mask, ending at i
    # We want to maximize the number of vertices in the path.
    # For n up to ~18-20, 2^n * n is feasible.
    
    if n <= 18:
        full_mask = (1 << n) - 1
        # dp[mask] is a bitset of which vertices can be the endpoint of a path covering exactly mask
        # Use list of integers as bitsets
        dp = [0] * (1 << n)
        # Initialize: single vertices
        for i in range(n):
            dp[1 << i] |= (1 << i)
        
        best_len = 1
        best_mask = 1
        best_end = 0
        
        for mask in range(1, 1 << n):
            if dp[mask] == 0:
                continue
            l = mask.bit_count()
            if l > best_len:
                best_len = l
                best_mask = mask
                best_end = dp[mask].bit_length() - 1  # highest set bit
            
            # Extend paths
            # For each endpoint i in dp[mask], try to go to neighbor j not in mask
            endpoints = dp[mask]
            while endpoints:
                i = (endpoints & -endpoints).bit_length() - 1
                endpoints &= endpoints - 1
                for j in local_adj[i]:
                    if not (mask & (1 << j)):
                        new_mask = mask | (1 << j)
                        dp[new_mask] |= (1 << j)
        
        # Reconstruct path from best_mask and best_end
        # We need to trace back. Let's store parent info or reconstruct.
        # Alternative: store the actual path during DP is expensive. 
        # Let's do a separate reconstruction: find a path of length best_len ending at best_end covering best_mask.
        
        # Actually, let's just do a DFS/BFS reconstruction from the DP table.
        # We know dp[best_mask] has bit best_end set. We need to find the sequence.
        # Let's re-run with parent tracking for the best solution only, or just use greedy from the mask.
        
        # Simpler: for reconstruction, do a reverse DP or just search.
        # Since n is small, let's just do a simple backtracking search for a path of length best_len.
        
        def find_path(mask, end, length):
            if length == 1:
                return [end]
            # Find previous vertex
            prev_mask = mask & ~(1 << end)
            for prev in local_adj[end]:
                if prev_mask & (1 << prev):
                    # Check if there's a path of length-1 ending at prev covering prev_mask
                    if dp[prev_mask] & (1 << prev):
                        subpath = find_path(prev_mask, prev, length - 1)
                        if subpath:
                            return subpath + [end]
            return None
        
        path_local = find_path(best_mask, best_end, best_len)
        if path_local:
            return [comp[i] for i in path_local]
    
    # Fallback: greedy for larger small components
    return None


def greedy_longest_path(start, adj_list, deadline):
    """Greedy longest path starting from a given vertex."""
    path = [start]
    seen = {start}
    while True:
        if time.time() > deadline:
            break
        current = path[-1]
        # Try all neighbors not yet visited
        candidates = [v for v in adj_list[current] if v not in seen]
        if not candidates:
            break
        # Choose neighbor with fewest unvisited neighbors (Warnsdorff-like)
        best_v = min(candidates, key=lambda v: (len([w for w in adj_list[v] if w not in seen]), v))
        path.append(best_v)
        seen.add(best_v)
    return path


def try_extend_path(path, adj_list, deadline):
    """Try to extend a path by adding vertices at either end."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to extend at the beginning
        if path:
            first = path[0]
            for nb in adj_list[first]:
                if nb not in set(path):
                    new_path = [nb] + path
                    path = new_path
                    changed = True
                    break
        # Try to extend at the end
        if path:
            last = path[-1]
            for nb in adj_list[last]:
                if nb not in set(path):
                    new_path = path + [nb]
                    path = new_path
                    changed = True
                    break
    return path


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_path = []
    
    # Process components from largest to smallest
    for comp in components:
        if time.time() > deadline:
            break
        
        if len(comp) <= 18:
            # Use exact DP
            exact_path = longest_path_exact(comp)
            if exact_path and len(exact_path) > len(best_path):
                best_path = exact_path
        else:
            # For larger components, use greedy with multiple restarts
            # Try starting from each vertex
            for v in comp:
                if time.time() > deadline:
                    break
                path = greedy_longest_path(v, adj_list, deadline)
                if len(path) > len(best_path):
                    best_path = path
            
            # Also try to extend the best path found in this component
            if best_path:
                best_path = try_extend_path(best_path, adj_list, deadline)
        
        # If we found a path covering the whole component, we can't do better in this component
        if len(best_path) == len(comp):
            continue
    
    # Final pass: try to extend the best path further
    if best_path:
        best_path = try_extend_path(best_path, adj_list, deadline)
    
    # Ensure we return a nonempty list
    if not best_path:
        best_path = [0]
    
    return best_path

# EVOLVE-BLOCK-END
