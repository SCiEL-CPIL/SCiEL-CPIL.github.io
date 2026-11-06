"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_list = P['edge_order']
total_edges = len(edge_list)

# Precompute edge position (1-based) for each pair
edge_pos = [[-1] * n for _ in range(n)]
for i, (u, v) in enumerate(edge_list):
    edge_pos[u][v] = i + 1
    edge_pos[v][u] = i + 1

def score_cycle(cycle):
    """Compute the score of a cycle: max 1-based arrival index of its edges."""
    max_idx = 0
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        idx = edge_pos[u][v]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def find_hc_in_prefix(k, deadline):
    """
    Find a Hamilton cycle using only edges with position <= k.
    Returns the cycle or None.
    """
    # Build adjacency: adj[u] = set of v such that edge_pos[u][v] <= k
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and edge_pos[u][v] <= k:
                adj[u].add(v)
    
    # Quick connectivity check
    visited = [False] * n
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    if count < n:
        return None
    
    # Backtracking with Warnsdorff's rule and pruning
    path = [0]
    on_path = [False] * n
    on_path[0] = True
    
    # Precompute for fast neighbor iteration
    # We'll use a list for path and check on_path
    
    def backtrack():
        if time.time() > deadline:
            return False
        
        if len(path) == n:
            if 0 in adj[path[-1]]:
                return True
            return False
        
        u = path[-1]
        
        # Get available neighbors, sorted by Warnsdorff's rule
        neighbors = []
        for v in adj[u]:
            if not on_path[v]:
                # Count how many unvisited neighbors v has
                deg = 0
                for w in adj[v]:
                    if not on_path[w]:
                        deg += 1
                neighbors.append((deg, v))
        
        if not neighbors:
            return False
        
        # Sort by degree (fewest options first)
        neighbors.sort()
        
        for deg, v in neighbors:
            on_path[v] = True
            path.append(v)
            if backtrack():
                return True
            path.pop()
            on_path[v] = False
            
        return False
    
    if backtrack():
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # First, get a baseline score with identity permutation
    identity = list(range(n))
    best_score = score_cycle(identity)
    best_cycle = identity
    
    # Try binary search for minimum K where HC exists
    lo = n  # minimum edges needed
    hi = best_score  # we know best_score works
    
    # Try smaller K values
    # We'll do a linear scan from n upward, but with a limit
    # Actually, binary search is better
    
    # Let's try a few values
    # Start with the known good score, try to improve
    for k in range(n, best_score):
        if time.time() > deadline:
            break
        cycle = find_hc_in_prefix(k, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_score = k
            break  # found the minimum, we're done
    
    # If we didn't find a better one, try some heuristic approaches
    if best_score == score_cycle(identity):
        # Try to find a cycle with lower score using a different heuristic
        # Try random restarts with time limit
        import random
        rng = random.Random(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            # Try a random permutation and check its score
            perm = list(range(n))
            rng.shuffle(perm)
            s = score_cycle(perm)
            if s < best_score:
                best_score = s
                best_cycle = perm
    
    return best_cycle

# EVOLVE-BLOCK-END
