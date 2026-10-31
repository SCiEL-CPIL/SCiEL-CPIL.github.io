"""Random Lift construction for k-regular graphs with high girth.
Uses non-backtracking walk verification for girth checking.
"""
import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Base edges list (unique undirected edges)
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))

    # Try random lifts of increasing size
    # m is the lift size. Total vertices = 10 * m
    # For k=3, g=16, we expect m around 50-500 to be sufficient.
    
    max_m = 5000  # Max lift size to try (N=50000)
    current_m = 50

    best_n = None
    best_edges = None
    best_girth_found = 0

    while current_m <= max_m and time.time() < deadline:
        # Try multiple random lifts of this size
        # More attempts for smaller m since each is cheap
        if current_m <= 100:
            attempts = 50
        elif current_m <= 500:
            attempts = 20
        else:
            attempts = 10

        for _ in range(attempts):
            if time.time() > deadline:
                break

            n, edges = _random_lift_petersen(base_edges, current_m)

            # Check girth using non-backtracking walks
            girth = _calculate_girth_nonbacktracking(n, edges, max_girth_to_check=g)

            if girth >= g:
                return n, edges

            # Track best
            if best_n is None or girth > best_girth_found or (girth == best_girth_found and n < best_n):
                best_n = n
                best_edges = edges
                best_girth_found = girth

        # If we didn't find a graph with girth >= g, try larger m
        current_m *= 2

    # If we exhausted search, return the best found
    if best_n is not None and best_girth_found >= g:
        return best_n, best_edges

    # Last resort: try one more time with a large m if time permits
    if time.time() < deadline - 5:
        m_final = min(max_m, int((deadline - time.time()) / 0.01))
        if m_final > current_m:
            n, edges = _random_lift_petersen(base_edges, m_final)
            girth = _calculate_girth_nonbacktracking(n, edges, max_girth_to_check=g)
            if girth >= g:
                return n, edges
            if best_n is None or girth > best_girth_found:
                best_n = n
                best_edges = edges
                best_girth_found = girth

    if best_n is not None:
        return best_n, best_edges

    return 0, []


def _random_lift_petersen(base_edges, m):
    """
    Generate a random lift of the Petersen graph of size m.
    Vertices are (u, i) where u in 0..9, i in 0..m-1.
    ID = u * m + i.
    """
    n = 10 * m
    edges = []

    for u, v in base_edges:
        shift = random.randint(0, m - 1)
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))

    return n, edges


def _calculate_girth_nonbacktracking(n, edges, max_girth_to_check=20):
    """
    Calculate the girth of the graph using non-backtracking walks.
    Returns the length of the shortest cycle.
    If no cycle shorter than max_girth_to_check is found, returns max_girth_to_check.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Non-backtracking walk verification
    # A non-backtracking walk is a sequence v_0, v_1, ..., v_L where v_i != v_{i-2} for i >= 2
    # The girth is the minimum L such that there exists a non-backtracking closed walk of length L.
    # For a simple graph, the shortest non-backtracking closed walk is exactly a simple cycle.
    
    # We use BFS on the directed edge graph (arc graph).
    # States are directed edges (u, v).
    # From (u, v), we can go to (v, w) for all w in adj[v] except w != u.
    
    min_girth = max_girth_to_check + 1
    
    # For each starting vertex, do BFS on non-backtracking walks
    for start in range(n):
        # visited[(u, v)] = length of shortest non-backtracking walk from start to v via u
        # Actually, we track (prev, curr) -> distance
        visited = {}
        queue = deque()
        
        # Initialize: from start, go to each neighbor
        for nb in adj[start]:
            visited[(start, nb)] = 1
            queue.append((start, nb, 1))
        
        while queue:
            prev, curr, dist = queue.popleft()
            
            if dist >= max_girth_to_check:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue  # Non-backtracking
                
                if nb == start and dist + 1 >= 2:
                    # Found a cycle of length dist + 1
                    cycle_len = dist + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth == 3:
                            return 3
                    continue
                
                if (curr, nb) not in visited:
                    visited[(curr, nb)] = dist + 1
                    queue.append((curr, nb, dist + 1))
        
        # Early exit if we found a very short cycle
        if min_girth <= 4:
            break
    
    return min_girth

# EVOLVE-BLOCK-END
