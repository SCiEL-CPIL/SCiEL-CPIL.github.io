"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 15}
N = P['n']
ALL_PAIRS = list(combinations(range(N), 2))

def is_valid(edges_set, n=N):
    """Check if the graph with edges in edges_set satisfies the constraint:
    No simple path s-u-v-t (3 edges, 4 distinct vertices) has deg(s) == deg(t) in the full graph."""
    # Compute degrees
    deg = [0] * n
    for (u, v) in edges_set:
        deg[u] += 1
        deg[v] += 1
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check every edge (u,v) and every s in adj[u]\{v}, t in adj[v]\{u} with s != t
    for (u, v) in edges_set:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return False
    return True

def greedy_add(edges_set, n=N):
    """Try to add more edges greedily. Returns the final edge set."""
    changed = True
    while changed:
        changed = False
        for (u, v) in ALL_PAIRS:
            if (u, v) in edges_set or (v, u) in edges_set:
                continue
            new_set = edges_set | {(u, v)}
            if is_valid(new_set, n):
                edges_set = new_set
                changed = True
    return edges_set

def random_search(edges_set, n=N, time_limit=85.0):
    """Randomized local search to find a larger valid graph."""
    best = edges_set
    best_size = len(edges_set)
    start = time.time()
    rng = random.Random(42)
    
    while time.time() - start < time_limit:
        # Try adding a random missing edge
        missing = [(u, v) for (u, v) in ALL_PAIRS if (u, v) not in best and (v, u) not in best]
        if not missing:
            break
        (u, v) = rng.choice(missing)
        candidate = best | {(u, v)}
        if is_valid(candidate, n):
            best = candidate
            best_size = len(best)
            # After adding, try to add more greedily
            best = greedy_add(best, n)
            best_size = len(best)
        
        # Occasionally try removing an edge and re-adding different ones
        if rng.random() < 0.3 and best_size > 0:
            # Pick a random edge to remove
            edge_list = list(best)
            rem = rng.choice(edge_list)
            candidate = best - {rem}
            # Now try to add up to 2 random edges
            for _ in range(2):
                missing = [(a, b) for (a, b) in ALL_PAIRS if (a, b) not in candidate and (b, a) not in candidate]
                if not missing:
                    break
                (a, b) = rng.choice(missing)
                candidate2 = candidate | {(a, b)}
                if is_valid(candidate2, n):
                    candidate = candidate2
            if len(candidate) > best_size:
                best = candidate
                best_size = len(best)
                best = greedy_add(best, n)
                best_size = len(best)
    
    return best

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # Start from star (14 edges)
    star = frozenset((0, v) for v in range(1, n))
    current = set(star)
    
    # Greedy phase
    current = greedy_add(current, n)
    
    # Randomized search phase
    if time.time() < deadline - 5.0:
        current = random_search(current, n, time_limit=min(80.0, deadline - time.time() - 5.0))
    
    # Final greedy polish
    current = greedy_add(current, n)
    
    # Convert to list of [u, v] with Python ints
    edges = [[int(u), int(v)] for (u, v) in sorted(current)]
    
    # Verify validity one last time
    es = set(tuple(e) for e in edges)
    if not is_valid(es, n):
        # Fallback to star
        edges = [[0, int(v)] for v in range(1, n)]
    
    return edges

# EVOLVE-BLOCK-END
