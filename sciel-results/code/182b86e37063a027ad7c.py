"""Bootstrap search program for Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools
import math

DEADLINE = 150  # seconds
SAFETY_MARGIN = 10  # seconds to return before kill
START_TIME = None


def is_regular_subgraph(vertices, adj, n):
    """Check if the induced subgraph on 'vertices' is regular."""
    k = len(vertices)
    if k < 6:
        return False
    
    # Calculate degrees within the subset
    # For speed, we can just count neighbors in the subset
    degs = []
    for v in vertices:
        d = 0
        for u in vertices:
            if u != v and adj[v][u]:
                d += 1
        degs.append(d)
    
    if all(d == degs[0] for d in degs[1:]):
        return True
    return False

def find_violation_fast(adj, n, max_k=10):
    """Find a regular induced subgraph of size >= 6, checking up to max_k.
    Returns the subset or None.
    This is used during local search for speed.
    """
    # Check smaller k first as they are more likely to exist and cheaper to find
    for k in range(6, min(max_k, n) + 1):
        for combo in itertools.combinations(range(n), k):
            if is_regular_subgraph(combo, adj, n):
                return combo
    return None

def has_regular_induced_6plus_full(adj, n):
    """Check if any subset of 6+ vertices induces a regular subgraph.
    This is the slow, exact check used for final validation.
    """
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            if is_regular_subgraph(combo, adj, n):
                return True, combo
    return False, None

def random_graph(n, p=0.5, seed=None):
    """Generate a random graph with given edge probability."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    adj = [[False] * n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
    return adj, edges

def get_edges_from_adj(adj, n):
    e = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                e.append([u, v])
    return e

def local_search(adj, n, deadline, max_iters=500):
    """Local search: flip edges to eliminate regular induced subgraphs."""
    rng = random.Random()
    current_adj = [row[:] for row in adj]
    
    # Check if already valid (using fast check for initial state, but we need to be careful)
    # For the initial check in local search, we use the fast version to decide if we need to start.
    # But if it's valid, we must return it. The caller will do the full check.
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find a violation using the fast method (checks up to k=10)
        violation = find_violation_fast(current_adj, n, max_k=10)
        
        if violation is None:
            # No small violations found. 
            # We should check for larger violations before declaring victory, 
            # but to save time in the loop, we assume if no k<=10, it's likely good.
            # The final validation in construct() will be exact.
            # However, if we stop here, we might return a graph with a k=12 violation.
            # To be safer, we can do a quick check for k > 10 only if we are near the end of iterations.
            # For now, break and let the final check decide.
            break
        
        # Flip a random edge within the violating subset
        v_list = list(violation)
        if len(v_list) < 2:
            break
            
        # Pick two distinct vertices from the subset
        u, v = rng.sample(v_list, 2)
        
        # Flip the edge
        current_adj[u][v] = not current_adj[u][v]
        current_adj[v][u] = current_adj[u][v]
        
    return current_adj

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    
    best_n = 0
    best_edges = []
    best_valid = False
    
    rng = random.Random(12345)
    
    # Prioritize higher n. Known max is 20, maybe 21.
    # We will try n=21, then 20, then 19...
    # If we find a valid one, we keep it.
    
    # To maximize score, we want the highest n possible.
    # We'll iterate through candidate n values.
    # Since 20 is known, we start there.
    
    target_ns = [21, 20, 19, 18]
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        # If we already have a valid graph with n >= current target, we might stop?
        # No, we want to maximize n. So we try 21 first.
        # If we find a valid 21, we stop.
        # If not, we try 20.
        
        if best_valid and best_n >= n:
            continue
            
        attempts = 0
        # Give more time to higher n
        max_attempts = 100 if n == 21 else 200
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Vary p. p=0.5 is often a good start for random graphs.
            # Maybe try a range.
            p = 0.4 + rng.random() * 0.2 # 0.4 to 0.6
            
            adj, edges = random_graph(n, p=p)
            
            # Quick check: does it have small regular induced subgraphs?
            # If not, it's a candidate.
            # We run local search to try to fix any violations.
            
            new_adj = local_search(adj, n, deadline, max_iters=200)
            
            # Exact verification
            is_valid, _ = has_regular_induced_6plus_full(new_adj, n)
            
            if is_valid:
                new_edges = get_edges_from_adj(new_adj, n)
                if n > best_n:
                    best_n = n
                    best_edges = new_edges
                    best_valid = True
                    # Found a valid graph for this n. 
                    # If n was 21, we are done.
                    # If n was 20, we can stop trying 20 and move to 19? 
                    # No, we want max n. If we found 20, we don't need 19.
                    # But we tried 21 first. If 21 failed, we try 20.
                    # If we find 20, we stop the loop for 20 and the outer loop continues to 19?
                    # The outer loop structure:
                    # for n in [21, 20, 19, 18]:
                    #   if best_n >= n: continue
                    #   ... search ...
                    #   if found: best_n = n
                    # So if we find 20, best_n becomes 20.
                    # Next iteration n=19. best_n (20) >= 19, so continue.
                    # This logic is correct.
                    break
        
        # If we found a valid graph at this n, we can stop looking at this n
        if best_valid and best_n >= n:
            continue

    # Fallback
    if not best_valid or best_n < 5:
        # Return a known small valid graph
        # Empty graph on 5 vertices
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
