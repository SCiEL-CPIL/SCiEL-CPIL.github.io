import time
import math
import random
import numpy as np

def _build_conflict_graph(N=256):
    """Build adjacency lists for the conflict graph."""
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                adj[i - 1].append(j - 1)
                adj[j - 1].append(i - 1)
    return adj

def _validate(selected, N=256):
    """Check that no two selected elements form a bad pair."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a == b:
                continue
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(adj, N, rng, order='degree'):
    """Greedy independent set construction."""
    selected = set()
    available = set(range(N))
    
    if order == 'degree':
        # Sort by degree ascending (pick low-degree vertices first)
        order_list = sorted(available, key=lambda v: len(adj[v]))
    elif order == 'random':
        order_list = list(available)
        rng.shuffle(order_list)
    else:
        order_list = list(available)
    
    for v in order_list:
        if v not in available:
            continue
        # Check if v conflicts with any already selected
        conflict = False
        for u in adj[v]:
            if u in selected:
                conflict = True
                break
        if not conflict:
            selected.add(v)
            # Remove v and its neighbors from available
            available.discard(v)
            for u in adj[v]:
                available.discard(u)
    
    return selected

def _local_search(selected, adj, N, max_iter=500, time_limit=None, start_time=None):
    """Local search: try to add vertices, or swap to improve."""
    selected = set(selected)
    best_size = len(selected)
    
    for iteration in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Try adding unselected vertices
        added = False
        for v in range(N):
            if v in selected:
                continue
            conflict = False
            for u in adj[v]:
                if u in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
                added = True
                if len(selected) > best_size:
                    best_size = len(selected)
        
        if not added:
            break
    
    return selected

def _swap_improve(selected, adj, N, max_iter=200, time_limit=None, start_time=None):
    """Try swaps: remove one vertex, add two non-conflicting ones."""
    selected = set(selected)
    best_size = len(selected)
    
    for iteration in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        # Try removing each vertex and adding back better ones
        for v in list(selected):
            # Temporarily remove v
            selected.discard(v)
            # Try to add vertices that were blocked by v
            added_any = False
            for u in range(N):
                if u in selected:
                    continue
                conflict = False
                for w in adj[u]:
                    if w in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(u)
                    added_any = True
            
            if len(selected) > best_size:
                best_size = len(selected)
                improved = True
            else:
                # Revert: remove what we added, put v back
                to_remove = [u for u in selected if u != v and any(w in selected for w in adj[u] if w != v)]
                # Simpler: just rebuild
                pass
            
            if not improved:
                selected.add(v)
        
        if not improved:
            break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    N = 256
    rng = random.Random(42)
    
    # Build conflict graph
    adj = _build_conflict_graph(N)
    
    # Strategy 1: Start with all odds (known good baseline), then add evens
    # Odds are always safe among themselves (odd*odd / (odd+odd) = odd*odd/2*odd, 
    # for odd a,b: a+b is even, a*b is odd, so a+b cannot divide a*b)
    odds = set(range(1, N + 1, 2))  # 1-indexed
    odds_0 = set(v - 1 for v in odds)  # 0-indexed
    
    # Greedily add even numbers that don't conflict with odds or each other
    selected = set(odds_0)
    evens_0 = [v - 1 for v in range(2, N + 1, 2)]
    # Sort by degree (conflict count)
    evens_0.sort(key=lambda v: len(adj[v]))
    
    for v in evens_0:
        if time.time() > deadline:
            break
        conflict = False
        for u in adj[v]:
            if u in selected:
                conflict = True
                break
        if not conflict:
            selected.add(v)
    
    # Local search: try to add more
    selected = _local_search(selected, adj, N, max_iter=1000, time_limit=deadline - time.time(), start_time=start_time)
    
    best = set(selected)
    best_size = len(selected)
    
    # Multiple randomized restarts
    while time.time() < deadline - 5.0:
        # Randomized greedy
        sel = _greedy_build(adj, N, rng, order='random')
        sel = _local_search(sel, adj, N, max_iter=200, time_limit=min(5.0, deadline - time.time()), start_time=start_time)
        
        if len(sel) > best_size:
            best = set(sel)
            best_size = len(sel)
        
        # Also try degree-ordered greedy with random perturbation
        sel2 = _greedy_build(adj, N, rng, order='degree')
        sel2 = _local_search(sel2, adj, N, max_iter=200, time_limit=min(5.0, deadline - time.time()), start_time=start_time)
        
        if len(sel2) > best_size:
            best = set(sel2)
            best_size = len(sel2)
        
        # Try starting from odds + random evens
        sel3 = set(odds_0)
        random_evens = list(evens_0)
        rng.shuffle(random_evens)
        for v in random_evens:
            conflict = False
            for u in adj[v]:
                if u in sel3:
                    conflict = True
                    break
            if not conflict:
                sel3.add(v)
        sel3 = _local_search(sel3, adj, N, max_iter=200, time_limit=min(5.0, deadline - time.time()), start_time=start_time)
        
        if len(sel3) > best_size:
            best = set(sel3)
            best_size = len(sel3)
    
    # Final polish: try to add any remaining vertices
    final = _local_search(best, adj, N, max_iter=500, time_limit=max(0, deadline - time.time()), start_time=start_time)
    
    result = sorted(v + 1 for v in final)
    
    # Validate
    if not _validate(result, N):
        # Fallback to odds only
        result = list(range(1, N + 1, 2))
    
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Valid: {_validate(result)}")
    print(f"First 20: {result[:20]}")
