# EVOLVE-BLOCK-START
import time
import random
import sys

def _build_conflict_graph(n_max=512):
    """
    Build conflict graph: edge (a,b) if there exists c in [1,n_max] 
    with a+b=2c (i.e., a,b,c form 3-term AP with c as mean).
    We want an independent set in this graph.
    """
    # For each pair (a,b), check if (a+b) is even and (a+b)//2 is in range
    # and != a and != b
    # Actually, we need: does there exist c != a, c != b such that a+b=2c?
    # c = (a+b)/2, need a+b even, c in [1,n_max], c != a, c != b
    # c != a means b != a (since c=(a+b)/2=a implies b=a)
    # c != b means a != b
    # So for a != b: conflict if (a+b) even and 1 <= (a+b)//2 <= n_max
    
    # Represent as adjacency sets
    adj = [set() for _ in range(n_max + 1)]
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            s = a + b
            if s % 2 == 0:
                c = s // 2
                if 1 <= c <= n_max and c != a and c != b:
                    adj[a].add(b)
                    adj[b].add(a)
    return adj

def _greedy_independent_set(adj, order):
    """Greedy independent set: process nodes in given order, add if no conflict with already-added."""
    selected = set()
    for v in order:
        if v not in selected:
            # Check if v conflicts with any selected node
            conflict = False
            for u in adj[v]:
                if u in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
    return selected

def _local_search_improve(adj, selected, max_iters=2000, time_limit=30.0):
    """
    Local search: try to add nodes, and if can't, try swaps (remove one, add two).
    """
    selected = set(selected)
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Try to add any node
        added = False
        for v in range(1, 513):
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
                break
        
        if added:
            continue
        
        # Try swaps: remove one, add one or two
        # Try 1-for-1 swap
        swapped = False
        nodes_list = list(selected)
        random.shuffle(nodes_list)
        for v in nodes_list:
            for w in range(1, 513):
                if w in selected:
                    continue
                # Remove v, check if w can be added
                # After removing v, check w against selected - {v}
                conflict = False
                for u in adj[w]:
                    if u in selected and u != v:
                        conflict = True
                        break
                if not conflict:
                    # Also need to check that removing v doesn't cause issues
                    # (it won't, since we're just removing)
                    selected.remove(v)
                    selected.add(w)
                    swapped = True
                    break
            if swapped:
                break
        
        if swapped:
            continue
        
        # Try 1-for-2 swap: remove v, add w1 and w2
        swapped = False
        nodes_list = list(selected)
        random.shuffle(nodes_list)
        for v in nodes_list:
            # Find pairs (w1, w2) that can both be added after removing v
            # and w1, w2 don't conflict with each other
            candidates = []
            for w in range(1, 513):
                if w in selected:
                    continue
                conflict = False
                for u in adj[w]:
                    if u in selected and u != v:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(w)
            
            # Try to find two non-conflicting candidates
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    w1, w2 = candidates[i], candidates[j]
                    if w2 not in adj[w1]:
                        selected.remove(v)
                        selected.add(w1)
                        selected.add(w2)
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                break
        
        if not swapped and not added:
            # Stuck, try random perturbation
            if len(selected) > 10:
                v = random.choice(list(selected))
                selected.remove(v)
            else:
                break
    
    return selected

def _verify_no_mean_subset(S):
    """
    Verify that no element in S is the mean of any subset of >= 2 OTHER elements.
    Uses bitset DP.
    """
    S = list(S)
    n = len(S)
    if n < 3:
        return True
    
    for idx, target in enumerate(S):
        # Check if target is mean of any subset of S\{target} with size >= 2
        others = [S[j] for j in range(n) if j != idx]
        m = len(others)
        if m < 2:
            continue
        
        # DP: dp[k] = bitset of achievable sums with k elements from others
        max_sum = m * 512
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << x)
        
        # Check: exists k >= 2 such that sum = k * target
        for k in range(2, m + 1):
            t = k * target
            if t <= max_sum and (dp[k] >> t) & 1:
                return False
    
    return True

def _fast_verify_2elem(S):
    """Fast check: no element is mean of 2 other elements (3-term AP check)."""
    S_set = set(S)
    for x in S:
        for a in S:
            if a == x:
                continue
            b = 2 * x - a
            if b != x and b != a and b in S_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Build conflict graph
    adj = _build_conflict_graph(512)
    
    # Precompute degree for ordering
    degrees = [len(adj[i]) for i in range(513)]
    
    best_set = set()
    best_len = 0
    
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Different ordering strategies
        if restart == 0:
            # Lowest degree first
            order = sorted(range(1, 513), key=lambda x: degrees[x])
        elif restart == 1:
            # Highest degree first
            order = sorted(range(1, 513), key=lambda x: -degrees[x])
        elif restart == 2:
            # Random
            order = list(range(1, 513))
            random.shuffle(order)
        else:
            # Random with some structure
            order = list(range(1, 513))
            random.shuffle(order)
        
        # Greedy independent set
        selected = _greedy_independent_set(adj, order)
        
        # Local search improvement
        if time.time() < deadline - 10.0:
            selected = _local_search_improve(adj, selected, max_iters=1000, time_limit=10.0)
        
        # Verify full condition
        if _verify_no_mean_subset(selected):
            if len(selected) > best_len:
                best_set = set(selected)
                best_len = len(selected)
                if best_len >= 48:
                    return list(best_set)
        else:
            # Filter: remove elements that violate
            # Try to find a valid subset
            pass
    
    # Final attempt: try to extend best_set
    if best_set and time.time() < deadline - 5.0:
        best_list = list(best_set)
        for cand in range(1, 513):
            if cand in best_set:
                continue
            test_set = list(best_set) + [cand]
            if _verify_no_mean_subset(test_set):
                best_set.add(cand)
                best_list.append(cand)
                if len(best_set) >= 48:
                    return list(best_set)
    
    return list(best_set)

# EVOLVE-BLOCK-END
