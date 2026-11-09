"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
import numpy as np

def _fast_sidon_check(s):
    """Check if set s is a Sidon set. Returns True if Sidon."""
    sums = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            val = si + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _get_sums_set(s):
    """Compute the set of all pairwise sums a+b for a<=b in s."""
    sums = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            sums.add(si + s[j])
    return sums

def _is_maximal_fast(s, N, sums_set):
    """
    Check if Sidon set s is inclusion-maximal in [1,N].
    """
    s_set = set(s)
    for x in range(1, N + 1):
        if x in s_set:
            continue
        if 2 * x in sums_set:
            continue
        blocked = False
        for a in s:
            if (x + a) in sums_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _build_greedy_sidon_heuristic(order, N):
    """
    Build a Sidon set.
    """
    selected = []
    sums = set()
    for x in order:
        new_sums = [2 * x]
        for a in selected:
            new_sums.append(x + a)
        
        if len(set(new_sums)) != len(new_sums):
            continue
            
        if sums.intersection(new_sums):
            continue
            
        selected.append(x)
        sums.update(new_sums)
        
    return selected

def _minimize_set_graph_heuristic(s, N):
    """
    Minimize s using the hypergraph hitting set heuristic.
    s: maximal Sidon set
    """
    s = list(s)
    s_set = set(s)
    
    # 1. Precompute hyperedges for each x not in s
    # hyperedges[x] = list of tuples (u, v, w) or (u, v) representing blocking collisions
    # We store them as frozensets of elements from s.
    
    # First, get sums and map sum -> list of pairs in s
    # sum_to_pairs[sum_val] = list of (a, b) such that a+b=sum_val, a<=b, a,b in s
    sum_to_pairs = {}
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i, n):
            val = si + s[j]
            if val not in sum_to_pairs:
                sum_to_pairs[val] = []
            sum_to_pairs[val].append((si, s[j]))
            
    hyperedges = {}
    for x in range(1, N + 1):
        if x in s_set:
            continue
            
        h_list = []
        
        # Check 2x
        if 2 * x in sum_to_pairs:
            for pair in sum_to_pairs[2 * x]:
                h_list.append(frozenset(pair))
                
        # Check x + a
        for a in s:
            val = x + a
            if val in sum_to_pairs:
                for pair in sum_to_pairs[val]:
                    # The collision is x+a = u+v
                    # Blocking set is {a, u, v}
                    h_list.append(frozenset([a, pair[0], pair[1]]))
                    
        if h_list:
            hyperedges[x] = h_list
            
    # 2. Initialize counts
    # cnt[x] = number of hyperedges H in hyperedges[x] such that H is subset of current K
    K = set(s)
    cnt = {}
    for x, h_list in hyperedges.items():
        cnt[x] = len(h_list) # All are subsets of K initially
        
    # 3. Iterative removal
    changed = True
    while changed:
        changed = False
        # Try to remove each element in K
        # To make it deterministic or random, we can shuffle K
        # But we want to be efficient.
        # Let's try removing elements one by one.
        # If we remove a, we need to check if any cnt[x] becomes 0.
        
        # Precompute for each a in K, which (x, H) pairs are affected
        # affected[a] = list of (x, H) where a in H and H subset of K
        
        # Since K changes, we recompute affected on the fly or maintain it.
        # Given small size, recompute is fine.
        
        for a in list(K):
            # Check if removing a is safe
            # For each x, count how many H in hyperedges[x] contain a and are subset of K
            # If cnt[x] - lost_count[x] == 0, unsafe.
            
            safe = True
            lost_counts = {}
            
            for x, h_list in hyperedges.items():
                lost = 0
                for H in h_list:
                    if a in H and H.issubset(K):
                        lost += 1
                if lost > 0:
                    if cnt[x] - lost == 0:
                        safe = False
                        break
                    lost_counts[x] = lost
                    
            if safe:
                # Commit removal
                K.remove(a)
                for x, lost in lost_counts.items():
                    cnt[x] -= lost
                changed = True
                break # Restart loop as K changed
                
    return list(K)

def construct():
    best_size = N + 1
    best_set = None
    start_time = time.time()
    time_limit = 28.0
    
    random.seed(42)
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        s = _build_greedy_sidon_heuristic(order, N)
        
        if len(s) >= best_size:
            pass
            
        s_min = _minimize_set_graph_heuristic(s, N)
        
        if len(s_min) < best_size:
            best_size = len(s_min)
            best_set = s_min[:]
            
        if best_set and time.time() - start_time < time_limit - 2.0:
             # Local search on best_set: try randomized removal order?
             # The graph heuristic already does a greedy removal.
             # We can try different removal orders by shuffling K before the while loop?
             # Or just rely on the outer loop generating different S.
             pass

    if best_set is None:
        return _build_greedy_sidon_heuristic(list(range(1, N + 1)), N)
        
    return best_set
# EVOLVE-BLOCK-END
