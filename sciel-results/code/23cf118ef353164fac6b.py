import time
import random
from collections import defaultdict

def _precompute_conflicts(N=180):
    """Precompute all forbidden triples (a,b,c) with a<b<c and 1/a=1/b+1/c."""
    triples = []
    triple_index = defaultdict(list)  # element -> list of (other1, other2)
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b and a != c:
                    triples.append((a, b, c))
                    triple_index[a].append((b, c))
                    triple_index[b].append((a, c))
                    triple_index[c].append((a, b))
    
    return triples, triple_index

def _greedy_build(triples, triple_index, N, S_set, deadline):
    """Greedily add elements that create no conflicts, choosing the one with fewest future conflicts."""
    candidates = [x for x in range(1, N + 1) if x not in S_set]
    
    while time.time() < deadline:
        best_candidate = None
        best_future_conflicts = float('inf')
        best_current_conflicts = float('inf')
        
        for x in candidates:
            if x in S_set:
                continue
            
            # Count current conflicts (triples where both other elements are in S)
            current_conflicts = 0
            # Count future conflicts (triples where one other element is in S, the other is a candidate)
            future_conflicts = 0
            
            for (o1, o2) in triple_index.get(x, []):
                o1_in = o1 in S_set
                o2_in = o2 in S_set
                if o1_in and o2_in:
                    current_conflicts += 1
                elif o1_in or o2_in:
                    future_conflicts += 1
            
            if current_conflicts == 0 and future_conflicts < best_future_conflicts:
                best_future_conflicts = future_conflicts
                best_candidate = x
        
        if best_candidate is None:
            break
        
        S_set.add(best_candidate)
        candidates.remove(best_candidate)
    
    return S_set

def _local_search_1flip(triples, triple_index, N, S_set, deadline):
    """Try 1-flip moves: remove an element and add another if it improves size."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S_set)
        
        for x in S_list:
            if time.time() > deadline:
                return S_set
            
            # Try removing x
            S_set.remove(x)
            
            # Try to add the best candidate
            best_add = None
            best_future = float('inf')
            
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                current_conflicts = 0
                future_conflicts = 0
                for (o1, o2) in triple_index.get(y, []):
                    o1_in = o1 in S_set
                    o2_in = o2 in S_set
                    if o1_in and o2_in:
                        current_conflicts += 1
                    elif o1_in or o2_in:
                        future_conflicts += 1
                
                if current_conflicts == 0 and future_conflicts < best_future:
                    best_future = future_conflicts
                    best_add = y
            
            if best_add is not None:
                S_set.add(best_add)
                improved = True
            else:
                S_set.add(x)
    
    return S_set

def _local_search_2flip(triples, triple_index, N, S_set, deadline):
    """Try 2-flip moves: remove two elements and add two others if it improves size."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S_set)
        n = len(S_list)
        
        for i in range(n):
            if time.time() > deadline:
                return S_set
            for j in range(i + 1, n):
                x, y = S_list[i], S_list[j]
                
                # Remove x and y
                S_set.remove(x)
                S_set.remove(y)
                
                # Try to add two candidates
                added = []
                for _ in range(2):
                    best_add = None
                    best_future = float('inf')
                    for z in range(1, N + 1):
                        if z in S_set:
                            continue
                        current_conflicts = 0
                        future_conflicts = 0
                        for (o1, o2) in triple_index.get(z, []):
                            o1_in = o1 in S_set
                            o2_in = o2 in S_set
                            if o1_in and o2_in:
                                current_conflicts += 1
                            elif o1_in or o2_in:
                                future_conflicts += 1
                        
                        if current_conflicts == 0 and future_conflicts < best_future:
                            best_future = future_conflicts
                            best_add = z
                    
                    if best_add is not None:
                        S_set.add(best_add)
                        added.append(best_add)
                    else:
                        break
                
                if len(added) == 2:
                    improved = True
                else:
                    # Revert
                    for z in added:
                        S_set.remove(z)
                    S_set.add(x)
                    S_set.add(y)
    
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 180
    triples, triple_index = _precompute_conflicts(N)
    
    # Strategy: multiple random restarts with greedy + local search
    best_S_set = set()
    best_size = 0
    
    # First, try a deterministic greedy from empty
    S_set = set()
    S_set = _greedy_build(triples, triple_index, N, S_set, deadline)
    S_set = _local_search_1flip(triples, triple_index, N, S_set, deadline)
    S_set = _local_search_2flip(triples, triple_index, N, S_set, deadline)
    
    if len(S_set) > best_size:
        best_size = len(S_set)
        best_S_set = set(S_set)
    
    # Random restarts
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 10:
        restart_count += 1
        
        # Random initial set: include elements with some probability
        S_set = set()
        for x in range(1, N + 1):
            if random.random() < 0.3:
                S_set.add(x)
        
        # Greedily add more
        S_set = _greedy_build(triples, triple_index, N, S_set, deadline)
        S_set = _local_search_1flip(triples, triple_index, N, S_set, deadline)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = set(S_set)
        
        # 2-flip if time permits
        if time.time() < deadline - 5:
            S_set = _local_search_2flip(triples, triple_index, N, S_set, deadline)
            if len(S_set) > best_size:
                best_size = len(S_set)
                best_S_set = set(S_set)
    
    return sorted(best_S_set)
