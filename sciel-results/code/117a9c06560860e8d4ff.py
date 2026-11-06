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

def _check_valid(S_set, x, triple_index):
    """Check if adding x to S_set creates any conflict."""
    for (o1, o2) in triple_index.get(x, []):
        if o1 in S_set and o2 in S_set:
            return False
    return True

def _greedy_build(triple_index, N, S_set, order, deadline):
    """Greedily add elements in the given order that create no conflicts."""
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _check_valid(S_set, x, triple_index):
            S_set.add(x)
    return S_set

def _local_search_1flip(triple_index, N, S_set, deadline, fixed_set=None):
    """Try 1-flip moves. If fixed_set is provided, those elements cannot be removed."""
    improved = True
    all_elements = list(range(1, N + 1))
    
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(S_set)
        
        for x in S_list:
            if time.time() > deadline:
                return S_set
            if fixed_set and x in fixed_set:
                continue
            
            # Try removing x
            S_set.remove(x)
            
            # Try to add the best candidate not in S_set
            best_add = None
            best_future = float('inf')
            
            for y in all_elements:
                if y in S_set:
                    continue
                if not _check_valid(S_set, y, triple_index):
                    continue
                
                # Count future conflicts
                future_conflicts = 0
                for (o1, o2) in triple_index.get(y, []):
                    o1_in = o1 in S_set
                    o2_in = o2 in S_set
                    if o1_in or o2_in:
                        future_conflicts += 1
                
                if future_conflicts < best_future:
                    best_future = future_conflicts
                    best_add = y
            
            if best_add is not None:
                S_set.add(best_add)
                improved = True
            else:
                S_set.add(x)
    
    return S_set

def _local_search_2swap(triple_index, N, S_set, deadline, fixed_set=None):
    """Try 2-swap moves: remove two elements, add two others."""
    improved = True
    all_elements = list(range(1, N + 1))
    
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(S_set)
        
        for i in range(len(S_list)):
            if time.time() > deadline:
                return S_set
            x1 = S_list[i]
            if fixed_set and x1 in fixed_set:
                continue
            
            for j in range(i + 1, len(S_list)):
                x2 = S_list[j]
                if fixed_set and x2 in fixed_set:
                    continue
                
                # Try removing x1 and x2
                S_set.remove(x1)
                S_set.remove(x2)
                
                # Try to add two candidates
                best_add1 = None
                best_add2 = None
                best_total_future = float('inf')
                
                # Find best first add
                for y1 in all_elements:
                    if y1 in S_set:
                        continue
                    if not _check_valid(S_set, y1, triple_index):
                        continue
                    
                    S_set.add(y1)
                    
                    # Find best second add
                    for y2 in all_elements:
                        if y2 in S_set or y2 == y1:
                            continue
                        if not _check_valid(S_set, y2, triple_index):
                            continue
                        
                        # Count future conflicts for both
                        future1 = 0
                        for (o1, o2) in triple_index.get(y1, []):
                            if o1 in S_set or o2 in S_set:
                                future1 += 1
                        future2 = 0
                        for (o1, o2) in triple_index.get(y2, []):
                            if o1 in S_set or o2 in S_set:
                                future2 += 1
                        
                        total_future = future1 + future2
                        if total_future < best_total_future:
                            best_total_future = total_future
                            best_add1 = y1
                            best_add2 = y2
                    
                    S_set.remove(y1)
                
                S_set.add(x1)
                S_set.add(x2)
                
                if best_add1 is not None and best_add2 is not None:
                    S_set.remove(x1)
                    S_set.remove(x2)
                    S_set.add(best_add1)
                    S_set.add(best_add2)
                    improved = True
    
    return S_set

def construct():
    start_time = time.time()
    # Use 85s deadline to be safe against 90s kill
    deadline = start_time + 85
    
    N = 180
    triples, triple_index = _precompute_conflicts(N)
    
    best_S_set = set()
    best_size = 0
    
    # Strategy 1: Start with all odds, greedily add evens
    S_set = set(range(1, N + 1, 2))
    evens = list(range(2, N + 1, 2))
    S_set = _greedy_build(triple_index, N, S_set, evens, deadline)
    S_set = _local_search_1flip(triple_index, N, S_set, deadline)
    S_set = _local_search_2swap(triple_index, N, S_set, deadline)
    
    if len(S_set) > best_size:
        best_size = len(S_set)
        best_S_set = set(S_set)
    
    # Strategy 2: Random restarts with different orders
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 5 and restart_count < 10:
        restart_count += 1
        
        # Randomly choose which odds to include
        odds = list(range(1, N + 1, 2))
        random.shuffle(odds)
        S_set = set()
        
        # Greedily add odds
        for x in odds:
            if time.time() > deadline:
                break
            if _check_valid(S_set, x, triple_index):
                S_set.add(x)
        
        # Greedily add evens
        evens = list(range(2, N + 1, 2))
        random.shuffle(evens)
        S_set = _greedy_build(triple_index, N, S_set, evens, deadline)
        
        # Polish with local search
        S_set = _local_search_1flip(triple_index, N, S_set, deadline)
        S_set = _local_search_2swap(triple_index, N, S_set, deadline)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = set(S_set)
    
    # Strategy 3: Try to add more elements by removing some
    if time.time() < deadline - 2:
        S_set = set(best_S_set)
        S_set = _local_search_1flip(triple_index, N, S_set, deadline)
        S_set = _local_search_2swap(triple_index, N, S_set, deadline)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = set(S_set)
    
    return sorted(best_S_set)
