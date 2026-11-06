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

def _greedy_build_evens(triple_index, N, S_set, deadline):
    """Greedily add even elements that create no conflicts with existing S_set (which contains all odds)."""
    # Candidates are only even numbers not already in S_set
    candidates = [x for x in range(2, N + 1, 2) if x not in S_set]
    
    while time.time() < deadline:
        best_candidate = None
        best_future_conflicts = float('inf')
        
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

def _local_search_1flip_evens(triple_index, N, S_set, deadline):
    """Try 1-flip moves on the EVEN subset only. Odds are fixed."""
    improved = True
    even_candidates = [x for x in range(2, N + 1, 2)]
    
    while improved and time.time() < deadline:
        improved = False
        S_evens = [x for x in S_set if x % 2 == 0]
        
        for x in S_evens:
            if time.time() > deadline:
                return S_set
            
            # Try removing x (even)
            S_set.remove(x)
            
            # Try to add the best even candidate
            best_add = None
            best_future = float('inf')
            
            for y in even_candidates:
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

def construct():
    start_time = time.time()
    # Use 55s deadline to be safe against 60s kill
    deadline = start_time + 55
    
    N = 180
    triples, triple_index = _precompute_conflicts(N)
    
    # Strategy:
    # 1. Start with all odd numbers (valid because no 3 distinct odds form a reciprocal triple)
    # 2. Greedily add even numbers
    # 3. Local search on even numbers
    
    best_S_set = set()
    best_size = 0
    
    # Initial set: All odds
    S_set = set(range(1, N + 1, 2))
    
    # Greedy add evens
    S_set = _greedy_build_evens(triple_index, N, S_set, deadline)
    
    # Local search 1-flip on evens
    S_set = _local_search_1flip_evens(triple_index, N, S_set, deadline)
    
    if len(S_set) > best_size:
        best_size = len(S_set)
        best_S_set = set(S_set)
        
    # Try a few random restarts on the even part if time permits
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 5 and restart_count < 5:
        restart_count += 1
        S_set = set(range(1, N + 1, 2))
        
        # Randomly shuffle the order of even candidates for greedy
        evens = list(range(2, N + 1, 2))
        random.shuffle(evens)
        
        # Modified greedy: iterate through shuffled evens, add if valid
        for x in evens:
            if time.time() > deadline:
                break
            valid = True
            for (o1, o2) in triple_index.get(x, []):
                if o1 in S_set and o2 in S_set:
                    valid = False
                    break
            if valid:
                S_set.add(x)
                
        # Polish with local search
        S_set = _local_search_1flip_evens(triple_index, N, S_set, deadline)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_S_set = set(S_set)

    return sorted(best_S_set)
