import time
import random

def construct():
    n = 128
    deadline = time.time() + 85.0
    
    # Valid set: no distinct a,b,c in S with a+b, a+c, b+c all in S.
    # Equivalently: graph G on S where edge (u,v) exists iff u+v in S must be triangle-free.
    
    def is_valid(S):
        S_set = set(S)
        S_list = sorted(S)
        # Build adjacency for pairs with sum in S
        adj = {x: set() for x in S_list}
        for i in range(len(S_list)):
            u = S_list[i]
            for j in range(i+1, len(S_list)):
                v = S_list[j]
                if u + v in S_set:
                    adj[u].add(v)
                    adj[v].add(u)
        # Check triangles
        for u in S_list:
            neighbors = adj[u]
            for v in neighbors:
                if v <= u:
                    continue
                for w in neighbors:
                    if w > v and w in adj[v]:
                        return False
        return True
    
    # Greedy construction: add elements one by one, checking validity
    def greedy_random(start_set=None, seed=None):
        if seed is not None:
            random.seed(seed)
        S = set(start_set) if start_set else set()
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for x in candidates:
            if x in S:
                continue
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
        return S
    
    # Local search with swaps
    def local_search(S, time_limit):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Try random swaps
            if len(S) == 0:
                break
            to_remove = random.choice(list(S))
            candidates = [x for x in range(1, n+1) if x not in S]
            if not candidates:
                break
            to_add = random.choice(candidates)
            new_S = (S - {to_remove}) | {to_add}
            if is_valid(new_S):
                S = new_S
                # After swap, try to add more
                remaining = [x for x in range(1, n+1) if x not in S]
                random.shuffle(remaining)
                for x in remaining:
                    if time.time() - start_time > time_limit:
                        break
                    test_S = S | {x}
                    if is_valid(test_S):
                        S = test_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    
            # Try random adds (if possible)
            candidates = [x for x in range(1, n+1) if x not in S]
            if candidates:
                x = random.choice(candidates)
                new_S = S | {x}
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
        
        return best_S
    
    # Initial valid set: use the property that if min(S) + second_min(S) > max(S), 
    # then no sums can be in S, so it's trivially valid.
    # For S = {k, k+1, ..., n}, min+k+1 = 2k+1. Need 2k+1 > n => k > (n-1)/2 = 63.5, so k >= 64.
    # S = {64..128} has size 65. Let's verify: 64+65=129 > 128. Yes.
    
    base_set = set(range(64, n+1))  # size 65
    
    # Try to expand from base_set
    current_best = base_set
    best_size = len(base_set)
    
    # Greedy add from base
    candidates = [x for x in range(1, n+1) if x not in base_set]
    random.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline:
            break
        new_S = current_best | {x}
        if is_valid(new_S):
            current_best = new_S
            best_size = len(current_best)
    
    # Local search from best
    start = time.time()
    remaining_time = deadline - start - 5.0  # safety margin
    if remaining_time > 5.0:
        improved = local_search(current_best, remaining_time)
        if len(improved) > best_size:
            current_best = improved
            best_size = len(improved)
    
    # Multiple restarts with different random greedy starts
    start = time.time()
    remaining_time = deadline - start - 2.0
    restart_budget = remaining_time / 3.0  # ~3 restarts
    
    for i in range(20):
        if time.time() > deadline - 2.0:
            break
        # Random subset of size ~40-50 as seed
        seed_size = random.randint(30, 50)
        seed_elements = random.sample(range(1, n+1), seed_size)
        S = set(seed_elements)
        
        # Check if seed is valid, if not, prune it
        if not is_valid(S):
            # Remove elements randomly until valid
            while not is_valid(S) and S:
                x = random.choice(list(S))
                S.discard(x)
        
        # Greedy complete
        candidates = [x for x in range(1, n+1) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
        
        # Local search
        if time.time() < deadline - 3.0:
            S = local_search(S, min(5.0, deadline - time.time() - 2.0))
        
        if len(S) > best_size:
            best_size = len(S)
            current_best = S
    
    return sorted(current_best)
