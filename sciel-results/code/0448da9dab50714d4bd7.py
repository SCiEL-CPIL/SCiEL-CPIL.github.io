import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    N = 1000
    best_set = []
    best_size = 0

    def try_greedy_cost(order, use_cost=True):
        """Greedy with optional cost-based selection for better heuristics."""
        S = []
        counts = [0] * (2 * N + 1)
        is_full = [False] * (2 * N + 1)
        candidates = list(order)
        
        while candidates:
            best_x = None
            best_cost = float('inf')
            best_candidates = []
            
            for x in candidates:
                # Quick reject: 2x already full
                if is_full[2 * x]:
                    continue
                
                valid = True
                cost = 0
                if counts[2 * x] == 1:
                    cost += 1
                
                # Single pass: check validity AND compute cost
                for y in S:
                    s = x + y
                    if is_full[s]:
                        valid = False
                        break
                    if counts[s] == 1:
                        cost += 1
                
                if not valid:
                    continue
                
                if cost < best_cost:
                    best_cost = cost
                    best_candidates = [x]
                elif cost == best_cost:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
            
            # Pick from best candidates (random tie-breaking for diversity)
            best_x = random.choice(best_candidates)
            S.append(best_x)
            
            # Update counts
            counts[2 * best_x] += 1
            if counts[2 * best_x] == 2:
                is_full[2 * best_x] = True
            for y in S[:-1]:
                s = best_x + y
                counts[s] += 1
                if counts[s] == 2:
                    is_full[s] = True
            
            # Remove from candidates
            candidates.remove(best_x)
        
        return S

    def try_greedy_fast(order):
        """Fast greedy without cost computation - just first valid element."""
        S = []
        counts = [0] * (2 * N + 1)
        is_full = [False] * (2 * N + 1)
        
        for x in order:
            if is_full[2 * x]:
                continue
            
            valid = True
            for y in S:
                if is_full[x + y]:
                    valid = False
                    break
            
            if valid:
                S.append(x)
                counts[2 * x] += 1
                if counts[2 * x] == 2:
                    is_full[2 * x] = True
                for y in S[:-1]:
                    s = x + y
                    counts[s] += 1
                    if counts[s] == 2:
                        is_full[s] = True
        
        return S

    def local_search(S, max_iters=200):
        """Local search: remove one element, try to add a better one."""
        S = list(S)
        S_set = set(S)
        counts = [0] * (2 * N + 1)
        is_full = [False] * (2 * N + 1)
        
        # Build counts
        for i, a in enumerate(S):
            counts[2 * a] += 1
            if counts[2 * a] == 2:
                is_full[2 * a] = True
            for j in range(i + 1, len(S)):
                s = a + S[j]
                counts[s] += 1
                if counts[s] == 2:
                    is_full[s] = True
        
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if not S:
                break
            
            # Pick random element to remove
            idx = random.randint(0, len(S) - 1)
            old_x = S[idx]
            
            # Remove old_x
            for i, y in enumerate(S):
                if y == old_x:
                    continue
                s = old_x + y
                counts[s] -= 1
                if counts[s] == 0:
                    is_full[s] = False
            counts[2 * old_x] -= 1
            if counts[2 * old_x] == 0:
                is_full[2 * old_x] = False
            S.pop(idx)
            S_set.discard(old_x)
            
            # Try to find a replacement
            found = False
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            random.shuffle(candidates)
            
            for x in candidates[:150]:
                if is_full[2 * x]:
                    continue
                
                valid = True
                for y in S:
                    if is_full[x + y]:
                        valid = False
                        break
                
                if valid:
                    # Add x
                    S.append(x)
                    S_set.add(x)
                    counts[2 * x] += 1
                    if counts[2 * x] == 2:
                        is_full[2 * x] = True
                    for y in S[:-1]:
                        s = x + y
                        counts[s] += 1
                        if counts[s] == 2:
                            is_full[s] = True
                    found = True
                    improved = True
                    break
            
            if not found:
                # Put old_x back
                S.append(old_x)
                S_set.add(old_x)
                for y in S[:-1]:
                    s = old_x + y
                    counts[s] += 1
                    if counts[s] == 2:
                        is_full[s] = True
                counts[2 * old_x] += 1
                if counts[2 * old_x] == 2:
                    is_full[2 * old_x] = True
        
        return S

    # Phase 1: Fast greedy with different orderings
    S = try_greedy_fast(range(1, N + 1))
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    S = try_greedy_fast(range(N, 0, -1))
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    # Phase 2: Cost-based greedy with random restarts
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        random.seed(seed * 12345 + 67890)
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = try_greedy_cost(order, use_cost=True)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Phase 3: More fast greedy restarts (cheap, so do many)
    while time.time() < deadline - 20:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = try_greedy_fast(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Phase 4: Local search on the best solution
    if time.time() < deadline - 5:
        improved_S = local_search(best_set, max_iters=300)
        if len(improved_S) > best_size:
            best_set = improved_S
            best_size = len(best_set)
    
    # Phase 5: Try local search on a few other good solutions
    if time.time() < deadline - 5:
        # Run a few more greedy and try local search on them
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = try_greedy_fast(order)
            if len(S) >= best_size - 2:  # Only search near-best solutions
                improved_S = local_search(S, max_iters=100)
                if len(improved_S) > best_size:
                    best_set = improved_S
                    best_size = len(best_set)
    
    return best_set
