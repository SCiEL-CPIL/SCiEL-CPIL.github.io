import time
import random

def construct():
    deadline = time.time() + 80
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # For each a, find divisors d of a^2 such that d < a.
    # b = a + d, c = a + a^2/d
    # Check b < c and c <= N
    
    # conflict_pairs[x] stores list of (y, z) such that {x, y, z} is forbidden
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Use a boolean array for fast membership
    in_S = [False] * (N + 1)
    
    def can_add(x, in_S):
        for y, z in conflict_pairs[x]:
            if in_S[y] and in_S[z]:
                return False
        return True
    
    def add_to_S(x, in_S):
        in_S[x] = True
    
    def remove_from_S(x, in_S):
        in_S[x] = False
    
    def S_size(in_S):
        return sum(in_S[1:])
    
    def S_set(in_S):
        return {i for i in range(1, N + 1) if in_S[i]}
    
    best_S = set()
    best_size = 0
    
    random.seed(42)
    
    # Multiple restarts with different strategies
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Upper half initialization
            in_S = [False] * (N + 1)
            for x in range(91, N + 1):
                in_S[x] = True
        elif restart == 1:
            # Lower half initialization
            in_S = [False] * (N + 1)
            for x in range(1, 91):
                in_S[x] = True
        elif restart == 2:
            # Random initialization with ~50% density
            in_S = [False] * (N + 1)
            for x in range(1, N + 1):
                if random.random() < 0.5:
                    in_S[x] = True
            # Remove violations
            changed = True
            while changed:
                changed = False
                for x in range(1, N + 1):
                    if in_S[x]:
                        for y, z in conflict_pairs[x]:
                            if in_S[y] and in_S[z]:
                                # Remove one of y or z (the larger one to keep smaller numbers)
                                remove_from_S(max(y, z), in_S)
                                changed = True
                                break
        else:
            # Perturbed version of best found so far
            in_S = [False] * (N + 1)
            for x in best_S:
                in_S[x] = True
            # Remove 5-15 random elements
            num_remove = random.randint(5, 15)
            if len(best_S) > num_remove:
                to_remove = random.sample(list(best_S), num_remove)
                for x in to_remove:
                    remove_from_S(x, in_S)
        
        # Greedy addition of remaining numbers
        for x in range(1, N + 1):
            if not in_S[x] and can_add(x, in_S):
                add_to_S(x, in_S)
        
        current_size = S_size(in_S)
        if current_size > best_size:
            best_size = current_size
            best_S = S_set(in_S)
        
        # Local search phase
        local_deadline = min(deadline - 2, time.time() + 5)
        stuck_count = 0
        max_stuck = 30
        
        for iteration in range(200000):
            if time.time() > local_deadline:
                break
            
            excluded = [x for x in range(1, N + 1) if not in_S[x]]
            if not excluded:
                break
            
            x = random.choice(excluded)
            
            if can_add(x, in_S):
                add_to_S(x, in_S)
                current_size = S_size(in_S)
                if current_size > best_size:
                    best_size = current_size
                    best_S = S_set(in_S)
                stuck_count = 0
                continue
            
            # Find conflicts
            conflicts = [(y, z) for y, z in conflict_pairs[x] if in_S[y] and in_S[z]]
            
            if not conflicts:
                add_to_S(x, in_S)
                current_size = S_size(in_S)
                if current_size > best_size:
                    best_size = current_size
                    best_S = S_set(in_S)
                stuck_count = 0
                continue
            
            # Try single swap: remove one element from conflict pairs
            candidates = set()
            for y, z in conflicts:
                candidates.add(y)
                candidates.add(z)
            
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            found_move = False
            for u in cand_list:
                remove_from_S(u, in_S)
                if can_add(x, in_S):
                    add_to_S(x, in_S)
                    current_size = S_size(in_S)
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S_set(in_S)
                    found_move = True
                    stuck_count = 0
                    break
                else:
                    add_to_S(u, in_S)
            
            if found_move:
                continue
            
            # Try 2-swap: remove two elements, add x
            if len(cand_list) >= 2:
                for _ in range(10):
                    u, v = random.sample(cand_list, 2)
                    remove_from_S(u, in_S)
                    remove_from_S(v, in_S)
                    if can_add(x, in_S):
                        add_to_S(x, in_S)
                        # Try to recover size by adding back one of u,v or another
                        for w in [u, v] + random.sample(excluded, min(3, len(excluded))):
                            if w != x and not in_S[w] and can_add(w, in_S):
                                add_to_S(w, in_S)
                                break
                        current_size = S_size(in_S)
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S_set(in_S)
                        found_move = True
                        stuck_count = 0
                        break
                    else:
                        add_to_S(u, in_S)
                        add_to_S(v, in_S)
            
            if found_move:
                continue
            
            # Kick: remove 1-3 random elements to escape
            if stuck_count >= max_stuck or random.random() < 0.02:
                kick_size = random.randint(1, 3)
                if S_size(in_S) > kick_size:
                    S_list = [x for x in range(1, N + 1) if in_S[x]]
                    kick_elements = random.sample(S_list, kick_size)
                    for e in kick_elements:
                        remove_from_S(e, in_S)
                    # Try to add back some elements
                    for _ in range(8):
                        remaining = [x for x in range(1, N + 1) if not in_S[x]]
                        if not remaining:
                            break
                        w = random.choice(remaining)
                        if can_add(w, in_S):
                            add_to_S(w, in_S)
                            current_size = S_size(in_S)
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S_set(in_S)
                    stuck_count = 0
                    continue
            
            stuck_count += 1
    
    return sorted(best_S)
