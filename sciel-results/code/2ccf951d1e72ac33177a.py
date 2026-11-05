import time
import random

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # For each a, find divisors d of a^2 such that d < a.
    # b = a + d, c = a + a^2/d
    # Check b < c and c <= N
    
    # conflict_pairs[x] will store a list of tuples (y, z)
    # such that {x, y, z} is a forbidden triple.
    # Note: The triple is undirected in the set context.
    # So if (a,b,c) is a triple, then:
    # conflict_pairs[a] contains (b,c)
    # conflict_pairs[b] contains (a,c)
    # conflict_pairs[c] contains (a,b)
    
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    # Triple (a, b, c)
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    def get_conflicts(x, S):
        """
        Returns a list of pairs (y, z) from conflict_pairs[x]
        such that both y and z are in S.
        """
        conflicts = []
        for y, z in conflict_pairs[x]:
            if y in S and z in S:
                conflicts.append((y, z))
        return conflicts

    def can_add(x, S):
        return len(get_conflicts(x, S)) == 0

    # Initial Solution: Upper half
    S = set(range(91, N + 1))
    
    # Greedy addition of smaller numbers
    for x in range(1, 91):
        if can_add(x, S):
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    random.seed(42)
    all_nums = set(range(1, N + 1))
    
    # Local Search
    # We will try to improve the set.
    # Moves:
    # 1. Add x (if possible) -> Size +1
    # 2. Swap: Remove u, Add x -> Size 0 (if x was blocked by u)
    # 3. Add x, Remove u, v -> Size -1 (Annealing/Kick)
    
    # To avoid infinite loops or getting stuck, we can use a simple counter
    # but given the time limit, we just run until deadline.
    
    for iteration in range(2000000):
        if time.time() > deadline:
            break
        
        excluded = all_nums - S
        if not excluded:
            break
            
        x = random.choice(tuple(excluded))
        conflicts = get_conflicts(x, S)
        
        if not conflicts:
            S.add(x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        
        # Try to resolve conflicts by removing elements
        # Conflicts are pairs (y, z). We need to remove at least one from each pair.
        
        # Try removing 1 element
        # Find an element that appears in all conflict pairs?
        # Or just try each candidate in the union of conflict pairs.
        candidates = set()
        for y, z in conflicts:
            candidates.add(y)
            candidates.add(z)
            
        found_swap = False
        # Shuffle candidates for diversity
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for u in cand_list:
            # If we remove u, does x become addable?
            # We need to check if there are any conflicts left in S \ {u}
            # We can just simulate:
            # Temporarily remove u
            S.remove(u)
            if can_add(x, S):
                S.add(x)
                # Size unchanged, but we might have enabled future moves
                # Accept swap always? Or with prob?
                # Let's accept if it doesn't decrease size.
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                found_swap = True
                break
            else:
                S.add(u) # Restore
        
        if found_swap:
            continue
            
        # Try removing 2 elements
        # This is more expensive. Only do if we are stuck or annealing.
        # We do this with low probability to escape local minima.
        if random.random() < 0.05: # 5% chance to try 2-removal
            # Get all candidates
            # Try pairs from candidates
            if len(cand_list) >= 2:
                # Random sample of pairs
                for _ in range(10):
                    u, v = random.sample(cand_list, 2)
                    S.remove(u)
                    S.remove(v)
                    if can_add(x, S):
                        S.add(x)
                        # Size decreased by 1
                        # We don't update best_S here unless size increases (which it doesn't)
                        # But this kick might help later
                        break
                    else:
                        S.add(u)
                        S.add(v)

    return sorted(best_S)
