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
    
    # Build a bitset representation for fast membership tests
    # For each x, store a list of (y, z) pairs, but also build
    # a fast check: for a given S, can we add x?
    
    # Precompute for each x, the list of pairs as (y, z)
    # We'll use a different approach: for each x, store a list of 
    # "blocking sets" - each entry is a pair (y, z) such that 
    # if both y and z are in S, then x cannot be added.
    
    # Use a bitmask for S where bit i represents whether element i+1 is in S
    # This allows fast bitwise AND operations
    
    def can_add_bitmask(x, S_mask):
        """Check if x can be added to set represented by S_mask (bit i = element i+1)."""
        for y, z in conflict_pairs[x]:
            # Check if both y and z are in S_mask
            # Element y corresponds to bit (y-1), element z to bit (z-1)
            if (S_mask >> (y-1)) & 1 and (S_mask >> (z-1)) & 1:
                return False
        return True
    
    def add_to_mask(S_mask, x):
        """Add element x to the bitmask."""
        return S_mask | (1 << (x-1))
    
    def remove_from_mask(S_mask, x):
        """Remove element x from the bitmask."""
        return S_mask & ~(1 << (x-1))
    
    def mask_to_set(S_mask):
        """Convert bitmask to a set."""
        return {i+1 for i in range(N) if (S_mask >> i) & 1}
    
    def set_to_mask(S):
        """Convert a set to a bitmask."""
        mask = 0
        for x in S:
            mask |= (1 << (x-1))
        return mask
    
    # Initial Solution: Upper half
    S = set(range(91, N + 1))
    S_mask = set_to_mask(S)
    
    # Greedy addition of smaller numbers
    for x in range(1, 91):
        if can_add_bitmask(x, S_mask):
            S.add(x)
            S_mask = add_to_mask(S_mask, x)
            
    best_S = set(S)
    best_mask = S_mask
    best_size = len(S)
    
    random.seed(42)
    all_nums = set(range(1, N + 1))
    
    # Local Search with 2-swap moves
    # Track stuck counter for adaptive kicking
    stuck_count = 0
    max_stuck = 50
    
    for iteration in range(5000000):
        if time.time() > deadline:
            break
        
        excluded = all_nums - S
        if not excluded:
            break
            
        x = random.choice(tuple(excluded))
        
        # Check if x can be added directly
        if can_add_bitmask(x, S_mask):
            S.add(x)
            S_mask = add_to_mask(S_mask, x)
            if len(S) > best_size:
                best_S = set(S)
                best_mask = S_mask
                best_size = len(S)
            stuck_count = 0
            continue
        
        # x is blocked. Find conflicts
        conflicts = [(y, z) for y, z in conflict_pairs[x] if y in S and z in S]
        
        if not conflicts:
            S.add(x)
            S_mask = add_to_mask(S_mask, x)
            if len(S) > best_size:
                best_S = set(S)
                best_mask = S_mask
                best_size = len(S)
            stuck_count = 0
            continue
        
        # Try single swap: remove one element from each conflict pair
        candidates = set()
        for y, z in conflicts:
            candidates.add(y)
            candidates.add(z)
            
        found_move = False
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for u in cand_list:
            S.discard(u)
            new_mask = remove_from_mask(S_mask, u)
            if can_add_bitmask(x, new_mask):
                S.add(x)
                S_mask = add_to_mask(new_mask, x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_mask = S_mask
                    best_size = len(S)
                found_move = True
                stuck_count = 0
                break
            else:
                S.add(u)
        
        if found_move:
            continue
        
        # Try 2-swap: remove u, v and add x, w
        # First, try to find w that can be added after removing u, v
        # This is more complex. Let's try a simpler kick first.
        
        # Kick: remove 1 or 2 random elements from S to escape local minimum
        if stuck_count >= max_stuck or random.random() < 0.03:
            # Remove 1-3 random elements from S
            kick_size = 1 if random.random() < 0.5 else 2
            if len(S) > kick_size:
                kick_elements = random.sample(list(S), kick_size)
                for e in kick_elements:
                    S.discard(e)
                    S_mask = remove_from_mask(S_mask, e)
                # Try to add some elements back
                for _ in range(5):
                    remaining_excluded = all_nums - S
                    if not remaining_excluded:
                        break
                    w = random.choice(tuple(remaining_excluded))
                    if can_add_bitmask(w, S_mask):
                        S.add(w)
                        S_mask = add_to_mask(S_mask, w)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_mask = S_mask
                            best_size = len(S)
                stuck_count = 0
                continue
        
        stuck_count += 1
        
        # If very stuck, try a more aggressive 2-removal to add x
        if stuck_count > max_stuck * 2:
            if len(cand_list) >= 2:
                for _ in range(20):
                    u, v = random.sample(cand_list, 2)
                    S.discard(u)
                    S.discard(v)
                    new_mask = remove_from_mask(remove_from_mask(S_mask, u), v)
                    if can_add_bitmask(x, new_mask):
                        S.add(x)
                        S_mask = add_to_mask(new_mask, x)
                        # Try to recover size
                        for _ in range(3):
                            remaining_excluded = all_nums - S
                            if not remaining_excluded:
                                break
                            w = random.choice(tuple(remaining_excluded))
                            if can_add_bitmask(w, S_mask):
                                S.add(w)
                                S_mask = add_to_mask(S_mask, w)
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_mask = S_mask
                                    best_size = len(S)
                        stuck_count = 0
                        break
                    else:
                        S.add(u)
                        S.add(v)
                        S_mask = add_to_mask(S_mask, u)
                        S_mask = add_to_mask(S_mask, v)
            stuck_count = 0
    
    return sorted(best_S)
