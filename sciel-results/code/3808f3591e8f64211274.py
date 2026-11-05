import time
import random

def construct():
    deadline = time.time() + 80
    N = 180
    
    # Precompute forbidden triples
    # conflict_pairs[x] contains list of (y, z) such that {x, y, z} is a forbidden triple
    # We store undirected edges in a hypergraph sense.
    # For a fixed x, if y and z are in S, we cannot add x.
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    # Triple is (a, b, c) with a < b < c? Not necessarily b < c, but we enforced b < c.
                    # Actually a < b and a < c is guaranteed since d > 0.
                    # We add the pair (b, c) to a's conflicts.
                    # And (a, c) to b's conflicts? No, the condition is symmetric for the set.
                    # If we have a, b, c in S, it's bad.
                    # So for any element x in the triple, the other two form a "blocking pair".
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
                    
    # Precompute bits
    bits = [0] * (N + 1)
    for i in range(1, N + 1):
        bits[i] = 1 << (i - 1)
        
    # Precompute conflict bits for fast checking
    # conflict_bits[x] is list of (bit_y, bit_z)
    conflict_bits = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y, z in conflict_pairs[x]:
            conflict_bits[x].append((bits[y], bits[z]))

    def can_add(x, mask):
        for by, bz in conflict_bits[x]:
            if (mask & by) and (mask & bz):
                return False
        return True

    def get_active_conflicts(x, mask):
        """Return list of (y, z) pairs where both y and z are in mask."""
        active = []
        for y, z, by, bz in [(y, z, bits[y], bits[z]) for y, z in conflict_pairs[x]]:
            if (mask & by) and (mask & bz):
                active.append((y, z))
        return active

    # Initial Solution Strategies
    # 1. Greedy 1..N
    # 2. Greedy N..1
    # 3. Random
    
    best_mask = 0
    best_size = 0
    
    def try_initialization(init_type):
        nonlocal best_mask, best_size
        mask = 0
        if init_type == 0:
            order = list(range(1, N + 1))
        elif init_type == 1:
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        for i in order:
            if can_add(i, mask):
                mask |= bits[i]
                
        # Local search to improve
        mask = local_search(mask)
        
        size = bin(mask).count("1")
        if size > best_size:
            best_mask = mask
            best_size = size
            return True
        return False

    def local_search(mask):
        """
        Perform local search starting from mask.
        Returns the best mask found during the search.
        """
        current_mask = mask
        current_size = bin(current_mask).count("1")
        stuck_count = 0
        max_stuck = 100
        # We use a simple random walk with restarts/kicks
        
        # Precompute list of all elements for random selection
        all_elements = list(range(1, N + 1))
        
        for _ in range(50000): # Max iterations per local search
            if time.time() > deadline:
                break
                
            # 1. Try to add a random missing element
            if random.random() < 0.5:
                # Pick random x not in mask
                # Optimization: pick from a small candidate set?
                # For now, just random
                x = random.choice(all_elements)
                if not (current_mask & bits[x]):
                    if can_add(x, current_mask):
                        current_mask |= bits[x]
                        current_size += 1
                        stuck_count = 0
                        continue
                    else:
                        # Try a targeted kick
                        active = get_active_conflicts(x, current_mask)
                        if active:
                            # Find a small hitting set for active conflicts
                            # Active conflicts are pairs (y, z)
                            # We need to remove a set R such that R intersects every (y, z)
                            # Since active conflicts are few, we can brute force small R
                            # Try removing 1 element from each conflict? No, one element can hit multiple.
                            # Let's try to find a minimum hitting set for the hypergraph of active conflicts.
                            # The hypergraph vertices are elements in the active conflicts.
                            # The hyperedges are the pairs.
                            
                            # Collect all vertices involved
                            involved = set()
                            for y, z in active:
                                involved.add(y)
                                involved.add(z)
                                
                            # If the number of involved elements is small, we can try all subsets?
                            # Or just try removing single elements, then pairs.
                            
                            # Try removing 1 element
                            found = False
                            for u in involved:
                                # Check if removing u allows adding x
                                # This is equivalent to checking if u hits ALL active conflicts?
                                # No, removing u only removes u. The other elements in conflicts remain.
                                # So x can be added if NO active conflict remains fully in the mask.
                                # An active conflict (y, z) remains if y != u and z != u.
                                # So u must be in EVERY active conflict.
                                # This is a very strong condition. Usually no single element hits all conflicts.
                                
                                # Let's just simulate:
                                new_mask = current_mask ^ bits[u]
                                if can_add(x, new_mask):
                                    current_mask = new_mask | bits[x]
                                    current_size += 1 # Removed u, added x
                                    stuck_count = 0
                                    found = True
                                    # Try to add back u or other elements
                                    # Greedy add back
                                    if can_add(u, current_mask):
                                        current_mask |= bits[u]
                                        current_size += 1
                                    break
                            
                            if found:
                                continue
                                
                            # Try removing 2 elements
                            # This is more expensive. Only do if stuck or with low prob
                            if stuck_count > 20 or random.random() < 0.1:
                                # Sample a few pairs of removals
                                involved_list = list(involved)
                                random.shuffle(involved_list)
                                limit = min(10, len(involved_list))
                                for i in range(limit):
                                    for j in range(i+1, limit):
                                        u = involved_list[i]
                                        v = involved_list[j]
                                        new_mask = current_mask ^ bits[u] ^ bits[v]
                                        if can_add(x, new_mask):
                                            current_mask = new_mask | bits[x]
                                            current_size += 1 # -2 + 1 = -1? No.
                                            # We removed u, v and added x.
                                            # Size change: -2 + 1 = -1.
                                            # This decreases size. We only do this if it leads to a larger size later?
                                            # Or if we can add back u, v?
                                            # Let's try to add back u, v and other missing elements.
                                            
                                            # Add x
                                            # Try to add back u
                                            if can_add(u, current_mask):
                                                current_mask |= bits[u]
                                                current_size += 1
                                            # Try to add back v
                                            if can_add(v, current_mask):
                                                current_mask |= bits[v]
                                                current_size += 1
                                            
                                            # Try to add other missing elements
                                            for _ in range(5):
                                                w = random.choice(all_elements)
                                                if not (current_mask & bits[w]):
                                                    if can_add(w, current_mask):
                                                        current_mask |= bits[w]
                                                        current_size += 1
                                                        
                                            stuck_count = 0
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    continue
                            
                            # If no small kick worked, do a random kick
                            if stuck_count >= max_stuck or random.random() < 0.05:
                                # Remove 3 random elements from mask
                                removed = []
                                for _ in range(3):
                                    u = random.choice(all_elements)
                                    if current_mask & bits[u]:
                                        removed.append(u)
                                        current_mask ^= bits[u]
                                        current_size -= 1
                                        
                                # Add x
                                if can_add(x, current_mask):
                                    current_mask |= bits[x]
                                    current_size += 1
                                    
                                # Greedily add back removed
                                random.shuffle(removed)
                                for u in removed:
                                    if can_add(u, current_mask):
                                        current_mask |= bits[u]
                                        current_size += 1
                                        
                                # Add other random
                                for _ in range(5):
                                    w = random.choice(all_elements)
                                    if not (current_mask & bits[w]):
                                        if can_add(w, current_mask):
                                            current_mask |= bits[w]
                                            current_size += 1
                                            
                                stuck_count = 0
                                continue
                            
                            stuck_count += 1
                            
            else:
                # Try a swap: remove u, add x
                # Pick x not in mask, u in mask
                x = random.choice(all_elements)
                if not (current_mask & bits[x]):
                    # Find u that conflicts with x?
                    # Or just random u?
                    # Random u is slow.
                    # Let's pick u from active conflicts of x
                    active = get_active_conflicts(x, current_mask)
                    if active:
                        # Pick a random active conflict
                        y, z = random.choice(active)
                        # Try removing y
                        if can_add(x, current_mask ^ bits[y]):
                            current_mask = (current_mask ^ bits[y]) | bits[x]
                            # Size unchanged
                            stuck_count = 0
                            continue
                        # Try removing z
                        if can_add(x, current_mask ^ bits[z]):
                            current_mask = (current_mask ^ bits[z]) | bits[x]
                            # Size unchanged
                            stuck_count = 0
                            continue
                    else:
                        # No active conflicts? Then can_add(x) should be true.
                        # If not, it's a bug or x is in mask (handled above).
                        pass
                stuck_count += 1

        return current_mask

    # Run initializations
    random.seed(42)
    try_initialization(0)
    if time.time() < deadline - 10:
        try_initialization(1)
    if time.time() < deadline - 5:
        try_initialization(2)
        
    # Final polish
    polish_mask = best_mask
    for i in range(1, N + 1):
        if not (polish_mask & bits[i]):
            if can_add(i, polish_mask):
                polish_mask |= bits[i]
                
    polish_size = bin(polish_mask).count("1")
    if polish_size > best_size:
        best_mask = polish_mask
        best_size = polish_size
        
    result = []
    for i in range(1, N + 1):
        if best_mask & bits[i]:
            result.append(i)
            
    return result
