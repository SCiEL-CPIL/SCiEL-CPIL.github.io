import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s
    
    N = 180
    # 1. Precompute conflicts
    # conflicts_with[x] is a list of tuples (y, z) such that {x, y, z} is a forbidden triple.
    # We store pairs (y, z) with y < z to ensure uniqueness in the pair representation if needed, 
    # but since we check 'y in S and z in S', order doesn't matter for the check.
    conflicts_with = {i: [] for i in range(1, N + 1)}
    
    # Generate all forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c => a = bc / (b+c)
    # Iterate a from 1 to N
    for a in range(1, N + 1):
        a2 = a * a
        # b = a + d, c = a + a2/d
        # d must divide a2
        # b > a => d > 0
        # c > b => a + a2/d > a + d => a2/d > d => d^2 < a2 => d < a
        # Also b, c <= N
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                # We need distinct elements a, b, c
                # Since d >= 1, b > a.
                # If d == e, then b == c, which is not allowed (distinct).
                if b == c:
                    continue
                if b <= N and c <= N:
                    # We have a triple {a, b, c}
                    # Add to conflicts
                    # For a: conflicts with (b, c)
                    conflicts_with[a].append((b, c))
                    # For b: conflicts with (a, c)
                    conflicts_with[b].append((a, c))
                    # For c: conflicts with (a, b)
                    conflicts_with[c].append((a, b))
                    
    # Helper: Check if x can be added to set S
    def can_add(x, S):
        for y, z in conflicts_with[x]:
            if y in S and z in S:
                return False
        return True

    # Helper: Calculate potential (number of addable elements)
    def get_potential(S):
        pot = 0
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    pot += 1
        return pot

    # Greedy Initialization
    def greedy_init():
        S = set()
        # Order by degree (number of conflicts) ascending? 
        # Elements with fewer conflicts are "safer" to add early.
        # Or descending? If we add high conflict elements first, we block many others.
        # Usually, adding "easy" elements first allows more "hard" elements to fit later?
        # Let's try sorting by len(conflicts_with[x]) ascending.
        order = sorted(range(1, N + 1), key=lambda x: len(conflicts_with[x]))
        
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Local Search
    def local_search(S, max_time=10):
        S = set(S)
        best_local = set(S)
        best_local_size = len(S)
        
        last_improvement_time = time.time()
        
        while time.time() - last_improvement_time < max_time and time.time() < DEADLINE:
            improved = False
            
            # 1. Try to add
            added = False
            # Shuffle to avoid bias
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, S):
                    S.add(x)
                    added = True
                    improved = True
                    break
            
            if added:
                last_improvement_time = time.time()
                if len(S) > best_local_size:
                    best_local_size = len(S)
                    best_local = set(S)
                continue
            
            # 2. Try to swap
            # If no add possible, try to swap u in S with v not in S
            # Goal: Maintain size, but increase potential (space for future adds)
            current_pot = get_potential(S)
            
            S_list = list(S)
            random.shuffle(S_list)
            
            # Try a limited number of swaps to keep it fast
            swaps_tried = 0
            max_swaps_to_try = 50
            best_swap = None
            best_swap_pot = -1
            
            for u in S_list:
                if time.time() > DEADLINE:
                    break
                # Candidates for v
                # v must not be in S
                # We can just iterate all v not in S
                # To be efficient, we can just check a few random v's
                for _ in range(5):
                    v = random.randint(1, N)
                    if v in S:
                        continue
                    if v == u:
                        continue
                    
                    # Check if S \ {u} \cup {v} is valid
                    # This is valid if:
                    # 1. v can be added to S \ {u}
                    # 2. No conflict is created involving v and other elements in S \ {u}
                    # Note: u is removed, so conflicts involving u are gone.
                    
                    # Efficient check:
                    # We need to ensure v doesn't conflict with any pair in S \ {u}.
                    # This is equivalent to: for all (y,z) in conflicts_with[v], 
                    # NOT (y in S \ {u} and z in S \ {u})
                    
                    valid_swap = True
                    for y, z in conflicts_with[v]:
                        # If y or z is u, they are removed, so no conflict from that side
                        # But we need to check if the OTHER one is in S
                        # Actually, the condition is: y in S' and z in S'
                        # S' = S - {u} + {v}
                        # If y == u or z == u, then y not in S' or z not in S' (unless the other is v? No, v is new)
                        # Wait, if y == u, then y is not in S'. So the pair (y,z) is not fully in S'.
                        # So we only need to check if both y and z are in S \ {u}.
                        
                        if y == u or z == u:
                            continue
                        
                        if y in S and z in S:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        # Calculate potential of S'
                        # S' = S - {u} + {v}
                        # We can estimate or calculate exactly.
                        # Exact calculation is safer.
                        # To avoid modifying S repeatedly, we can simulate or just accept if it looks promising?
                        # Let's just calculate potential for this specific candidate if valid.
                        # This is expensive. 
                        # Alternative: Accept swap if valid and random?
                        # Better: Accept swap if valid. Then in the next iteration, we try to add.
                        # If the swap opens up space, we will find it.
                        # To guide the swap, we can use a lighter heuristic:
                        # Count how many elements become "unblocked"?
                        
                        # Let's just take the first valid swap we find.
                        # This is a "random walk" in the space of valid sets of same size.
                        S.remove(u)
                        S.add(v)
                        improved = True
                        swaps_tried += 1
                        break
                if improved:
                    break
            
            if improved:
                # After a swap, size is same. 
                # We don't update best_local unless size increases (which it doesn't in swap).
                # But we might have escaped a local optimum.
                # We should still track best_local by size.
                pass
            else:
                # No add, no valid swap found in this pass
                # Perform a Kick: Remove a random element to break structure
                if S:
                    u = random.choice(list(S))
                    S.remove(u)
                    # This decreases size. 
                    # We hope that re-adding allows a larger set eventually.
                    # To prevent shrinking indefinitely, we can be conservative.
                    # Only kick if we've been stuck for a while.
                    # For now, simple kick.
        
        return best_local

    # Main Loop
    random.seed(42)
    
    # 1. Greedy Start
    S = greedy_init()
    S = local_search(S, max_time=5)
    
    best_set = set(S)
    best_size = len(S)
    
    # 2. Multiple Restarts with different seeds/orders if time permits
    # We can try greedy with different orderings
    while time.time() < DEADLINE - 5:
        # Randomize order for greedy
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        
        S = local_search(S, max_time=2)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    return list(best_set)
