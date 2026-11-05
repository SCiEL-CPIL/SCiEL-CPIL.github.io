import random
import time

def construct():
    N = 180
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    # Precompute all valid triples (a, b, c) with distinct elements
    # such that 1/a = 1/b + 1/c
    # a(b+c) = bc
    # c = ab / (a+b)
    # We need to store for each x, the pairs (u, v) such that (x, u, v) is a triple.
    # Note: The equation is symmetric in b and c. a is the "result".
    # So for a fixed x, it can play the role of a, b, or c.
    
    triples_with = [[] for _ in range(N + 1)]
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # (a, b, c) is a valid triple
                    # For a: pair (b, c)
                    triples_with[a].append((b, c))
                    # For b: pair (a, c)
                    triples_with[b].append((a, c))
                    # For c: pair (a, b)
                    triples_with[c].append((a, b))
                    
    def is_valid_add(x, in_set):
        """Check if adding x to in_set creates a conflict."""
        for u, v in triples_with[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    def find_conflict(in_set):
        """Return a conflicting triple (a, b, c) if exists, else None."""
        # Iterate through all possible x in set
        for x in range(1, N + 1):
            if not in_set[x]:
                continue
            for u, v in triples_with[x]:
                if in_set[u] and in_set[v]:
                    return (x, u, v)
        return None

    best_size = 0
    best_set = None

    # Multiple restarts
    while time.time() < deadline:
        # Initialize with a random subset
        # Heuristic: Start with larger numbers?
        # Let's try random subset of size 150
        current_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # Try to build a valid set by adding elements
        # Start with empty, add randomly?
        # Or start with all, remove conflicts?
        # Let's try starting with a random permutation and adding if valid.
        # This is a greedy construction.
        
        # Better: Start with a "dense" random set and repair.
        # Let's try starting with all numbers >= 60 (size 121) and adding smaller ones.
        # Or just random.
        
        # Let's use a random restart strategy with local search.
        # 1. Create a random set of size ~140
        # 2. Repair it
        # 3. Local search (add, swap)
        
        init_size = 140
        chosen = set(random.sample(range(1, N + 1), init_size))
        for x in chosen:
            current_set[x] = True
        
        # Repair phase
        # While there is a conflict, remove one element
        # To be faster, we can pick a conflict and remove the "most connected" element?
        # Or just random.
        max_repair_steps = 500
        steps = 0
        while steps < max_repair_steps:
            conflict = find_conflict(current_set)
            if conflict is None:
                break
            a, b, c = conflict
            # Remove one randomly
            to_remove = random.choice([a, b, c])
            current_set[to_remove] = False
            steps += 1
            
        # Now current_set is valid.
        # Try to add more elements
        added = True
        while added:
            added = False
            # Shuffle candidates to break symmetry
            candidates = [x for x in range(1, N + 1) if not current_set[x]]
            random.shuffle(candidates)
            for x in candidates:
                if is_valid_add(x, current_set):
                    current_set[x] = True
                    added = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        
        # Local Search: 1-swap
        # Try to remove y and add x if it allows adding more later?
        # Simple 1-swap: if removing y and adding x keeps it valid, do it?
        # This doesn't change size. We need to change size.
        # So 1-swap is only useful if it enables a subsequent add.
        # Or 2-swap: remove y1, y2, add x1, x2. Size +0.
        # To increase size, we need to find a set of swaps that results in a net gain.
        # This is complex.
        
        # Alternative: Just do "Add" phase repeatedly with different orderings?
        # The "Add" phase above adds ALL possible elements. So it reaches a maximal set.
        # Different random starts might lead to different maximal sets of different sizes.
        
        size = sum(current_set)
        if size > best_size:
            best_size = size
            best_set = current_set[:]
            
        # If we have time, try to improve best_set with swaps?
        # Let's stick to multiple restarts for now.
        
    # Final check
    if best_set is None:
        # Fallback to seed
        return list(range(91, 181))
        
    return [i for i in range(1, N + 1) if best_set[i]]

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
