import random
from itertools import combinations
from math import lcm, gcd

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute conflict map
    # conflict_map[a][b] is a set of c such that {a,b,c} is a forbidden triple
    # Note: The condition is symmetric.
    # We only need to store for a < b.
    conflict_map = {}
    for a in range(1, N + 1):
        conflict_map[a] = {}
        for b in range(a + 1, N + 1):
            conflict_map[a][b] = set()
            
    # Fill conflict map
    # For each triple, if forbidden, add c to map[a][b], b to map[a][c], a to map[b][c]
    # Optimization: Iterate c and find pairs? Or just iterate all triples?
    # N=128, C(128,3) ~ 340k. This is fast.
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            # Add to map
            # Ensure keys are sorted for consistent access if needed, 
            # but we access via specific a,b.
            # We will access conflict_map[min(x,y)][max(x,y)]
            # So let's store in a way that handles order.
            # Actually, let's just use a dict of sets where key is frozenset({a,b})?
            # Or just a 2D list/dict.
            # Let's use a dict of dicts: cm[a][b] and cm[b][a]
            pass
            
    # Re-implementation of conflict map for efficiency
    # cm[a][b] = set of c
    cm = {}
    for a in range(1, N + 1):
        cm[a] = {}
        for b in range(1, N + 1):
            cm[a][b] = set()
            
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            cm[a][b].add(c)
            cm[b][a].add(c)
            cm[a][c].add(b)
            cm[c][a].add(b)
            cm[b][c].add(a)
            cm[c][b].add(a)

    # State: 
    # in_set: set of elements currently in S
    # block_count[x]: number of pairs {a,b} in S such that x is in cm[a][b]
    # If block_count[x] == 0, x is addable.
    
    def add_element(x, in_set, block_count):
        in_set.add(x)
        for y in in_set:
            if y == x: continue
            for z in cm[x][y]:
                block_count[z] += 1
                
    def remove_element(x, in_set, block_count):
        in_set.remove(x)
        for y in in_set:
            if y == x: continue
            for z in cm[x][y]:
                block_count[z] -= 1

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    for _ in range(100):
        in_set = set()
        block_count = {i: 0 for i in range(1, N + 1)}
        
        # Greedy Phase: Add elements with minimal impact
        # Impact of adding x: number of currently addable elements that become blocked
        # We approximate by just adding elements that are addable, 
        # prioritizing those that block fewer others.
        
        # Simple Greedy: Sort by number of conflicts in the full universe?
        # Or just random order with "add if possible"
        # Let's use a smarter greedy: 
        # At each step, among addable elements, pick the one that minimizes the number of 
        # newly blocked elements.
        
        # To make it fast, we might just pick a random addable element.
        # But "minimal impact" is better.
        
        # Let's try: 
        # 1. Get all addable elements.
        # 2. If empty, break greedy phase.
        # 3. Pick one randomly.
        
        # Actually, let's do a "Kicks" local search after a simple greedy init.
        
        # Simple Greedy Init
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if block_count[x] == 0:
                add_element(x, in_set, block_count)
                
        # Local Search Phase
        # Goal: Increase size by finding swaps that unlock new additions.
        # Move: Remove x, if this makes some y addable, add y.
        # This keeps size same. 
        # To increase size, we need to remove x and add y AND z? 
        # Or simply: The greedy phase might get stuck.
        # We can try to "shake" the set:
        # Randomly remove 1 element, then greedily add as many as possible.
        
        for step in range(50): # 50 restarts of local search within this main restart
            # Try to improve
            # Pick a random element to remove
            if not in_set: break
            x = random.choice(list(in_set))
            remove_element(x, in_set, block_count)
            
            # Now try to add any addable element
            added_any = False
            # Get addable
            addable = [y for y in range(1, N + 1) if y not in in_set and block_count[y] == 0]
            if addable:
                # Add one randomly
                y = random.choice(addable)
                add_element(y, in_set, block_count)
                added_any = True
                # Try to add more?
                # Let's just do one swap per kick to keep it simple and fast.
                # Or, let's run a greedy fill after the kick.
                # Greedy fill:
                while True:
                    addable = [y for y in range(1, N + 1) if y not in in_set and block_count[y] == 0]
                    if not addable: break
                    y = random.choice(addable)
                    add_element(y, in_set, block_count)
                    
            if len(in_set) > best_size:
                best_size = len(in_set)
                best_set = set(in_set)
                
    if not best_set:
        return [1]
        
    return list(best_set)
