import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses randomized local search with conflict counting.
    """
    m = 205
    k = 2
    deadline = time.time() + 580.0  # 580s safety margin before 630s kill

    # 1. Precompute forbidden differences and adjacency
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency list
    # neighbors[v] is a list of u such that (u-v) % m is in forbidden
    neighbors = [[] for _ in range(m)]
    for v in range(m):
        for f in forbidden:
            u = (v + f) % m
            neighbors[v].append(u)
            # Note: The graph is undirected. 
            # If u is neighbor of v, v is neighbor of u.
            # We build the full adjacency list for both directions to simplify conflict updates.
            # Actually, since (v+f) and (v-f) are both in forbidden (if f is square, -f is square),
            # the graph is symmetric.
            # Let's just build neighbors[v] to contain all u != v such that (u-v)%m in forbidden.
            
    # Re-verify adjacency construction for correctness and efficiency
    # For each v, find all u such that (u - v) % m is in forbidden
    neighbors = [[] for _ in range(m)]
    for v in range(m):
        for f in forbidden:
            u1 = (v + f) % m
            u2 = (v - f) % m
            if u1 != v:
                neighbors[v].append(u1)
            if u2 != v and u2 != u1:
                neighbors[v].append(u2)
        # Remove duplicates just in case, though forbidden set handles unique diffs
        # But u1 and u2 could be same if 2f=0, which is not the case for odd m.
        # However, different f values could lead to same u? No, f is unique diff.
        # So neighbors[v] might have duplicates if we iterate f and -f separately?
        # No, we iterate f in forbidden. (v+f) and (v-f) are distinct for m=205 (odd).
        # But do we add both?
        # If f is in forbidden, is -f in forbidden?
        # Yes, if x^2 = f, then (-x)^2 = f. Wait.
        # Forbidden set is {x^2 mod 205}.
        # If d is a square, is -d a square?
        # 205 = 5 * 41.
        # Legendre symbol (-1/p).
        # p=5: (-1/5) = -1. So -1 is not a square mod 5.
        # p=41: (-1/41) = 1 (since 41 = 1 mod 4). So -1 is a square mod 41.
        # By CRT, -1 is a square mod 205 iff it is a square mod 5 AND mod 41.
        # Since it's not a square mod 5, -1 is NOT a square mod 205.
        # Therefore, if d is a square, -d is NOT a square.
        # So the set of forbidden differences F is NOT symmetric under negation.
        # However, the condition is: neither (a-b) nor (b-a) is in F.
        # This means the graph is directed?
        # "For every pair ... neither (a-b) mod 205 nor (b-a) mod 205 may be in F."
        # This is equivalent to: (a-b) not in F AND (b-a) not in F.
        # Since (b-a) = -(a-b), this means: d not in F AND -d not in F.
        # So we can define an undirected graph where edge (u,v) exists if (u-v) mod m in F OR (v-u) mod m in F.
        # Let's construct this undirected adjacency list.
        
    neighbors = [[] for _ in range(m)]
    for v in range(m):
        for f in forbidden:
            u1 = (v + f) % m
            u2 = (v - f) % m
            if u1 != v and u1 not in neighbors[v]:
                neighbors[v].append(u1)
            if u2 != v and u2 not in neighbors[v]:
                neighbors[v].append(u2)
        # Optimization: convert to set for faster "in" check? No, list is fine for small degree.
        # But we need to update conflict counts.
        
    # To speed up conflict updates, we can store neighbors as lists.
    # Degree is roughly |F|. |F| is number of non-zero squares.
    # Number of squares mod 205:
    # Squares mod 5: {0, 1, 4} (3 values). Non-zero: {1, 4} (2 values).
    # Squares mod 41: (41+1)/2 = 21 values. Non-zero: 20 values.
    # By CRT, number of squares mod 205 is 2 * 20 = 40?
    # Actually, map x -> x^2.
    # Total squares including 0: 1 (from 0) + 2*20 = 41?
    # Let's just compute the size of `forbidden`.
    
    # 2. State Variables
    in_set = np.zeros(m, dtype=bool)
    conflict_count = np.zeros(m, dtype=int)
    current_size = 0
    best_size = 0
    best_state = np.zeros(m, dtype=bool)
    
    def add_vertex(v):
        nonlocal current_size
        in_set[v] = True
        current_size += 1
        for u in neighbors[v]:
            conflict_count[u] += 1
            
    def remove_vertex(v):
        nonlocal current_size
        in_set[v] = False
        current_size -= 1
        for u in neighbors[v]:
            conflict_count[u] -= 1
            
    def try_greedy_fill():
        # Find all v with conflict_count[v] == 0 and not in_set[v]
        # To make it random, we can collect candidates and pick one.
        # Or just iterate and pick the first one found? Randomizing helps escape symmetry.
        candidates = []
        for v in range(m):
            if not in_set[v] and conflict_count[v] == 0:
                candidates.append(v)
        
        added_any = False
        while candidates:
            # Pick a random candidate
            idx = random.randint(0, len(candidates) - 1)
            v = candidates.pop(idx)
            
            # It's possible that adding previous vertices changed conflict_count[v]?
            # No, we only add v if conflict_count[v] == 0.
            # When we add v, it increases conflict_count of its neighbors.
            # So other candidates in the list might now have conflict_count > 0.
            # We should check again before adding.
            if in_set[v] or conflict_count[v] != 0:
                continue
                
            add_vertex(v)
            added_any = True
            
            # Update candidates list: remove neighbors of v from candidates if they are there?
            # Or just let the "check again" handle it.
            # Since m is small (205), re-scanning or maintaining list is fine.
            # Let's just re-scan to be safe and simple, or filter the list.
            # Filtering the list is O(N).
            new_candidates = []
            for c in candidates:
                if not in_set[c] and conflict_count[c] == 0:
                    new_candidates.append(c)
            candidates = new_candidates
            
        return added_any

    # 3. Main Loop
    random.seed(42)
    
    # Initial Greedy Phase
    try_greedy_fill()
    if current_size > best_size:
        best_size = current_size
        best_state = in_set.copy()
        
    while time.time() < deadline:
        # Check if we can add more
        if not try_greedy_fill():
            # Stuck. Need to kick.
            if current_size > best_size:
                best_size = current_size
                best_state = in_set.copy()
                
            # If we are at best size or better, and stuck, we must kick to explore.
            # If current_size < best_size, we might still kick to try to recover/improve.
            
            # Kick: Remove 1 to 3 random vertices from the set
            if current_size == 0:
                # Empty set, restart
                break
                
            kick_size = random.choice([1, 1, 1, 2, 2, 3])
            if kick_size > current_size:
                kick_size = current_size
                
            # Select vertices to remove
            indices = np.where(in_set)[0]
            if len(indices) < kick_size:
                kick_size = len(indices)
            if kick_size == 0:
                continue
                
            # Randomly select kick_size indices
            removed_indices = random.sample(list(indices), kick_size)
            
            for v in removed_indices:
                remove_vertex(v)
                
            # After kicking, try to fill again
            # The loop will continue and call try_greedy_fill()
            
    # Return the best state found
    result_indices = np.where(best_state)[0]
    return result_indices.astype(np.int64)
