import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute conflicts
    # adj[i] is a set of vertices that conflict with i
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    # Initial solution: all odd numbers
    S = set(range(1, N + 1, 2))
    
    # Helper to check if a vertex can be added
    def can_add(v, S):
        # Check if any neighbor of v is in S
        # Optimization: iterate over smaller of adj[v] or S? 
        # adj[v] is likely small.
        for u in adj[v]:
            if u in S:
                return False
        return True

    # Greedy addition phase
    # Try to add any vertex that is not in S and has no conflicts with S
    # We do this in a loop until no more can be added
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # To make it deterministic or random? Random might help escape local minima later, 
        # but for initial fill, let's try adding all possible.
        # Actually, adding one by one might block others? 
        # If v1 and v2 are both free, adding v1 doesn't block v2 if they don't conflict.
        # If they do conflict, we can only add one.
        # So we should pick carefully.
        # Let's find all free vertices.
        free_verts = []
        for v in vertices:
            if v not in S and can_add(v, S):
                free_verts.append(v)
                
        if free_verts:
            # Add one. Which one?
            # To maximize future options, maybe add the one with fewest conflicts with other free vertices?
            # Or just add the smallest.
            # Let's add the one with the fewest neighbors in the set of free_verts?
            # Or just random.
            v_to_add = random.choice(free_verts)
            S.add(v_to_add)
            changed = True
            
    # Local Search / Swap Phase
    # Try to improve by swapping one element out and adding one or more in.
    # We will run this for a fixed number of iterations or until time limit.
    
    best_S = set(S)
    best_size = len(S)
    
    # We can use a simple random restart / hill climbing
    # Since N is small, we can try all single swaps in a loop.
    
    # Function to try all single swaps
    def try_swaps(S):
        improved = False
        # Iterate over a copy of S to allow modification
        # To be efficient, we check each u in S
        for u in list(S):
            # Tentatively remove u
            S.remove(u)
            
            # Find all vertices v not in S (and not u) that can now be added
            addable = []
            for v in vertices:
                if v in S or v == u:
                    continue
                if can_add(v, S):
                    addable.append(v)
                    
            if addable:
                # If we can add at least 1, we gain size.
                # How many to add? All of them?
                # If we add all, we might block others, but we gain immediately.
                # Let's add all.
                S.update(addable)
                improved = True
            else:
                # Revert
                S.add(u)
                
        return improved

    # Run swap loop
    # We need to be careful with time.
    # Each pass over all u in S takes O(|S| * N * avg_degree).
    # |S| ~ 130, N=256. 130 * 256 * ~10 = 300k ops. Very fast.
    # We can do many passes.
    
    while time.time() < deadline:
        if not try_swaps(S):
            break
            
        # If we improved, we might want to continue swapping until stable
        # The loop above breaks if no improvement in a full pass.
        
    # After local search, if time remains, try random restarts?
    # Or just return best found.
    
    # Ensure we return the best one found during the process if we track it
    # In the above, S is updated in place.
    # We should track the best size.
    
    # Let's re-structure to keep best_S
    # Actually, the swap logic only improves size. It never decreases.
    # So S at the end is the best.
    
    return list(S)
