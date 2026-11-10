import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    N = 256
    # 1. Build the conflict graph
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Fast greedy initialization with randomized ordering
    def greedy_init(seed):
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        selected = set()
        for v in order:
            # Check if v conflicts with any selected vertex
            conflict = False
            for s in selected:
                if adj[v] & (1 << s):
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
        return selected
    
    # 3. Local search: try to swap out a selected vertex for an unselected one
    def local_search(selected, time_limit):
        selected = set(selected)
        unselected = set(range(1, N + 1)) - selected
        improved = True
        last_improve = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            # Try removing each selected vertex and see if we can add more
            for v in list(selected):
                # Temporarily remove v
                selected.remove(v)
                unselected.add(v)
                
                # Try to add as many unselected vertices as possible
                added = []
                for u in list(unselected):
                    conflict = False
                    for s in selected:
                        if adj[u] & (1 << s):
                            conflict = True
                            break
                    if not conflict:
                        selected.add(u)
                        unselected.remove(u)
                        added.append(u)
                
                # If we added more than we removed (1), keep the change
                if len(added) > 1:
                    improved = True
                    last_improve = time.time()
                else:
                    # Revert
                    selected.add(v)
                    unselected.remove(v)
                    for u in added:
                        selected.remove(u)
                        unselected.add(u)
        
        return selected
    
    # 4. Run multiple greedy + local search iterations
    best_set = set()
    best_size = 0
    num_seeds = 20
    ls_time_budget = 2.0  # seconds per local search
    
    for seed in range(num_seeds):
        if time.time() > deadline - 30:
            break
        gs = greedy_init(seed)
        ls_result = local_search(gs, min(time.time() + ls_time_budget, deadline - 30))
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_set = ls_result
    
    # 5. Branch and Bound with strong incumbent
    best_bb_size = best_size
    best_bb_set = 0
    for i in best_set:
        best_bb_set |= (1 << i)
    
    def popcount(x):
        return x.bit_count()
    
    # Convert best_set to bitset for comparison
    def set_to_bitset(s):
        b = 0
        for i in s:
            b |= (1 << i)
        return b
    
    initial_best_bitset = set_to_bitset(best_set)
    
    def solve(candidates, current_size, current_bitset):
        nonlocal best_bb_size, best_bb_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > best_bb_size:
                best_bb_size = current_size
                best_bb_set = current_bitset
            return
            
        # Upper bound
        if current_size + popcount(candidates) <= best_bb_size:
            return
            
        # Select vertex with max degree in candidates
        max_deg = -1
        v = -1
        c = candidates
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            deg = popcount(adj[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = idx
            c ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        new_candidates = candidates & ~adj[v] & ~(1 << v)
        solve(new_candidates, current_size + 1, current_bitset | (1 << v))
        
        # Branch 2: Exclude v
        new_candidates = candidates & ~(1 << v)
        solve(new_candidates, current_size, current_bitset)
    
    # Only run B&B if we have time
    if time.time() < deadline - 5:
        initial_candidates = 0
        for i in range(1, N + 1):
            initial_candidates |= (1 << i)
        
        solve(initial_candidates, 0, 0)
    
    # Use the best result
    final_size = max(best_size, best_bb_size)
    if best_bb_size > best_size:
        final_set = set()
        for i in range(1, N + 1):
            if best_bb_set & (1 << i):
                final_set.add(i)
    else:
        final_set = best_set
    
    if not final_set:
        return list(range(1, 257, 2))
    
    return list(final_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                print(f"Conflict found: {a}, {b}")
                sys.exit(1)
    print("Valid solution")
