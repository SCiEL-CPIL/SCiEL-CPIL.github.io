import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
                    cycles.append(e1)
                    e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
                    cycles.append(e2)
                    e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
                    cycles.append(e3)
    return cycles

def construct():
    deadline = time.time() + 55
    
    cycles = build_four_cycles()
    # For each edge, store the list of cycle indices it belongs to
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for i, cycle in enumerate(cycles):
        for e in cycle:
            cycles_by_edge[e].append(i)
    
    # We want to find a subset S of edges such that for all cycles C, |S intersect C| <= 2.
    # We maximize |S|.
    # Then we assign distinct colors to edges in S, and 0 to edges not in S.
    # This yields |S| + 1 colors.
    
    # Binary state: x[e] = 1 if e in S, else 0.
    x = [0] * NUM_EDGES
    
    # Helper to check if adding e to S is valid (assuming x is current state)
    # Returns True if for all cycles containing e, count of 1s in cycle (excluding e) < 2.
    # i.e., count of 1s in cycle (including e) <= 2.
    def can_add(e):
        for ci in cycles_by_edge[e]:
            c = cycles[ci]
            count = 0
            for oe in c:
                if oe == e:
                    continue
                if x[oe]:
                    count += 1
            if count >= 2:
                return False
        return True

    # Helper to check if removing e from S is valid (always valid)
    # Helper to check if swapping e_out (1->0) and e_in (0->1) is valid
    # We need to check cycles containing e_in or e_out.
    # Actually, just check all cycles affected.
    def can_swap(e_out, e_in):
        # Cycles affected are those containing e_out or e_in
        affected = set(cycles_by_edge[e_out])
        affected.update(cycles_by_edge[e_in])
        for ci in affected:
            c = cycles[ci]
            count = 0
            for oe in c:
                if oe == e_in:
                    # e_in is being added, so treat as 1
                    count += 1
                elif oe == e_out:
                    # e_out is being removed, so treat as 0
                    continue
                else:
                    if x[oe]:
                        count += 1
            if count > 2:
                return False
        return True

    # Greedy Initialization
    # Sort edges by degree (number of cycles) descending? Or random?
    # Let's try random order, add if possible.
    rng = random.Random(42)
    edge_order = list(range(NUM_EDGES))
    rng.shuffle(edge_order)
    
    for e in edge_order:
        if can_add(e):
            x[e] = 1
    
    best_x = x[:]
    best_count = sum(x)
    
    # Local Search
    # 1. Try to add any 0 that can be added.
    # 2. Try to swap 1->0 and 0->1 if it allows adding more later? 
    #    Simple hill climbing: just add if possible.
    #    If stuck, try random swaps that maintain validity but change structure?
    #    Or try to remove an edge that is "blocking" many others, then re-add.
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Add any possible edge
        for e in edge_order:
            if time.time() > deadline:
                break
            if x[e] == 0 and can_add(e):
                x[e] = 1
                improved = True
        
        # Pass 2: If no improvement, try swaps
        if not improved:
            # Try to find a swap that might help?
            # Actually, if we can't add any edge, we are at a local maximum for this heuristic.
            # We can try to remove an edge and see if we can add 2 others?
            # Or just perturb.
            
            # Perturbation: Randomly flip some bits, then re-optimize?
            # Or try a random swap that is valid (count doesn't increase, but structure changes)
            # Let's try to find a valid swap (1->0, 0->1) that doesn't change count but might unblock.
            # This is slow. Let's just do a random restart if stuck for a while.
            
            # Simple perturbation: Pick a random 1, set to 0. Pick a random 0, try to set to 1.
            ones = [i for i in range(NUM_EDGES) if x[i]==1]
            zeros = [i for i in range(NUM_EDGES) if x[i]==0]
            
            if ones and zeros:
                e_out = rng.choice(ones)
                e_in = rng.choice(zeros)
                
                if can_swap(e_out, e_in):
                    x[e_out] = 0
                    x[e_in] = 1
                    improved = True
        
        cur_count = sum(x)
        if cur_count > best_count:
            best_count = cur_count
            best_x = x[:]
        
        if not improved:
            # Stuck. Do a larger perturbation.
            # Remove 3 random 1s, then re-run greedy add.
            ones = [i for i in range(NUM_EDGES) if x[i]==1]
            if len(ones) > 3:
                to_remove = rng.sample(ones, 3)
                for e in to_remove:
                    x[e] = 0
                # Re-add greedily
                for e in edge_order:
                    if x[e] == 0 and can_add(e):
                        x[e] = 1
                
                cur_count = sum(x)
                if cur_count > best_count:
                    best_count = cur_count
                    best_x = x[:]
    
    # Construct final colors
    colors = [0] * NUM_EDGES
    next_color = 1
    for e in range(NUM_EDGES):
        if best_x[e] == 1:
            colors[e] = next_color
            next_color += 1
    
    return colors
