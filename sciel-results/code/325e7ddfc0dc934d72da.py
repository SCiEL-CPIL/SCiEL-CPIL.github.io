import time
import math
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_triples(N=180):
    """
    Precompute all (a, b, c) with distinct a,b,c in [1,N] such that 1/a = 1/b + 1/c.
    Equation: 1/a = 1/b + 1/c => c = ab / (b-a).
    We require b > a and c to be an integer in [1, N] distinct from a and b.
    Note: Since 1/a = 1/b + 1/c, a must be smaller than both b and c.
    """
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 180
    
    triples = _precompute_triples(N)
    num_triples = len(triples)
    
    # Map each element to the list of triple indices it participates in
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
        
    # We want to find a Minimum Hitting Set R such that every triple has at least one element in R.
    # The solution S is [1..N] \ R.
    # Maximize |S| == Minimize |R|.
    
    best_removed_size = N
    best_removed_set = set(range(1, N + 1))
    
    # We will run multiple restarts with different random initializations or greedy variations.
    # Strategy 1: Greedy Hitting Set
    # Strategy 2: Local Search (Delete/Swap) starting from Greedy
    
    for seed in range(100):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        
        # 1. Greedy Hitting Set Initialization
        # We maintain coverage count.
        # We want to add elements to R until all triples are covered.
        # Greedy: Pick element that covers the most *currently uncovered* triples.
        
        R = set()
        active_triples = set(range(num_triples))
        
        # Precompute element degrees in the full hypergraph
        deg = {i: len(elem_triples[i]) for i in range(1, N + 1)}
        
        while active_triples:
            # Find x with max deg[x] among candidates not in R
            # To avoid O(N) scan for max every time, we can just scan. N=180 is small.
            max_deg = -1
            best_candidate = None
            candidates = []
            for x in range(1, N + 1):
                if x not in R:
                    if deg[x] > max_deg:
                        max_deg = deg[x]
                        best_candidate = x
                        candidates = [x]
                    elif deg[x] == max_deg:
                        candidates.append(x)
                        
            if best_candidate is None:
                break
                
            x = random.choice(candidates)
            R.add(x)
            
            # Update active_triples and deg
            for t_idx in elem_triples[x]:
                if t_idx in active_triples:
                    active_triples.remove(t_idx)
                    a, b, c = triples[t_idx]
                    for y in (a, b, c):
                        deg[y] -= 1
                        
        current_removed_size = len(R)
        
        # 2. Local Search Phase
        # Goal: Minimize |R|.
        # Moves:
        # A. Delete: Remove x from R if it doesn't uncover any triple.
        # B. Swap: Remove x, Add y.
        
        # Rebuild cov_count from R
        cov_count = [0] * num_triples
        for x in R:
            for t_idx in elem_triples[x]:
                cov_count[t_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try Delete Moves
            R_list = list(R)
            random.shuffle(R_list)
            
            for x in R_list:
                if time.time() > deadline - 2.0:
                    break
                if x not in R:
                    continue
                    
                # Check if x can be removed
                can_remove = True
                for t_idx in elem_triples[x]:
                    if cov_count[t_idx] == 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove x
                    R.remove(x)
                    for t_idx in elem_triples[x]:
                        cov_count[t_idx] -= 1
                    current_removed_size -= 1
                    improved = True
                    break # Restart the while loop to re-evaluate all
            
            if not improved:
                # Try Swap Moves
                swap_found = False
                R_list = list(R)
                random.shuffle(R_list)
                
                for x in R_list:
                    if time.time() > deadline - 2.0:
                        break
                    if x not in R:
                        continue
                        
                    # Find uncovered triples if x is removed
                    uncovered = []
                    for t_idx in elem_triples[x]:
                        if cov_count[t_idx] == 1:
                            uncovered.append(t_idx)
                            
                    if not uncovered:
                        continue # Should have been deleted
                        
                    # Find intersection of elements in all triples in 'uncovered'
                    common = set(triples[uncovered[0]])
                    for t_idx in uncovered[1:]:
                        common.intersection_update(set(triples[t_idx]))
                        
                    # Filter common to exclude elements already in R
                    candidates = [y for y in common if y not in R]
                    
                    if candidates:
                        # We can swap x with any of these candidates.
                        y = random.choice(candidates)
                        
                        # Perform Swap
                        R.remove(x)
                        for t_idx in elem_triples[x]:
                            cov_count[t_idx] -= 1
                            
                        R.add(y)
                        for t_idx in elem_triples[y]:
                            cov_count[t_idx] += 1
                            
                        swap_found = True
                        improved = True
                        break # Restart to try deletes again
                        
                if not swap_found:
                    # No moves possible. Stuck in local optimum.
                    pass
        
        if current_removed_size < best_removed_size:
            best_removed_size = current_removed_size
            best_removed_set = set(R)
            
    # Construct final set
    final_set = set(range(1, N + 1)) - best_removed_set
    
    return list(final_set)

# EVOLVE-BLOCK-END
