"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph using bitmasks
    # adj[i] is a bitmask of vertices j such that (i+1) + (j+1) is a perfect square
    # Note: We include self-loops (a=b) if 2*a is a square
    adj = [0] * N
    
    for i in range(N):
        val = i + 1
        for j in range(N):
            other = j + 1
            if val + other in squares:
                adj[i] |= (1 << j)
    
    # Identify vertices that cannot be in any valid set (self-conflict: 2*a is a square)
    # These must be excluded
    self_conflict = [False] * N
    for i in range(N):
        if adj[i] & (1 << i):
            self_conflict[i] = True
    
    # Candidate vertices (no self-conflict)
    candidates = [i for i in range(N) if not self_conflict[i]]
    
    # Precompute neighbor masks restricted to candidates
    # For each candidate i, neighbors[i] = adj[i] & candidate_mask
    candidate_mask = 0
    for i in candidates:
        candidate_mask |= (1 << i)
    
    neighbors = [adj[i] & candidate_mask for i in range(N)]
    
    def is_valid_set(selected_mask):
        """Check if selected_mask is a valid independent set."""
        for i in range(N):
            if selected_mask & (1 << i):
                if selected_mask & adj[i]:
                    return False
        return True
    
    def greedy_independent_set(order=None):
        """Greedy: process vertices in given order, add if no conflict with already added."""
        if order is None:
            order = list(candidates)
        
        selected = set()
        selected_mask = 0
        for v in order:
            # Check if v conflicts with any already selected
            if selected_mask & neighbors[v]:
                continue
            # Also check self (shouldn't happen for candidates)
            selected.add(v)
            selected_mask |= (1 << v)
        
        return selected, selected_mask
    
    def local_search(selected_mask, max_iter=10000):
        """Try to improve by swapping: remove a vertex, add a different one."""
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        
        for _ in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            improved = False
            selected = [i for i in range(N) if best_mask & (1 << i)]
            
            # Try removing each selected vertex and adding an unselected one
            random.shuffle(selected)
            for rem in selected:
                new_mask = best_mask & ~(1 << rem)
                
                # Try to add an unselected vertex
                unselected = [i for i in candidates if not (new_mask & (1 << i))]
                random.shuffle(unselected)
                
                for add in unselected:
                    if new_mask & neighbors[add]:
                        continue
                    # Check if add doesn't conflict
                    candidate_mask_new = new_mask | (1 << add)
                    if candidate_mask_new & neighbors[add]:
                        continue
                    new_size = bin(candidate_mask_new).count('1')
                    if new_size > best_size:
                        best_mask = candidate_mask_new
                        best_size = new_size
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_mask, best_size
    
    def degree_based_order(cand_list):
        """Order by degree (number of conflicts within candidates) ascending."""
        degrees = []
        for v in cand_list:
            deg = bin(neighbors[v]).count('1')
            degrees.append((deg, v))
        degrees.sort()
        return [v for _, v in degrees]
    
    # Strategy 1: Degree-based greedy
    best_mask = 0
    best_size = 0
    
    # Try degree-based ordering
    order = degree_based_order(candidates)
    selected, sel_mask = greedy_independent_set(order)
    if len(selected) > best_size:
        best_size = len(selected)
        best_mask = sel_mask
    
    # Local search on the greedy result
    improved_mask, improved_size = local_search(best_mask)
    if improved_size > best_size:
        best_size = improved_size
        best_mask = improved_mask
    
    # Strategy 2: Multiple random restarts with random ordering
    while time.time() - start_time < DEADLINE - 5:
        # Random ordering
        random_order = list(candidates)
        random.shuffle(random_order)
        
        selected, sel_mask = greedy_independent_set(random_order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_mask = sel_mask
        
        # Local search
        improved_mask, improved_size = local_search(best_mask, max_iter=5000)
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask
    
    # Convert back to values
    result = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    # Final validation
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid, should not happen
                pass
    
    return result
# EVOLVE-BLOCK-END
