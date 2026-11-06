import time
import random
from itertools import combinations

# Precompute popcounts for all 16-bit masks
N = 16
ALL_MASK = (1 << N) - 1
POP_COUNTS = [0] * (1 << N)
for i in range(1 << N):
    POP_COUNTS[i] = bin(i).count('1')

# Precompute all possible triples as bitmasks
ALL_TRIPLES = []
TRIPLE_TO_INDEX = {}
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            idx = len(ALL_TRIPLES)
            ALL_TRIPLES.append(mask)
            TRIPLE_TO_INDEX[mask] = idx

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute conflict matrix
# conflict[i][j] is True if ALL_TRIPLES[i] and ALL_TRIPLES[j] share > 1 point
CONFLICT = [[False] * NUM_TRIPLES for _ in range(NUM_TRIPLES)]
for i in range(NUM_TRIPLES):
    mi = ALL_TRIPLES[i]
    for j in range(i + 1, NUM_TRIPLES):
        mj = ALL_TRIPLES[j]
        if POP_COUNTS[mi & mj] > 1:
            CONFLICT[i][j] = True
            CONFLICT[j][i] = True

def calc_alpha(edges_indices):
    """Calculate independence number of the hypergraph defined by edges_indices."""
    if not edges_indices:
        return N
    
    bad = [False] * (1 << N)
    
    for idx in edges_indices:
        e = ALL_TRIPLES[idx]
        # Mark all supersets of e as bad
        comp = ALL_MASK ^ e
        sub = comp
        while True:
            bad[e | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    max_alpha = 0
    for i in range(1 << N):
        if not bad[i]:
            pc = POP_COUNTS[i]
            if pc > max_alpha:
                max_alpha = pc
                if max_alpha == N:
                    break
    return max_alpha

def random_greedy_packing():
    """Generate a random maximal linear packing using conflict matrix."""
    active = set()
    available = list(range(NUM_TRIPLES))
    random.shuffle(available)
    
    for idx in available:
        # Check if idx conflicts with any active edge
        valid = True
        for a in active:
            if CONFLICT[idx][a]:
                valid = False
                break
        if valid:
            active.add(idx)
            
    return list(active)

def local_search(edges_indices, time_limit):
    """Improve the independence number via local search using conflict matrix."""
    current_alpha = calc_alpha(edges_indices)
    best_alpha = current_alpha
    best_edges = edges_indices[:]
    
    active_set = set(edges_indices)
    
    while time.time() < time_limit:
        improved = False
        
        # Try removing one edge and adding another
        # To speed up, we can randomly sample edges to remove
        # Or iterate through all, but that's 560 * 560 = 313k checks per pass.
        # With 120s, we can do many passes.
        
        for i in range(len(edges_indices)):
            if time.time() >= time_limit:
                break
                
            idx_to_remove = edges_indices[i]
            active_set.remove(idx_to_remove)
            
            # Find a replacement
            # We can pick a random candidate from ALL_TRIPLES that doesn't conflict with current active_set
            # Or iterate through a shuffled list
            
            candidates = list(range(NUM_TRIPLES))
            random.shuffle(candidates)
            
            found_replacement = False
            for new_idx in candidates:
                if new_idx == idx_to_remove:
                    continue
                
                # Check conflict with active_set
                valid = True
                for a in active_set:
                    if CONFLICT[new_idx][a]:
                        valid = False
                        break
                
                if valid:
                    # Candidate is valid
                    candidate_edges = list(active_set) + [new_idx]
                    # Optimization: Only calculate alpha if this might be better?
                    # Hard to know without calculating.
                    # But we can check if new_idx covers more "gaps"?
                    # For now, just calculate.
                    new_alpha = calc_alpha(candidate_edges)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = candidate_edges
                        active_set.add(new_idx)
                        improved = True
                        found_replacement = True
                        break # Break candidate loop, restart outer while?
                        # Actually, if we break the candidate loop, we should restart the outer while to re-evaluate all edges with the new set
                    elif new_alpha == best_alpha:
                        # Tie-breaking: prefer more edges? Or random?
                        # For now, ignore ties to avoid oscillation
                        pass
            
            if improved:
                # Re-sync edges list with active_set
                edges_indices = list(active_set)
                break # Restart outer while loop with new edges
        
        if not improved:
            # No single swap improved alpha.
            # Try adding an edge if not maximal
            maximal = True
            candidates = list(range(NUM_TRIPLES))
            random.shuffle(candidates)
            
            for new_idx in candidates:
                if new_idx in active_set:
                    continue
                valid = True
                for a in active_set:
                    if CONFLICT[new_idx][a]:
                        valid = False
                        break
                if valid:
                    # Not maximal
                    candidate_edges = list(active_set) + [new_idx]
                    new_alpha = calc_alpha(candidate_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = candidate_edges
                        active_set.add(new_idx)
                        edges_indices = list(active_set)
                        maximal = False
                        break
            if maximal:
                break # Local optimum reached
                
    return best_edges, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial seed: Random greedy
    best_edges = random_greedy_packing()
    best_alpha = calc_alpha(best_edges)
    
    # Random restarts
    while time.time() < deadline:
        # Generate a new random packing
        candidate_edges = random_greedy_packing()
        candidate_alpha = calc_alpha(candidate_edges)
        
        if candidate_alpha < best_alpha:
            best_alpha = candidate_alpha
            best_edges = candidate_edges[:]
        
        # Try to improve the candidate
        if time.time() < deadline - 10:
            improved_edges, improved_alpha = local_search(candidate_edges, deadline - 5)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_edges = improved_edges[:]
                
        # Also try to improve the current best if time permits and it's different
        if time.time() < deadline - 10 and best_edges != candidate_edges:
            improved_edges, improved_alpha = local_search(best_edges, deadline - 5)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_edges = improved_edges[:]

    # Convert best_indices back to list of triples [a, b, c]
    result = []
    for idx in best_edges:
        mask = ALL_TRIPLES[idx]
        triple = []
        for i in range(N):
            if mask & (1 << i):
                triple.append(i)
        result.append(triple)
        
    return result
