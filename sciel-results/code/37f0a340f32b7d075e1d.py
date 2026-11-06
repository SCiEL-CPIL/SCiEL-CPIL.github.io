"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Returns:
    cycles: list of lists of edge indices, each representing a 4-cycle.
    cycles_for_edge: list of lists, cycles_for_edge[e] contains indices of cycles containing e.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    def get_idx(u, v):
        if u > v: u, v = v, u
        return edge_to_idx[(u, v)]
        
    cycles = []
    cycles_for_edge = [[] for _ in range(NUM_EDGES)]
    
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles
                    # 1: a-b, b-c, c-d, d-a
                    cyc1 = [get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)]
                    # 2: a-b, b-d, d-c, c-a
                    cyc2 = [get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)]
                    # 3: a-c, c-b, b-d, d-a
                    cyc3 = [get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)]
                    
                    for cyc in [cyc1, cyc2, cyc3]:
                        cid = len(cycles)
                        cycles.append(cyc)
                        for e in cyc:
                            cycles_for_edge[e].append(cid)
                            
    return cycles, cycles_for_edge

CYCLES, CYCLES_FOR_EDGE = _precompute_cycles()
NUM_CYCLES = len(CYCLES)

def _greedy_min_S():
    """
    Greedily find a small set S such that every 4-cycle has at least 2 edges in S.
    Equivalent to: Maximize Y = E \ S such that every 4-cycle has at most 2 edges in Y.
    """
    in_Y = [True] * NUM_EDGES
    cycle_counts = [4] * NUM_CYCLES # Initially all 4 edges are in Y
    bad_cycles = list(range(NUM_CYCLES)) # All cycles have count 4 > 2
    
    S = set()
    
    while bad_cycles:
        # Count frequency of each edge in bad_cycles
        # Only consider edges that are currently in Y
        freq = Counter()
        for cid in bad_cycles:
            for e in CYCLES[cid]:
                if in_Y[e]:
                    freq[e] += 1
        
        if not freq:
            # This shouldn't happen if bad_cycles is not empty
            break
            
        # Pick edge with max frequency
        # Tie-breaking: random to allow exploration? Or fixed?
        # Let's pick the one with max freq. If tie, pick smallest index for determinism in one run,
        # but we will run multiple times with random tie-breaking.
        max_freq = max(freq.values())
        candidates = [e for e, f in freq.items() if f == max_freq]
        
        # Random choice among candidates to diversify runs
        e_to_remove = random.choice(candidates)
        
        if not in_Y[e_to_remove]:
             # Should not happen as we only count in_Y edges
             continue

        # Remove e_to_remove from Y
        in_Y[e_to_remove] = False
        S.add(e_to_remove)
        
        # Update cycle counts
        for cid in CYCLES_FOR_EDGE[e_to_remove]:
            cycle_counts[cid] -= 1
            if cycle_counts[cid] <= 2:
                # Remove from bad_cycles
                # To avoid O(N) removal, we can use a flag or just filter later
                # For simplicity, we'll mark it and filter at the start of next loop?
                # Or just use a list and filter.
                pass
                
        # Filter bad_cycles
        # This is O(N_cycles) per step. 5460 * 120 is fine.
        bad_cycles = [cid for cid in bad_cycles if cycle_counts[cid] > 2]
        
    return S

def construct():
    start_time = time.time()
    best_S = set(range(NUM_EDGES))
    best_size = NUM_EDGES
    
    # Run multiple random restarts to find a smaller S
    for seed in range(200):
        if time.time() - start_time > 55:
            break
        random.seed(seed)
        S = _greedy_min_S()
        if len(S) < best_size:
            best_size = len(S)
            best_S = S
            
    # Construct colors
    colors = [0] * NUM_EDGES
    next_color = 1
    for e in range(NUM_EDGES):
        if e in best_S:
            colors[e] = 0
        else:
            colors[e] = next_color
            next_color += 1
            
    return colors

# EVOLVE-BLOCK-END
