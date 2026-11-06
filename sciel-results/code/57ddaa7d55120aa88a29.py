"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
NUM_VERTS = 1 << N  # 512

# Precompute neighbors for all vertices
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_VERTS)]

# Precompute bit_count lookup for all possible values (0 to 2**N - 1)
# Actually we need bit_count of XOR results, which are in [0, 2**N - 1]
BC = np.zeros(NUM_VERTS, dtype=np.uint8)
for i in range(1, NUM_VERTS):
    BC[i] = BC[i >> 1] + (i & 1)

def construct():
    """Construct a long induced path (snake) in Q_9."""
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Start with the greedy snake
    best_path = _greedy_snake()
    best_len = len(best_path)
    
    # Try multiple random seeds and beam searches
    rng = np.random.default_rng(42)
    
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Random starting vertex
        start = int(rng.integers(0, NUM_VERTS))
        
        # Beam search with varying widths
        beam_width = 50 + seed * 10  # increase beam width over time
        if beam_width > 500:
            beam_width = 500
        
        path = _beam_search(start, beam_width, deadline - 2.0)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    return best_path


def _greedy_snake():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any non-consecutive vertex in path
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _beam_search(start, beam_width, deadline):
    """Beam search to find a long induced path starting from 'start'."""
    # Each beam element: (path, used_set_as_frozenset)
    # For efficiency, use a list of paths and maintain used as a set
    
    # Initialize beam with single-element paths
    beam = [([start], {start})]
    
    while True:
        if time.time() > deadline:
            break
        
        # Expand each beam element
        candidates = []
        for path, used in beam:
            cur = path[-1]
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                # Check if v is adjacent to any non-consecutive vertex in path
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    new_path = path + [v]
                    new_used = used | {v}
                    candidates.append((new_path, new_used))
        
        if not candidates:
            break
        
        # Score candidates: prefer longer paths, break ties randomly
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: (-len(x[0]), random.random()))
        beam = candidates[:beam_width]
        
        # If no candidate can be extended further, we're done
        # Check if all beam elements are stuck
        all_stuck = True
        for path, used in beam:
            cur = path[-1]
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used:
                    valid = True
                    for p in path[:-1]:
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        all_stuck = False
                        break
            if not all_stuck:
                break
        if all_stuck:
            break
    
    # Return the longest path in the beam
    if beam:
        longest = max(beam, key=lambda x: len(x[0]))
        return longest[0]
    return [start]


# EVOLVE-BLOCK-START
# The construct function is defined above, but we need to keep the interface.
# The EVOLVE-BLOCK is just a marker; the actual code is above.
# EVOLVE-BLOCK-END
