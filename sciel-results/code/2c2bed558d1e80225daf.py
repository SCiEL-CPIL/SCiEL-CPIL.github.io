"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
import numpy as np

# Precompute neighbor masks for O(1) forbidden set updates
# neighbors_mask[v] is an integer with bits set for all neighbors of v
neighbors_mask = [0] * TOTAL_VERTICES
neighbors_list = [[] for _ in range(TOTAL_VERTICES)]
for v in range(TOTAL_VERTICES):
    mask = 0
    for b in range(N):
        u = v ^ (1 << b)
        mask |= (1 << u)
        neighbors_list[v].append(u)
    neighbors_mask[v] = mask

def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _beam_search_construct(deadline, beam_width=100, seed=42):
    """Optimized Beam search using bitmask for forbidden neighbors."""
    rng = random.Random(seed)
    
    # Beam: list of (path_list, used_mask, forbidden_mask, last_vertex)
    beam = []
    # Start with a few random vertices
    for _ in range(min(beam_width, 10)):
        sv = rng.randint(0, TOTAL_VERTICES - 1)
        # initial forbidden is 0
        beam.append(([sv], 1 << sv, 0, sv))
    
    best_path = []
    best_len = 0
    
    while time.time() < deadline:
        if time.time() > deadline - 0.5:
            break
        
        new_beam = []
        
        for path, used_mask, forbidden_mask, last in beam:
            # Candidates are neighbors of last
            # We need to check if they are in used_mask or forbidden_mask
            # Note: forbidden_mask contains neighbors of path[:-1]
            # So if v is in forbidden_mask, it's adjacent to some p in path[:-1] -> invalid
            # If v is in used_mask, it's already in path -> invalid
            
            candidates = neighbors_list[last]
            valid_nexts = []
            
            for v in candidates:
                bit = 1 << v
                if (used_mask & bit) == 0 and (forbidden_mask & bit) == 0:
                    valid_nexts.append(v)
            
            if not valid_nexts:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                continue
            
            # Heuristic: Score by number of further valid moves (1-ply lookahead on the new state)
            # To keep it fast, we might just take all valid nexts if few, or score them.
            # Let's score by "degree in remaining graph" roughly
            scored = []
            for v in valid_nexts:
                # New state would have forbidden_mask' = forbidden_mask | neighbors_mask[last]
                new_forbidden = forbidden_mask | neighbors_mask[last]
                # Count how many neighbors of v are valid for the step AFTER this one
                # Those neighbors w must not be in used_mask | (1<<v)
                # And must not be in new_forbidden | neighbors_mask[v]
                # This is getting complex. Simple heuristic: random or count available neighbors of v.
                
                # Simple score: number of neighbors of v not in used_mask
                # (Ignoring forbidden for speed, as it's a rough proxy for flexibility)
                score = 0
                for w in neighbors_list[v]:
                    if (used_mask & (1 << w)) == 0:
                        score += 1
                scored.append((score, v))
            
            scored.sort(key=lambda x: -x[0])
            
            # Take top K
            top_k = min(5, len(scored))
            for i in range(top_k):
                v = scored[i][1]
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                # Update forbidden mask: add neighbors of 'last' to forbidden set
                # Because 'last' becomes part of path[:-1] for the next step
                new_forbidden = forbidden_mask | neighbors_mask[last]
                new_beam.append((new_path, new_used, new_forbidden, v))
        
        if not new_beam:
            break
        
        # Prune beam
        # Sort by length, then maybe by flexibility (number of valid nexts from current last)
        # To avoid sorting huge lists, we can just take top beam_width by length
        new_beam.sort(key=lambda x: -len(x[0]))
        beam = new_beam[:beam_width]
        
        # Track best
        for path, used, forb, last in beam:
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
        
        # Check if all dead
        all_dead = True
        for path, used, forb, last in beam:
            for v in neighbors_list[last]:
                bit = 1 << v
                if (used & bit) == 0 and (forb & bit) == 0:
                    all_dead = False
                    break
            if not all_dead:
                break
        if all_dead:
            break
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_path = []
    best_len = 0
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        path = _beam_search_construct(deadline, beam_width=100, seed=seed)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback: simple greedy from vertex 0
    path, used = [0], 1
    forbidden = 0
    while True:
        cur = path[-1]
        nxt = None
        for v in neighbors_list[cur]:
            bit = 1 << v
            if (used & bit) == 0 and (forbidden & bit) == 0:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used |= (1 << nxt)
        forbidden |= neighbors_mask[cur]

# EVOLVE-BLOCK-END
