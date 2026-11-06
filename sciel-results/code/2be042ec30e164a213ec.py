import time
import random
import sys
import numpy as np

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    sys.setrecursionlimit(10000)

    # We will use a beam search approach.
    # State: (path_tuple, visited_bitset, current_vertex)
    # To make it fast, we represent the beam as a list of states.
    # We periodically vectorize the "valid next steps" check if the beam is large, 
    # but for N=9, even pure Python with bit operations is fast enough if we keep the beam width moderate.
    
    # A key optimization: Precompute for each vertex v, the mask of all its neighbors.
    # Actually, we already have NEIGHBORS.
    
    # The bottleneck in the current program is the `for v in path[:-1]` loop inside the candidate check.
    # We can optimize the chord check:
    # A candidate `nb` is invalid if it is adjacent to any `v` in `path[:-1]`.
    # `nb` is adjacent to `v` iff `nb ^ v` has exactly one bit set.
    # This is hard to vectorize easily in pure Python without NumPy on the path itself.
    
    # However, we can use a different trick:
    # Maintain a `blocked` bitset.
    # When we add a vertex `v` to the path, which future vertices become blocked?
    # Any vertex `u` such that `u` is adjacent to `v` AND `u` is NOT the next vertex in the path?
    # No, the rule is: non-consecutive vertices must differ by >= 2 bits.
    # So if we are at `current` and consider `nb` as next:
    # `nb` must not be adjacent to any `v` in `path[:-1]`.
    # Note: `nb` is adjacent to `current` (by definition of being a neighbor).
    # So we only need to check against `path[:-1]`.
    
    # Let's try a "Lookahead Beam Search" with aggressive pruning.
    # We will maintain a list of (path, visited, current).
    
    def greedy_seed():
        """Generate a decent seed quickly."""
        path = [0]
        visited = 1
        current = 0
        rnd = random.Random(42)
        
        for _ in range(100):
            valids = []
            for nb in NEIGHBORS[current]:
                if (visited >> nb) & 1:
                    continue
                # Check chord
                ok = True
                for v in path[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    valids.append(nb)
            if not valids:
                break
            next_v = rnd.choice(valids)
            path.append(next_v)
            visited |= (1 << next_v)
            current = next_v
        return path, visited, current

    # Initial seed
    seed_path, seed_visited, seed_current = greedy_seed()
    best_path = list(seed_path)
    best_len = len(best_path) - 1
    best_visited = seed_visited
    
    # Beam Search Parameters
    beam_widths = [10, 20, 50, 100]
    
    # We will run multiple restarts.
    # Each restart starts from a slightly perturbed seed or a new random start.
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
            
        # Randomize the start for diversity in later restarts
        if restart == 0:
            # Use the initial greedy seed
            beam = [(tuple(seed_path), seed_visited, seed_current)]
        else:
            # Generate a new short greedy path from a random start
            rnd = random.Random(restart * 1000)
            start_v = rnd.randint(0, SIZE - 1)
            path = [start_v]
            visited = 1 << start_v
            current = start_v
            
            # Greedy fill up to length 20-40
            target_len = 20 + rnd.randint(0, 20)
            for _ in range(target_len):
                valids = []
                for nb in NEIGHBORS[current]:
                    if (visited >> nb) & 1:
                        continue
                    ok = True
                    for v in path[:-1]:
                        if (nb ^ v).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        valids.append(nb)
                if not valids:
                    break
                next_v = rnd.choice(valids)
                path.append(next_v)
                visited |= (1 << next_v)
                current = next_v
            
            if len(path) > best_len + 1:
                # Update best if we just found a longer seed?
                # Unlikely, but possible.
                pass
                
            beam = [(tuple(path), visited, current)]
            
            # If the seed is already very long, maybe skip beam search?
            # No, beam search might extend it.

        # Run Beam Search
        # We will expand the beam step-by-step.
        # To keep it fast, we limit the depth of the beam search per restart.
        
        for width in beam_widths:
            if time.time() > deadline - 2:
                break
                
            # Expand the beam to `width`
            # We do this in batches to manage memory and time.
            # Actually, standard beam search:
            # 1. Take current beam states.
            # 2. Generate all valid extensions for each state.
            # 3. Score them.
            # 4. Keep top `width`.
            
            # We will iterate until we can't extend anymore or time runs out.
            # To avoid infinite loops on very short beams, we limit the number of "beam steps".
            
            for step in range(100): # Max 100 beam steps per restart/width combo?
                if time.time() > deadline - 2:
                    break
                
                new_beam = []
                
                for path_tuple, visited, current in beam:
                    # Generate valid nexts for this state
                    valids = []
                    for nb in NEIGHBORS[current]:
                        if (visited >> nb) & 1:
                            continue
                        # Chord check
                        ok = True
                        # Optimization: Check recent vertices first? 
                        # Or just iterate.
                        # Since path is a tuple, path[:-1] is a bit expensive.
                        # Let's convert to list once? No, tuple is fine.
                        for v in path_tuple[:-1]:
                            if (nb ^ v).bit_count() == 1:
                                ok = False
                                break
                        if ok:
                            valids.append(nb)
                    
                    if not valids:
                        # Dead end. 
                        # If this path is longer than best, update best.
                        if len(path_tuple) - 1 > best_len:
                            best_len = len(path_tuple) - 1
                            best_path = list(path_tuple)
                        continue
                    
                    # Score each valid next
                    # Heuristic: Prefer nexts that leave more options? (Least Constrained Value)
                    # Or prefer nexts that are "central"?
                    # Let's use a simple heuristic: Random tie-breaking, but keep all.
                    # Actually, for beam search, we usually keep the top K based on a score.
                    # If we don't have a strong score, we can just keep the first K or random K.
                    # Let's use a "lookahead" score: count valid nexts for the next step.
                    
                    scored_valids = []
                    for nb in valids:
                        # Calculate lookahead score
                        # How many valid moves from nb?
                        # This is expensive. 
                        # Alternative: Just keep all valids, and if too many, sample.
                        
                        # Let's try a cheaper heuristic: 
                        # Prefer nb if it has fewer "conflicts" with the rest of the path?
                        # Actually, let's just add them to new_beam and prune at the end.
                        
                        new_path = path_tuple + (nb,)
                        new_visited = visited | (1 << nb)
                        new_beam.append((new_path, new_visited, nb))
                    
                if not new_beam:
                    break
                
                # Prune new_beam to size `width`
                if len(new_beam) > width:
                    # How to score?
                    # 1. Length (all are same length in this step, so no diff)
                    # 2. Lookahead: number of valid next steps from the new current.
                    # Let's compute lookahead for pruning.
                    
                    scored_beam = []
                    for p, v, c in new_beam:
                        # Count valid nexts from c
                        count = 0
                        for nb in NEIGHBORS[c]:
                            if (v >> nb) & 1:
                                continue
                            ok = True
                            for x in p[:-1]:
                                if (nb ^ x).bit_count() == 1:
                                    ok = False
                                    break
                            if ok:
                                count += 1
                        # We want to maximize count (most options)
                        scored_beam.append((count, p, v, c))
                    
                    # Sort by count descending
                    scored_beam.sort(key=lambda x: x[0], reverse=True)
                    
                    # Keep top `width`
                    beam = [(p, v, c) for count, p, v, c in scored_beam[:width]]
                else:
                    beam = new_beam
                    
                # Update best if any path in beam is longer
                # Since all paths in beam are length L, check once
                if len(beam[0][0]) - 1 > best_len:
                    best_len = len(beam[0][0]) - 1
                    best_path = list(beam[0][0])
                    
                # If beam size is 0, break
                if not beam:
                    break

    # Final validation and return
    # Ensure best_path is valid
    if not best_path:
        return [0]
        
    # Quick sanity check
    # for i in range(len(best_path)-1):
    #     if (best_path[i] ^ best_path[i+1]).bit_count() != 1:
    #         return [0] # Fallback
    # for i in range(len(best_path)):
    #     for j in range(len(best_path)):
    #         if abs(i-j) > 1 and (best_path[i] ^ best_path[j]).bit_count() == 1:
    #             return [0]
            
    return best_path

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(len(p)):
            if abs(i-j) > 1 and (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
