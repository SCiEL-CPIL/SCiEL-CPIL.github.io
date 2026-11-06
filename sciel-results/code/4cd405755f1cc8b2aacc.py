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
    deadline = start_time + 115.0  # Safety margin
    
    sys.setrecursionlimit(10000)

    # Helper: Check if 'candidate' can be added to 'path' given 'visited'
    # path is the list so far. candidate is the next vertex.
    # We need to ensure candidate is not adjacent to any vertex in path[:-1]
    # and candidate is not in visited.
    def is_valid_next(candidate, path, visited):
        if (visited >> candidate) & 1:
            return False
        # Check chord condition against all previous vertices except the last one
        # (last one is adjacent by definition of neighbor)
        # Optimization: Only check if Hamming distance to any v in path[:-1] is 1
        # Note: path[-1] is the current end. candidate is neighbor of path[-1].
        # So we check path[:-1].
        # If path has length 1, path[:-1] is empty, so valid.
        for v in path[:-1]:
            if (candidate ^ v).bit_count() == 1:
                return False
        return True

    def greedy_initial(start_v=None, rnd=None):
        """Generate a reasonably long snake quickly using a greedy heuristic with MCV-lite."""
        if start_v is None:
            start_v = 0 if rnd is None else rnd.randint(0, SIZE - 1)
        
        path = [start_v]
        visited = 1 << start_v
        current = start_v
        
        max_len = 200
        for _ in range(max_len):
            valid_nexts = []
            for nb in NEIGHBORS[current]:
                if (visited >> nb) & 1:
                    continue
                if is_valid_next(nb, path, visited):
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                break
            
            if rnd is None:
                # Deterministic greedy: pick the one that leaves most options? 
                # For speed, just pick first or random. 
                # Let's use a simple "most constrained" lite: 
                # Count neighbors of nb that are valid.
                best_nb = valid_nexts[0]
                best_score = -1
                for nb in valid_nexts:
                    # Score: number of valid next-neighbors
                    score = 0
                    for nb2 in NEIGHBORS[nb]:
                        if (visited >> nb2) & 1:
                            continue
                        # Check if nb2 is valid after adding nb
                        # path + [nb]
                        # nb2 must not be adjacent to path[:-1]
                        ok = True
                        for v in path[:-1]:
                            if (nb2 ^ v).bit_count() == 1:
                                ok = False
                                break
                        if ok:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_nb = nb
                next_v = best_nb
            else:
                next_v = rnd.choice(valid_nexts)
                
            path.append(next_v)
            visited |= (1 << next_v)
            current = next_v
            
        return path

    # Helper: Fast Backtracking to find a path between start and end
    # start: vertex before the gap
    # end: vertex after the gap
    # blocked: bitset of vertices that are in the rest of the path (cannot be used)
    # target_len: desired number of vertices in the gap
    # returns: list of vertices [v1, ..., vk] such that start-v1-...-vk-end is valid
    # and no vertex in the gap is in blocked, and no chords form with the rest of the path.
    # Note: The 'rest of the path' includes start and end.
    # So when checking validity of a new vertex v, we must ensure it is not adjacent to 
    # any vertex in (path_without_gap).
    
    def solve_gap(start, end, blocked, target_len, time_limit, rnd):
        """
        Finds a path of length target_len (number of edges in the gap, so target_len+1 vertices? 
        No, let's say we want to replace k vertices. 
        If we remove vertices i..j, the gap is between path[i-1] and path[j+1].
        We want to find a sequence of vertices v1...vk such that:
        path[i-1] - v1 - ... - vk - path[j+1] is a valid induced path segment.
        And none of v1...vk are in blocked.
        And no chords are formed with path_without_gap.
        """
        if time.time() > time_limit:
            return None
        
        # The "fixed" part of the path is everything except the gap.
        # Let's pass the full path without the gap vertices to check chords against.
        # Actually, it's easier to pass the 'blocked' set and the 'fixed' path list.
        
        # We are searching for a path of exactly 'target_len' vertices? 
        # Or at most? 
        # Usually we want to fill the gap. If we can't fill it, we can try shorter?
        # If we find a shorter path, the total path length decreases. 
        # In ILS, we might accept shorter if it allows future improvements, 
        # but usually we want to maintain or increase length.
        # Let's try to find a path of exactly target_len vertices first.
        
        # State: current vertex, list of vertices in gap, visited_in_gap
        # Start: start
        # End: end
        
        # We need to ensure that the new vertices are not adjacent to any vertex in the fixed path.
        # Let's precompute the set of "forbidden" vertices for the gap.
        # A vertex v is forbidden if it is in blocked OR it is adjacent to any vertex in fixed_path.
        # Wait, if v is adjacent to a vertex in fixed_path, and v is not in fixed_path, 
        # then v forms a chord with that fixed vertex.
        # So, forbidden = blocked | {u | exists w in fixed_path such that dist(u,w)==1}
        
        # However, 'start' and 'end' are in fixed_path.
        # The new path must start at start and end at end.
        # So v1 must be neighbor of start. v1 must not be in forbidden (except start itself? no, v1 is new).
        # Actually, v1 is adjacent to start. That's allowed (consecutive).
        # But v1 must NOT be adjacent to any OTHER vertex in fixed_path.
        
        # Let's refine the check in the search.
        
        def search(current, gap_path, visited_gap, depth):
            if time.time() > time_limit:
                return None
            
            if depth == target_len:
                # Check if current can connect to end
                if current == end:
                    return gap_path
                # Check if end is neighbor of current
                # And check chord condition for end? 
                # end is in fixed_path. 
                # The new vertex 'current' is adjacent to 'end'. This is allowed.
                # But 'current' must not be adjacent to any OTHER vertex in fixed_path.
                # This check is done when we added 'current' to gap_path.
                # So if current is neighbor of end, we are good.
                if end in NEIGHBORS[current]:
                    return gap_path
                return None
            
            # Try neighbors of current
            candidates = NEIGHBORS[current]
            # Shuffle for diversity
            rnd.shuffle(candidates)
            
            for nb in candidates:
                if nb in visited_gap:
                    continue
                if nb in blocked:
                    continue
                # Check chord condition: nb must not be adjacent to any vertex in fixed_path
                # EXCEPT if nb is 'start' (it's not, nb is new) 
                # AND we must allow nb to be adjacent to 'current' (it is, by definition).
                # Wait, 'current' is in gap_path. 
                # The chord condition applies to non-consecutive vertices.
                # nb is consecutive to current. So that's fine.
                # nb must NOT be adjacent to any vertex in fixed_path.
                
                is_chord = False
                for v in fixed_path:
                    if (nb ^ v).bit_count() == 1:
                        is_chord = True
                        break
                
                if is_chord:
                    continue
                
                # Also, if depth == target_len - 1, nb must be able to reach end
                # We check that in the next recursion or here?
                # If we add nb, depth becomes depth+1.
                # If depth+1 == target_len, then in the next call, we check if nb connects to end.
                
                new_gap = gap_path + [nb]
                new_visited = visited_gap | (1 << nb)
                
                res = search(nb, new_gap, new_visited, depth + 1)
                if res is not None:
                    return res
            
            return None
        
        # Precompute fixed_path
        # The caller should pass the path with the gap removed.
        # For this helper, we need to pass fixed_path.
        # Let's modify the signature or handle it outside.
        # To keep it simple, we'll pass fixed_path as a closure variable or argument.
        # Let's make it an argument.
        pass # Placeholder, will implement in main logic

    # Revised solve_gap that takes fixed_path
    def solve_gap_impl(start, end, blocked, fixed_path, target_len, time_limit, rnd):
        if time.time() > time_limit:
            return None
        
        def search(current, gap_path, visited_gap, depth):
            if time.time() > time_limit:
                return None
            
            if depth == target_len:
                if end in NEIGHBORS[current]:
                    # Final check: current must not be adjacent to any other fixed_path vertex?
                    # This was checked when current was added.
                    return gap_path
                return None
            
            candidates = NEIGHBORS[current]
            # Heuristic: prefer candidates that have more valid next steps?
            # For speed, just shuffle.
            rnd.shuffle(candidates)
            
            for nb in candidates:
                if (visited_gap >> nb) & 1:
                    continue
                if (blocked >> nb) & 1:
                    continue
                
                # Chord check against fixed_path
                is_chord = False
                for v in fixed_path:
                    if (nb ^ v).bit_count() == 1:
                        is_chord = True
                        break
                if is_chord:
                    continue
                
                new_gap = gap_path + [nb]
                new_visited = visited_gap | (1 << nb)
                
                res = search(nb, new_gap, new_visited, depth + 1)
                if res is not None:
                    return res
            
            return None
        
        # Start search from start
        # Depth 0 means we are at 'start', gap_path is empty
        # We need to find target_len vertices.
        # So we need to make target_len steps.
        # Step 1: pick nb1 from start.
        # ...
        # Step target_len: pick nb_target_len.
        # Then check if nb_target_len is connected to end.
        
        # Wait, if target_len is 1, we pick nb1. Then check if nb1 connects to end.
        # So the search depth logic:
        # search(current, gap_path, visited, depth)
        # depth is the number of vertices added to gap_path so far.
        # If depth == target_len, we check connection to end.
        
        # Initial call: current=start, gap_path=[], visited=0, depth=0
        return search(start, [], 0, 0)

    # ILS Strategy
    best_path = greedy_initial()
    best_len = len(best_path) - 1
    best_path_global = list(best_path)
    
    # Sort vertices by degree or something? No, random is fine.
    
    # Temperature for Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.95
    
    current_path = list(best_path)
    current_len = len(current_path) - 1
    
    # Precompute bitsets for current path
    def path_to_bitset(path):
        b = 0
        for v in path:
            b |= (1 << v)
        return b
    
    current_visited = path_to_bitset(current_path)
    
    iter_count = 0
    while time.time() < deadline - 5.0:
        iter_count += 1
        rnd = random.Random(iter_count)
        
        # 1. Pick a segment to disrupt
        # Don't disrupt the very first or last few vertices? 
        # Actually, disrupting anywhere is fine, but we need start and end to be defined.
        # Let's pick i and j such that 1 <= i < j < len(current_path) - 1
        # Gap is current_path[i..j]
        # Fixed part: current_path[0..i-1] + current_path[j+1..end]
        
        path_len = len(current_path)
        if path_len < 5:
            break
            
        # Randomly select segment length
        # Small segments are easier to fix, large segments are harder but more impactful
        # Let's try segment length between 3 and 10
        seg_len = rnd.randint(3, min(10, path_len - 4))
        
        # Select start index i
        # i can be from 1 to path_len - seg_len - 2 (leaving room for end)
        min_i = 1
        max_i = path_len - seg_len - 2
        if max_i < min_i:
            continue
            
        i = rnd.randint(min_i, max_i)
        j = i + seg_len - 1
        
        start_v = current_path[i-1]
        end_v = current_path[j+1]
        
        # Build blocked set: all vertices in current_path EXCEPT the gap
        # Gap vertices: current_path[i..j]
        # Fixed vertices: current_path[0..i-1] and current_path[j+1..end]
        
        fixed_path = current_path[:i] + current_path[j+1:]
        blocked = 0
        for v in fixed_path:
            blocked |= (1 << v)
        
        # Try to solve the gap
        # We want to replace seg_len vertices with seg_len vertices.
        # If we fail, we might try seg_len - 1? 
        # If we find a longer path, that's even better.
        # Let's first try to find a path of exactly seg_len.
        
        new_gap = solve_gap_impl(start_v, end_v, blocked, fixed_path, seg_len, time.time() + 2.0, rnd)
        
        if new_gap is not None:
            # Construct new path
            new_path = current_path[:i] + new_gap + current_path[j+1:]
            new_len = len(new_path) - 1
            
            # Validate the new path quickly (just in case)
            valid = True
            # Check consecutive
            for k in range(len(new_path)-1):
                if (new_path[k] ^ new_path[k+1]).bit_count() != 1:
                    valid = False
                    break
            # Check chords
            if valid:
                for k in range(len(new_path)):
                    for l in range(k+2, len(new_path)):
                        if (new_path[k] ^ new_path[l]).bit_count() == 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Accept if longer, or with probability exp(-delta/T) if shorter
                if new_len >= current_len:
                    current_path = new_path
                    current_len = new_len
                    current_visited = path_to_bitset(current_path)
                    
                    if current_len > best_len:
                        best_len = current_len
                        best_path_global = list(current_path)
                else:
                    delta = current_len - new_len
                    prob = np.exp(-delta / T)
                    if rnd.random() < prob:
                        current_path = new_path
                        current_len = new_len
                        current_visited = path_to_bitset(current_path)
        else:
            # Try to find a shorter path? 
            # If we can't fill the gap, maybe a shorter gap works?
            # Or maybe we should try to EXTEND the path?
            # For now, let's try to find a path of length seg_len - 1
            # This reduces path length, but might allow future extensions?
            # Actually, if we can't find a valid path of length seg_len, 
            # it's likely the gap is "closed". 
            # Let's try to find ANY valid path between start and end that doesn't use blocked.
            # We can try decreasing target_len.
            found = False
            for try_len in range(seg_len - 1, 0, -1):
                if time.time() > deadline - 5.0:
                    break
                new_gap = solve_gap_impl(start_v, end_v, blocked, fixed_path, try_len, time.time() + 0.5, rnd)
                if new_gap is not None:
                    new_path = current_path[:i] + new_gap + current_path[j+1:]
                    new_len = len(new_path) - 1
                    # Accept if it's not too much shorter? 
                    # Or just accept to escape local minimum?
                    # Let's accept if valid.
                    valid = True
                    for k in range(len(new_path)-1):
                        if (new_path[k] ^ new_path[k+1]).bit_count() != 1:
                            valid = False
                            break
                    if valid:
                        for k in range(len(new_path)):
                            for l in range(k+2, len(new_path)):
                                if (new_path[k] ^ new_path[l]).bit_count() == 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        current_path = new_path
                        current_len = new_len
                        current_visited = path_to_bitset(current_path)
                        found = True
                        break
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Occasionally, restart from scratch if stuck?
        # Or just keep going.
        
    return best_path_global

if __name__ == "__main__":
    p = construct()
    print(f"Length: {len(p)-1}")
    # Validate
    valid = True
    for i in range(len(p)-1):
        if (p[i] ^ p[i+1]).bit_count() != 1:
            valid = False
            break
    for i in range(len(p)):
        for j in range(i+2, len(p)):
            if (p[i] ^ p[j]).bit_count() == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
