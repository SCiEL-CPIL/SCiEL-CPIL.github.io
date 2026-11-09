import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 120s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Bidirectional growth with Warnsdorff's rule.
    Instead of growing from one end, we grow from both ends (start and end)
    simultaneously. At each step, we pick the end that has fewer valid moves
    (Warnsdorff's rule) and extend that end with the move that leaves the
    fewest options for the next step (to avoid trapping).
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    def is_valid_extension(path, used_set, candidate, exclude_indices=None):
        """
        Check if candidate can be appended/prepended to path.
        1. candidate not in used_set
        2. candidate not adjacent to any vertex in path except the one it's connecting to.
        
        exclude_indices: set of indices in path that we don't need to check adjacency against.
        Usually, if extending the end (last), we don't check against path[-1].
        If extending the start (first), we don't check against path[0].
        """
        if candidate in used_set:
            return False
        
        # We need to check adjacency against all vertices in path EXCEPT the one we are connecting to.
        # The caller handles which vertex is the 'connector'.
        # Actually, it's safer to just check all vertices in path, but skip the specific one if needed.
        # However, since candidate is a neighbor of the end, it IS adjacent to the end.
        # We must ensure it is NOT adjacent to any OTHER vertex.
        
        for i, p in enumerate(path):
            if exclude_indices and i in exclude_indices:
                continue
            if (candidate ^ p).bit_count() == 1:
                return False
        return True

    def bidirectional_grow(start_vertex=0, time_limit=None):
        """
        Grow a snake bidirectionally.
        """
        # Initialize with a small seed path to avoid trivial cases
        # Start with just one vertex? Or a small random path?
        # Let's start with a random vertex and grow from it.
        # To make it bidirectional, we can start with a random edge or just grow from one point
        # but alternate sides?
        # Actually, standard bidirectional growth for paths often starts with a central segment.
        # Let's start with a random vertex v0.
        # Then pick a random neighbor v1. Path: [v0, v1].
        # Now we have two ends: v0 and v1.
        # We grow v0 backwards and v1 forwards?
        # Or we treat the path as having a "left" end and a "right" end.
        
        # Let's start with a random vertex.
        v0 = start_vertex
        path = [v0]
        used = {v0}
        
        # To have two ends, we need at least 2 vertices.
        # Pick a random neighbor to start the second end.
        if not neighbors[v0]:
            return path
            
        # Randomly pick a neighbor to extend
        # But we want to be careful not to trap ourselves immediately.
        # For the first step, any neighbor is fine.
        v1 = random.choice(neighbors[v0])
        path.append(v1)
        used.add(v1)
        
        # Now we have ends at path[0] and path[-1].
        # We will alternate or choose based on heuristic.
        
        while True:
            # Check time
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end (prepending)
            # A move v is valid if:
            # 1. v not in used
            # 2. v not adjacent to any p in path[1:] (since v will be adjacent to path[0])
            #    Note: v is adjacent to path[0]. We must check against path[1:].
            #    Also, if len(path) > 1, path[0] is the connector.
            
            valid_left = []
            for v in neighbors[left_end]:
                if v in used:
                    continue
                # Check against path[1:]
                # Optimization: if v is adjacent to path[1], it's invalid.
                # We need to check against all p in path[1:]
                is_valid = True
                for p in path[1:]:
                    if (v ^ p).bit_count() == 1:
                        is_valid = False
                        break
                if is_valid:
                    valid_left.append(v)
            
            # Get valid moves for right end (appending)
            valid_right = []
            for v in neighbors[right_end]:
                if v in used:
                    continue
                # Check against path[:-1]
                is_valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        is_valid = False
                        break
                if is_valid:
                    valid_right.append(v)
            
            if not valid_left and not valid_right:
                break
            
            # Heuristic: Warnsdorff's rule
            # Pick the end with fewer options?
            # Or pick the move that results in fewer options for the next step?
            # Standard Warnsdorff: pick the neighbor with the fewest onward moves.
            
            # Let's calculate the "cost" of each move for both sides.
            # For a move v on the left:
            #   New path: [v] + path
            #   Next left end is v.
            #   Count valid moves from v (excluding path[0] which is now connected).
            #   Actually, we just count how many valid moves v would have if we added it.
            
            def count_onward_moves(candidate, current_path, is_left):
                """
                Estimate how many moves will be available after adding candidate.
                """
                # Simulate adding candidate
                # If is_left, new end is candidate. Path becomes [candidate] + current_path
                # Valid moves for candidate must not be adjacent to current_path[1:] (if left)
                # or current_path[:-1] (if right)
                
                count = 0
                for w in neighbors[candidate]:
                    if w in used:
                        continue
                    # Check adjacency to the rest of the path
                    # The "connector" is current_path[0] if left, current_path[-1] if right.
                    # We must check against all OTHER vertices.
                    
                    # To speed this up, we can just check against the whole path, 
                    # but we know candidate is adjacent to the connector.
                    # So we check against path excluding the connector.
                    
                    if is_left:
                        # Connector is current_path[0]
                        # Check against current_path[1:]
                        # Note: candidate is NOT in current_path yet.
                        # If w is adjacent to any p in current_path[1:], it's invalid.
                        # Also, if w is in used, it's invalid.
                        # We already checked w in used.
                        for p in current_path[1:]:
                            if (w ^ p).bit_count() == 1:
                                break
                        else:
                            count += 1
                    else:
                        # Connector is current_path[-1]
                        # Check against current_path[:-1]
                        for p in current_path[:-1]:
                            if (w ^ p).bit_count() == 1:
                                break
                        else:
                            count += 1
                return count

            # Evaluate moves
            best_move_left = None
            min_score_left = float('inf')
            for v in valid_left:
                # Score: number of onward moves. Lower is better (Warnsdorff).
                # To add randomness, we can add a small noise or pick from top K lowest.
                score = count_onward_moves(v, path, True)
                if score < min_score_left:
                    min_score_left = score
                    best_move_left = v
                # Tie-breaking: keep first or random? Let's collect candidates with min score.
            
            best_move_right = None
            min_score_right = float('inf')
            for v in valid_right:
                score = count_onward_moves(v, path, False)
                if score < min_score_right:
                    min_score_right = score
                    best_move_right = v

            # If one side has no moves, pick the other.
            if not valid_left:
                next_v = best_move_right
                path.append(next_v)
                used.add(next_v)
            elif not valid_right:
                next_v = best_move_left
                path.insert(0, next_v)
                used.add(next_v)
            else:
                # Choose between left and right.
                # Heuristic: Pick the side with fewer valid moves (more constrained)?
                # Or pick the move with the lowest score (most constrained future)?
                # Let's compare the "tightness" of the best move on each side.
                # If min_score_left < min_score_right, the left move is more constrained (fewer future options).
                # Warnsdorff suggests picking the most constrained move.
                
                # Also, consider the number of available moves right now.
                # If one side has very few moves, maybe we should pick that side to avoid getting stuck?
                # Actually, if a side has few moves, it's risky. But Warnsdorff says pick the move with fewest onward moves.
                
                # Let's use the score of the best move.
                if min_score_left < min_score_right:
                    next_v = best_move_left
                    path.insert(0, next_v)
                    used.add(next_v)
                elif min_score_right < min_score_left:
                    next_v = best_move_right
                    path.append(next_v)
                    used.add(next_v)
                else:
                    # Tie: random
                    if random.random() < 0.5:
                        next_v = best_move_left
                        path.insert(0, next_v)
                        used.add(next_v)
                    else:
                        next_v = best_move_right
                        path.append(next_v)
                        used.add(next_v)

        return path

    # Multi-start
    attempts = 0
    while time.time() - start_time < DEADLINE:
        # Randomize start vertex?
        # Due to symmetry, start=0 is fine, but randomizing the initial neighbor choice is key.
        # The bidirectional_grow function handles the randomness in the first step and subsequent choices.
        
        p = bidirectional_grow(start_vertex=0, time_limit=DEADLINE - (time.time() - start_time))
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
        attempts += 1
        
        # Optional: try a different starting configuration?
        # E.g., start with a random edge.
        if attempts % 10 == 0:
            # Try starting with a random edge
            v0 = random.randint(0, MAX_VERTS - 1)
            v1 = random.choice(neighbors[v0])
            # We can modify bidirectional_grow to accept an initial path
            # For now, the internal randomness in the first step covers this partially.
            # But let's explicitly randomize the start vertex for diversity in restarts.
            pass

    if best_path is None:
        # Fallback to seed
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
        return path

    return best_path

# EVOLVE-BLOCK-END
