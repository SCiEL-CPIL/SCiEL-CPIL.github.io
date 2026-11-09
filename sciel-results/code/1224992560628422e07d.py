import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 120s limit (or 150s if evaluator allows, but 118 is safe)

# Precompute neighbors and adjacency masks
neighbors = [[] for _ in range(MAX_VERTS)]
adj_mask = [0] * MAX_VERTS

for v in range(MAX_VERTS):
    for b in range(N):
        nb = v ^ (1 << b)
        neighbors[v].append(nb)
        adj_mask[v] |= (1 << nb)

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Optimized Bidirectional growth with Warnsdorff's rule using bitmask operations.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Local variables for speed
    max_verts = MAX_VERTS
    neighbors_local = neighbors
    adj_mask_local = adj_mask
    
    def bidirectional_grow(start_vertex=0):
        """
        Grow a snake bidirectionally with optimized bitmask checks.
        """
        v0 = start_vertex
        path = [v0]
        used = 1 << v0
        
        # Pick a random neighbor to start the second end
        v1 = random.choice(neighbors_local[v0])
        path.append(v1)
        used |= (1 << v1)
        
        # Pre-compute the "forbidden" mask for the current ends?
        # For left end (index 0), forbidden is adj_mask[left_end] & used ^ (1 << left_end)
        # For right end (index -1), forbidden is adj_mask[right_end] & used ^ (1 << right_end)
        
        def get_valid_moves(end_idx, end_vertex, used_mask):
            """
            Get list of valid candidates to extend from end_vertex.
            end_idx is the index of the end vertex in the path.
            """
            # The candidate 'c' must be:
            # 1. Neighbor of end_vertex
            # 2. Not in used_mask
            # 3. Not adjacent to any other vertex in the path.
            #    adj_mask[c] & (used_mask ^ (1 << end_vertex)) == 0
            
            valid = []
            # Optimization: We can iterate neighbors of end_vertex
            for c in neighbors_local[end_vertex]:
                if used_mask & (1 << c):
                    continue
                # Check if c is adjacent to any other used vertex
                # used_mask without the end_vertex
                other_used = used_mask ^ (1 << end_vertex)
                if adj_mask_local[c] & other_used:
                    continue
                valid.append(c)
            return valid

        def count_onward_moves(candidate, end_vertex, used_mask):
            """
            Count valid moves from candidate if it were added.
            candidate will be the new end.
            The new used mask will be used_mask | (1 << candidate).
            The new end is candidate.
            The other end of the path is the opposite end, but for Warnsdorff we just look at neighbors of candidate.
            A neighbor w of candidate is valid if:
            1. w not in new_used
            2. w not adjacent to any vertex in path except candidate.
            """
            new_used = used_mask | (1 << candidate)
            other_used = new_used ^ (1 << candidate) # This is just used_mask
            # Wait, the path is the same, we just added candidate.
            # The vertices in the path are the old path + candidate.
            # So "other_used" for candidate's perspective is the old used_mask.
            
            count = 0
            for w in neighbors_local[candidate]:
                if new_used & (1 << w):
                    continue
                # Check if w is adjacent to any vertex in old path
                # old path vertices are in used_mask
                if adj_mask_local[w] & used_mask:
                    continue
                count += 1
            return count

        while True:
            # Time check
            if time.time() - start_time > DEADLINE:
                return path

            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end
            valid_left = get_valid_moves(0, left_end, used)
            
            # Get valid moves for right end
            valid_right = get_valid_moves(len(path)-1, right_end, used)
            
            if not valid_left and not valid_right:
                break
            
            # Evaluate scores
            best_moves_left = []
            min_score_left = float('inf')
            if valid_left:
                for v in valid_left:
                    score = count_onward_moves(v, left_end, used)
                    if score < min_score_left:
                        min_score_left = score
                        best_moves_left = [v]
                    elif score == min_score_left:
                        best_moves_left.append(v)
            
            best_moves_right = []
            min_score_right = float('inf')
            if valid_right:
                for v in valid_right:
                    score = count_onward_moves(v, right_end, used)
                    if score < min_score_right:
                        min_score_right = score
                        best_moves_right = [v]
                    elif score == min_score_right:
                        best_moves_right.append(v)

            if not valid_left:
                next_v = random.choice(best_moves_right)
                path.append(next_v)
                used |= (1 << next_v)
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                path.insert(0, next_v)
                used |= (1 << next_v)
            else:
                # Choose side based on score (tightness)
                # If scores are equal, random
                if min_score_left < min_score_right:
                    next_v = random.choice(best_moves_left)
                    path.insert(0, next_v)
                    used |= (1 << next_v)
                elif min_score_right < min_score_left:
                    next_v = random.choice(best_moves_right)
                    path.append(next_v)
                    used |= (1 << next_v)
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(best_moves_left)
                        path.insert(0, next_v)
                        used |= (1 << next_v)
                    else:
                        next_v = random.choice(best_moves_right)
                        path.append(next_v)
                        used |= (1 << next_v)

        return path

    # Multi-start
    # Use random start vertices to break symmetry
    # To improve diversity, we can also randomize the neighbor choice in the first step more aggressively
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        p = bidirectional_grow(start_vertex=v0)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        # Fallback
        return [0]

    return best_path

# EVOLVE-BLOCK-END
