import time
import random
import sys
from collections import deque

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit

# Precompute neighbors
neighbors = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

# Precompute adjacency masks
adj_mask = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in neighbors[v]:
        m |= (1 << nb)
    adj_mask[v] = m

def construct():
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Local variables for speed
    neighbors_local = neighbors
    adj_mask_local = adj_mask
    max_verts = MAX_VERTS
    
    def get_valid_extensions(end_vertex, used_mask):
        """
        Get valid candidates to extend from end_vertex.
        A candidate v is valid if:
        1. v is not in used_mask
        2. v is not adjacent to any vertex in used_mask except end_vertex
        """
        valid = []
        end_bit = 1 << end_vertex
        other_used = used_mask ^ end_bit
        for c in neighbors_local[end_vertex]:
            if used_mask & (1 << c):
                continue
            if adj_mask_local[c] & other_used:
                continue
            valid.append(c)
        return valid

    def count_onward_moves(candidate, used_mask):
        """
        Count valid moves from candidate if it were added to the path.
        used_mask is the mask BEFORE adding candidate.
        The new end is candidate.
        A neighbor w of candidate is a valid onward move if:
        1. w is not in used_mask
        2. w is not adjacent to any vertex in used_mask
        (Note: w is adjacent to candidate, but candidate is not in used_mask yet, 
         so we just check against current used_mask)
        """
        count = 0
        for w in neighbors_local[candidate]:
            if used_mask & (1 << w):
                continue
            if adj_mask_local[w] & used_mask:
                continue
            count += 1
        return count

    def greedy_grow(start_v0, start_v1, time_limit):
        path = deque([start_v0, start_v1])
        used_mask = (1 << start_v0) | (1 << start_v1)
        
        while True:
            if time.time() > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            valid_left = get_valid_extensions(left_end, used_mask)
            valid_right = get_valid_extensions(right_end, used_mask)
            
            if not valid_left and not valid_right:
                break
            
            # Determine which side to extend
            if not valid_left:
                side = 'right'
                valid = valid_right
            elif not valid_right:
                side = 'left'
                valid = valid_left
            else:
                # Warnsdorff's rule: pick move with fewest onward moves
                # Add slight randomization to avoid getting stuck in local minima
                min_score_left = min(count_onward_moves(v, used_mask) for v in valid_left)
                min_score_right = min(count_onward_moves(v, used_mask) for v in valid_right)
                
                # Randomize choice of side if scores are close
                if min_score_left < min_score_right:
                    side = 'left'
                    valid = valid_left
                elif min_score_right < min_score_left:
                    side = 'right'
                    valid = valid_right
                else:
                    if random.random() < 0.5:
                        side = 'left'
                        valid = valid_left
                    else:
                        side = 'right'
                        valid = valid_right
            
            if not valid:
                break
                
            # Pick best move from valid
            # Sort by onward moves, then pick randomly from top few for diversity
            scored = [(count_onward_moves(v, used_mask), v) for v in valid]
            scored.sort(key=lambda x: x[0])
            
            # Pick from top 3 or random if many ties
            top_k = min(3, len(scored))
            # Weighted random selection from top_k
            # Simple: pick random from top_k
            next_v = scored[random.randint(0, top_k - 1)][1]
            
            # Update
            new_bit = 1 << next_v
            used_mask |= new_bit
            
            if side == 'left':
                path.appendleft(next_v)
            else:
                path.append(next_v)
                
        return list(path)

    # Multi-start
    # We can run many starts now due to bitset speed
    while time.time() - start_time < DEADLINE:
        # Random start
        v0 = random.randint(0, max_verts - 1)
        v1 = random.choice(neighbors_local[v0])
        
        # Run greedy growth
        p = greedy_grow(v0, v1, DEADLINE)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        return [0]
        
    return best_path
