import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit

# Precompute neighbors
neighbors = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

# Precompute adjacency masks for conflict checking
# adj_mask[v] is a bitset of all vertices adjacent to v
adj_mask = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in neighbors[v]:
        m |= (1 << nb)
    adj_mask[v] = m


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Parallel multi-start greedy with backtracking and Warnsdorff's rule.
    Uses bitmasks for fast validity checks.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    neighbors_local = neighbors
    adj_mask_local = adj_mask
    max_verts = MAX_VERTS
    
    def get_valid_extensions(end_vertex, used_mask):
        """Get list of valid candidates to extend from end_vertex."""
        valid = []
        for c in neighbors_local[end_vertex]:
            if used_mask & (1 << c):
                continue
            # c must not be adjacent to any other vertex in the path except end_vertex
            other_used = used_mask ^ (1 << end_vertex)
            if adj_mask_local[c] & other_used:
                continue
            valid.append(c)
        return valid
    
    def count_onward_moves(candidate, used_mask):
        """Count valid moves from candidate if it were added."""
        new_used = used_mask | (1 << candidate)
        count = 0
        for w in neighbors_local[candidate]:
            if new_used & (1 << w):
                continue
            if adj_mask_local[w] & (used_mask ^ (1 << candidate)):
                continue
            count += 1
        return count
    
    def greedy_grow(path, used_mask, max_iters=500, random_factor=0.2):
        """
        Grow path greedily using Warnsdorff's rule with some randomization.
        Returns (path, used_mask, length).
        """
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            valid_left = get_valid_extensions(left_end, used_mask)
            valid_right = get_valid_extensions(right_end, used_mask)
            
            if not valid_left and not valid_right:
                break
            
            # Choose which end to extend
            if not valid_left:
                side = 'right'
            elif not valid_right:
                side = 'left'
            else:
                # Score both ends
                min_left = min(count_onward_moves(v, used_mask) for v in valid_left)
                min_right = min(count_onward_moves(v, used_mask) for v in valid_right)
                
                if min_left < min_right:
                    side = 'left'
                elif min_right < min_left:
                    side = 'right'
                else:
                    side = 'left' if random.random() < 0.5 else 'right'
            
            # Get valid moves for chosen side
            if side == 'left':
                valid = valid_left
            else:
                valid = valid_right
            
            if not valid:
                break
            
            # Score candidates
            scored = [(count_onward_moves(v, used_mask), v) for v in valid]
            scored.sort(key=lambda x: x[0])
            
            # Pick with some randomization
            if random.random() < random_factor:
                # Pick from top half
                top_half = max(1, len(scored) // 2)
                idx = random.randint(0, top_half - 1)
            else:
                idx = 0
            
            next_v = scored[idx][1]
            new_used = used_mask | (1 << next_v)
            
            if side == 'left':
                path.insert(0, next_v)
            else:
                path.append(next_v)
            
            used_mask = new_used
        
        return path, used_mask, len(path)
    
    def backtrack_and_retry(path, used_mask, depth=3):
        """
        When stuck, try backtracking to a previous branching point and try a different move.
        Returns (new_path, new_used_mask, success).
        """
        if len(path) < 2:
            return path, used_mask, False
        
        # Try backtracking from the end
        for i in range(min(depth, len(path) - 1)):
            # Remove last i+1 vertices and try alternative
            remove_count = i + 1
            body = path[:-remove_count] if remove_count < len(path) else []
            if not body:
                continue
                
            body_used = 0
            for v in body:
                body_used |= (1 << v)
                
            end_vertex = body[-1]
            valid = []
            for c in neighbors_local[end_vertex]:
                if body_used & (1 << c):
                    continue
                if adj_mask_local[c] & (body_used ^ (1 << end_vertex)):
                    continue
                valid.append(c)
                
            if valid:
                random.shuffle(valid)
                next_v = valid[0]
                new_used = body_used | (1 << next_v)
                new_path = body + [next_v]
                # Try to regrow
                new_path, new_used, new_len = greedy_grow(new_path, new_used, max_iters=50, random_factor=0.3)
                return new_path, new_used, True
        
        return path, used_mask, False
    
    # Multi-start with backtracking
    num_starts = 0
    
    while time.time() - start_time < DEADLINE:
        num_starts += 1
        
        # Initialize with a random starting edge
        v0 = random.randint(0, max_verts - 1)
        path = [v0]
        used = 1 << v0
        
        v1 = random.choice(neighbors_local[v0])
        path.append(v1)
        used |= (1 << v1)
        
        # Greedy growth with backtracking
        current_len = len(path)
        stuck_count = 0
        
        for _ in range(500):  # Max iterations per start
            if time.time() - start_time > DEADLINE:
                break
            
            path, used, current_len = greedy_grow(path, used, max_iters=100, random_factor=0.1)
            
            if current_len > best_len:
                best_len = current_len
                best_path = path[:]
            
            # Check if stuck
            left_end = path[0]
            right_end = path[-1]
            valid_left = get_valid_extensions(left_end, used)
            valid_right = get_valid_extensions(right_end, used)
            
            if not valid_left and not valid_right:
                stuck_count += 1
                if stuck_count > 3:
                    # Try backtracking
                    new_path, new_used, success = backtrack_and_retry(path, used, depth=3)
                    if success and len(new_path) > 0:
                        path = new_path
                        used = new_used
                        stuck_count = 0
                    else:
                        break
            else:
                stuck_count = 0
        
    if best_path is None:
        # Fallback
        return [0]

    return best_path
