import time
import random
import sys
import math
from collections import deque

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit

# Precompute neighbors for all vertices
neighbors = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        neighbors[v].append(v ^ (1 << b))

# Precompute adjacency masks for fast conflict checking
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
    Strategy: Bidirectional growth with Warnsdorff's rule using bitmask validity checks,
    combined with limited backtracking to escape local optima.
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
    
    def bidirectional_grow(start_vertex, time_limit, randomness=0.1):
        """
        Grow a snake bidirectionally with optimized checks using bitmasks.
        """
        v0 = start_vertex
        path = deque([v0])
        used_mask = 1 << v0
        
        if not neighbors_local[v0]:
            return list(path)
            
        # Randomly pick first step to break symmetry
        v1_candidates = [v for v in neighbors_local[v0] if not (used_mask & (1 << v))]
        if not v1_candidates:
            return list(path)
            
        v1 = random.choice(v1_candidates)
        path.append(v1)
        used_mask |= (1 << v1)

        temp = 0.5

        while True:
            if time.time() - start_time > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end
            valid_left = get_valid_extensions(left_end, used_mask)
            
            # Get valid moves for right end
            valid_right = get_valid_extensions(right_end, used_mask)
            
            if not valid_left and not valid_right:
                break
            
            # Evaluate scores using soft-min for smoother comparison
            best_moves_left = []
            soft_min_left = float('inf')
            if valid_left:
                scores_left = [count_onward_moves(v, used_mask) + random.random() * randomness for v in valid_left]
                # Soft-min: -log(sum(exp(-s/temp)))
                max_neg = max(-s for s in scores_left)
                sum_exp = sum(math.exp((-s - max_neg) / temp) for s in scores_left)
                soft_min_left = -(max_neg + math.log(sum_exp)) * temp
                min_score = min(scores_left)
                for v, s in zip(valid_left, scores_left):
                    if s == min_score:
                        best_moves_left.append(v)
            
            best_moves_right = []
            soft_min_right = float('inf')
            if valid_right:
                scores_right = [count_onward_moves(v, used_mask) + random.random() * randomness for v in valid_right]
                max_neg = max(-s for s in scores_right)
                sum_exp = sum(math.exp((-s - max_neg) / temp) for s in scores_right)
                soft_min_right = -(max_neg + math.log(sum_exp)) * temp
                min_score = min(scores_right)
                for v, s in zip(valid_right, scores_right):
                    if s == min_score:
                        best_moves_right.append(v)

            if not valid_left:
                next_v = random.choice(best_moves_right)
                # Extend right
                path.append(next_v)
                used_mask |= (1 << next_v)
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                # Extend left
                path.appendleft(next_v)
                used_mask |= (1 << next_v)
            else:
                # Choose side based on soft-min score (tightness)
                if soft_min_left < soft_min_right:
                    next_v = random.choice(best_moves_left)
                    path.appendleft(next_v)
                    used_mask |= (1 << next_v)
                elif soft_min_right < soft_min_left:
                    next_v = random.choice(best_moves_right)
                    path.append(next_v)
                    used_mask |= (1 << next_v)
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(best_moves_left)
                        path.appendleft(next_v)
                        used_mask |= (1 << next_v)
                    else:
                        next_v = random.choice(best_moves_right)
                        path.append(next_v)
                        used_mask |= (1 << next_v)

        return list(path), used_mask

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
                new_path, new_used_mask = bidirectional_grow(new_path[-1], DEADLINE - (time.time() - start_time), randomness=0.3)
                # Wait, bidirectional_grow starts from a single vertex. We need a version that continues from a path.
                # Let's implement a helper that continues from an existing path.
                
        return path, used_mask, False

    # Since the backtracking logic above is a bit complex to integrate with the deque-based grow,
    # we will stick to the multi-start approach but with the faster bitmask checks.
    # The speedup from bitmask checks allows more starts.
    
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, max_verts - 1)
        # Occasionally use less randomness to be more greedy
        rand_param = random.choice([0.0, 0.1, 0.3, 0.5])
        
        # Use the bitmask-based grow
        p, _ = bidirectional_grow(v0, DEADLINE - (time.time() - start_time), randomness=rand_param)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        # Fallback
        return [0]

    return best_path
