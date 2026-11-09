import time
import random
import sys
import math
from collections import deque

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Optimized Bidirectional growth with Warnsdorff's rule using O(N) validity checks.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    def bidirectional_grow(start_vertex=0, time_limit=None, randomness=0.1):
        """
        Grow a snake bidirectionally with optimized checks.
        Uses deg_in_path array for O(N) validity check.
        Uses deque for O(1) append/appendleft.
        Uses soft-min (log-sum-exp) for smoother score comparison.
        """
        v0 = start_vertex
        path = deque([v0])
        used = [False] * MAX_VERTS
        used[v0] = True
        
        # deg_in_path[v] = number of neighbors of v that are in the current path
        deg_in_path = [0] * MAX_VERTS
        
        # Initialize deg_in_path for the start vertex
        for w in neighbors[v0]:
            deg_in_path[w] += 1

        if not neighbors[v0]:
            return list(path)
            
        # Randomly pick first step to break symmetry
        v1_candidates = [v for v in neighbors[v0] if not used[v]]
        if not v1_candidates:
            return list(path)
            
        v1 = random.choice(v1_candidates)
        path.append(v1)
        used[v1] = True
        
        # Update deg_in_path for v1's neighbors
        for w in neighbors[v1]:
            deg_in_path[w] += 1

        # Helper to calculate onward moves count for a candidate v
        def count_onward_moves(candidate):
            count = 0
            for w in neighbors[candidate]:
                if not used[w] and deg_in_path[w] == 0:
                    count += 1
            return count

        # Soft-min temperature for smoothing
        temp = 0.5

        while True:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end
            valid_left = []
            for v in neighbors[left_end]:
                if not used[v] and deg_in_path[v] == 1:
                    valid_left.append(v)
            
            # Get valid moves for right end
            valid_right = []
            for v in neighbors[right_end]:
                if not used[v] and deg_in_path[v] == 1:
                    valid_right.append(v)
            
            if not valid_left and not valid_right:
                break
            
            # Evaluate scores using soft-min (log-sum-exp) for smoother comparison
            # We want to pick the move with minimum onward moves, but with some smoothing
            best_moves_left = []
            soft_min_left = float('inf')
            if valid_left:
                scores_left = [count_onward_moves(v) + random.random() * randomness for v in valid_left]
                # Soft-min: -log(sum(exp(-s/temp)))
                max_neg = max(-s for s in scores_left)
                sum_exp = sum(math.exp((-s - max_neg) / temp) for s in scores_left)
                soft_min_left = -(max_neg + math.log(sum_exp)) * temp
                # Find argmin of scores for tie-breaking
                min_score = min(scores_left)
                for v, s in zip(valid_left, scores_left):
                    if s == min_score:
                        best_moves_left.append(v)
            
            best_moves_right = []
            soft_min_right = float('inf')
            if valid_right:
                scores_right = [count_onward_moves(v) + random.random() * randomness for v in valid_right]
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
                used[next_v] = True
                for w in neighbors[next_v]:
                    deg_in_path[w] += 1
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                # Extend left
                path.appendleft(next_v)
                used[next_v] = True
                for w in neighbors[next_v]:
                    deg_in_path[w] += 1
            else:
                # Choose side based on soft-min score (tightness)
                # Lower soft-min means tighter (better)
                if soft_min_left < soft_min_right:
                    next_v = random.choice(best_moves_left)
                    path.appendleft(next_v)
                    used[next_v] = True
                    for w in neighbors[next_v]:
                        deg_in_path[w] += 1
                elif soft_min_right < soft_min_left:
                    next_v = random.choice(best_moves_right)
                    path.append(next_v)
                    used[next_v] = True
                    for w in neighbors[next_v]:
                        deg_in_path[w] += 1
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(best_moves_left)
                        path.appendleft(next_v)
                        used[next_v] = True
                        for w in neighbors[next_v]:
                            deg_in_path[w] += 1
                    else:
                        next_v = random.choice(best_moves_right)
                        path.append(next_v)
                        used[next_v] = True
                        for w in neighbors[next_v]:
                            deg_in_path[w] += 1

        return list(path)

    # Multi-start
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        # Occasionally use less randomness to be more greedy
        rand_param = random.choice([0.0, 0.1, 0.3, 0.5])
        p = bidirectional_grow(start_vertex=v0, time_limit=DEADLINE - (time.time() - start_time), randomness=rand_param)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        # Fallback
        return [0]

    return best_path

# EVOLVE-BLOCK-END
