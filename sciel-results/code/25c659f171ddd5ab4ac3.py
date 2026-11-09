import time
import random
import sys
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
            
            # Evaluate scores (Warnsdorff: min onward moves)
            best_moves_left = []
            min_score_left = float('inf')
            if valid_left:
                for v in valid_left:
                    score = count_onward_moves(v)
                    # Add some randomness
                    score += random.random() * randomness
                    if score < min_score_left:
                        min_score_left = score
                        best_moves_left = [v]
                    elif score == min_score_left:
                        best_moves_left.append(v)
            
            best_moves_right = []
            min_score_right = float('inf')
            if valid_right:
                for v in valid_right:
                    score = count_onward_moves(v)
                    score += random.random() * randomness
                    if score < min_score_right:
                        min_score_right = score
                        best_moves_right = [v]
                    elif score == min_score_right:
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
                # Choose side based on score (tightness)
                # If scores are equal, random
                if min_score_left < min_score_right:
                    next_v = random.choice(best_moves_left)
                    path.appendleft(next_v)
                    used[next_v] = True
                    for w in neighbors[next_v]:
                        deg_in_path[w] += 1
                elif min_score_right < min_score_left:
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
        rand_param = random.choice([0.0, 0.1, 0.5])
        p = bidirectional_grow(start_vertex=v0, time_limit=DEADLINE - (time.time() - start_time), randomness=rand_param)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        # Fallback
        return [0]

    return best_path

# EVOLVE-BLOCK-END
