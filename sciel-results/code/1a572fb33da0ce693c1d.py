import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
NUM_SUBSETS = 1 << N

# Precompute bit indices for fast lsb extraction
# Or just use bit_length
def construct():
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin
    
    best_trans_a = [0] * N
    best_trans_b = [0] * N
    best_score = -1
    
    # Helper to compute the shortest reset word length (threshold)
    def compute_threshold(trans_a, trans_b):
        # Precompute next states for all subsets
        # next_a[M] is the subset reached from M by applying trans_a
        next_a = [0] * NUM_SUBSETS
        next_b = [0] * NUM_SUBSETS
        
        # next_a[0] = 0
        for M in range(1, NUM_SUBSETS):
            lsb = M & -M
            i = lsb.bit_length() - 1
            rest = M ^ lsb
            next_a[M] = next_a[rest] | (1 << trans_a[i])
            next_b[M] = next_b[rest] | (1 << trans_b[i])
        
        # BFS from FULL_MASK to any singleton
        # visited[M] = distance from Full to M
        visited = [-1] * NUM_SUBSETS
        visited[FULL_MASK] = 0
        q = deque([FULL_MASK])
        
        # A mask is a singleton if M & (M-1) == 0 and M != 0
        while q:
            curr = q.popleft()
            dist = visited[curr]
            
            # Check if singleton
            if curr != 0 and (curr & (curr - 1)) == 0:
                return dist
            
            # Explore neighbors
            next_mask_a = next_a[curr]
            if visited[next_mask_a] == -1:
                visited[next_mask_a] = dist + 1
                q.append(next_mask_a)
                
            next_mask_b = next_b[curr]
            if visited[next_mask_b] == -1:
                visited[next_mask_b] = dist + 1
                q.append(next_mask_b)
                
        # If we exit the loop, it's not synchronizing
        return -1

    # Initialize with a random synchronizing automaton
    # Strategy: Random mappings, but ensure synchronization if possible
    # Actually, just start random and hope it syncs, or use a known base and perturb.
    # A simple cycle + reset is safe but low score.
    # Let's start with a random automaton. If it doesn't sync, threshold is -1.
    # We want to maximize threshold, so we only care about sync ones.
    
    # To ensure we start with a valid (synchronizing) automaton, 
    # we can start with the seed structure and mutate.
    cur_a = [(i + 1) % N for i in range(N)]
    cur_b = [0] * N
    
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score > best_score:
        best_score = cur_score
        best_trans_a = cur_a[:]
        best_trans_b = cur_b[:]
        
    iterations = 0
    while time.time() - start_time < deadline:
        iterations += 1
        
        # Mutate: pick a random state and a random letter, change its destination
        state = random.randint(0, N - 1)
        letter = random.randint(0, 1)
        
        old_dest = cur_a[state] if letter == 0 else cur_b[state]
        new_dest = random.randint(0, N - 1)
        
        if letter == 0:
            cur_a[state] = new_dest
        else:
            cur_b[state] = new_dest
            
        new_score = compute_threshold(cur_a, cur_b)
        
        if new_score > best_score:
            best_score = new_score
            best_trans_a = cur_a[:]
            best_trans_b = cur_b[:]
            # If we found a very high score, we might want to stick around
            # but keep exploring
            
        # Acceptance strategy:
        # If new_score > cur_score: Accept (Hill Climbing)
        # If new_score < cur_score: Revert (Strict Hill Climbing)
        # To escape local maxima, we can do Random Restarts or Simulated Annealing.
        # Given the landscape, strict hill climbing with random restarts is robust.
        
        if new_score < cur_score:
            # Revert
            if letter == 0:
                cur_a[state] = old_dest
            else:
                cur_b[state] = old_dest
        else:
            cur_score = new_score
            
        # Occasionally do a random restart to escape local optima
        if iterations % 500 == 0:
            # Generate a new random automaton
            # Bias towards synchronizing structures?
            # Simple random often fails to sync.
            # Let's try a "random cycle + noise" or just random.
            # Actually, if we revert, we stay in the current local optimum.
            # Let's force a restart every 500 steps to explore new basins.
            cur_a = [random.randint(0, N - 1) for _ in range(N)]
            cur_b = [random.randint(0, N - 1) for _ in range(N)]
            cur_score = compute_threshold(cur_a, cur_b)
            if cur_score > best_score:
                best_score = cur_score
                best_trans_a = cur_a[:]
                best_trans_b = cur_b[:]
            # If it doesn't sync, cur_score is -1. We will just keep mutating from this invalid state?
            # Better: if invalid, revert to best known? Or just keep mutating hoping it becomes valid.
            # Mutating an invalid automaton might make it valid.
            
    return [best_trans_a, best_trans_b]
