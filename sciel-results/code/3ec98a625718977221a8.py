"""Improved Snake-in-the-Box solver for Q_9 using Greedy DFS with Lookahead."""
N = 9
SIZE = 1 << N
import time
import random
import sys

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_path = []
    best_len = 0
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(SIZE)]
    for v in range(SIZE):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Global variables for the DFS
    path = []
    used = [False] * SIZE
    block_count = [0] * SIZE
    time_limit = deadline

    def add_vertex(v):
        path.append(v)
        used[v] = True
        for w in neighbors[v]:
            block_count[w] += 1

    def remove_vertex():
        v = path.pop()
        used[v] = False
        for w in neighbors[v]:
            block_count[w] -= 1

    def score_candidate(v):
        """
        Heuristic score for a candidate vertex v.
        We count how many of v's neighbors would be valid next steps if v were added.
        A neighbor w is valid if:
        1. not used[w]
        2. block_count[w] == 1 (after adding v, block_count[w] will be incremented if w is neighbor of v)
        
        Wait, block_count is updated *after* adding v.
        So, to score v, we simulate the update:
        For each w in neighbors[v]:
            new_block_count_w = block_count[w] + (1 if w == v else 0) # No, v is not neighbor of itself.
            # Actually, adding v increments block_count for all w in neighbors[v].
            # So for a neighbor w of v, its new block count is block_count[w] + 1.
            # For a non-neighbor w, its block count is unchanged.
            
            # We want to know if w is a valid *next* step from v.
            # w is a valid next step from v if:
            # - w is a neighbor of v (which it is, by definition of the loop)
            # - not used[w]
            # - The only neighbor of w in the path (including v) is v.
            #   This means block_count[w] (before adding v) must be 0.
            #   Because after adding v, block_count[w] becomes 1.
            #   If block_count[w] was > 0, it becomes > 1, and w is invalid.
            
            # So, count w in neighbors[v] such that not used[w] and block_count[w] == 0.
        """
        score = 0
        for w in neighbors[v]:
            if not used[w] and block_count[w] == 0:
                score += 1
        return score

    def dfs():
        nonlocal best_path, best_len
        
        if time.time() > time_limit:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        if len(path) == SIZE:
            return
            
        cur = path[-1]
        
        # Get valid candidates
        candidates = []
        for v in neighbors[cur]:
            if not used[v] and block_count[v] == 1:
                candidates.append(v)
                
        if not candidates:
            return
            
        # Score and sort candidates
        # We want to try the most flexible ones first
        # To add randomness, we can shuffle ties or add noise
        scored_candidates = [(score_candidate(v), v) for v in candidates]
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Add some randomness to the top candidates to avoid deterministic loops
        # If there are many candidates with the same high score, shuffle them
        # Simple approach: shuffle the list, then stable sort by score?
        # Or just shuffle if the top score is common.
        
        # Let's just shuffle the candidates if there are more than 1,
        # but bias the shuffle towards high scores?
        # Simpler: Just shuffle the candidates, then pick the one with max score among the first K?
        # No, standard Greedy DFS: Sort by score, pick first.
        # To avoid getting stuck in local optima, we can randomize the tie-breaking.
        
        # Let's group by score and pick randomly from the highest score group.
        if scored_candidates:
            max_score = scored_candidates[0][0]
            top_candidates = [v for s, v in scored_candidates if s == max_score]
            if len(top_candidates) > 1:
                random.shuffle(top_candidates)
                v = top_candidates[0]
            else:
                v = top_candidates[0]
        else:
            return
            
        # Add v
        add_vertex(v)
        
        dfs()
        
        # Backtrack
        remove_vertex()

    # Run multiple restarts
    # Each restart is a full DFS with the heuristic.
    # The DFS will explore the most promising path first.
    # If it gets stuck, it backtracks.
    # We limit the time for each restart?
    # Or just run until the global deadline.
    
    # Since DFS can be deep, we run it until time is up or it finishes.
    # If it finishes (finds a Hamiltonian path), we stop.
    # Otherwise, we start a new random restart.
    
    num_restarts = 0
    while time.time() < deadline - 1.0:
        num_restarts += 1
        # Reset state
        path = []
        used = [False] * SIZE
        block_count = [0] * SIZE
        
        start = random.randint(0, SIZE - 1)
        path.append(start)
        used[start] = True
        for w in neighbors[start]:
            block_count[w] += 1
            
        dfs()
        
        # If we found a very long path, we might want to stop early?
        # No, keep trying to improve.
        
        # If the DFS finished quickly (stuck early), restart immediately.
        # If it took a long time, it might be close to the limit.
        
        # Check if we should break
        if time.time() > deadline - 5.0:
            break
            
    if not best_path:
        return [0]
        
    return best_path
