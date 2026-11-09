import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute adjacency masks for all vertices
    # adj_mask[v] is a bitmask where bit u is 1 if u is adjacent to v
    adj_mask = [0] * MAX_VERTS
    for v in range(MAX_VERTS):
        mask = 0
        for b in range(N):
            u = v ^ (1 << b)
            mask |= (1 << u)
        adj_mask[v] = mask

    best_path = [0, 1]
    best_len = 2

    def solve_once():
        """
        Runs one instance of bidirectional growth with Warnsdorff heuristic.
        Returns the path found.
        """
        if time.time() - start_time > DEADLINE:
            return []
            
        # Initialize path with two random adjacent vertices
        v0 = random.randint(0, MAX_VERTS - 1)
        # Pick a random neighbor
        # Find set bits in adj_mask[v0]
        neighbors_v0 = []
        mask = adj_mask[v0]
        while mask:
            # Get index of lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neighbors_v0.append(idx)
            mask ^= lsb
        v1 = random.choice(neighbors_v0)
        
        path = [v0, v1]
        used_mask = (1 << v0) | (1 << v1)
        
        # To make path extension efficient, we maintain the path as a list.
        # Extending at the end is O(1). Extending at the beginning is O(N).
        # Since N is small (max ~200), O(N) insertion is acceptable but we can optimize 
        # by always growing the right end and reversing at the end if needed?
        # No, bidirectional is key. Let's just use list insert/append.
        
        while True:
            if time.time() - start_time > DEADLINE:
                break
                
            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end
            # A move v is valid if:
            # 1. v is neighbor of left_end (bit set in adj_mask[left_end])
            # 2. v is not used (bit NOT set in used_mask)
            # 3. v is not adjacent to any OTHER used vertex.
            #    This means: adj_mask[v] & (used_mask ^ (1 << left_end)) == 0
            
            left_candidates = []
            right_candidates = []
            
            # Helper to get valid candidates from an end vertex
            def get_valids(end_v):
                # Start with all neighbors of end_v
                avail = adj_mask[end_v] & ~used_mask
                
                cands = []
                m = avail
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    
                    # Check condition 3: v must not be adjacent to any used vertex except end_v
                    # other_used = used_mask without end_v
                    if adj_mask[v] & (used_mask ^ (1 << end_v)) == 0:
                        cands.append(v)
                return cands

            left_cands = get_valids(left_end)
            right_cands = get_valids(right_end)
            
            if not left_cands and not right_cands:
                break
            
            # Score candidates using Warnsdorff's rule
            # Score = number of unvisited neighbors
            def get_scores(cands, end_v):
                scores = []
                for v in cands:
                    # Count unvisited neighbors of v
                    # neighbors of v: adj_mask[v]
                    # unvisited: ~used_mask
                    # Note: v itself is not in its own adj_mask
                    score = bin(adj_mask[v] & ~used_mask).count('1')
                    scores.append(score)
                return scores

            left_scores = []
            right_scores = []
            
            if left_cands:
                left_scores = get_scores(left_cands, left_end)
            if right_cands:
                right_scores = get_scores(right_cands, right_end)
            
            # Choose side and move
            chosen_move = None
            chosen_side = None
            
            if not left_cands:
                chosen_side = 'right'
                # Pick min score
                min_s = min(right_scores)
                # Tie-breaking: random among min scores
                min_cands = [c for c, s in zip(right_cands, right_scores) if s == min_s]
                chosen_move = random.choice(min_cands)
            elif not right_cands:
                chosen_side = 'left'
                min_s = min(left_scores)
                min_cands = [c for c, s in zip(left_cands, left_scores) if s == min_s]
                chosen_move = random.choice(min_cands)
            else:
                # Both sides have moves
                min_left_s = min(left_scores)
                min_right_s = min(right_scores)
                
                # Heuristic: Prefer the side with the lower minimum score (most constrained)
                # If tie, random side
                if min_left_s < min_right_s:
                    chosen_side = 'left'
                    min_cands = [c for c, s in zip(left_cands, left_scores) if s == min_left_s]
                    chosen_move = random.choice(min_cands)
                elif min_right_s < min_left_s:
                    chosen_side = 'right'
                    min_cands = [c for c, s in zip(right_cands, right_scores) if s == min_right_s]
                    chosen_move = random.choice(min_cands)
                else:
                    # Tie in min scores. 
                    # Maybe prefer the side with fewer total options? Or random?
                    # Let's pick randomly
                    if random.random() < 0.5:
                        chosen_side = 'left'
                        min_cands = [c for c, s in zip(left_cands, left_scores) if s == min_left_s]
                        chosen_move = random.choice(min_cands)
                    else:
                        chosen_side = 'right'
                        min_cands = [c for c, s in zip(right_cands, right_scores) if s == min_right_s]
                        chosen_move = random.choice(min_cands)
            
            # Execute move
            if chosen_side == 'left':
                path.insert(0, chosen_move)
            else:
                path.append(chosen_move)
            
            used_mask |= (1 << chosen_move)
            
        return path

    # Run multiple attempts
    attempts = 0
    while time.time() - start_time < DEADLINE:
        path = solve_once()
        if path and len(path) > best_len:
            best_len = len(path)
            best_path = path
        attempts += 1
        
        # Safety check: if we've been running for a while, ensure we don't exceed hard limit
        if time.time() - start_time > DEADLINE:
            break

    return best_path
