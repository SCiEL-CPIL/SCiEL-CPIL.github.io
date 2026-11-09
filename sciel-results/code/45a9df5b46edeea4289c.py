import time
import random

N = 9

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_path = []
    
    # Helper functions defined inside construct to capture N and deadline
    
    def get_candidates(v, path, used, excluded_idx):
        """
        Get valid neighbors of v to extend the path.
        excluded_idx is the index of the vertex in path that v is adjacent to.
        """
        candidates = []
        for b in range(N):
            nb = v ^ (1 << b)
            if nb in used:
                continue
            
            # Check if nb can be added
            # The "excluded" vertex for nb will be v.
            # So we check if nb is adjacent to any other vertex in path.
            # This is the same logic as can_add, but we are checking nb.
            
            # Inline check for speed
            valid = True
            for b2 in range(N):
                nb2 = nb ^ (1 << b2)
                if nb2 == v:
                    continue
                if nb2 in used:
                    valid = False
                    break
            
            if valid:
                candidates.append(nb)
        return candidates

    def score_candidate_with_prev(v, prev, path, used):
        """
        Score candidate v given we are coming from prev.
        Returns the number of valid neighbors of v (excluding prev).
        """
        count = 0
        for b in range(N):
            nb = v ^ (1 << b)
            if nb == prev:
                continue
            if nb in used:
                continue
            
            # Check if nb is a valid extension
            # nb must not be adjacent to any other vertex in path
            valid = True
            for b2 in range(N):
                nb2 = nb ^ (1 << b2)
                if nb2 == v:
                    continue
                if nb2 in used:
                    valid = False
                    break
            
            if valid:
                count += 1
        return count

    def greedy_bidirectional(start_vertex, randomize_ties=True):
        path = [start_vertex]
        used = {start_vertex}
        
        while True:
            if time.time() > deadline:
                return path
            
            # Try left
            left_end = path[0]
            left_cands = get_candidates(left_end, path, used, 0)
            
            # Try right
            right_end = path[-1]
            right_cands = get_candidates(right_end, path, used, len(path)-1)
            
            if not left_cands and not right_cands:
                return path
            
            # Decide which end to extend
            if not left_cands:
                # Must extend right
                # Pick best candidate from right_cands
                best_cand = None
                best_score = float('inf')
                best_cands = []
                for c in right_cands:
                    s = score_candidate_with_prev(c, right_end, path, used)
                    if s < best_score:
                        best_score = s
                        best_cands = [c]
                    elif s == best_score:
                        best_cands.append(c)
                
                if randomize_ties and len(best_cands) > 1:
                    best_cand = random.choice(best_cands)
                else:
                    best_cand = best_cands[0]
                    
                path.append(best_cand)
                used.add(best_cand)
                continue
            
            if not right_cands:
                # Must extend left
                best_cand = None
                best_score = float('inf')
                best_cands = []
                for c in left_cands:
                    s = score_candidate_with_prev(c, left_end, path, used)
                    if s < best_score:
                        best_score = s
                        best_cands = [c]
                    elif s == best_score:
                        best_cands.append(c)
                
                if randomize_ties and len(best_cands) > 1:
                    best_cand = random.choice(best_cands)
                else:
                    best_cand = best_cands[0]
                    
                path.insert(0, best_cand)
                used.add(best_cand)
                continue
            
            # Both have candidates
            # Strategy: Pick the end that has the candidate with the lowest score.
            
            best_left_cand = None
            best_left_score = float('inf')
            best_left_cands = []
            for c in left_cands:
                s = score_candidate_with_prev(c, left_end, path, used)
                if s < best_left_score:
                    best_left_score = s
                    best_left_cands = [c]
                elif s == best_left_score:
                    best_left_cands.append(c)
            
            best_right_cand = None
            best_right_score = float('inf')
            best_right_cands = []
            for c in right_cands:
                s = score_candidate_with_prev(c, right_end, path, used)
                if s < best_right_score:
                    best_right_score = s
                    best_right_cands = [c]
                elif s == best_right_score:
                    best_right_cands.append(c)
            
            # Compare
            if best_left_score < best_right_score:
                if randomize_ties and len(best_left_cands) > 1:
                    best_left_cand = random.choice(best_left_cands)
                else:
                    best_left_cand = best_left_cands[0]
                path.insert(0, best_left_cand)
                used.add(best_left_cand)
            elif best_right_score < best_left_score:
                if randomize_ties and len(best_right_cands) > 1:
                    best_right_cand = random.choice(best_right_cands)
                else:
                    best_right_cand = best_right_cands[0]
                path.append(best_right_cand)
                used.add(best_right_cand)
            else:
                # Tie in score. 
                # Randomly pick an end?
                if randomize_ties:
                    if random.random() < 0.5:
                        if len(best_left_cands) > 1:
                            best_left_cand = random.choice(best_left_cands)
                        else:
                            best_left_cand = best_left_cands[0]
                        path.insert(0, best_left_cand)
                        used.add(best_left_cand)
                    else:
                        if len(best_right_cands) > 1:
                            best_right_cand = random.choice(best_right_cands)
                        else:
                            best_right_cand = best_right_cands[0]
                        path.append(best_right_cand)
                        used.add(best_right_cand)
                else:
                    # Deterministic fallback
                    if len(left_cands) <= len(right_cands):
                        best_left_cand = best_left_cands[0]
                        path.insert(0, best_left_cand)
                        used.add(best_left_cand)
                    else:
                        best_right_cand = best_right_cands[0]
                        path.append(best_right_cand)
                        used.add(best_right_cand)

    # Run multiple times with different random starts
    # We can also vary the random seed for tie-breaking
    
    for i in range(200): # Number of restarts
        if time.time() > deadline:
            break
        
        # Random start vertex
        start_v = random.randint(0, (1 << N) - 1)
        
        path = greedy_bidirectional(start_v, randomize_ties=True)
        
        if len(path) > len(best_path):
            best_path = path
    
    return best_path
