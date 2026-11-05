"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest snake in Q_9.
    Uses a greedy approach with lookahead heuristic and optimized 'forbidden neighbors' set.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_len = 0
    best_path = []
    
    # Precompute neighbors for all vertices to speed up iteration
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(512)]
    
    # Precompute bit_count for XOR differences (Hamming distance)
    # dist[a][b] = popcount(a ^ b)
    dist = [[(a ^ b).bit_count() for b in range(512)] for a in range(512)]
    
    def _score_candidate(nxt, cur, used, blocked):
        """
        Score a candidate next vertex. Higher is better.
        Heuristic: prefer vertices that have more valid future options.
        """
        # Count how many of nxt's neighbors are still available (not used, not blocked, not cur)
        # This approximates how many steps we can continue from nxt
        score = 0
        for nb in neighbors[nxt]:
            if nb == cur:
                continue
            if nb not in used and nb not in blocked:
                score += 1
        return score
    
    def _greedy_with_lookahead(start, use_lookahead=True):
        """
        Run greedy extension from start vertex.
        If use_lookahead, pick the candidate with highest score; else random.
        """
        path = [start]
        used = {start}
        blocked = set()
        
        while True:
            if time.time() > deadline:
                break
                
            cur = path[-1]
            candidates = []
            for nb in neighbors[cur]:
                if nb not in used and nb not in blocked:
                    candidates.append(nb)
            
            if not candidates:
                break
            
            if use_lookahead and len(candidates) > 1:
                # Pick candidate with best lookahead score
                best_score = -1
                best_nxt = None
                for cand in candidates:
                    s = _score_candidate(cand, cur, used, blocked)
                    if s > best_score:
                        best_score = s
                        best_nxt = cand
                # Add some randomness: with 30% probability pick random instead
                if random.random() < 0.3:
                    nxt = random.choice(candidates)
                else:
                    nxt = best_nxt
            else:
                nxt = random.choice(candidates)
            
            path.append(nxt)
            used.add(nxt)
            
            # Update blocked set
            for nb in neighbors[cur]:
                if nb != nxt:
                    blocked.add(nb)
                    
            cur = nxt
            
        return path
    
    # Run multiple strategies
    while time.time() < deadline:
        # Alternate between pure random and lookahead-guided greedy
        use_lookahead = random.random() < 0.5
        
        start = random.randint(0, 511)
        path = _greedy_with_lookahead(start, use_lookahead)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
    return best_path
# EVOLVE-BLOCK-END
