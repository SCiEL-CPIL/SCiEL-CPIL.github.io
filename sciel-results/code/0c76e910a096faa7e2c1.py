import time
import random
import numpy as np

N = 9
MAX_TIME = 118.0
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    
    # Precompute neighbors as a numpy array for fast access
    neighbors = np.empty((NUM_VERTICES, N), dtype=np.uint32)
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    best_path = None
    best_len = 0
    
    rng = np.random.default_rng(seed=42)
    
    def try_greedy(start_v, strategy="max_degree"):
        """
        Try to build a snake starting from start_v.
        strategy: 
          - "max_degree": pick neighbor with max available degree in next step
          - "min_degree": pick neighbor with min available degree (Weinstein)
          - "random": pick random valid neighbor
          - "lookahead3": pick neighbor that maximizes the min-degree of its valid neighbors
        """
        path = [start_v]
        used = np.zeros(NUM_VERTICES, dtype=bool)
        used[start_v] = True
        
        # forbidden[v] is True if v is adjacent to any internal vertex in path[:-1]
        forbidden = np.zeros(NUM_VERTICES, dtype=bool)
        
        cur = start_v
        
        while True:
            # Find valid next vertices
            nbrs = neighbors[cur]
            
            valid_next = []
            for n in nbrs:
                if not used[n] and not forbidden[n]:
                    valid_next.append(n)
            
            if not valid_next:
                break
            
            if strategy == "random":
                next_v = valid_next[rng.integers(len(valid_next))]
            elif strategy == "min_degree":
                # Weinstein: pick the vertex with the fewest available moves
                # This forces the path to be "tight" and often longer
                min_score = 100
                best_candidates = []
                for n in valid_next:
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            score += 1
                    if score < min_score:
                        min_score = score
                        best_candidates = [n]
                    elif score == min_score:
                        best_candidates.append(n)
                next_v = best_candidates[rng.integers(len(best_candidates))]
            elif strategy == "lookahead3":
                # For each candidate n, find valid neighbors nn.
                # For each nn, find valid neighbors nnn.
                # Score n by the minimum number of valid moves from nn.
                # This tries to avoid dead ends 2 steps ahead.
                
                # Pre-compute forbidden_next for speed?
                # forbidden_next = forbidden | neighbors(cur)
                # We can simulate this check without copying the whole array
                # if we just check `not forbidden[nn] and nn not in nbrs`
                
                best_score = -1
                best_candidates = []
                
                for n in valid_next:
                    # Get valid neighbors of n
                    nn_candidates = []
                    for nn in neighbors[n]:
                        # nn is valid if not used, not forbidden, and not cur (cur is in nbrs, so nn != cur is implied if nn not in nbrs? No, nn is neighbor of n. cur is neighbor of n. nn could be cur.)
                        # But nn must not be in forbidden.
                        # If nn is cur, it is in used, so it's filtered.
                        if not used[nn] and not forbidden[nn]:
                            # Also, nn must not be adjacent to cur?
                            # If nn is adjacent to cur, then nn is in neighbors[cur].
                            # Since cur becomes internal, neighbors[cur] become forbidden.
                            # So if nn is in neighbors[cur], it should be forbidden in the next step.
                            # But for the score of n, we are looking at moves *from* n.
                            # The move to nn is valid if nn is not forbidden *now* and not used.
                            # However, after moving to n, cur becomes internal.
                            # So nn must not be adjacent to cur.
                            # Is nn adjacent to cur?
                            # dist(n, cur) = 1.
                            # dist(nn, n) = 1.
                            # dist(nn, cur) can be 0 (nn=cur), 2, or ...
                            # If nn == cur, dist=0. (Filtered by used)
                            # If nn is another neighbor of cur, dist(nn, cur)=1? No, in hypercube, neighbors of n are at dist 1 from n.
                            # cur is at dist 1 from n.
                            # The neighbors of n are: cur, and N-1 others.
                            # The others are at dist 2 from cur.
                            # So only cur is at dist 1 from cur among neighbors of n.
                            # So if nn != cur, dist(nn, cur) = 2.
                            # So nn is NOT adjacent to cur.
                            # Thus, nn is not forbidden by cur.
                            # So the check `not forbidden[nn]` is sufficient?
                            # Wait, `forbidden` currently contains neighbors of path[:-1].
                            # cur is in path[:-1] (it's the second to last).
                            # So `forbidden` ALREADY contains neighbors of cur?
                            # No. `forbidden` is updated *after* moving.
                            # In the loop:
                            # 1. cur is path[-1].
                            # 2. forbidden contains neighbors of path[:-1].
                            # 3. We pick n.
                            # 4. We check if n is in used or forbidden.
                            # 5. We move to n.
                            # 6. We update forbidden to include neighbors of cur.
                            
                            # So when evaluating n, `forbidden` does NOT include neighbors of cur.
                            # But if we move to n, and then to nn, nn must not be adjacent to cur.
                            # As established, if nn != cur, dist(nn, cur)=2.
                            # So nn is not adjacent to cur.
                            # So the condition `not forbidden[nn]` is correct for the *current* state.
                            # But we need to ensure that the move to nn is valid *after* cur becomes internal.
                            # Since nn is not adjacent to cur (unless nn=cur), it is safe.
                            
                            nn_candidates.append(nn)
                    
                    if not nn_candidates:
                        score = 0
                    else:
                        # Score n by the minimum number of valid moves from its valid neighbors
                        min_next_degree = 100
                        for nn in nn_candidates:
                            deg = 0
                            for nnn in neighbors[nn]:
                                # nnn must not be used
                                # nnn must not be forbidden (current forbidden)
                                # nnn must not be adjacent to n?
                                # If we move n -> nn, n becomes internal.
                                # So nnn must not be adjacent to n.
                                # nnn is neighbor of nn. n is neighbor of nn.
                                # So nnn could be n. (Filtered by used)
                                # If nnn != n, dist(nnn, n) = 2.
                                # So nnn is not adjacent to n.
                                # So `not forbidden[nnn]` is sufficient?
                                # `forbidden` contains neighbors of path[:-1].
                                # n is not in path[:-1] yet (it's the candidate).
                                # So `forbidden` does not contain neighbors of n.
                                # But after moving to nn, n becomes internal.
                                # So nnn must not be adjacent to n.
                                # As above, if nnn != n, dist(nnn, n)=2.
                                # So it's safe.
                                
                                if not used[nnn] and not forbidden[nnn]:
                                    deg += 1
                            if deg < min_next_degree:
                                min_next_degree = deg
                        
                        score = min_next_degree
                    
                    if score > best_score:
                        best_score = score
                        best_candidates = [n]
                    elif score == best_score:
                        best_candidates.append(n)
                
                next_v = best_candidates[rng.integers(len(best_candidates))]
            else: # max_degree
                best_score = -1
                best_candidates = []
                for n in valid_next:
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden[nn]:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_candidates = [n]
                    elif score == best_score:
                        best_candidates.append(n)
                next_v = best_candidates[rng.integers(len(best_candidates))]
            
            # Add next_v to path
            path.append(next_v)
            used[next_v] = True
            
            # Update forbidden: cur becomes internal, so its neighbors are now forbidden
            for nc in nbrs:
                forbidden[nc] = True
            
            cur = next_v
        
        return path
    
    # Strategy: run many restarts with different starting vertices and strategies
    # Weights: max_degree is usually best, but lookahead3 might find longer ones
    # Let's mix: 50% max_degree, 30% lookahead3, 20% min_degree
    
    while time.time() - start_time < MAX_TIME:
        # Random start
        start_v = rng.integers(NUM_VERTICES)
        
        # Random strategy choice
        r = rng.random()
        if r < 0.5:
            strategy = "max_degree"
        elif r < 0.8:
            strategy = "lookahead3"
        else:
            strategy = "min_degree"
        
        path = try_greedy(start_v, strategy)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    if best_path is None:
        # Fallback: simple greedy from 0
        path = [0]
        used = {0}
        forbidden = set()
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or v in forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            # Add neighbors of cur to forbidden
            for b in range(N):
                forbidden.add(cur ^ (1 << b))
        best_path = path
    
    return best_path
