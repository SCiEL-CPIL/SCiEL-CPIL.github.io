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
          - "random": pick random valid neighbor
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
            else:
                # Lookahead: for each candidate n, count how many valid moves from n
                # Valid moves from n are neighbors nn such that:
                #   nn not in used
                #   nn not in forbidden_next
                # where forbidden_next = forbidden | neighbors(cur)
                
                # Compute forbidden_next
                forbidden_next = forbidden.copy()
                for nc in nbrs:
                    forbidden_next[nc] = True
                
                best_score = -1
                best_candidates = []
                
                for n in valid_next:
                    # Count valid moves from n
                    score = 0
                    for nn in neighbors[n]:
                        if not used[nn] and not forbidden_next[nn]:
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
    # Use a mix of strategies to explore different parts of the search space
    
    while time.time() - start_time < MAX_TIME:
        # Random start
        start_v = rng.integers(NUM_VERTICES)
        
        # Random strategy choice
        r = rng.random()
        if r < 0.7:
            strategy = "max_degree"
        else:
            strategy = "random"
        
        path = try_greedy(start_v, strategy)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
    
    if best_path is None:
        # Fallback: simple greedy from 0
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path
    
    return best_path
