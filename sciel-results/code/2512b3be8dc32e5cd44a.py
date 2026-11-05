import time
import random

N = 9
V = 1 << N  # 512
TIME_LIMIT = 115.0

# Precompute neighbor masks
# neighbor_mask[v] is a 512-bit integer where bit i is 1 if i is a neighbor of v
neighbor_mask = [0] * V
for v in range(V):
    mask = 0
    for b in range(N):
        nb = v ^ (1 << b)
        mask |= (1 << nb)
    neighbor_mask[v] = mask

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = []
    best_len = 0
    rng = random.Random(42)
    
    # Precompute bit positions for fast iteration? 
    # We iterate over bits of a 512-bit int.
    # Candidates are neighbors of cur.
    
    def get_candidates(cur, used_mask, excl_mask):
        """Return list of valid candidate vertices."""
        # Neighbors of cur
        # We can iterate through the 9 neighbors directly
        cands = []
        for b in range(N):
            v = cur ^ (1 << b)
            # Check if v is used
            if (used_mask >> v) & 1:
                continue
            # Check if v is excluded
            if (excl_mask >> v) & 1:
                continue
            cands.append(v)
        return cands

    def score_candidate(v, cur, used_mask, excl_mask):
        """
        Score candidate v.
        State after moving to v:
        new_used = used_mask | (1 << v)
        new_excl = excl_mask | neighbor_mask[cur]
        
        Score is the number of valid moves from v in the new state.
        """
        new_excl = excl_mask | neighbor_mask[cur]
        new_used = used_mask | (1 << v)
        
        count = 0
        for b in range(N):
            w = v ^ (1 << b)
            if (new_used >> w) & 1:
                continue
            if (new_excl >> w) & 1:
                continue
            count += 1
        return count

    def run_greedy():
        nonlocal best_path, best_len
        path = [0]
        used_mask = 1
        excl_mask = 0
        
        while True:
            cur = path[-1]
            cands = get_candidates(cur, used_mask, excl_mask)
            
            if not cands:
                break
            
            if len(cands) == 1:
                v = cands[0]
            else:
                # Score each candidate
                best_score = 100
                best_cands = []
                for c in cands:
                    s = score_candidate(c, cur, used_mask, excl_mask)
                    if s < best_score:
                        best_score = s
                        best_cands = [c]
                    elif s == best_score:
                        best_cands.append(c)
                v = rng.choice(best_cands)
            
            path.append(v)
            used_mask |= (1 << v)
            excl_mask |= neighbor_mask[cur]
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path.copy()
                
            if time.time() > deadline:
                return

    # Run multiple restarts
    # Vary the starting vertex? Or just the random seed for tie-breaking.
    # The problem is symmetric, so starting at 0 is fine.
    # We can also try starting with a random first move if we allow it, but path must start at 0? 
    # The interface says "list of distinct integers... in order". It doesn't force start at 0.
    # But usually snakes are considered up to isomorphism.
    # Let's stick to starting at 0 to be safe, but randomize the tie-breaking.
    
    while time.time() < deadline - 1.0:
        run_greedy()
        
    return best_path
