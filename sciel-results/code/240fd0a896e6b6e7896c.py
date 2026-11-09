"""Search-based snake construction for Q_9."""
import time
import random

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Uses randomized greedy with multiple restarts and extension attempts.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_path = None
    best_len = 0
    
    def greedy_extend(path, used, forbidden, rng, max_steps=500):
        """Extend path using randomized greedy."""
        path = list(path)
        used = set(used)
        forbidden = set(forbidden)
        
        while len(path) < max_steps + len(path):  # Safety
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    candidates.append(v)
            if not candidates:
                break
            v = rng.choice(candidates)
            path.append(v)
            used.add(v)
            # Update forbidden: add neighbors of cur (the old last element)
            for b2 in range(N):
                nb = cur ^ (1 << b2)
                if nb != v:
                    forbidden.add(nb)
        
        return path, used, forbidden
    
    # Run multiple iterations
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        rng = random.Random(iteration * 7919 + 104729)
        
        path = [0]
        used = {0}
        forbidden = set()
        
        path, used, forbidden = greedy_extend(path, used, forbidden, rng, max_steps=500)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Try to extend the current best if we have time
        if best_len > 100 and time.time() < deadline - 3:
            ext_rng = random.Random(iteration * 31 + 41)
            bp = list(best_path)
            bu = set(best_path)
            # Rebuild forbidden
            bforb = set()
            for i in range(len(bp) - 1):
                for b in range(N):
                    bforb.add(bp[i] ^ (1 << b))
            bp, bu, bforb = greedy_extend(bp, bu, bforb, ext_rng, max_steps=50)
            if len(bp) > best_len:
                best_len = len(bp)
                best_path = bp
    
    if best_path is None:
        path, used = [0], {0}
        forbidden = set()
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v not in used and v not in forbidden:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            for b2 in range(N):
                nb = cur ^ (1 << b2)
                if nb != nxt:
                    forbidden.add(nb)
        best_path = path
    
    return best_path
# EVOLVE-BLOCK-END
