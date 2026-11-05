"""Improved snake-in-the-box for Q_9 using Warnsdorff's rule with bitset optimization."""
import random
import sys
import time

N = 9
FULL_MASK = (1 << N) - 1

# Precompute neighbors for all vertices for speed
NEIGHBORS = []
for v in range(1 << N):
    nbrs = []
    for b in range(N):
        nbrs.append(v ^ (1 << b))
    NEIGHBORS.append(nbrs)

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using Warnsdorff's rule.
    This heuristic tends to produce longer paths than simple greedy or random walk.
    """
    best_path = None
    best_len = 0
    
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110
    
    # We run many iterations of the Warnsdorff greedy algorithm
    # with random tie-breaking to explore the solution space.
    
    while time.time() - start_time < time_limit:
        # Random start
        start = rng.randint(0, (1 << N) - 1)
        path = [start]
        used = {start}
        
        # Forbidden set: vertices at distance 1 from P[:-1]
        # Initially P[:-1] is empty, so Forbidden is empty.
        # But we need to update it as we go.
        # Forbidden = Union of Neighbors(p) for p in path[:-1]
        
        forbidden = set()
        
        while True:
            cur = path[-1]
            
            # Candidates must be:
            # 1. Neighbors of cur
            # 2. Not in used
            # 3. Not in forbidden (dist 1 from path[:-1])
            
            candidates = []
            for nb in NEIGHBORS[cur]:
                if nb in used:
                    continue
                if nb in forbidden:
                    continue
                candidates.append(nb)
                
            if not candidates:
                break
            
            # Warnsdorff's Rule: Pick the candidate with the fewest onward moves.
            # Onward moves for candidate 'c' are neighbors of 'c' that would be valid
            # in the next step.
            # Next step state:
            # used' = used U {c}
            # forbidden' = forbidden U Neighbors(cur)
            # (Because 'cur' becomes part of path[:-1] after we move to 'c')
            
            new_forbidden_base = forbidden
            # Note: We don't need to construct the full set for each candidate,
            # we can just check if a neighbor 'n' of 'c' is in used' or forbidden'.
            
            min_score = N + 1
            best_candidates = []
            
            for c in candidates:
                # Score: count of valid next-next steps
                score = 0
                for nn in NEIGHBORS[c]:
                    if nn in used:
                        continue
                    if nn == c: # just in case, though neighbors are distinct
                        continue
                    # nn is valid if it's not in used' and not in forbidden'
                    # used' = used U {c}
                    if nn in used or nn == c:
                        continue
                    # forbidden' = forbidden U Neighbors(cur)
                    if nn in forbidden:
                        continue
                    if nn in NEIGHBORS[cur]:
                        continue
                    score += 1
                
                if score < min_score:
                    min_score = score
                    best_candidates = [c]
                elif score == min_score:
                    best_candidates.append(c)
            
            # Tie-breaking: pick randomly from best_candidates
            next_v = rng.choice(best_candidates)
            
            path.append(next_v)
            used.add(next_v)
            
            # Update forbidden set
            # The new forbidden set is the old forbidden set PLUS the neighbors of 'cur'
            # because 'cur' is now in path[:-1]
            for nb in NEIGHBORS[cur]:
                forbidden.add(nb)
            
            # Optional: If path is getting long, we might want to break early if we hit a dead end
            # but the loop handles that.
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
            # If we find a very long path, we might want to stop or keep going
            # The task is to maximize length. Let's keep going until time limit.
            # However, Warnsdorff is deterministic given tie-breaks.
            # We can try to "repair" or extend, but for now, restarts are the main driver.
            
    if best_path is None:
        # Fallback
        return [0]
        
    return best_path
# EVOLVE-BLOCK-END
