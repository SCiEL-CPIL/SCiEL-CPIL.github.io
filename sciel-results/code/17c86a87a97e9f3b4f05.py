"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import deque

N = 9
TIME_LIMIT = 115  # seconds, leave buffer before 120s

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using constraint propagation.
    We maintain a 'blocked' set of vertices that are adjacent to any vertex in the
    current path except the last one. This allows O(1) validity checks for
    potential next vertices.
    """
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def greedy_extend_efficient(start_vertex, start_time, randomize=False, seed=None, bit_order=None):
        """
        Efficient greedy extension using blocked set.
        """
        if seed is not None:
            random.seed(seed)
            
        path = [start_vertex]
        used = {start_vertex}
        
        # blocked contains vertices adjacent to path[0]
        # Any new vertex v must not be in blocked (unless it's the next one, 
        # but we check candidates against blocked before adding to path)
        # Actually, the condition is: v must not be adjacent to any p in path[:-1].
        # So, blocked should contain neighbors of all p in path[:-1].
        # Initially, path = [v0]. path[:-1] is empty. So blocked is empty?
        # No, if we add v1, then v1 must not be adjacent to v0? 
        # Wait. "non-consecutive vertices differ in at least two bits".
        # v0 and v1 are consecutive. So they CAN be adjacent.
        # v0 and v2 are non-consecutive. So v2 must NOT be adjacent to v0.
        # So, when we are at v_k and want to pick v_{k+1}:
        # v_{k+1} must be adjacent to v_k.
        # v_{k+1} must NOT be adjacent to v_i for i < k.
        
        # So, `blocked` should be the set of neighbors of {v_0, ..., v_{k-1}}.
        # Initially k=0. path=[v0]. path[:-1] is empty. blocked = empty set.
        # Candidates for v1 are neighbors of v0.
        # None of them are in blocked (empty). So all are valid.
        # After picking v1, path=[v0, v1].
        # Now we want v2.
        # v2 must be neighbor of v1.
        # v2 must NOT be neighbor of v0.
        # So blocked should now contain neighbors of v0.
        
        blocked = set()
        
        while True:
            if time.time() - start_time > TIME_LIMIT - 2:
                return path
                
            cur = path[-1]
            candidates = []
            
            # Get neighbors of cur
            if bit_order:
                neighs = [cur ^ (1 << b) for b in bit_order]
            else:
                neighs = neighbors[cur]
                
            for v in neighs:
                if v in used:
                    continue
                if v in blocked:
                    continue
                candidates.append(v)
                
            if not candidates:
                break
                
            if randomize:
                random.shuffle(candidates)
                # Heuristic: pick candidate with most free neighbors
                # Free neighbors = neighbors not in used and not in blocked
                # Note: v's neighbors that are in blocked are forbidden for FUTURE steps?
                # No, blocked is for the NEXT step's constraint relative to current path.
                # If we pick v, it becomes the new cur.
                # The new blocked will be old_blocked U neighbors(cur).
                # So we want v to have many neighbors that are NOT in (used U old_blocked U {cur})?
                # Actually, we want to maximize future options.
                # Score: count of neighbors of v that are not in used and not in blocked.
                # (Neighbors in blocked are forbidden for the step AFTER v? 
                #  Let's trace:
                #  Path: ..., cur, v.
                #  Next vertex w must be neighbor of v.
                #  w must not be neighbor of any p in path[:-1] (which includes cur).
                #  So w cannot be in neighbors(cur).
                #  Current blocked contains neighbors of path[:-2] (before cur was added? No).
                #  Let's re-verify blocked update.
                #  Start: path=[v0]. blocked={}.
                #  Pick v1. path=[v0, v1].
                #  Update blocked: blocked = blocked U neighbors(v0).
                #  Now pick v2. v2 must be neighbor of v1. v2 not in blocked (neighbors of v0).
                #  This is correct. v2 is not adjacent to v0.
                #  Pick v2. path=[v0, v1, v2].
                #  Update blocked: blocked = blocked U neighbors(v1).
                #  Now blocked contains neighbors of v0 and v1.
                #  Pick v3. v3 must be neighbor of v2. v3 not in blocked.
                #  So v3 not adjacent to v0 or v1. Correct.
                
                # So, to score v (candidate for next step):
                # We want to know how many options we have for the step AFTER v.
                # Options for next-next step (w) are neighbors of v that are not in used
                # and not in (current_blocked U neighbors(cur)).
                # Because after adding v, the new blocked will be current_blocked U neighbors(cur).
                # Wait, no.
                # If we pick v, the new path is [..., cur, v].
                # The new blocked for the NEXT selection (w) will be:
                # old_blocked U neighbors(cur).
                # So w must be neighbor of v, not in used, and not in (old_blocked U neighbors(cur)).
                
                new_blocked_preview = blocked | set(neighbors[cur])
                
                best_v = None
                best_score = -1
                for v in candidates:
                    score = 0
                    for nb in neighbors[v]:
                        if nb not in used and nb not in new_blocked_preview:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                    # Add small random noise to break ties
                    if random.random() < 0.01:
                        score += 0.1
                        if score > best_score:
                            best_score = score
                            best_v = v
                chosen = best_v
            else:
                new_blocked_preview = blocked | set(neighbors[cur])
                best_v = None
                best_score = -1
                for v in candidates:
                    score = 0
                    for nb in neighbors[v]:
                        if nb not in used and nb not in new_blocked_preview:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
                chosen = best_v

            path.append(chosen)
            used.add(chosen)
            # Update blocked: add neighbors of the previous current (cur)
            # Because cur is now in path[:-1] for the next step
            for nb in neighbors[cur]:
                blocked.add(nb)
                
            if len(path) > best_len:
                # Update global best
                # We need to return the path, so we can't easily update nonlocal here without closure issues
                # We'll return the path and let the caller update
                pass
        
        return path

    start_time = time.time()
    
    # Strategy 1: Deterministic greedy with different bit orderings
    # Try a few promising bit orderings
    import itertools
    # Instead of all permutations, try a few random ones and the identity
    bit_orders_to_try = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        [0, 2, 4, 6, 8, 1, 3, 5, 7], # even then odd
        [1, 3, 5, 7, 0, 2, 4, 6, 8]  # odd then even
    ]
    
    for _ in range(20):
        bit_orders_to_try.append(random.sample(range(N), N))
        
    for bo in bit_orders_to_try:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        path = greedy_extend_efficient(0, start_time, randomize=False, bit_order=bo)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Strategy 2: Randomized greedy with multiple restarts
    random.seed(42)
    for _ in range(500):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        start_v = random.randint(0, (1 << N) - 1)
        bo = random.sample(range(N), N)
        path = greedy_extend_efficient(start_v, start_time, randomize=True, seed=random.randint(0, 100000), bit_order=bo)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    # Strategy 3: Local search / Backtracking on best path
    # We can try to extend the best path found so far with a more aggressive search
    # Since greedy is fast, we can try to "repair" or "extend" by backtracking a few steps
    # For now, let's do a few more randomized restarts focusing on the best structure
    
    return best_path
# EVOLVE-BLOCK-END
