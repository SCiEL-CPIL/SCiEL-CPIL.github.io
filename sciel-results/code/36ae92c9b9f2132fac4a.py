import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
SIZE = 1 << N
DEADLINE = 110.0  # Safety margin for 120s limit

# Precompute neighbors
NEIGHBORS = [[] for _ in range(MAX_VERTS)]
for v in range(MAX_VERTS):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute blocked-set lookups for O(1) conflict checking
# For a vertex v, blocked[v] is a bitmask of vertices that cannot be added
# if v is in the path (excluding v's immediate neighbors which are allowed as path extensions)
# Actually, for induced path: if v is in path, any other vertex u in path must not be adjacent to v.
# So when adding a new vertex x to the path, x must not be adjacent to any vertex in the path
# except the one it connects to.

# Let's use a different approach: maintain a "forbidden" set.
# When we have a path, for each vertex v in the path, all neighbors of v are "blocked" 
# from being added to the path (unless they are the immediate predecessor/successor).
# But this is dynamic. 

# Better: use the bitmask approach from inspiration 1.
# ADJ_MASK[v] has bit i set iff v and i are adjacent.
ADJ_MASK = [0] * MAX_VERTS
for v in range(MAX_VERTS):
    m = 0
    for nb in NEIGHBORS[v]:
        m |= (1 << nb)
    ADJ_MASK[v] = m


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy:
    1. Multi-start bidirectional greedy growth with Warnsdorff heuristic.
    2. Local search via backtracking with O(1) blocked-set lookups.
    3. Insertion search to fill gaps.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_path = None
    best_len = 0
    rng = random.Random(42)
    
    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    def is_valid_tail(v, path, used_mask):
        """Check if v can be appended to the tail of path."""
        if used_mask & (1 << v):
            return False
        # v connects to path[-1], check against path[:-1]
        other_mask = used_mask ^ (1 << path[-1])
        if ADJ_MASK[v] & other_mask:
            return False
        return True

    def is_valid_head(v, path, used_mask):
        """Check if v can be prepended to the head of path."""
        if used_mask & (1 << v):
            return False
        # v connects to path[0], check against path[1:]
        other_mask = used_mask ^ (1 << path[0])
        if ADJ_MASK[v] & other_mask:
            return False
        return True

    def count_onward(v, path, used_mask, is_tail):
        """Count valid moves from v if v is added as tail or head."""
        new_used = used_mask | (1 << v)
        count = 0
        if is_tail:
            # v is new tail, next move w from v
            # w must not be in new_used, and w must not be adj to path[:-1]
            other_mask = used_mask  # all old vertices
            for w in NEIGHBORS[v]:
                if new_used & (1 << w):
                    continue
                # w must not be adj to any vertex in old path except path[-1]
                # But w is adj to v, and v is adj to path[-1], so w is at dist 2 from path[-1]
                # Check w against path[:-1]
                if ADJ_MASK[w] & (used_mask ^ (1 << path[-1])):
                    continue
                count += 1
        else:
            # v is new head, next move w from v
            other_mask = used_mask
            for w in NEIGHBORS[v]:
                if new_used & (1 << w):
                    continue
                if ADJ_MASK[w] & (used_mask ^ (1 << path[0])):
                    continue
                count += 1
        return count

    def bidirectional_grow(v0, v1, time_limit):
        """Grow a snake bidirectionally from v0 and v1."""
        if (v0 ^ v1).bit_count() != 1:
            return None
        
        path = [v0, v1]
        used_mask = (1 << v0) | (1 << v1)
        
        while time.time() < time_limit:
            # Get valid moves for tail
            tail_moves = []
            for v in NEIGHBORS[path[-1]]:
                if is_valid_tail(v, path, used_mask):
                    cnt = count_onward(v, path, used_mask, True)
                    tail_moves.append((cnt, v))
            
            # Get valid moves for head
            head_moves = []
            for v in NEIGHBORS[path[0]]:
                if is_valid_head(v, path, used_mask):
                    cnt = count_onward(v, path, used_mask, False)
                    head_moves.append((cnt, v))
            
            if not tail_moves and not head_moves:
                break
            
            # Choose which end to extend
            if not tail_moves:
                extend_head = True
            elif not head_moves:
                extend_head = False
            else:
                min_tail = min(m[0] for m in tail_moves)
                min_head = min(m[0] for m in head_moves)
                if min_tail < min_head:
                    extend_head = False
                elif min_head < min_tail:
                    extend_head = True
                else:
                    extend_head = rng.random() < 0.5
            
            if extend_head:
                min_sub = min(m[0] for m in head_moves)
                candidates = [m[1] for m in head_moves if m[0] <= min_sub + 1]
                v = rng.choice(candidates)
                path.insert(0, v)
                used_mask |= (1 << v)
            else:
                min_sub = min(m[0] for m in tail_moves)
                candidates = [m[1] for m in tail_moves if m[0] <= min_sub + 1]
                v = rng.choice(candidates)
                path.append(v)
                used_mask |= (1 << v)
            
            if len(path) > best_len:
                update_best(path)
        
        return path

    def local_search(path, time_limit):
        """
        Local search via backtracking with O(1) blocked-set lookups.
        Try to extend the path by truncating the end and using DFS.
        """
        current_best = path[:]
        current_best_len = len(current_best)
        
        for k in range(1, 12):
            if time.time() > time_limit:
                break
            if len(current_best) <= k:
                break
            
            truncated = current_best[:-k]
            if len(truncated) < 2:
                continue
            
            # Rebuild state
            used_mask = 0
            for v in truncated:
                used_mask |= (1 << v)
            
            def dfs_extend(cur_path, depth, max_depth=20):
                nonlocal current_best, current_best_len
                
                if time.time() > time_limit:
                    return
                
                if len(cur_path) > current_best_len:
                    current_best = cur_path[:]
                    current_best_len = len(current_best)
                    update_best(current_best)
                
                if depth >= max_depth:
                    return
                
                v_end = cur_path[-1]
                
                # Gather valid moves
                valid_moves = []
                for u in NEIGHBORS[v_end]:
                    if used_mask & (1 << u):
                        continue
                    # u connects to v_end, check against cur_path[:-1]
                    other_mask = used_mask ^ (1 << v_end)
                    if ADJ_MASK[u] & other_mask:
                        continue
                    # Heuristic: count onward moves
                    cnt = 0
                    for w in NEIGHBORS[u]:
                        if (used_mask | (1 << u)) & (1 << w):
                            continue
                        if ADJ_MASK[w] & (used_mask ^ (1 << v_end)):
                            continue
                        cnt += 1
                    valid_moves.append((cnt, u))
                
                if not valid_moves:
                    return
                
                valid_moves.sort(key=lambda x: x[0])
                beam_width = 4 if depth < 5 else 2
                candidates = valid_moves[:beam_width]
                
                for cnt, u in candidates:
                    if time.time() > time_limit:
                        return
                    
                    # Add u
                    used_mask |= (1 << u)
                    cur_path.append(u)
                    
                    dfs_extend(cur_path, depth + 1)
                    
                    # Remove u
                    cur_path.pop()
                    used_mask ^= (1 << u)
            
            dfs_extend(truncated[:], 0)
            
            if len(current_best) > len(path):
                path = current_best[:]
        
        return current_best

    def insertion_search(path, time_limit):
        """
        Try to insert vertices into gaps in the path to extend it.
        """
        current_path = path[:]
        current_used = 0
        for v in current_path:
            current_used |= (1 << v)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Pick a random gap
            pos = rng.randint(1, len(current_path) - 1)
            a = current_path[pos - 1]
            c = current_path[pos]
            
            # Find v such that v~a, v~c, v not in used, v not adj to others
            candidates = []
            for v in NEIGHBORS[a]:
                if v == c or (current_used & (1 << v)):
                    continue
                if (v ^ c).bit_count() != 1:
                    continue
                # Check validity against rest of path
                other_mask = current_used ^ (1 << a) ^ (1 << c)
                if ADJ_MASK[v] & other_mask:
                    continue
                candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                current_path.insert(pos, v)
                current_used |= (1 << v)
                improved = True
                update_best(current_path)
        
        return current_path

    # --- Main Loop ---
    
    # Phase 1: Quick Greedy/Bidirectional starts to get a baseline
    for i in range(30):
        if time.time() > deadline - 30:
            break
        v0 = rng.randint(0, MAX_VERTS - 1)
        v1 = rng.choice(NEIGHBORS[v0])
        p = bidirectional_grow(v0, v1, time.time() + 1.0)
        if p:
            update_best(p)
    
    # Phase 2: Intensive Local Search on the best path found
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        if not best_path:
            break
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 2.0:
            break
        
        # Local search
        improved_path = local_search(best_path[:], time.time() + remaining * 0.7)
        
        if improved_path and len(improved_path) > best_len:
            best_len = len(improved_path)
            best_path = improved_path[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Insertion search
        if time.time() < deadline - 5:
            inserted_path = insertion_search(best_path[:], time.time() + remaining * 0.2)
            if inserted_path and len(inserted_path) > best_len:
                best_len = len(inserted_path)
                best_path = inserted_path[:]
                no_improve_count = 0
        
        # If stuck, try fresh starts
        if no_improve_count >= 3 and time.time() < deadline - 15:
            no_improve_count = 0
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                v0 = rng.randint(0, MAX_VERTS - 1)
                v1 = rng.choice(NEIGHBORS[v0])
                p = bidirectional_grow(v0, v1, time.time() + 0.5)
                if p:
                    update_best(p)
    
    if best_path is None:
        return [0]
    
    return best_path

if __name__ == "__main__":
    path = construct()
    print(f"Length: {len(path)}")
    # Verify
    valid = True
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            print(f"Invalid edge at {i}")
            valid = False
            break
    if valid:
        for i in range(len(path)):
            for j in range(i+2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    print(f"Invalid non-consecutive adjacency at {i}, {j}")
                    valid = False
                    break
            if not valid:
                break
    print(f"Valid: {valid}")
