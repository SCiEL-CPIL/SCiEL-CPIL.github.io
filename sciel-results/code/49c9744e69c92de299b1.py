# EVOLVE-BLOCK-START

import time
import os
import multiprocessing
import random

def _precompute_neighbors():
    """Precompute neighbors and neighbor masks for all vertices."""
    neighbors = [[] for _ in range(TOTAL_VERTICES)]
    neighbor_masks = [0] * TOTAL_VERTICES
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            nb = v ^ (1 << b)
            neighbors[v].append(nb)
            neighbor_masks[v] |= (1 << nb)
    return neighbors, neighbor_masks

# Global precomputed data
_NEIGHBORS, _NEIGHBOR_MASKS = _precompute_neighbors()

def _worker(args):
    """
    Worker process: runs multiple DFS searches with specific seeds.
    """
    seed_start, seed_end, time_limit, start_time_global = args
    
    best_path = [0, 1, 3]
    
    # Symmetry Breaking: Fix 0 -> 1 -> 3
    # 0: 000000000
    # 1: 000000001 (flip bit 0)
    # 3: 000000011 (flip bit 1)
    
    initial_path = [0, 1, 3]
    initial_used = (1 << 0) | (1 << 1) | (1 << 3)
    
    # Calculate initial banned mask
    initial_banned = 0
    # Neighbors of 0, excluding 1
    for b in range(1, N):
        initial_banned |= (1 << (1 << b))
        
    # Neighbors of 1, excluding 0 and 3
    for b in range(N):
        if b == 0: continue # neighbor 0
        nb = 1 ^ (1 << b)
        if nb == 3: continue
        initial_banned |= (1 << nb)
        
    # Neighbors of 3, excluding 1 and 2? 
    # Wait, when we move to 3, we ban neighbors of 1.
    # The banned set for the *next* step (from 3) will include neighbors of 1.
    # But the `banned` variable in DFS represents "vertices that cannot be used in the future".
    # When at cur=1, we moved to 3.
    # The new banned set is `old_banned | neighbors(1)`.
    # So `initial_banned` for the DFS starting at 3 should be:
    # (Neighbors of 0 excl 1) | (Neighbors of 1 excl 0,3)
    
    # Let's re-verify the DFS logic.
    # dfs(cur, used, banned, ...)
    # banned contains vertices that are neighbors of ANY previous vertex, except the immediate predecessor.
    # So when starting at 3, the predecessors are 0 and 1.
    # Banned should be (Neighbors of 0) U (Neighbors of 1), excluding 1 and 3 and 0?
    # Actually, `used` contains 0, 1, 3.
    # `banned` should contain any vertex $v$ such that $v$ is adjacent to some $u \in \{0, 1\}$ and $v \notin \{0, 1, 3\}$.
    
    # Neighbors of 0: 1, 2, 4, 8, 16, 32, 64, 128, 256.
    # Excluding 1 (in used/path).
    # So 2, 4, 8, 16, 32, 64, 128, 256 are banned.
    
    # Neighbors of 1: 0, 3, 5, 9, 17, 33, 65, 129, 257.
    # Excluding 0, 3 (in used/path).
    # So 5, 9, 17, 33, 65, 129, 257 are banned.
    
    # So initial_banned is the union of these sets.
    
    def dfs_randomized(cur, used, banned, depth, rng, current_path):
        nonlocal best_path
        
        if time.time() - start_time_global > time_limit:
            return
        
        if depth > len(best_path):
            best_path = list(current_path)
        
        cand_mask = _NEIGHBOR_MASKS[cur] & ~used & ~banned
        if cand_mask == 0:
            return
        
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            candidates.append(v)
            cm ^= lsb
            
        def heuristic_score(v):
            new_used = used | (1 << v)
            next_banned = banned | _NEIGHBOR_MASKS[cur]
            next_cand_mask = _NEIGHBOR_MASKS[v] & ~new_used & ~next_banned
            return next_cand_mask.bit_count()
            
        rng.shuffle(candidates)
        candidates.sort(key=heuristic_score, reverse=True)
        
        for v in candidates:
            if time.time() - start_time_global > time_limit:
                return
            
            current_path.append(v)
            new_used = used | (1 << v)
            new_banned = banned | _NEIGHBOR_MASKS[cur]
            
            dfs_randomized(v, new_used, new_banned, depth + 1, rng, current_path)
            
            current_path.pop()
            
            if time.time() - start_time_global > time_limit:
                return

    for i in range(seed_start, seed_end):
        if time.time() - start_time_global > time_limit:
            break
        rng = random.Random(i)
        current_path = [0, 1, 3]
        dfs_randomized(3, initial_used, initial_banned, 3, rng, current_path)

    return best_path

def construct():
    num_workers = os.cpu_count() or 4
    time_limit = 115.0
    start_time = time.time()
    
    total_seeds = 2000
    seeds_per_worker = total_seeds // num_workers
    
    tasks = []
    for w in range(num_workers):
        seed_start = w * seeds_per_worker
        seed_end = (w + 1) * seeds_per_worker if w < num_workers - 1 else total_seeds
        tasks.append((seed_start, seed_end, time_limit, start_time))
        
    if num_workers == 1:
        best_path = _worker(tasks[0])
    else:
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(_worker, tasks)
            best_path = max(results, key=len)
        except Exception:
            # Fallback to single process if multiprocessing fails
            best_path = _worker((0, total_seeds, time_limit, start_time))
            
    return best_path

# EVOLVE-BLOCK-END
