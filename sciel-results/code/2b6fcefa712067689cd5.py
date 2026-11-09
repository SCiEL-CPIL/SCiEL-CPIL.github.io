"""Covering design C(v,k,t) — Optimized Greedy Set Cover with Randomized Restarts."""
import itertools
import time
import random

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 100.0   # 100s safety margin
    
    points = list(range(v))
    
    # 1. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets and map to indices
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 3. Precompute block -> t-subsets mapping
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
        
    # 4. Precompute t-subset -> blocks mapping
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    def run_greedy(randomize=False):
        block_gain = [len(subs) for subs in block_tsubs]
        is_covered = [0] * N_t
        uncovered_count = N_t
        selected_blocks = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            best_block_idx = -1
            max_gain = -1
            candidates = []
            
            if randomize:
                # Find max gain, then collect all blocks with that gain
                # To speed up, we can just find max and then scan for ties?
                # Or just pick randomly from top K.
                # For efficiency, let's find the max gain first.
                for i in range(N_blocks):
                    if block_gain[i] > max_gain:
                        max_gain = block_gain[i]
                        candidates = [i]
                    elif block_gain[i] == max_gain:
                        candidates.append(i)
                
                if not candidates:
                    break
                
                best_block_idx = random.choice(candidates)
            else:
                for i in range(N_blocks):
                    if block_gain[i] > max_gain:
                        max_gain = block_gain[i]
                        best_block_idx = i
            
            if best_block_idx == -1 or max_gain == 0:
                break
                
            selected_blocks.append(all_blocks[best_block_idx])
            
            for si in block_tsubs[best_block_idx]:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        if not blocks:
            return blocks
        
        block_subs = []
        for b in blocks:
            subs = set()
            for s in itertools.combinations(b, t):
                subs.add(t_idx[s])
            block_subs.append(subs)
            
        n = len(blocks)
        cov = [0] * N_t
        for subs in block_subs:
            for si in subs:
                cov[si] += 1
                
        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_subs[i]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_subs[i]:
                        cov[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    # 1. Run deterministic greedy for a safe baseline
    sol_base = run_greedy(randomize=False)
    best_sol = sol_base
    best_len = len(sol_base)
    
    # 2. Run randomized restarts to potentially find a smaller cover
    # We have plenty of time. Let's run as many as possible.
    # Each greedy run is O(N_blocks * N_t) roughly in the worst case, but usually fast.
    # For v=19, k=8, t=5, N_blocks ~ 75582, N_t ~ 11628.
    # The inner loop is tight.
    
    attempts = 0
    while time.time() < DEADLINE - 5.0: # Leave 5s for reduction
        sol_rand = run_greedy(randomize=True)
        if len(sol_rand) < best_len:
            best_len = len(sol_rand)
            best_sol = sol_rand
            # If we find a very good one, we might want to try more
            # But if we hit the theoretical lower bound or close to it, we can stop?
            # No, just keep going until time runs out.
        attempts += 1
        
    # 3. Reduce the best solution found
    best_sol = try_reduce(best_sol)
    
    return best_sol
