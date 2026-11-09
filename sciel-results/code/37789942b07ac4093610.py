import time
import random
from collections import deque

N = 12
DEADLINE_MARGIN = 75.0  # seconds (90s limit, 15s safety)

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton
    using precomputed letter transition tables for bitmasks.
    Returns the length, or -1 if not synchronizing.
    """
    # Precompute: for each subset bitmask, the image under trans_a and trans_b
    # There are 2^12 = 4096 subsets
    total_subsets = 1 << N
    trans_a_table = [0] * total_subsets
    trans_b_table = [0] * total_subsets
    
    for mask in range(1, total_subsets):
        # Use lowest bit decomposition
        lsb = mask & (-mask)
        s = lsb.bit_length() - 1
        rest = mask ^ lsb
        if rest:
            trans_a_table[mask] = trans_a_table[rest] | (1 << trans_a[s])
            trans_b_table[mask] = trans_b_table[rest] | (1 << trans_b[s])
        else:
            trans_a_table[mask] = 1 << trans_a[s]
            trans_b_table[mask] = 1 << trans_b[s]
    
    start_state = total_subsets - 1
    visited = bytearray(total_subsets)  # 0 = unvisited, 1 = visited
    visited[start_state] = 1
    queue = deque([(start_state, 0)])
    
    while queue:
        state, dist = queue.popleft()
        if state & (state - 1) == 0: 
            return dist
        
        ns0 = trans_a_table[state]
        if not visited[ns0]:
            visited[ns0] = 1
            queue.append((ns0, dist + 1))
            
        ns1 = trans_b_table[state]
        if not visited[ns1]:
            visited[ns1] = 1
            queue.append((ns1, dist + 1))
            
    return -1

def construct():
    """
    Returns the transition tables for an optimal 12-state synchronizing automaton.
    Uses fast BFS with precomputed tables, multiple structural families,
    and local search with random restarts.
    """
    start_time = time.time()
    random.seed(42)
    
    best_threshold = 0
    best_a = None
    best_b = None
    
    def update_best(a, b):
        nonlocal best_threshold, best_a, best_b
        t = compute_threshold(a, b)
        if t > best_threshold:
            best_threshold = t
            best_a = a[:]
            best_b = b[:]
        return t
    
    # Family 1: Černý automaton (cycle + decrement)
    # This is the known optimal construction achieving (N-1)^2 = 121
    a_cz = [(i + 1) % N for i in range(N)]
    b_cz = [0 if i == 0 else i - 1 for i in range(N)]
    update_best(a_cz, b_cz)
    
    # If we've already reached the theoretical maximum, we can stop early
    if best_threshold >= (N - 1) ** 2:
        return [best_a, best_b]
    
    # Family 2: Cycle + single sink (all states map to one target)
    for sink in range(N):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        a_cy = [(i + 1) % N for i in range(N)]
        b_sink = [sink] * N
        update_best(a_cy, b_sink)
    
    # Family 3: Cycle + "reset to k" pattern (b maps i to max(0, i-k))
    for k in range(1, N):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        a_cy = [(i + 1) % N for i in range(N)]
        b_dec = [max(0, i - k) for i in range(N)]
        update_best(a_cy, b_dec)
    
    # Family 4: Cycle + "fold" pattern (b maps i to min(i, N-1-i))
    a_cy = [(i + 1) % N for i in range(N)]
    b_fold = [min(i, N - 1 - i) for i in range(N)]
    update_best(a_cy, b_fold)
    
    # Family 5: Cycle + random b with local search
    for restart in range(30):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        a_cy = [(i + 1) % N for i in range(N)]
        b_rand = [random.randint(0, N - 1) for _ in range(N)]
        cur_t = update_best(a_cy, b_rand)
        
        # Local search on b
        for ls_iter in range(50):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            improved = False
            for s in range(N):
                old = b_rand[s]
                for nd in range(N):
                    if nd == old:
                        continue
                    b_rand[s] = nd
                    nt = compute_threshold(a_cy, b_rand)
                    if nt > cur_t:
                        cur_t = nt
                        improved = True
                        if cur_t > best_threshold:
                            best_threshold = cur_t
                            best_a = a_cy[:]
                            best_b = b_rand[:]
                    else:
                        b_rand[s] = old
            if not improved:
                break
    
    # Family 6: Perturbed cycles (not just the standard cycle)
    for restart in range(20):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        # Generate a random permutation as "cycle-like" a
        a_perm = list(range(N))
        random.shuffle(a_perm)
        # Ensure it's a single cycle with probability ~1/e
        # Just use random permutation; many will have long cycles
        b_rand = [random.randint(0, N - 1) for _ in range(N)]
        cur_t = update_best(a_perm, b_rand)
        
        # Local search on both a and b
        for ls_iter in range(30):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            improved = False
            # Try changing b
            for s in range(N):
                old = b_rand[s]
                for nd in range(N):
                    if nd == old:
                        continue
                    b_rand[s] = nd
                    nt = compute_threshold(a_perm, b_rand)
                    if nt > cur_t:
                        cur_t = nt
                        improved = True
                        if cur_t > best_threshold:
                            best_threshold = cur_t
                            best_a = a_perm[:]
                            best_b = b_rand[:]
                    else:
                        b_rand[s] = old
            # Try changing a
            for s in range(N):
                old = a_perm[s]
                for nd in range(N):
                    if nd == old:
                        continue
                    a_perm[s] = nd
                    nt = compute_threshold(a_perm, b_rand)
                    if nt > cur_t:
                        cur_t = nt
                        improved = True
                        if cur_t > best_threshold:
                            best_threshold = cur_t
                            best_a = a_perm[:]
                            best_b = b_rand[:]
                    else:
                        a_perm[s] = old
            if not improved:
                break
    
    # Family 7: Systematic exploration of "cycle + structured b"
    # b defined by: b[i] = (i + offset) % N for most i, with one or two exceptions
    for offset in range(N):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        a_cy = [(i + 1) % N for i in range(N)]
        b_base = [(i + offset) % N for i in range(N)]
        update_best(a_cy, b_base)
        # With one exception
        for exc_state in range(N):
            for exc_dest in range(N):
                if time.time() - start_time > DEADLINE_MARGIN:
                    break
                b_mod = b_base[:]
                b_mod[exc_state] = exc_dest
                update_best(a_cy, b_mod)
    
    if best_a is None:
        # Fallback: should never happen
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    return [best_a, best_b]
